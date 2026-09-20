"""Keyframe-capped propagation with anchor-prepended chunks (spec 5.3)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from ..engine.types import ChunkOOMError, Engine
from ..store import Clip, LabelObject, ProjectStore
from .errors import BadRequest, PropagationFailed
from .prompts import load_frame
from .ranges import add_range

Emit = Callable[[dict], None]


@dataclass
class PropagateResult:
    processed: int = 0
    written: int = 0
    cleared: int = 0
    lost_at: int | None = None
    cancelled: bool = False
    oom_retries: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def plan_indices(clip: Clip, obj: LabelObject, from_frame: int, direction: Literal["forward", "backward"]) -> list[int]:
    keyframes = sorted(k.frame_idx for k in obj.keyframes)
    if from_frame not in keyframes:
        raise BadRequest(f"frame {from_frame} is not a keyframe; accept a prompt there first")
    if direction == "forward":
        stop = min((k for k in keyframes if k > from_frame), default=clip.frame_count)
        return list(range(from_frame + 1, stop))
    stop = max((k for k in keyframes if k < from_frame), default=-1)
    return list(range(from_frame - 1, stop, -1))


def _record_range(store: ProjectStore, clip_id: str, indices: list[int]) -> None:
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        obj = clip.objects[0]
        obj.propagated_ranges = add_range(obj.propagated_ranges, min(indices), max(indices))
        store.write_clip(clip)


def _run_direction(
    store: ProjectStore,
    engine: Engine,
    clip: Clip,
    object_id: int,
    from_frame: int,
    seed: np.ndarray,
    indices: list[int],
    *,
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    total: int,
    result: PropagateResult,
    emit: Emit,
    cancel: Callable[[], bool],
) -> None:
    deleted = set(clip.deleted_frames)
    anchor_idx, anchor_mask = from_frame, seed
    pending = list(indices)
    size = max(min_chunk_size, chunk_size)
    empties, first_empty = 0, None

    while pending and not result.cancelled and result.lost_at is None:
        chunk = pending[:size]
        frames = [load_frame(store, clip, anchor_idx)] + [load_frame(store, clip, i) for i in chunk]
        done_in_chunk: list[int] = []
        oom = False
        try:
            with closing(engine.propagate(frames, anchor_mask, cancel_cb=cancel)) as masks:
                for offset, mask in masks:
                    if offset == 0:
                        continue  # the anchor's own output is discarded
                    idx = chunk[offset - 1]
                    done_in_chunk.append(idx)
                    result.processed += 1
                    if mask is None:
                        if store.delete_mask(clip.id, object_id, idx):
                            result.cleared += 1
                        empties += 1
                        first_empty = idx if first_empty is None else first_empty
                    else:
                        empties, first_empty = 0, None
                        anchor_idx, anchor_mask = idx, mask
                        if idx not in deleted:
                            store.write_mask(clip.id, object_id, idx, mask)
                            result.written += 1
                    emit({
                        "type": "frame", "frame": idx,
                        "has_mask": mask is not None and idx not in deleted,
                        "done": result.processed, "total": total,
                    })
                    if empties >= lost_after:
                        result.lost_at = first_empty
                        break  # leaving the `with` closes the generator and frees the GPU lock
        except ChunkOOMError as e:
            oom = True
            if size <= min_chunk_size:
                raise PropagationFailed(
                    f"CUDA out of memory even at chunk size {size}; lower propagate.min_chunk_size "
                    "or free GPU memory (check nvidia-smi)"
                ) from e
            size = max(min_chunk_size, size // 2)
            result.oom_retries += 1
            emit({"type": "log", "message": f"out of memory; retrying with chunk size {size}"})

        if done_in_chunk:
            _record_range(store, clip.id, done_in_chunk)
            pending = pending[len(done_in_chunk):]
        if cancel():
            result.cancelled = True
        elif not done_in_chunk and not oom:
            raise PropagationFailed("the engine returned no frames for a non-empty chunk")


def run_propagation(
    store: ProjectStore,
    engine: Engine,
    clip_id: str,
    from_frame: int,
    direction: Literal["forward", "backward", "both"],
    *,
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    emit: Emit | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PropagateResult:
    emit = emit or (lambda event: None)
    cancel = cancel or (lambda: False)
    clip = store.read_clip(clip_id)
    if not clip.objects:
        raise BadRequest("nothing to propagate: accept a prompt first")
    obj = clip.objects[0]
    directions = ["forward", "backward"] if direction == "both" else [direction]
    plans = [plan_indices(clip, obj, from_frame, d) for d in directions]
    seed = store.read_mask(clip_id, obj.id, from_frame)
    if seed is None:
        raise BadRequest(f"keyframe {from_frame} has no mask on disk")

    result = PropagateResult()
    total = sum(len(p) for p in plans)
    emit({"type": "progress", "done": 0, "total": total})
    for indices in plans:
        if result.cancelled:
            break
        result.lost_at = None  # a lost forward pass must not suppress the backward pass
        _run_direction(
            store, engine, clip, obj.id, from_frame, seed, indices,
            chunk_size=chunk_size, min_chunk_size=min_chunk_size, lost_after=lost_after,
            total=total, result=result, emit=emit, cancel=cancel,
        )
    return result
