"""Keyframe-capped propagation with anchor-prepended chunks (spec 5.3); the next chunk is read and
preprocessed while the current one runs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from ..engine.types import ChunkOOMError, Engine
from ..store import Clip, LabelObject, ProjectStore
from .errors import BadRequest, PropagationFailed
from .prefetch import ChunkPrefetcher
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
    lost_by_direction: dict[str, int] = field(default_factory=dict)

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


def _record_range(store: ProjectStore, clip_id: str, object_id: int, indices: list[int]) -> None:
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        obj = next(o for o in clip.objects if o.id == object_id)
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
    direction: str,
    *,
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    total: int,
    result: PropagateResult,
    emit: Emit,
    cancel: Callable[[], bool],
    prefetch: ChunkPrefetcher,
) -> int:
    """Run one direction's chunk loop.

    Returns the (possibly OOM-backed-off) chunk size actually used, so that a following
    direction in a "both" run starts from it instead of re-attempting a size that just OOM'd.
    """
    anchor_idx, anchor_mask = from_frame, seed
    pending = list(indices)
    size = max(min_chunk_size, chunk_size)
    empties, first_empty = 0, None

    while pending:
        # Deletes/restores stay allowed during a run: re-read every chunk so a frame restored
        # mid-run gets its mask written starting from the next chunk.
        deleted = set(store.read_clip(clip.id).deleted_frames)
        chunk = pending[:size]
        frames = prefetch.take(anchor_idx, chunk)
        # Predict the next chunk (this one completes and its last frame with a mask becomes the anchor) and
        # start reading and preprocessing it now, so that work overlaps this chunk's GPU time instead of
        # following it. A wrong prediction costs only the preprocessing: the prefetcher re-reads nothing it
        # already holds.
        if len(pending) > size:
            prefetch.schedule(chunk[-1], pending[size:2 * size])
        done_in_chunk: list[int] = []
        oom_error: ChunkOOMError | None = None
        direction_lost_frame: int | None = None
        try:
            with closing(engine.propagate(frames, anchor_mask, cancel_cb=cancel)) as masks:
                for offset, mask in masks:
                    if offset == 0:
                        continue  # the anchor's own output is discarded
                    expected = len(done_in_chunk) + 1
                    if offset != expected or offset > len(chunk):
                        raise PropagationFailed(f"engine yielded offset {offset}, expected {expected}")
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
                        direction_lost_frame = first_empty
                        break  # leaving the `with` closes the generator and frees the GPU lock
        except ChunkOOMError as e:
            oom_error = e
        finally:
            # Frames landed in this chunk must be recorded on EVERY exit -- normal end, lost,
            # cancel, OOM, or any other exception escaping the loop -- before it propagates.
            if done_in_chunk:
                _record_range(store, clip.id, object_id, done_in_chunk)
                pending = pending[len(done_in_chunk):]
        del frames

        if oom_error is not None:
            result.oom_retries += 1
            if size <= min_chunk_size:
                if not done_in_chunk:
                    raise PropagationFailed(
                        f"CUDA out of memory even at chunk size {size}; lower propagate.min_chunk_size "
                        "or free GPU memory (check nvidia-smi)"
                    ) from oom_error
                # Progress was made at the floor: keep the size, retry the remaining pending frames.
            else:
                size = max(min_chunk_size, size // 2)
            emit({"type": "log", "message": f"out of memory; retrying with chunk size {size}"})
            # Drop the traceback so the failed chunk's prepared tensors are freed before the retry
            # reads the next one -- it would otherwise keep the generator frame (and everything it
            # held) alive through prefetch.take().
            oom_error = None
            continue

        if direction_lost_frame is not None:
            result.lost_by_direction[direction] = direction_lost_frame
            break

        # Cancellation is latched from whether the engine actually stopped short of the chunk
        # while cancel() is true -- not from re-polling cancel() after a chunk that completed.
        # A cancel that arrives after a complete chunk with frames still pending is caught by
        # the next chunk instead (the engine yields nothing there -> stopped short).
        stopped_short = len(done_in_chunk) < len(chunk)
        if stopped_short:
            if cancel():
                result.cancelled = True
                break
            if done_in_chunk:
                raise PropagationFailed(
                    f"the engine stopped after {len(done_in_chunk)} of {len(chunk)} frames without being cancelled"
                )
            raise PropagationFailed("the engine returned no frames for a non-empty chunk")

    return size


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
    """Propagate the clip's single labeled object from `from_frame` in `direction`.

    `emit` and `cancel` are both called while the engine's `propagate()` generator is open, i.e.
    while the GPU lock is held. Both must return quickly: no blocking I/O, no blocking on a slow
    consumer, no raising, and no calling back into the engine -- a slow or misbehaving callback
    stalls the GPU lock for every other caller. (The jobs layer's `Job.emit` is a plain in-memory
    list append, which satisfies this.)
    """
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
    size = chunk_size
    with ChunkPrefetcher(store, clip, engine) as prefetch:
        for d, indices in zip(directions, plans):
            if result.cancelled:
                break
            size = _run_direction(
                store, engine, clip, obj.id, from_frame, seed, indices, d,
                chunk_size=size, min_chunk_size=min_chunk_size, lost_after=lost_after,
                total=total, result=result, emit=emit, cancel=cancel, prefetch=prefetch,
            )
    for d in directions:
        if d in result.lost_by_direction:
            result.lost_at = result.lost_by_direction[d]
            break
    return result
