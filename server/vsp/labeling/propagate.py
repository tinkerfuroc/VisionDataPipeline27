"""Keyframe-capped propagation of one or more objects with anchor-prepended chunks (spec 5.3, multi-object
spec §6); the next chunk is read and preprocessed while the current one runs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from ..engine.types import ChunkOOMError, Engine
from ..store import Clip, LabelObject, NotFound, ProjectStore
from .errors import BadRequest, PropagationFailed
from .prefetch import ChunkPrefetcher
from .ranges import add_range

Emit = Callable[[dict], None]


@dataclass
class PropagateResult:
    processed: int = 0  # frames covered by at least one object
    written: int = 0  # masks written, over all objects
    cleared: int = 0
    lost_at: int | None = None  # the frame the page seeks to (spec §6)
    cancelled: bool = False
    oom_retries: int = 0
    lost_by_direction: dict[str, dict[int, int]] = field(default_factory=dict)
    lost: dict[int, int] = field(default_factory=dict)
    skipped: list[int] = field(default_factory=list)
    objects: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def plan_indices(clip: Clip, obj: LabelObject, from_frame: int, direction: Literal["forward", "backward"]) -> list[int]:
    """Frames a run from `from_frame` covers: up to, not including, the next keyframe in `direction`.

    Any frame may be a start (a keyframe, or a propagated frame when a stopped run is resumed); whether it
    has a mask to seed from is checked by `resolve_participants`. Keyframes are never overwritten, so they
    bound the run.
    """
    if not 0 <= from_frame < clip.frame_count:
        raise BadRequest(f"frame {from_frame} is out of range 0..{clip.frame_count - 1}")
    keyframes = sorted(k.frame_idx for k in obj.keyframes)
    if direction == "forward":
        stop = min((k for k in keyframes if k > from_frame), default=clip.frame_count)
        return list(range(from_frame + 1, stop))
    stop = max((k for k in keyframes if k < from_frame), default=-1)
    return list(range(from_frame - 1, stop, -1))


def resolve_participants(
    store: ProjectStore,
    clip: Clip,
    from_frame: int,
    objects: Literal["all"] | list[int],
    *,
    strict: bool,
) -> tuple[list[tuple[LabelObject, np.ndarray]], list[int]]:
    """The objects a run from `from_frame` propagates, each with its seed mask, and the ids skipped.

    `strict` (the route, validating a request): an unknown id is NotFound and a listed object without a mask on
    `from_frame` is BadRequest. Not strict (the job, which re-reads the clip): both are skipped instead, so an object
    deleted between the request and the job's start does not fail the run.
    """
    if not 0 <= from_frame < clip.frame_count:
        raise BadRequest(f"frame {from_frame} is out of range 0..{clip.frame_count - 1}")
    if from_frame in clip.deleted_frames:
        raise BadRequest(f"frame {from_frame} is deleted; restore it or propagate from another frame")
    by_id = {o.id: o for o in clip.objects}
    explicit = objects != "all"
    wanted = list(by_id) if not explicit else list(dict.fromkeys(objects))
    participants: list[tuple[LabelObject, np.ndarray]] = []
    skipped: list[int] = []
    for oid in wanted:
        obj = by_id.get(oid)
        if obj is None:
            if strict:
                raise NotFound(f"object {oid} not found in clip {clip.id}")
            skipped.append(oid)
            continue
        seed = store.read_mask(clip.id, oid, from_frame)
        if seed is None:
            if strict and explicit:
                raise BadRequest(
                    f"object {oid} has no mask on frame {from_frame}; propagate it from a frame where it has one"
                )
            skipped.append(oid)
            continue
        participants.append((obj, seed))
    if not participants:
        if not clip.objects:
            raise BadRequest("nothing to propagate: accept a prompt first")
        raise BadRequest(f"frame {from_frame} has no mask; propagate from a masked frame or prompt here first")
    return participants, skipped


def _record_range(store: ProjectStore, clip_id: str, object_id: int, indices: list[int]) -> None:
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        obj = next((o for o in clip.objects if o.id == object_id), None)
        if obj is None:  # deleted while the run was going (the API forbids it, but never crash the run on it)
            return
        obj.propagated_ranges = add_range(obj.propagated_ranges, min(indices), max(indices))
        store.write_clip(clip)


@dataclass
class _Track:
    """One object's progress through one direction's frame list."""

    obj_id: int
    limit: int  # how many frames of the direction's list this object may cover (up to its next keyframe)
    anchor_idx: int  # its last frame with a mask: the chunk it runs in is seeded there
    anchor_mask: np.ndarray
    pos: int = 0  # index of its next frame in the direction's list
    empties: int = 0
    first_empty: int | None = None
    finished: bool = False  # lost

    @property
    def active(self) -> bool:
        return not self.finished and self.pos < self.limit


def _run_direction(
    store: ProjectStore,
    engine: Engine,
    clip: Clip,
    tracks: list[_Track],
    frames_list: list[int],
    direction: str,
    *,
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    base: int,
    total: int,
    result: PropagateResult,
    emit: Emit,
    cancel: Callable[[], bool],
    prefetch: ChunkPrefetcher,
) -> int:
    """Run one direction: engine calls over chunks, one per group of objects sharing an anchor frame.

    Each object keeps the single-object rule of old: it is seeded on its own last masked frame, which is prepended
    to the chunk. Objects whose anchors (and positions) coincide share one engine call -- the usual case; an object
    that missed a frame falls behind into its own group until it is found again or lost. Returns the (possibly
    OOM-backed-off) chunk size, so a following direction starts from it.
    """
    size = max(min_chunk_size, chunk_size)
    while True:
        live = [t for t in tracks if t.active]
        if not live:
            return size
        groups: dict[tuple[int, int], list[_Track]] = {}
        for t in live:
            groups.setdefault((t.pos, t.anchor_idx), []).append(t)
        (pos, anchor_idx), members = min(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1])))
        end = max(t.limit for t in members)
        chunk = frames_list[pos:min(pos + size, end)]
        # Deletes/restores stay allowed during a run: re-read every chunk so a frame restored mid-run gets its mask
        # written starting from the next chunk.
        deleted = set(store.read_clip(clip.id).deleted_frames)
        frames = prefetch.take(anchor_idx, chunk)
        # Predict this group's next chunk and start reading it now (see ChunkPrefetcher: a wrong prediction -- an
        # OOM back-off, another group running next, an anchor that is not the chunk's last frame -- only costs a
        # rebuild from the frames the prefetcher still holds).
        after = frames_list[pos + size:min(pos + 2 * size, end)]
        others = [t for t in live if t not in members]
        # only when this group will run next: no other group may be left behind this chunk's end
        if after and all(t.pos >= pos + len(chunk) for t in others):
            prefetch.schedule(chunk[-1], after)
        seeds = {t.obj_id: t.anchor_mask for t in members}
        covered: dict[int, list[int]] = {t.obj_id: [] for t in members}
        yielded = 0
        oom_error: ChunkOOMError | None = None
        try:
            with closing(engine.propagate(frames, seeds, cancel_cb=cancel)) as outputs:
                for offset, masks in outputs:
                    if offset == 0:
                        continue  # the anchor's own output is discarded
                    if offset != yielded + 1 or offset > len(chunk):
                        raise PropagationFailed(f"engine yielded offset {offset}, expected {yielded + 1}")
                    if set(masks) != set(seeds):
                        raise PropagationFailed(f"engine yielded objects {sorted(masks)}, expected {sorted(seeds)}")
                    yielded = offset
                    idx = chunk[offset - 1]
                    report = []
                    for t in members:
                        if not t.active:
                            continue  # bounded or lost earlier in this chunk: its outputs are ignored
                        mask = masks[t.obj_id]
                        t.pos += 1
                        covered[t.obj_id].append(idx)
                        if mask is None:
                            if store.delete_mask(clip.id, t.obj_id, idx):
                                result.cleared += 1
                            t.empties += 1
                            t.first_empty = idx if t.first_empty is None else t.first_empty
                        else:
                            t.empties, t.first_empty = 0, None
                            t.anchor_idx, t.anchor_mask = idx, mask
                            if idx not in deleted:
                                store.write_mask(clip.id, t.obj_id, idx, mask)
                                result.written += 1
                        report.append({"id": t.obj_id, "has_mask": mask is not None and idx not in deleted})
                        if t.empties >= lost_after:
                            t.finished = True
                            result.lost_by_direction.setdefault(direction, {})[t.obj_id] = t.first_empty
                    done = base + max(t.pos for t in tracks)
                    emit({
                        "type": "frame", "frame": idx, "objects": report,
                        # legacy shim: removed in plan B -- covers only `report`, i.e. the objects of *this*
                        # engine call (`members`) that emitted the event; an object in a lagging group not yet
                        # at `idx`, or one already bounded/lost, is left out of both `report` and this OR.
                        # Fine for the single-object legacy web app, which only ever has one object to ask about.
                        "has_mask": any(r["has_mask"] for r in report),
                        "done": done, "total": total,
                    })
                    if not any(t.active for t in members):
                        break  # leaving the `with` closes the generator and frees the GPU lock
        except ChunkOOMError as e:
            oom_error = e
        finally:
            # Frames landed in this call must be recorded on EVERY exit -- normal end, lost, cancel, OOM, or any
            # other exception escaping the loop -- before it propagates.
            for oid, idxs in covered.items():
                if idxs:
                    _record_range(store, clip.id, oid, idxs)
        del frames

        if oom_error is not None:
            result.oom_retries += 1
            if size <= min_chunk_size:
                if yielded == 0:
                    raise PropagationFailed(
                        f"CUDA out of memory even at chunk size {size}; lower propagate.min_chunk_size "
                        "or free GPU memory (check nvidia-smi)"
                    ) from oom_error
            else:
                size = max(min_chunk_size, size // 2)
            emit({"type": "log", "message": f"out of memory; retrying with chunk size {size}"})
            oom_error = None  # drop the traceback so the failed chunk's tensors are freed before the retry
            continue

        # Cancellation is latched from whether the engine actually stopped short of the chunk while cancel() is
        # true -- not from re-polling cancel() after a call that completed (see the single-object history).
        if yielded < len(chunk) and any(t.active for t in members):
            if cancel():
                result.cancelled = True
                return size
            if yielded:
                raise PropagationFailed(
                    f"the engine stopped after {yielded} of {len(chunk)} frames without being cancelled"
                )
            raise PropagationFailed("the engine returned no frames for a non-empty chunk")


def run_propagation(
    store: ProjectStore,
    engine: Engine,
    clip_id: str,
    from_frame: int,
    direction: Literal["forward", "backward", "both"],
    *,
    objects: Literal["all"] | list[int] = "all",
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    emit: Emit | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PropagateResult:
    """Propagate `objects` (default: every object with a mask on `from_frame`) from `from_frame` in `direction`.

    `emit` and `cancel` are both called while the engine's `propagate()` generator is open, i.e.
    while the GPU lock is held. Both must return quickly: no blocking I/O, no blocking on a slow
    consumer, no raising, and no calling back into the engine -- a slow or misbehaving callback
    stalls the GPU lock for every other caller. (The jobs layer's `Job.emit` is a plain in-memory
    list append, which satisfies this.)
    """
    emit = emit or (lambda event: None)
    cancel = cancel or (lambda: False)
    clip = store.read_clip(clip_id)
    participants, skipped = resolve_participants(store, clip, from_frame, objects, strict=False)
    directions = ["forward", "backward"] if direction == "both" else [direction]
    plans = {d: {obj.id: plan_indices(clip, obj, from_frame, d) for obj, _ in participants} for d in directions}
    frames_lists = {d: max(plans[d].values(), key=len) for d in directions}

    result = PropagateResult(skipped=skipped, objects=[obj.id for obj, _ in participants])
    total = sum(len(f) for f in frames_lists.values())
    emit({"type": "progress", "done": 0, "total": total})
    size = chunk_size
    base = 0
    with ChunkPrefetcher(store, clip, engine) as prefetch:
        for d in directions:
            if result.cancelled:
                break
            tracks = [_Track(obj.id, len(plans[d][obj.id]), from_frame, seed) for obj, seed in participants]
            size = _run_direction(
                store, engine, clip, tracks, frames_lists[d], d,
                chunk_size=size, min_chunk_size=min_chunk_size, lost_after=lost_after,
                base=base, total=total, result=result, emit=emit, cancel=cancel, prefetch=prefetch,
            )
            covered = max((t.pos for t in tracks), default=0)
            result.processed += covered
            base += covered
    for d in directions:
        for oid, frame in result.lost_by_direction.get(d, {}).items():
            result.lost.setdefault(oid, frame)
    for d in directions:
        lost_here = result.lost_by_direction.get(d)
        if lost_here:
            result.lost_at = min(lost_here.values(), key=lambda f: abs(f - from_frame))
            break
    return result
