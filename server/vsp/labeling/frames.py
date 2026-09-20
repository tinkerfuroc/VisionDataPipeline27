"""Frame cleanup (spec 5.4). Deleting a frame only flags it; no file is ever removed."""

from __future__ import annotations

from ..store import Clip, ProjectStore
from .errors import BadRequest
from .ranges import subtract_range


def _check(clip: Clip, idx: int) -> None:
    if not 0 <= idx < clip.frame_count:
        raise BadRequest(f"frame {idx} is out of range 0..{clip.frame_count - 1}")


def _set_deleted(store: ProjectStore, clip_id: str, indices: range, deleted: bool) -> Clip:
    store.clip_dir(clip_id)  # raises NotFound before the lock table grows an entry for a bogus id
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        for idx in (indices.start, indices.stop - 1):
            _check(clip, idx)
        current = set(clip.deleted_frames)
        current = current | set(indices) if deleted else current - set(indices)
        clip.deleted_frames = sorted(current)
        store.write_clip(clip)
        return clip


def delete_frame(store: ProjectStore, clip_id: str, idx: int) -> Clip:
    return _set_deleted(store, clip_id, range(idx, idx + 1), True)


def restore_frame(store: ProjectStore, clip_id: str, idx: int) -> Clip:
    return _set_deleted(store, clip_id, range(idx, idx + 1), False)


def prune_from(store: ProjectStore, clip_id: str, from_idx: int) -> Clip:
    clip = store.read_clip(clip_id)
    _check(clip, from_idx)
    return _set_deleted(store, clip_id, range(from_idx, clip.frame_count), True)


def clear_range(store: ProjectStore, clip_id: str, start: int, end: int) -> Clip:
    store.clip_dir(clip_id)  # raises NotFound before the lock table grows an entry for a bogus id
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        _check(clip, start)
        _check(clip, end)
        if start > end:
            raise BadRequest(f"start {start} is after end {end}")
        for obj in clip.objects:
            keyframes = {k.frame_idx for k in obj.keyframes}
            for idx in range(start, end + 1):
                if idx not in keyframes:
                    store.delete_mask(clip.id, obj.id, idx)
            obj.propagated_ranges = subtract_range(obj.propagated_ranges, start, end)
        store.write_clip(clip)
        return clip
