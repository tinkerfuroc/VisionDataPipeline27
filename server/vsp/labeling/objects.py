"""Object-level operations: find, relabel, delete (multi-object spec §7)."""

from __future__ import annotations

from ..store import Clip, LabelObject, NotFound, ProjectStore
from .errors import BadRequest


def find_object(clip: Clip, object_id: int) -> LabelObject:
    for obj in clip.objects:
        if obj.id == object_id:
            return obj
    raise NotFound(f"object {object_id} not found in clip {clip.id}")


def check_class(store: ProjectStore, clip: Clip, class_id: int) -> None:
    if class_id not in {c.id for c in store.read_project(clip.project_id).classes}:
        raise BadRequest(f"unknown class id {class_id}")


def resolve_target(
    store: ProjectStore, clip: Clip, object_id: int | None, class_id: int | None
) -> LabelObject:
    """The object a write targets: `object_id=None` creates one (needs `class_id`); an int writes to
    that object (and `class_id` must then be None -- relabel it instead)."""
    from .prompts import create_object  # local: prompts.py imports resolve_target from here

    if object_id is None:
        if class_id is None:
            raise BadRequest("class_id is required to create a new object")
        check_class(store, clip, class_id)
        return create_object(clip, class_id)
    obj = find_object(clip, object_id)
    if class_id is not None:
        raise BadRequest("class_id cannot be given with object_id; relabel the object instead")
    return obj


def relabel_object(store: ProjectStore, clip_id: str, object_id: int, class_id: int) -> Clip:
    store.clip_dir(clip_id)  # raises NotFound before the lock table grows an entry for a bogus id
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        obj = find_object(clip, object_id)
        check_class(store, clip, class_id)
        obj.class_id = class_id
        store.write_clip(clip)
        return clip


def delete_object(store: ProjectStore, clip_id: str, object_id: int) -> Clip:
    store.clip_dir(clip_id)  # raises NotFound before the lock table grows an entry for a bogus id
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        find_object(clip, object_id)
        clip.objects = [o for o in clip.objects if o.id != object_id]
        store.write_clip(clip)  # the entry goes first: a crash between the two leaves orphan files, not a
        store.delete_object_masks(clip_id, object_id)  # listed object without masks
        return clip
