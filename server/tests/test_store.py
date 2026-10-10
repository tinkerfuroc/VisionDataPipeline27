import os
import threading

import numpy as np
import pytest
from vdp.labeling.prompts import create_object
from vdp.store import (
    Clip,
    ClipSource,
    Keyframe,
    LabelObject,
    NotFound,
    ProjectStore,
    Prompt,
    StoreError,
    atomic_write_bytes,
    encode_mask_png,
    now_iso,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "data")


def _clip(store, project_id, frame_count=5):
    clip_id, _cdir = store.new_clip_dir(project_id)
    clip = Clip(
        id=clip_id, project_id=project_id, source=ClipSource(type="video", name="a.mp4"),
        fps=30.0, stride=1, width=64, height=48, frame_count=frame_count, created_at="2026-09-20T00:00:00+00:00",
    )
    store.write_clip(clip)
    return clip


def test_project_roundtrip_and_classes(store):
    p = store.create_project("Board A")
    assert p.id.startswith("board-a-")
    c1 = store.add_class(p.id, "connector")
    c2 = store.add_class(p.id, "capacitor")
    assert (c1.id, c2.id) == (0, 1)
    assert c1.color != c2.color
    again = store.read_project(p.id)
    assert [c.name for c in again.classes] == ["connector", "capacitor"]
    assert [x.id for x in store.list_projects()] == [p.id]


def test_duplicate_class_name_rejected(store):
    p = store.create_project("p")
    store.add_class(p.id, "connector")
    with pytest.raises(StoreError, match="already exists"):
        store.add_class(p.id, "connector")


def test_add_class_is_atomic_under_concurrent_writers(store):
    p = store.create_project("p")
    names = [f"c{i}" for i in range(8)]
    barrier = threading.Barrier(len(names))
    results: list[object] = [None] * len(names)

    def worker(i: int) -> None:
        barrier.wait(5)
        results[i] = store.add_class(p.id, names[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(names))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert all(results)  # every worker returned a ClassDef; a raised exception would leave a None
    classes = store.read_project(p.id).classes
    assert len(classes) == 8
    assert len({c.id for c in classes}) == 8
    assert sorted(c.name for c in classes) == sorted(names)


def test_set_class_color_persists_and_lower_cases(store):
    p = store.create_project("p")
    c = store.add_class(p.id, "connector")
    updated = store.set_class_color(p.id, c.id, "#ABCDEF")
    assert updated.color == "#abcdef"
    again = store.read_project(p.id)
    assert again.classes[0].color == "#abcdef"
    # a colour change only reaches NEW exports' mask PNG palette (mask_png.py reads c.color from the live
    # project); an export already on disk carries its own export.json classes and is untouched by this.


def test_set_class_color_rejects_a_bad_hex_string(store):
    p = store.create_project("p")
    c = store.add_class(p.id, "connector")
    with pytest.raises(StoreError, match="invalid colour"):
        store.set_class_color(p.id, c.id, "red")
    with pytest.raises(StoreError, match="invalid colour"):
        store.set_class_color(p.id, c.id, "#abc")


def test_set_class_color_unknown_class_or_project(store):
    p = store.create_project("p")
    c = store.add_class(p.id, "connector")
    with pytest.raises(NotFound):
        store.set_class_color(p.id, c.id + 1, "#ffffff")
    with pytest.raises(NotFound):
        store.set_class_color("nope-000000", c.id, "#ffffff")


def test_clip_roundtrip_preserves_int_keys(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    clip.original_names = {0: "frame_1.png", 1: "frame_10.png"}
    clip.deleted_frames = [3]
    clip.objects = [LabelObject(id=1, class_id=0, keyframes=[
        Keyframe(frame_idx=2, prompt=Prompt(type="box", payload={"xyxy": [1, 2, 3, 4]}), created_at="t")
    ], propagated_ranges=[(2, 4)])]
    store.write_clip(clip)
    got = store.read_clip(clip.id)
    assert got.original_names == {0: "frame_1.png", 1: "frame_10.png"}
    assert got.objects[0].propagated_ranges == [(2, 4)]
    assert got.objects[0].keyframes[0].prompt.payload == {"xyxy": [1, 2, 3, 4]}
    assert [c.id for c in store.list_clips(p.id)] == [clip.id]


def test_clip_lookup_by_id_and_not_found(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    assert store.clip_dir(clip.id).name == clip.id
    with pytest.raises(NotFound):
        store.read_clip("doesnotexist")


def test_list_clips_skips_half_imported_dirs(store):
    p = store.create_project("p")
    store.new_clip_dir(p.id)  # directory exists, clip.json not yet written
    assert store.list_clips(p.id) == []


def test_mask_roundtrip_and_absent(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 5:30] = True
    store.write_mask(clip.id, 1, 2, mask)
    got = store.read_mask(clip.id, 1, 2)
    assert got.dtype == bool and got.shape == (48, 64)
    assert np.array_equal(got, mask)
    assert store.read_mask(clip.id, 1, 3) is None
    assert store.mask_indices(clip.id, 1) == [2]
    assert store.delete_mask(clip.id, 1, 2) is True
    assert store.delete_mask(clip.id, 1, 2) is False


def test_empty_mask_is_never_written(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    with pytest.raises(StoreError, match="empty mask"):
        store.write_mask(clip.id, 1, 0, np.zeros((48, 64), dtype=bool))
    assert store.mask_indices(clip.id, 1) == []


def test_frame_states_priority(store):
    p = store.create_project("p")
    clip = _clip(store, p.id, frame_count=5)
    m = np.ones((48, 64), dtype=bool)
    for i in (1, 2, 3):
        store.write_mask(clip.id, 1, i, m)
    clip.objects = [LabelObject(id=1, class_id=0, keyframes=[
        Keyframe(frame_idx=1, prompt=Prompt(type="clicks", payload={}), created_at="t")])]
    clip.deleted_frames = [3]
    assert store.frame_states(clip) == ["empty", "keyframe", "masked", "deleted", "empty"]


def test_atomic_write_leaves_nothing_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "x" / "clip.json"
    atomic_write_bytes(target, b"old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert [f.name for f in target.parent.iterdir()] == ["clip.json"]  # no stray temp file


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "UPPER", ".hidden"])
def test_path_traversal_ids_rejected(store, bad):
    with pytest.raises(StoreError):
        store.read_project(bad)
    with pytest.raises(StoreError):
        store.clip_dir(bad)


def test_delete_clip(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    store.delete_clip(clip.id)
    with pytest.raises(NotFound):
        store.read_clip(clip.id)


def test_encode_mask_png_matches_what_write_mask_stores(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    mask = np.zeros((48, 64), dtype=bool)
    mask[5:15, 8:20] = True
    data = encode_mask_png(mask)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    store.write_mask(clip.id, 1, 0, mask)
    assert store.mask_path(clip.id, 1, 0).read_bytes() == data
    with pytest.raises(StoreError, match="empty mask"):
        encode_mask_png(np.zeros((4, 4), dtype=bool))


def test_object_states_are_per_object_and_deleted_frames_show_in_every_lane(store, make_clip):
    _, clip = make_clip(n_frames=6)
    a = create_object(clip, 0)
    b = create_object(clip, 0)
    clip.deleted_frames = [5]
    store.write_clip(clip)
    m = np.ones((60, 120), dtype=bool)
    store.write_mask(clip.id, a.id, 1, m)
    store.write_mask(clip.id, b.id, 2, m)
    store.write_mask(clip.id, b.id, 5, m)
    b.keyframes = [Keyframe(frame_idx=2, prompt=Prompt(type="paint", payload={}), created_at=now_iso())]
    assert store.object_states(clip, a) == ["empty", "masked", "empty", "empty", "empty", "deleted"]
    assert store.object_states(clip, b) == ["empty", "empty", "keyframe", "empty", "empty", "deleted"]
    assert store.frame_states(clip) == ["empty", "masked", "keyframe", "empty", "empty", "deleted"]


def test_delete_object_masks_removes_only_that_object(store, make_clip):
    _, clip = make_clip(n_frames=3)
    m = np.ones((60, 120), dtype=bool)
    store.write_mask(clip.id, 1, 0, m)
    store.write_mask(clip.id, 2, 0, m)
    store.delete_object_masks(clip.id, 2)
    assert store.mask_indices(clip.id, 1) == [0] and store.mask_indices(clip.id, 2) == []
    store.delete_object_masks(clip.id, 7)  # nothing there: no error
