from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest
from vdp.export.plan import (
    ExportOptions,
    build_plan,
    check_request,
    classes_from_manifest,
    options_from_manifest,
)
from vdp.ingest import FrameDirSource, import_clip
from vdp.labeling.errors import BadRequest
from vdp.store.models import LabelObject


def _mask(x0=10, size=20):
    m = np.zeros((60, 120), bool)
    m[10 : 10 + size, x0 : x0 + size] = True
    return m


def _label(store, clip, frames, class_id=0, size=20):
    clip.objects = [LabelObject(id=1, class_id=class_id)]
    store.write_clip(clip)
    for i in frames:
        store.write_mask(clip.id, 1, i, _mask(size=size))
    return store.read_clip(clip.id)


def _extra_clip(store, project_id, tmp_path, n=6):
    src = tmp_path / f"more_{uuid.uuid4().hex[:6]}"
    src.mkdir()
    for i in range(n):
        assert cv2.imwrite(str(src / f"{i:04d}.png"), np.full((60, 120, 3), 90, np.uint8))
    return import_clip(store, project_id, FrameDirSource(src))


def test_contiguous_split_puts_the_tail_of_the_clip_in_val(store, make_clip):
    project, clip = make_clip(n_frames=10)
    _label(store, clip, range(10))
    plan = build_plan(store, project.id, [clip.id], ExportOptions())
    assert plan.options.split_mode == "contiguous"
    assert [f.frame_idx for f in plan.frames if f.split == "val"] == [8, 9]
    assert [f.frame_idx for f in plan.frames if f.split == "train"] == list(range(8))
    assert plan.frames[0].stem == f"{clip.id}_0000"  # the original file stem of a frame-directory clip
    assert plan.frames[0].objects[0].class_id == 0


def test_deleted_frames_are_left_out_and_stride_thins_what_remains(store, make_clip):
    project, clip = make_clip(n_frames=10)
    clip = _label(store, clip, range(10))
    clip.deleted_frames = [1, 2]
    store.write_clip(clip)
    plan = build_plan(store, project.id, [clip.id], ExportOptions(stride=3, val_fraction=0.0))
    assert [f.frame_idx for f in plan.frames] == [0, 5, 8]  # kept = 0,3,4,5,6,7,8,9 -> every third
    assert {f.split for f in plan.frames} == {"train"}


def test_include_empty_keeps_or_drops_frames_without_a_usable_mask(store, make_clip):
    project, clip = make_clip(n_frames=6)
    clip = _label(store, clip, [0, 1, 2])
    store.write_mask(clip.id, 1, 3, _mask(size=5))  # 25 px: below min_area_px, so "absent"
    keep = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0))
    assert [len(f.objects) for f in keep.frames] == [1, 1, 1, 0, 0, 0]
    drop = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, include_empty=False))
    assert [f.frame_idx for f in drop.frames] == [0, 1, 2]


def test_per_clip_split_never_shares_a_clip_between_train_and_val(store, make_clip, tmp_path):
    project, first = make_clip(n_frames=6)
    _label(store, first, range(6))
    others = [_extra_clip(store, project.id, tmp_path) for _ in range(2)]
    for c in others:
        _label(store, c, range(6))
    ids = [first.id, *(c.id for c in others)]
    plan = build_plan(store, project.id, ids, ExportOptions())
    assert plan.options.split_mode == "per_clip"
    assert len(plan.val_clip_ids) == 1  # ceil(0.2 * 3)
    by_clip = {cid: {f.split for f in plan.frames if f.clip_id == cid} for cid in ids}
    assert all(len(splits) == 1 for splits in by_clip.values())
    assert by_clip[plan.val_clip_ids[0]] == {"val"}
    chosen = build_plan(store, project.id, ids, ExportOptions(val_clip_ids=(first.id,)))
    assert {f.split for f in chosen.frames if f.clip_id == first.id} == {"val"}
    stems = [f.stem for f in chosen.frames]
    assert len(stems) == len(set(stems))


def test_invalid_requests_are_refused(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=4)
    _label(store, clip, range(4))
    _other_project, foreign = make_clip(n_frames=4)
    ok = ExportOptions()
    with pytest.raises(BadRequest, match="at least one clip"):
        check_request(store, project.id, [], ["coco"], ok)
    with pytest.raises(BadRequest, match="at least one format"):
        check_request(store, project.id, [clip.id], [], ok)
    with pytest.raises(BadRequest, match="unknown format"):
        check_request(store, project.id, [clip.id], ["voc"], ok)
    with pytest.raises(BadRequest, match="does not belong"):
        check_request(store, project.id, [foreign.id], ["coco"], ok)
    with pytest.raises(BadRequest, match="at least two clips"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(split_mode="per_clip"))
    with pytest.raises(BadRequest, match="every selected clip"):
        second = _extra_clip(store, project.id, tmp_path)
        check_request(store, project.id, [clip.id, second.id], ["coco"], ExportOptions(val_clip_ids=(clip.id, second.id)))
    with pytest.raises(BadRequest, match="per-clip split only"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(val_clip_ids=(clip.id,)))
    with pytest.raises(BadRequest, match="nothing to export"):
        build_plan(store, project.id, [clip.id], ExportOptions(include_empty=False, min_area_px=10_000))


def test_stems_stay_unique_when_an_original_name_looks_like_another_frames_index(store, make_clip):
    project, clip = make_clip(n_frames=8)
    clip = _label(store, clip, range(8))
    clip.original_names = {**clip.original_names, 3: "000007.png", 4: "dup.png", 5: "dup.jpg", 6: "a b/c.png"}
    store.write_clip(clip)
    plan = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0))
    stems = {f.frame_idx: f.stem for f in plan.frames}
    assert len(set(stems.values())) == 8
    assert stems[3] == f"{clip.id}_000003" and stems[7] != stems[3]  # "000007" is frame 7's own index form
    assert stems[4] == f"{clip.id}_dup" and stems[5] == f"{clip.id}_000005"
    assert stems[6] == f"{clip.id}_c"


def test_the_contiguous_tail_is_taken_after_empty_frames_are_dropped(store, make_clip):
    project, clip = make_clip(n_frames=10)
    _label(store, clip, range(6))  # masks on 0-5 only
    plan = build_plan(store, project.id, [clip.id], ExportOptions(include_empty=False))
    assert [(f.frame_idx, f.split) for f in plan.frames] == [
        (0, "train"), (1, "train"), (2, "train"), (3, "train"), (4, "val"), (5, "val"),
    ]
    project2, late = make_clip(n_frames=10)
    _label(store, late, range(6, 10))  # masks on 6-9 only
    plan2 = build_plan(store, project2.id, [late.id], ExportOptions(include_empty=False))
    assert [(f.frame_idx, f.split) for f in plan2.frames] == [(6, "train"), (7, "train"), (8, "train"), (9, "val")]


def test_a_clip_without_masks_contributes_only_background_frames_or_nothing(store, make_clip, tmp_path):
    project, first = make_clip(n_frames=4)
    _label(store, first, range(4))
    bare = _extra_clip(store, project.id, tmp_path, n=4)
    ids = [first.id, bare.id]
    keep = build_plan(store, project.id, ids, ExportOptions(val_clip_ids=()))
    assert sum(1 for f in keep.frames if f.clip_id == bare.id) == 4
    drop = build_plan(store, project.id, ids, ExportOptions(val_clip_ids=(), include_empty=False))
    assert {f.clip_id for f in drop.frames} == {first.id}


def test_duplicate_formats_are_refused(store, make_clip):
    project, clip = make_clip(n_frames=4)
    with pytest.raises(BadRequest, match="listed twice"):
        check_request(store, project.id, [clip.id], ["coco", "coco"], ExportOptions())


def _cup_and_plate(store, make_clip, n=4):
    """Classes cup (0), bowl (1), plate (2); a cup on every frame, a plate on the first half; no bowl anywhere."""
    project, clip = make_clip(n_frames=n, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=2)]
    store.write_clip(clip)
    cup = np.zeros((60, 120), bool)
    cup[10:30, 5:30] = True
    plate = np.zeros((60, 120), bool)
    plate[10:30, 70:100] = True
    for i in range(n):
        store.write_mask(clip.id, 1, i, cup)
    for i in range(n // 2):
        store.write_mask(clip.id, 2, i, plate)
    return project, store.read_clip(clip.id)


def test_selected_classes_are_renumbered_in_project_class_order(store, make_clip):
    project, clip = _cup_and_plate(store, make_clip)
    only_plate = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, class_ids=(2,)))
    assert [(c.id, c.name) for c in only_plate.classes] == [(0, "plate")]
    assert only_plate.class_map == {0: 2}
    assert only_plate.options.class_ids == (2,)
    assert [[(o.object_id, o.class_id) for o in f.objects] for f in only_plate.frames] == [[(2, 0)], [(2, 0)], [], []]

    both = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, class_ids=(2, 0)))
    assert [(c.id, c.name) for c in both.classes] == [(0, "cup"), (1, "plate")]
    assert both.class_map == {0: 0, 1: 2}
    assert both.options.class_ids == (0, 2)  # resolved: sorted
    assert [(o.object_id, o.class_id) for o in both.frames[0].objects] == [(1, 0), (2, 1)]


def test_every_class_selected_is_the_identity_map(store, make_clip):
    project, clip = _cup_and_plate(store, make_clip)
    plan = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0))
    assert plan.options.class_ids is None
    assert [c.id for c in plan.classes] == [0, 1, 2] and plan.class_map == {0: 0, 1: 1, 2: 2}
    assert [(o.object_id, o.class_id) for o in plan.frames[0].objects] == [(1, 0), (2, 2)]


def test_dropped_objects_leave_their_frames_to_include_empty_and_the_split_follows(store, make_clip):
    project, clip = _cup_and_plate(store, make_clip, n=10)  # plate on frames 0-4
    keep = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, class_ids=(2,)))
    assert [len(f.objects) for f in keep.frames] == [1] * 5 + [0] * 5
    drop = build_plan(store, project.id, [clip.id], ExportOptions(class_ids=(2,), include_empty=False))
    # the contiguous tail is taken from the 5 frames left after the filter: ceil(0.2 * 5) = 1
    assert [(f.frame_idx, f.split) for f in drop.frames] == [
        (0, "train"), (1, "train"), (2, "train"), (3, "train"), (4, "val"),
    ]
    with pytest.raises(BadRequest, match="nothing to export"):  # bowl: selected, but on no frame
        build_plan(store, project.id, [clip.id], ExportOptions(class_ids=(1,), include_empty=False))
    empty = build_plan(store, project.id, [clip.id], ExportOptions(class_ids=(1,)))
    assert len(empty.frames) == 10 and not any(f.objects for f in empty.frames)


def test_class_selection_is_validated(store, make_clip):
    project, clip = make_clip(n_frames=2, classes=("cup", "bowl"))
    with pytest.raises(BadRequest, match="at least one class"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=()))
    with pytest.raises(BadRequest, match="unknown class 5"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=(0, 5)))
    with pytest.raises(BadRequest, match="class is listed twice"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=(1, 1)))
    assert check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=(1, 0))).class_ids == (0, 1)
    assert check_request(store, project.id, [clip.id], ["coco"], ExportOptions()).class_ids is None


def test_manifest_helpers_restore_the_options_and_the_class_map():
    stored = {"split_mode": "contiguous", "val_clip_ids": [], "val_fraction": 0.2, "stride": 1,
              "include_empty": True, "min_area_px": 64, "class_ids": [2]}
    opts = options_from_manifest(stored)
    assert opts.class_ids == (2,) and opts.val_clip_ids == () and opts.split_mode == "contiguous"
    old = {k: v for k, v in stored.items() if k != "class_ids"}
    assert options_from_manifest({**old, "val_clip_ids": None}).class_ids is None
    assert options_from_manifest({**old, "val_clip_ids": None}).val_clip_ids == ()

    classes, class_map = classes_from_manifest({"classes": [
        {"id": 1, "project_class_id": 4, "name": "plate", "color": "#00ff00", "yolo_id": 1, "coco_id": 2,
         "mask_value": 2},
        {"id": 0, "project_class_id": 2, "name": "cup", "color": "#ff0000", "yolo_id": 0, "coco_id": 1,
         "mask_value": 1},
    ]})
    assert [(c.id, c.name, c.color) for c in classes] == [(0, "cup", "#ff0000"), (1, "plate", "#00ff00")]
    assert class_map == {0: 2, 1: 4}
    _, old_map = classes_from_manifest({"classes": [
        {"id": 0, "name": "a", "color": "#000000", "yolo_id": 0, "coco_id": 1, "mask_value": 1},
        {"id": 1, "name": "b", "color": "#111111", "yolo_id": 1, "coco_id": 2, "mask_value": 2},
    ]})
    assert old_map == {0: 0, 1: 1}  # an export written before class selection maps to itself
