from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from vdp.export import ExportCancelled, ExportOptions, run_export
from vdp.store.models import LabelObject, RegionRange


def _labelled(store, make_clip, n=6):
    project, clip = make_clip(n_frames=n)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(n):
        store.write_mask(clip.id, 1, i, mask)
    return project, clip


def test_run_export_writes_every_format_a_manifest_and_reports_progress(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    seen = []
    result = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), tmp_path / "exports",
                        progress=lambda done, total: seen.append((done, total)))
    out = tmp_path / "exports" / result["export_id"]
    assert result["path"] == str(out) and result["formats"] == ["yolo_seg", "masks"]
    assert result["counts"] == {"train": {"images": 4, "annotated": 4, "empty": 0}, "val": {"images": 2, "annotated": 2, "empty": 0}}
    assert len(list((out / "images").glob("*.jpg"))) == 6
    assert (out / "yolo_seg" / "data.yaml").is_file() and (out / "masks" / "masks" / "val").is_dir()
    assert not (out / "coco").exists()
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["options"]["split_mode"] == "contiguous" and manifest["options"]["min_area_px"] == 64
    assert manifest["clips"] == [{"id": clip.id, "name": clip.source.name, "train": 4, "val": 2}]
    assert manifest["classes"] == [{"id": 0, "project_class_id": 0, "name": "part", "color": manifest["classes"][0]["color"], "yolo_id": 0, "coco_id": 1, "mask_value": 1}]
    assert manifest["options"]["class_ids"] is None
    assert seen[-1] == (6 * 4, 6 * 4) and [d for d, _ in seen] == sorted(d for d, _ in seen)
    # the store was only read: no stray file, and the export holds copies, not links to store frames
    import os
    assert os.stat(store.frame_path(clip.id, 0)).st_nlink == 1


def test_a_cancelled_export_leaves_nothing_behind(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 8

    with pytest.raises(ExportCancelled):
        run_export(store, project.id, [clip.id], ["coco"], ExportOptions(), tmp_path / "exports", cancel=cancel)
    assert list((tmp_path / "exports").iterdir()) == []


def test_run_export_with_a_name_slugs_the_folder_id(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    result = run_export(store, project.id, [clip.id], ["yolo_seg"], ExportOptions(), tmp_path / "exports",
                        name="Cups v2")
    assert result["export_id"] == "cups-v2" and result["name"] == "Cups v2"
    manifest = json.loads((tmp_path / "exports" / "cups-v2" / "export.json").read_text())
    assert manifest["name"] == "Cups v2"

    second = run_export(store, project.id, [clip.id], ["yolo_seg"], ExportOptions(), tmp_path / "exports",
                        name="Cups v2")
    assert second["export_id"] == "cups-v2-2"


def test_run_export_without_a_name_gets_a_timestamp_id_and_null_name(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    result = run_export(store, project.id, [clip.id], ["yolo_seg"], ExportOptions(), tmp_path / "exports")
    assert result["name"] is None
    assert len(result["export_id"]) == len("20260926-104433-a1b2c3")
    manifest = json.loads((tmp_path / "exports" / result["export_id"] / "export.json").read_text())
    assert manifest["name"] is None


def test_a_failed_named_export_removes_its_reserved_folder(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(ExportCancelled):
        run_export(store, project.id, [clip.id], ["coco"], ExportOptions(), tmp_path / "exports",
                   cancel=cancel, name="Cups v2")
    assert list((tmp_path / "exports").iterdir()) == []


def test_the_manifest_maps_export_ids_back_to_project_classes(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=4, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=2)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(4):
        store.write_mask(clip.id, 1, i, mask)
    result = run_export(store, project.id, [clip.id], ["yolo_seg"], ExportOptions(class_ids=(2,)),
                        tmp_path / "exports")
    manifest = json.loads((tmp_path / "exports" / result["export_id"] / "export.json").read_text())
    assert manifest["options"]["class_ids"] == [2]
    assert manifest["classes"] == [{
        "id": 0, "project_class_id": 2, "name": "plate", "color": project.classes[2].color,
        "yolo_id": 0, "coco_id": 1, "mask_value": 1,
    }]


def test_overlap_px_is_reported_for_the_mask_format_only(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    result = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), tmp_path / "exports")
    manifest = json.loads((tmp_path / "exports" / result["export_id"] / "export.json").read_text())
    formats = {f["name"]: f for f in manifest["formats"]}
    assert formats["masks"]["overlap_px"] == 0  # one object per frame
    assert "overlap_px" not in formats["yolo_seg"]


def _regioned(store, make_clip, n=6):
    project, clip = _labelled(store, make_clip, n=n)  # mask rows 10:40, cols 20:60 on 120x60 frames
    clip = store.read_clip(clip.id)
    clip.objects.append(LabelObject(id=2, class_id=0))
    store.write_clip(clip)
    hidden = np.zeros((60, 120), bool)
    hidden[5:20, 90:110] = True  # wholly inside the block below
    for i in range(n):
        store.write_mask(clip.id, 2, i, hidden)
    clip = store.read_clip(clip.id)
    clip.regions = [RegionRange(start=0, end=2, crop=(10, 0, 115, 50), blocks=[(40, 10, 60, 40), (85, 0, 115, 25)])]
    store.write_clip(clip)
    return project, clip


def test_export_crops_greys_and_cuts_labels(store, make_clip, tmp_path):
    project, clip = _regioned(store, make_clip)
    result = run_export(store, project.id, [clip.id], ["yolo_seg", "yolo_det", "coco", "masks"],
                        ExportOptions(), tmp_path / "exports")
    out = tmp_path / "exports" / result["export_id"]
    first = cv2.imread(str(out / "images" / f"{clip.id}_0000.jpg"))
    assert first.shape[:2] == (50, 105)
    assert abs(int(first[20, 40].mean()) - 114) <= 3  # block (40..60 -> 30..50 after the crop) is grey
    last = cv2.imread(str(out / "images" / f"{clip.id}_0005.jpg"))
    assert last.shape[:2] == (60, 120)
    assert (out / "images" / f"{clip.id}_0005.jpg").read_bytes() == store.frame_path(clip.id, 5).read_bytes()

    det = (out / "yolo_det" / "labels" / "train" / f"{clip.id}_0000.txt").read_text().split()
    assert len(det) == 5  # object 2 is wholly blocked: absent
    cls, cx, cy, w, h = det[0], *map(float, det[1:])
    # kept part of object 1: x 20..40 in frame = 10..30 after the crop; y 10..40
    assert cls == "0" and abs(cx - 20 / 105) < 0.01 and abs(w - 20 / 105) < 0.01 and abs(cy - 25 / 50) < 0.01
    assert abs(h - 30 / 50) < 0.01

    coco = json.loads((out / "coco" / "annotations" / "instances_train.json").read_text())
    img0 = next(i for i in coco["images"] if i["file_name"].endswith("_0000.jpg"))
    assert (img0["width"], img0["height"]) == (105, 50)
    assert [a["area"] for a in coco["annotations"] if a["image_id"] == img0["id"]] == [20 * 30]

    manifest = json.loads((out / "export.json").read_text())
    assert manifest["regions_applied"] == 3
