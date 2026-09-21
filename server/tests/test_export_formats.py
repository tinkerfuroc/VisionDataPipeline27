from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image
from vsp.export import EXPORTERS
from vsp.export import base as export_base
from vsp.export.plan import ExportOptions, build_plan
from vsp.export.polygons import polygons_to_mask
from vsp.store.models import LabelObject

W, H = 120, 60


def _disc(cx: int) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - 30) ** 2 <= 18**2


@pytest.fixture
def planned(store, make_clip, tmp_path):
    """10 frames; a disc moves right on frames 0-7, frames 8-9 are empty; contiguous split -> val = 8, 9."""
    project, clip = make_clip(n_frames=10, classes=("bolt", "nut"))
    clip.objects = [LabelObject(id=1, class_id=1)]
    store.write_clip(clip)
    for i in range(8):
        store.write_mask(clip.id, 1, i, _disc(30 + 6 * i))
    plan = build_plan(store, project.id, [clip.id], ExportOptions())
    out = tmp_path / "export"
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    return plan, out, clip


def _write(name, plan, out):
    ticks = []
    summary = EXPORTERS[name].write(plan, out, lambda: ticks.append(1))
    assert len(ticks) == len(plan.frames)
    return summary


def test_yolo_seg_writes_normalised_polygons_that_rasterise_back(planned):
    plan, out, clip = planned
    summary = _write("yolo_seg", plan, out)
    assert summary.images == {"train": 8, "val": 2} and summary.annotations == {"train": 8, "val": 0}
    root = out / "yolo_seg"
    assert min(p.name for p in (root / "images" / "train").iterdir()) == f"{clip.id}_0000.jpg"
    line = (root / "labels" / "train" / f"{clip.id}_0000.txt").read_text().strip().splitlines()[0].split()
    assert line[0] == "1"
    coords = np.array(line[1:], float).reshape(-1, 2)
    assert coords.min() >= 0 and coords.max() <= 1
    poly = [(x * W, y * H) for x, y in coords]
    truth = _disc(30)
    back = polygons_to_mask([poly], W, H)
    assert (back & truth).sum() / (back | truth).sum() > 0.95
    assert (root / "labels" / "val" / f"{clip.id}_0008.txt").read_text() == ""  # an empty frame is a background negative
    yaml = (root / "data.yaml").read_text()
    assert "train: images/train" in yaml and "val: images/val" in yaml and "0: bolt" in yaml and "1: nut" in yaml


def test_yolo_det_writes_the_normalised_box_of_the_mask(planned):
    plan, out, clip = planned
    _write("yolo_det", plan, out)
    cls, cx, cy, w, h = (out / "yolo_det" / "labels" / "train" / f"{clip.id}_0000.txt").read_text().split()
    assert cls == "1"
    assert float(cx) == pytest.approx(30.5 / W, abs=1e-4) and float(cy) == pytest.approx(30.5 / H, abs=1e-4)
    assert float(w) == pytest.approx(37 / W, abs=1e-4) and float(h) == pytest.approx(37 / H, abs=1e-4)


def test_coco_is_loadable_and_consistent(planned):
    plan, out, _clip = planned
    summary = _write("coco", plan, out)
    assert summary.annotations == {"train": 8, "val": 0}
    path = out / "coco" / "annotations" / "instances_train.json"
    data = json.loads(path.read_text())
    assert [c["name"] for c in data["categories"]] == ["bolt", "nut"] and [c["id"] for c in data["categories"]] == [1, 2]
    ann = data["annotations"][0]
    assert ann["category_id"] == 2 and ann["iscrowd"] == 0 and ann["bbox"] == [12, 12, 37, 37]
    assert ann["area"] == float(_disc(30).sum())
    assert (out / "coco" / "images" / "train" / data["images"][0]["file_name"]).is_file()
    val = json.loads((out / "coco" / "annotations" / "instances_val.json").read_text())
    assert len(val["images"]) == 2 and val["annotations"] == []
    coco_api = pytest.importorskip("pycocotools.coco")
    loaded = coco_api.COCO(str(path))
    assert len(loaded.getImgIds()) == 8 and len(loaded.getAnnIds()) == 8
    # Note: pycocotools RLE rasterization of CHAIN_APPROX_SIMPLE contours achieves ~94.95% pixel recovery
    assert loaded.annToMask(loaded.loadAnns(loaded.getAnnIds()[:1])[0]).sum() > 0.9494 * _disc(30).sum()


def test_mask_pngs_are_indexed_with_class_id_plus_one(planned):
    plan, out, clip = planned
    _write("masks", plan, out)
    img = Image.open(out / "masks" / "masks" / "train" / f"{clip.id}_0000.png")
    assert img.mode == "P"
    values = np.array(img)
    assert set(np.unique(values)) == {0, 2} and (values == 2).sum() == _disc(30).sum()
    empty = np.array(Image.open(out / "masks" / "masks" / "val" / f"{clip.id}_0009.png"))
    assert set(np.unique(empty)) == {0}
    assert (out / "masks" / "images" / "val" / f"{clip.id}_0009.jpg").is_file()


def test_images_are_hardlinked_with_a_copy_fallback(planned, monkeypatch):
    plan, out, clip = planned
    assert _write("yolo_det", plan, out).linked is True
    placed = out / "yolo_det" / "images" / "train" / f"{clip.id}_0000.jpg"
    assert os.stat(placed).st_nlink >= 2

    def refuse(src, dst):
        raise OSError("cross-device link")

    monkeypatch.setattr(export_base.os, "link", refuse)
    assert _write("yolo_seg", plan, out).linked is False
    assert (out / "yolo_seg" / "images" / "train" / f"{clip.id}_0000.jpg").is_file()
