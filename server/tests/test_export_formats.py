from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml as yaml_lib
from PIL import Image
from vdp.export import EXPORTERS
from vdp.export import base as export_base
from vdp.export.coco import coco_annotations
from vdp.export.mask_png import paint
from vdp.export.plan import ExportOptions, build_plan
from vdp.export.polygons import mask_bbox, mask_to_polygons, polygons_to_mask
from vdp.export.yolo import yolo_det_lines, yolo_seg_lines
from vdp.store.models import LabelObject

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
    parsed = yaml_lib.safe_load(yaml)
    assert parsed["train"] == "images/train" and parsed["val"] == "images/val" and parsed["names"] == {0: "bolt", 1: "nut"}
    # the polygon and the detection box of the same object agree (one coordinate convention for every format)
    assert coords[:, 0].min() * W == pytest.approx(12.25, abs=1e-3) and coords[:, 0].max() * W == pytest.approx(48.75, abs=1e-3)


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
    back = loaded.annToMask(loaded.loadAnns(loaded.getAnnIds()[:1])[0]).astype(bool)
    assert (back & _disc(30)).sum() / (back | _disc(30)).sum() > 0.99  # as a real COCO consumer rasterises it


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


def test_data_yaml_survives_awkward_class_names(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=4, classes=("bolt: m6", "nut #steel", 'say "hi"'))
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    for i in range(4):
        store.write_mask(clip.id, 1, i, _disc(40))
    plan = build_plan(store, project.id, [clip.id], ExportOptions())
    out = tmp_path / "export"
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    EXPORTERS["yolo_det"].write(plan, out, lambda: None)
    names = yaml_lib.safe_load((out / "yolo_det" / "data.yaml").read_text())["names"]
    assert names == {0: "bolt: m6", 1: "nut #steel", 2: 'say "hi"'}


def _export_all(store, project, clip, tmp_path, options):
    plan = build_plan(store, project.id, [clip.id], options)
    out = tmp_path / "export_all"
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    for name in EXPORTERS:
        EXPORTERS[name].write(plan, out, lambda: None)
    return plan, out


def test_a_stray_pixel_does_not_make_the_formats_disagree_about_the_object(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=2)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    noisy = np.zeros((H, W), bool)
    noisy[10:30, 10:30] = True  # the object: 400 px
    noisy[55, 115] = True  # one stray pixel, routine in tracker output
    for i in range(2):
        store.write_mask(clip.id, 1, i, noisy)
    plan, out = _export_all(store, project, clip, tmp_path, ExportOptions(val_fraction=0.0))
    stem = plan.frames[0].stem
    _, cx, cy, w, h = map(float, (out / "yolo_det" / "labels" / "train" / f"{stem}.txt").read_text().split())
    assert (cx * W, cy * H, w * W, h * H) == pytest.approx((20, 20, 20, 20), abs=1e-3)
    seg = np.array((out / "yolo_seg" / "labels" / "train" / f"{stem}.txt").read_text().split()[1:], float).reshape(-1, 2)
    assert seg[:, 0].max() * W < 30 and seg[:, 1].max() * H < 30
    ann = json.loads((out / "coco" / "annotations" / "instances_train.json").read_text())["annotations"][0]
    assert ann["bbox"] == [10, 10, 20, 20] and ann["area"] == 400.0
    png = np.array(Image.open(out / "masks" / "masks" / "train" / f"{stem}.png"))
    assert (png > 0).sum() == 400


def test_a_mask_of_scattered_specks_is_absent_in_every_format(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=3)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    specks = np.zeros((H, W), bool)
    specks[::6, ::10] = True  # 120 isolated pixels: well over min_area_px in total, no real blob
    store.write_mask(clip.id, 1, 0, specks)
    store.write_mask(clip.id, 1, 1, _disc(40))
    plan, out = _export_all(store, project, clip, tmp_path, ExportOptions(val_fraction=0.0))
    assert [len(f.objects) for f in plan.frames] == [0, 1, 0]
    stem = plan.frames[0].stem
    assert (out / "yolo_seg" / "labels" / "train" / f"{stem}.txt").read_text() == ""
    assert (out / "yolo_det" / "labels" / "train" / f"{stem}.txt").read_text() == ""
    assert not np.array(Image.open(out / "masks" / "masks" / "train" / f"{stem}.png")).any()
    data = json.loads((out / "coco" / "annotations" / "instances_train.json").read_text())
    assert len(data["annotations"]) == 1


def test_every_image_has_its_label_annotation_and_mask_in_the_same_split(planned):
    plan, out, _clip = planned
    for name in EXPORTERS:
        EXPORTERS[name].write(plan, out, lambda: None)
    for split in ("train", "val"):
        stems = {f.stem for f in plan.of_split(split)}
        for fmt in ("yolo_seg", "yolo_det"):
            assert {p.stem for p in (out / fmt / "images" / split).iterdir()} == stems
            assert {p.stem for p in (out / fmt / "labels" / split).iterdir()} == stems
        assert {p.stem for p in (out / "masks" / "images" / split).iterdir()} == stems
        assert {p.stem for p in (out / "masks" / "masks" / split).iterdir()} == stems
        coco = json.loads((out / "coco" / "annotations" / f"instances_{split}.json").read_text())
        assert {Path(i["file_name"]).stem for i in coco["images"]} == stems
        assert {p.stem for p in (out / "coco" / "images" / split).iterdir()} == stems


def _cup_and_plate(store, make_clip, n=2):
    project, clip = make_clip(n_frames=n, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=2)]
    store.write_clip(clip)
    cup = np.zeros((H, W), bool)
    cup[10:30, 5:30] = True  # 500 px
    plate = np.zeros((H, W), bool)
    plate[10:30, 70:100] = True  # 600 px
    for i in range(n):
        store.write_mask(clip.id, 1, i, cup)
        store.write_mask(clip.id, 2, i, plate)
    return project, clip


def _export_to(store, project, clip, out, options):
    plan = build_plan(store, project.id, [clip.id], options)
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    for name in EXPORTERS:
        EXPORTERS[name].write(plan, out, lambda: None)
    return plan


def _rgb(color: str) -> list[int]:
    return [int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]


def test_one_selected_class_is_export_id_0_in_every_format(store, make_clip, tmp_path):
    project, clip = _cup_and_plate(store, make_clip)
    out = tmp_path / "plate"
    plan = _export_to(store, project, clip, out, ExportOptions(val_fraction=0.0, class_ids=(2,)))
    stem = plan.frames[0].stem
    seg = (out / "yolo_seg" / "labels" / "train" / f"{stem}.txt").read_text().splitlines()
    assert [ln.split()[0] for ln in seg] == ["0"]
    assert (out / "yolo_det" / "labels" / "train" / f"{stem}.txt").read_text().split()[0] == "0"
    assert yaml_lib.safe_load((out / "yolo_seg" / "data.yaml").read_text())["names"] == {0: "plate"}
    coco = json.loads((out / "coco" / "annotations" / "instances_train.json").read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(1, "plate")]
    assert [a["category_id"] for a in coco["annotations"]] == [1, 1]  # one plate per frame, no cup
    png = Image.open(out / "masks" / "masks" / "train" / f"{stem}.png")
    values = np.array(png)
    assert set(np.unique(values)) == {0, 1} and (values == 1).sum() == 600
    assert png.getpalette()[3:6] == _rgb(project.classes[2].color)  # value 1 is drawn in the plate's colour


def test_two_selected_classes_keep_project_order_with_a_gap_closed(store, make_clip, tmp_path):
    project, clip = _cup_and_plate(store, make_clip)
    out = tmp_path / "cup_plate"
    plan = _export_to(store, project, clip, out, ExportOptions(val_fraction=0.0, class_ids=(0, 2)))
    stem = plan.frames[0].stem
    seg = (out / "yolo_seg" / "labels" / "train" / f"{stem}.txt").read_text().splitlines()
    assert sorted(ln.split()[0] for ln in seg) == ["0", "1"]
    assert yaml_lib.safe_load((out / "yolo_det" / "data.yaml").read_text())["names"] == {0: "cup", 1: "plate"}
    coco = json.loads((out / "coco" / "annotations" / "instances_train.json").read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(1, "cup"), (2, "plate")]
    assert sorted({a["category_id"] for a in coco["annotations"]}) == [1, 2]
    png = Image.open(out / "masks" / "masks" / "train" / f"{stem}.png")
    values = np.array(png)
    assert set(np.unique(values)) == {0, 1, 2} and (values == 1).sum() == 500 and (values == 2).sum() == 600
    assert png.getpalette()[6:9] == _rgb(project.classes[2].color)


def test_paint_draws_the_smaller_object_on_top_whatever_the_order_and_counts_the_overlap():
    big = np.zeros((H, W), bool)
    big[10:50, 10:60] = True  # 2000 px
    small = np.zeros((H, W), bool)
    small[20:30, 20:30] = True  # 100 px, inside big
    for objects in ([(2, small), (1, big)], [(1, big), (2, small)]):
        canvas, overlap = paint(objects, H, W)
        assert canvas.dtype == np.uint8
        assert (canvas == 2).sum() == 100 and (canvas == 1).sum() == 1900 and overlap == 100
    apart = np.zeros((H, W), bool)
    apart[0:5, 100:110] = True
    assert paint([(1, apart), (2, small)], H, W)[1] == 0
    assert paint([], H, W)[1] == 0


def test_the_mask_image_keeps_a_small_object_visible_on_a_large_one(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=2, classes=("screw", "board"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=1)]  # the small one comes first
    store.write_clip(clip)
    screw = np.zeros((H, W), bool)
    screw[20:40, 40:60] = True  # 400 px, lying on the board
    board = np.zeros((H, W), bool)
    board[5:55, 10:110] = True  # 5000 px
    for i in range(2):
        store.write_mask(clip.id, 1, i, screw)
        store.write_mask(clip.id, 2, i, board)
    plan = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0))
    out = tmp_path / "overlap"
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    summary = EXPORTERS["masks"].write(plan, out, lambda: None)
    values = np.array(Image.open(out / "masks" / "masks" / "train" / f"{plan.frames[0].stem}.png"))
    assert (values == 1).sum() == 400 and (values == 2).sum() == 4600
    assert summary.overlap_px == 2 * 400
    assert asdict(summary)["overlap_px"] == 800
    assert "overlap_px" not in asdict(EXPORTERS["yolo_seg"].write(plan, out, lambda: None))




def _square(h=40, w=60, y=(10, 30), x=(20, 50)):
    m = np.zeros((h, w), bool)
    m[y[0]:y[1], x[0]:x[1]] = True
    return m


def test_pure_label_functions():
    objs = [(1, _square())]
    det = yolo_det_lines(objs, 60, 40)
    assert det == ["1 0.583333 0.500000 0.500000 0.500000"]
    seg = yolo_seg_lines(objs, 60, 40)
    assert len(seg) == 1 and seg[0].startswith("1 ")
    (ann,) = coco_annotations(objs)
    assert ann["category_id"] == 2 and ann["bbox"] == [20, 10, 30, 20] and ann["area"] == 600.0
    assert "id" not in ann and "image_id" not in ann
    assert coco_annotations([(0, np.zeros((5, 5), bool))]) == []


# ---- frozen copy of the pre-refactor per-frame logic ------------------------------------------------------------
def _old_seg(frame):
    out = []
    for obj in frame.objects:
        for polygon in mask_to_polygons(export_base.read_mask(obj)):
            coords = " ".join(f"{x / frame.width:.6f} {y / frame.height:.6f}" for x, y in polygon)
            out.append(f"{obj.class_id} {coords}")
    return out


def _old_det(frame):
    out = []
    for obj in frame.objects:
        box = mask_bbox(export_base.read_mask(obj))
        if box is None:
            continue
        x, y, w, h = box
        out.append(
            f"{obj.class_id} {(x + w / 2) / frame.width:.6f} {(y + h / 2) / frame.height:.6f} "
            f"{w / frame.width:.6f} {h / frame.height:.6f}"
        )
    return out


def _old_palette(plan):
    palette = [0, 0, 0] * 256
    for c in plan.classes:
        if 0 <= c.id + 1 <= 255:
            rgb = c.color.lstrip("#")
            palette[(c.id + 1) * 3 : (c.id + 1) * 3 + 3] = [int(rgb[i : i + 2], 16) for i in (0, 2, 4)]
    return palette


def _old_coco(plan, split):
    annotations, image_id = [], 0
    for frame in plan.of_split(split):
        image_id += 1
        for obj in frame.objects:
            mask = export_base.read_mask(obj)
            polygons = mask_to_polygons(mask)
            box = mask_bbox(mask)
            if not polygons or box is None:
                continue
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": obj.class_id + 1,
                "segmentation": [[round(v, 2) for point in polygon for v in point] for polygon in polygons],
                "bbox": list(box), "area": float(mask.sum()), "iscrowd": 0,
            })
    return annotations


def test_exporters_are_byte_identical_to_the_frozen_pre_refactor_logic(planned):
    plan, out, _clip = planned
    for name in ("yolo_seg", "yolo_det", "masks", "coco"):
        _write(name, plan, out)
    for frame in plan.frames:
        for name, old in (("yolo_seg", _old_seg), ("yolo_det", _old_det)):
            expect = "".join(f"{ln}\n" for ln in old(frame))
            assert (out / name / "labels" / frame.split / f"{frame.stem}.txt").read_text() == expect
        canvas, _ = paint([(o.class_id + 1, export_base.read_mask(o)) for o in frame.objects], frame.height, frame.width)
        img = Image.fromarray(canvas)
        img.putpalette(_old_palette(plan))
        old_png = out / "old.png"
        img.save(old_png)
        assert (out / "masks" / "masks" / frame.split / f"{frame.stem}.png").read_bytes() == old_png.read_bytes()
    for split in ("train", "val"):
        body = json.loads((out / "coco" / "annotations" / f"instances_{split}.json").read_text())
        assert body["annotations"] == _old_coco(plan, split)
