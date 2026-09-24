from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from vsp.augment.derive import AugmentOptions, check_augment, run_augment
from vsp.export import ExportCancelled, ExportOptions, run_export
from vsp.labeling.errors import BadRequest
from vsp.store import NotFound
from vsp.store.models import LabelObject
from vsp.train.models import list_exports


def _labelled(store, make_clip, n=6):
    project, clip = make_clip(n_frames=n)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(n):
        store.write_mask(clip.id, 1, i, mask)
    return project, clip


def _make_pool(data_root: Path, n: int = 3, seed: int = 0) -> None:
    d = data_root / "backgrounds" / "dtd" / "x"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(d / f"bg_{i}.jpg"), img)


def _snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*") if p.is_file()
    }


def _export(store, project, clip, formats, data_root):
    return run_export(store, project.id, [clip.id], formats, ExportOptions(), data_root / "exports")


def test_derive_writes_val_unchanged_and_train_with_composites(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "coco", "masks"], data_root)
    _make_pool(data_root)
    src_dir = Path(src["path"])
    before = _snapshot(src_dir)

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=2, max_objects=2, seed=1))
    new_dir = Path(result["path"])
    assert new_dir != src_dir and new_dir.parent == src_dir.parent

    src_val_labels = sorted(os.listdir(src_dir / "yolo_seg" / "labels" / "val"))
    new_val_labels = sorted(os.listdir(new_dir / "yolo_seg" / "labels" / "val"))
    assert new_val_labels == src_val_labels
    for name in src_val_labels:
        assert (new_dir / "yolo_seg" / "labels" / "val" / name).read_bytes() == \
            (src_dir / "yolo_seg" / "labels" / "val" / name).read_bytes()

    real_train = len(os.listdir(src_dir / "yolo_seg" / "images" / "train"))
    new_train_images = sorted(os.listdir(new_dir / "yolo_seg" / "images" / "train"))
    manifest = json.loads((new_dir / "export.json").read_text())

    assert manifest["augment"]["synthetic_images"] == real_train * 2  # none skipped at this scale
    assert len(new_train_images) == real_train + manifest["augment"]["synthetic_images"]

    syn_names = [n for n in new_train_images if n.startswith("syn_")]
    assert len(syn_names) == manifest["augment"]["synthetic_images"]
    for name in syn_names:
        label_name = Path(name).with_suffix(".txt").name
        lines = (new_dir / "yolo_seg" / "labels" / "train" / label_name).read_text().splitlines()
        assert len(lines) >= 1

    assert manifest["derived_from"] == src["export_id"]
    assert manifest["augment"]["per_image"] == 2
    src_manifest = json.loads((src_dir / "export.json").read_text())
    assert [f["name"] for f in manifest["formats"]] == [f["name"] for f in src_manifest["formats"]]

    from pycocotools.coco import COCO

    coco = COCO(str(new_dir / "coco" / "annotations" / "instances_train.json"))
    syn_image_ids = [im["id"] for im in coco.dataset["images"] if im["file_name"].startswith("syn_")]
    assert syn_image_ids
    for image_id in syn_image_ids:
        assert len(coco.getAnnIds(imgIds=[image_id])) >= 1

    assert _snapshot(src_dir) == before


def test_same_seed_same_composites(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    opts = AugmentOptions(per_image=2, max_objects=2, seed=3)

    r1 = run_augment(data_root, src["export_id"], opts)
    r2 = run_augment(data_root, src["export_id"], opts)

    def train_labels(path):
        labels_dir = Path(path) / "yolo_seg" / "labels" / "train"
        return {p.name: p.read_bytes() for p in labels_dir.glob("syn_*.txt")}

    labels1, labels2 = train_labels(r1["path"]), train_labels(r2["path"])
    assert labels1 and labels1 == labels2


def test_cancel_removes_output(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)

    with pytest.raises(ExportCancelled):
        run_augment(data_root, src["export_id"], AugmentOptions(), cancel=lambda: True)

    assert sorted(p.name for p in (data_root / "exports").iterdir()) == [src["export_id"]]


def test_check_augment_errors(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"

    with pytest.raises(NotFound):
        check_augment(data_root, "nope", AugmentOptions())

    no_masks = _export(store, project, clip, ["yolo_seg"], data_root)
    with pytest.raises(BadRequest):
        check_augment(data_root, no_masks["export_id"], AugmentOptions())

    with_masks = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    with pytest.raises(BadRequest):
        check_augment(data_root, with_masks["export_id"], AugmentOptions())  # empty pool

    _make_pool(data_root)
    with pytest.raises(BadRequest):
        check_augment(data_root, with_masks["export_id"], AugmentOptions(per_image=11))


def test_list_exports_reports_formats_and_derived_from(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=2))

    by_id = {e.id: e for e in list_exports(data_root)}
    assert by_id[src["export_id"]].derived_from is None
    assert by_id[src["export_id"]].formats == ("yolo_seg", "masks")
    derived = by_id[result["export_id"]]
    assert derived.derived_from == src["export_id"] and "masks" in derived.formats
