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


def _labelled_at(store, make_clip, n, indices):
    """Like `_labelled`, but only the given frame indices get a mask (the rest are unlabelled)."""
    project, clip = make_clip(n_frames=n)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in indices:
        store.write_mask(clip.id, 1, i, mask)
    return project, clip


def _multi(store, make_clip, n=6):
    """Two touching same-class objects plus an overlapping other-class object, on every frame -- the
    semantic `masks` PNG merges/occludes these, so real labels must never be rebuilt from it."""
    project, clip = make_clip(n_frames=n, classes=("a", "b"))
    clip.objects = [
        LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=0), LabelObject(id=3, class_id=1),
    ]
    store.write_clip(clip)
    m1 = np.zeros((60, 120), bool)
    m1[10:30, 5:30] = True
    m2 = np.zeros((60, 120), bool)
    m2[10:30, 29:50] = True  # touches m1, same class
    m3 = np.zeros((60, 120), bool)
    m3[20:45, 20:45] = True  # overlaps m1, other class
    for i in range(n):
        store.write_mask(clip.id, 1, i, m1)
        store.write_mask(clip.id, 2, i, m2)
        store.write_mask(clip.id, 3, i, m3)
    return project, clip


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


def test_augmented_copy_of_a_named_source_gets_a_derived_name(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    manifest_path = Path(src["path"]) / "export.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["name"] = "Cups v2"
    manifest_path.write_text(json.dumps(manifest))

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=3, seed=0))
    assert result["name"] == "Cups v2 ×3 augmented"
    new_manifest = json.loads((Path(result["path"]) / "export.json").read_text())
    assert new_manifest["name"] == "Cups v2 ×3 augmented"


def test_augmented_copy_name_is_truncated_to_stay_within_max_name(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    manifest_path = Path(src["path"]) / "export.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["name"] = "x" * 80  # an already-max-length source name
    manifest_path.write_text(json.dumps(manifest))

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=3, seed=0))
    assert len(result["name"]) <= 80
    assert result["name"].endswith(" ×3 augmented")
    new_manifest = json.loads((Path(result["path"]) / "export.json").read_text())
    assert new_manifest["name"] == result["name"]


def test_augmented_copy_of_an_unnamed_source_is_named_from_its_id(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=3, seed=0))
    assert result["name"] == f"{src['export_id']} ×3 augmented"


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


def test_multi_object_labels_are_copied_verbatim_not_relabelled(store, make_clip, tmp_path):
    """Reviewer-reported defect: rebuilding real labels from the semantic `masks` PNG merges touching
    same-class objects and drops occluded pixels. Real frames' labels must be byte-identical to the
    source's, for every format, on both splits."""
    project, clip = _multi(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "yolo_det", "coco", "masks"], data_root)
    _make_pool(data_root)
    src_dir = Path(src["path"])

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    new_dir = Path(result["path"])

    for fmt, rel in (("yolo_seg", "labels"), ("yolo_det", "labels"), ("masks", "masks")):
        for split in ("train", "val"):
            src_split = src_dir / fmt / rel / split
            new_split = new_dir / fmt / rel / split
            names = sorted(p.name for p in src_split.iterdir())
            new_names = {p.name for p in new_split.iterdir()}
            assert names  # every real frame has a label/mask file, even an empty one
            if split == "val":
                assert set(names) == new_names  # val never gains synthetic frames
            else:
                assert set(names) <= new_names  # train keeps every real frame, plus synthetic ones
            for name in names:
                assert (new_split / name).read_bytes() == (src_split / name).read_bytes()

    src_coco_val = (src_dir / "coco" / "annotations" / "instances_val.json").read_bytes()
    new_coco_val = (new_dir / "coco" / "annotations" / "instances_val.json").read_bytes()
    assert src_coco_val == new_coco_val

    src_coco_train = json.loads((src_dir / "coco" / "annotations" / "instances_train.json").read_text())
    new_coco_train = json.loads((new_dir / "coco" / "annotations" / "instances_train.json").read_text())
    src_by_id = {a["id"]: a for a in src_coco_train["annotations"]}
    new_by_id = {a["id"]: a for a in new_coco_train["annotations"]}
    assert set(src_by_id) <= set(new_by_id)
    for ann_id, ann in src_by_id.items():
        assert new_by_id[ann_id] == ann  # every source annotation present, unchanged
    assert len(new_by_id) > len(src_by_id)  # synthetic annotations were appended
    assert len(new_by_id) == len(new_coco_train["annotations"])  # every id unique


def test_check_augment_rejects_deriving_from_a_derived_export(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    derived = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))

    with pytest.raises(BadRequest, match="already an augmented copy"):
        check_augment(data_root, derived["export_id"], AugmentOptions())


def test_check_augment_rejects_no_train_objects(store, make_clip, tmp_path):
    project, clip = _labelled_at(store, make_clip, 6, [4, 5])  # only the val frames are labelled
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)

    with pytest.raises(BadRequest, match="no train-split objects"):
        check_augment(data_root, src["export_id"], AugmentOptions())


def test_check_augment_requires_yolo_seg_format(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["masks"], data_root)  # no yolo_seg
    _make_pool(data_root)

    with pytest.raises(BadRequest):
        check_augment(data_root, src["export_id"], AugmentOptions())


def test_empty_val_frames_are_kept(store, make_clip, tmp_path):
    project, clip = _labelled_at(store, make_clip, 6, [0, 1, 2, 3, 5])  # val frame 4 is unlabelled
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    src_dir = Path(src["path"])

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    new_dir = Path(result["path"])

    src_val_dir = src_dir / "yolo_seg" / "labels" / "val"
    new_val_dir = new_dir / "yolo_seg" / "labels" / "val"
    src_val_names = sorted(p.name for p in src_val_dir.iterdir())
    assert sorted(p.name for p in new_val_dir.iterdir()) == src_val_names
    assert any(not (src_val_dir / n).read_text().strip() for n in src_val_names)  # at least one is empty
    for name in src_val_names:
        assert (new_val_dir / name).read_bytes() == (src_val_dir / name).read_bytes()


def test_progress_is_monotonic_with_a_fixed_total(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)

    seen = []
    run_augment(
        data_root, src["export_id"], AugmentOptions(per_image=2, seed=0),
        progress=lambda done, total: seen.append((done, total)),
    )

    assert seen
    assert len({total for _, total in seen}) == 1  # the total is fixed up front, never moved
    dones = [done for done, _ in seen]
    assert dones == sorted(dones)  # done never goes backwards


def test_a_derived_copy_keeps_the_class_map_and_the_export_ids(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=6, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=2)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(6):
        store.write_mask(clip.id, 1, i, mask)
    data_root = tmp_path / "data"
    src = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(class_ids=(2,)),
                     data_root / "exports")
    _make_pool(data_root)
    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    manifest = json.loads((Path(result["path"]) / "export.json").read_text())
    assert [(c["id"], c["project_class_id"], c["name"]) for c in manifest["classes"]] == [(0, 2, "plate")]
    assert manifest["options"]["class_ids"] == [2]
    syn = list((Path(result["path"]) / "yolo_seg" / "labels" / "train").glob("syn_*.txt"))
    assert syn
    for p in syn:
        assert {line.split()[0] for line in p.read_text().splitlines()} == {"0"}


def test_a_source_manifest_without_project_class_id_maps_to_itself(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    manifest_path = Path(src["path"]) / "export.json"
    old = json.loads(manifest_path.read_text())
    for c in old["classes"]:
        del c["project_class_id"]  # as an export written before class selection
    del old["options"]["class_ids"]
    manifest_path.write_text(json.dumps(old))

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    manifest = json.loads((Path(result["path"]) / "export.json").read_text())
    assert [(c["id"], c["project_class_id"]) for c in manifest["classes"]] == [(0, 0)]
    assert manifest["options"]["class_ids"] is None
