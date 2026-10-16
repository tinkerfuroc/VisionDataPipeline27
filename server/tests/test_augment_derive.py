from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from vdp.augment.derive import (
    AugmentOptions,
    _cutouts_and_sizes,
    _merged_format_summaries,
    check_augment,
    run_augment,
)
from vdp.export import ExportCancelled, ExportOptions, run_export
from vdp.export.base import ExportSummary
from vdp.export.mask_png import MaskSummary
from vdp.labeling.errors import BadRequest
from vdp.store import NotFound
from vdp.store.models import LabelObject
from vdp.train.models import list_exports


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


def test_a_derived_copy_keeps_the_sources_regions_applied(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    manifest_path = Path(src["path"]) / "export.json"
    old = json.loads(manifest_path.read_text())
    old["regions_applied"] = 2
    manifest_path.write_text(json.dumps(old))
    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    manifest = json.loads((Path(result["path"]) / "export.json").read_text())
    assert manifest["regions_applied"] == 2
    del old["regions_applied"]  # an export written before regions existed
    manifest_path.write_text(json.dumps(old))
    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=1))
    assert json.loads((Path(result["path"]) / "export.json").read_text())["regions_applied"] == 0


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


def test_merged_summaries_sum_overlap_px_and_leave_other_formats_alone():
    counts = {"images": {"train": 4, "val": 1}, "annotations": {"train": 4, "val": 1}, "linked": True}
    src = [{"name": "masks", **counts, "overlap_px": 30}, {"name": "yolo_seg", **counts}]
    synth = [MaskSummary("masks", overlap_px=5), ExportSummary("yolo_seg")]
    merged = {m["name"]: m for m in _merged_format_summaries(src, synth, True)}
    assert merged["masks"]["overlap_px"] == 35
    assert "overlap_px" not in merged["yolo_seg"]
    older = _merged_format_summaries([{"name": "masks", **counts}], [MaskSummary("masks", overlap_px=0)], True)
    assert older[0]["overlap_px"] == 0  # a source written before overlap_px counts as 0


def test_the_derived_mask_summary_carries_the_source_overlap(store, make_clip, tmp_path):
    project, clip = _multi(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    src_masks = next(f for f in json.loads((Path(src["path"]) / "export.json").read_text())["formats"]
                     if f["name"] == "masks")
    new_masks = next(f for f in json.loads((Path(result["path"]) / "export.json").read_text())["formats"]
                     if f["name"] == "masks")
    # _multi: m1&m2 share column 29 on rows 10-19 (10 px), and rows 20-29, columns 20-44 hold 2+ objects (250 px)
    assert src_masks["overlap_px"] == 6 * 260
    assert new_masks["overlap_px"] == src_masks["overlap_px"]  # composites never overlap: one owner per pixel


def _train_images(src) -> int:
    return json.loads((Path(src["path"]) / "export.json").read_text())["counts"]["train"]["images"]


def test_cutouts_are_per_object_when_the_source_has_coco(store, make_clip, tmp_path):
    project, clip = _multi(store, make_clip)  # two touching "a" objects and an overlapping "b" on every frame
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "coco", "masks"]
    src = _export(store, project, clip, formats, data_root)
    ticks = []
    cutouts, sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: ticks.append(1))
    n = _train_images(src)
    assert mode == "per_object"
    assert sizes == [(120, 60)] * n and len(ticks) == n
    assert len(cutouts) == 3 * n
    # paint() paints largest first and a later (smaller) object overwrites an earlier (larger) one wherever they
    # overlap: painted order is m3 (625) then m1 (500) then m2 (420).
    # m2 is painted last, so it wins every pixel it covers: 420 (no loss).
    # m1 is painted before m2, so it only loses to m2: m1 ^ m2 is column 29, rows 10-29 = 1 x 20 = 20 px ->
    # 500 - 20 = 480.
    # m3 is painted first, so it loses to both m1 (rows 20-29 x cols 20-29 = 10 x 10 = 100 px) and m2 (rows 20-29 x
    # cols 29-44 = 10 x 16 = 160 px); those two losses share column 29, rows 20-29 (10 px), so together they take
    # 100 + 160 - 10 = 250 px -> 625 - 250 = 375.
    assert sorted((c.class_id, int(c.mask.sum())) for c in cutouts[:3]) == [(0, 420), (0, 480), (1, 375)]


def test_cutouts_keep_a_fully_contained_object_and_the_rest_of_its_host(store, make_clip, tmp_path):
    """A screw entirely inside a board: the old `mask & (owners == 1)` rule claimed every one of the screw's
    pixels for "more than one owner" and dropped it completely, and holed the board where the screw sat. `paint`
    paints the board (bigger) first and the screw (smaller) on top, so the screw's cutout is its own full area
    and the board keeps whatever is left over."""
    project, clip = make_clip(n_frames=1, classes=("board", "screw"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=1)]
    store.write_clip(clip)
    board = np.zeros((60, 120), bool)
    board[5:45, 5:45] = True  # 40x40 = 1600 px
    screw = np.zeros((60, 120), bool)
    screw[15:25, 15:25] = True  # 10x10 = 100 px, entirely inside the board
    store.write_mask(clip.id, 1, 0, board)
    store.write_mask(clip.id, 2, 0, screw)
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "coco", "masks"]
    src = _export(store, project, clip, formats, data_root)

    cutouts, _sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None)
    assert mode == "per_object"
    by_class = {c.class_id: int(c.mask.sum()) for c in cutouts}
    assert by_class == {1: 100, 0: 1600 - 100}  # the screw keeps its full 10x10; the board keeps the rest


def test_cutouts_restore_a_hole_from_the_mask_png(store, make_clip, tmp_path):
    """COCO polygons are outer contours only (`mask_to_polygons` uses `RETR_EXTERNAL`): a ring's polygon,
    refilled by `polygons_to_mask`, is a solid square with its hole filled in. Intersecting the painted canvas
    with the semantic `masks` PNG (which keeps the true, holed mask) restores the hole; a dot sitting inside
    that hole still gets its own full cutout, since `paint` puts it on top of the (reconstructed, hole-filled)
    ring."""
    project, clip = make_clip(n_frames=1, classes=("ring", "dot"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=1)]
    store.write_clip(clip)
    ring = np.zeros((60, 120), bool)
    ring[10:50, 10:50] = True  # 40x40 outer = 1600 px
    ring[20:34, 20:34] = False  # 14x14 hole = 196 px -> the ring itself is 1600 - 196 = 1404 px
    dot = np.zeros((60, 120), bool)
    dot[22:32, 22:32] = True  # 10x10 = 100 px, entirely inside the hole
    store.write_mask(clip.id, 1, 0, ring)
    store.write_mask(clip.id, 2, 0, dot)
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "coco", "masks"]
    src = _export(store, project, clip, formats, data_root)

    cutouts, _sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None)
    assert mode == "per_object"
    by_class = {c.class_id: int(c.mask.sum()) for c in cutouts}
    assert int(ring.sum()) == 1404
    assert by_class[0] == 1404  # cut without the (polygon-filled) hole, matching the ring mask's own pixel count
    assert by_class[1] == 100  # the dot keeps its full cutout


def test_cutouts_resolve_shared_pixels_per_class_a_washer_keeps_its_hole_on_a_board(store, make_clip, tmp_path):
    """A washer (a ring) lying on a larger board of ANOTHER class: the real (semantic PNG) mask shows the board
    through the washer's hole, because the board is painted first (largest first) and the washer -- smaller, and
    with a genuine hole -- is painted on top of it, leaving the hole showing whatever is under it. Resolving
    shared pixels one class at a time (instead of on one canvas shared by every object in the frame) keeps that
    hole: the washer's own canvas is built only from the washer's (outer-contour, hole-filled) polygon, but is
    then intersected only with the washer's OWN class in the semantic PNG, so the board's pixels in the hole are
    never claimed by the washer."""
    project, clip = make_clip(n_frames=1, classes=("washer", "board"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=1)]
    store.write_clip(clip)
    board = np.zeros((60, 120), bool)
    board[5:55, 5:105] = True  # 50x100 = 5000 px, covers the washer's whole bounding box
    washer = np.zeros((60, 120), bool)
    washer[10:50, 10:50] = True  # 40x40 outer = 1600 px
    washer[20:40, 20:40] = False  # 20x20 hole = 400 px -> the washer itself is 1600 - 400 = 1200 px
    store.write_mask(clip.id, 1, 0, washer)
    store.write_mask(clip.id, 2, 0, board)
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "coco", "masks"]
    src = _export(store, project, clip, formats, data_root)

    cutouts, _sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None)
    assert mode == "per_object"
    by_class = {c.class_id: int(c.mask.sum()) for c in cutouts}
    assert int(washer.sum()) == 1200
    assert by_class[0] == 1200  # the washer's hole is restored: the board's pixels there are not counted as its own
    assert by_class[1] == 5000 - 1200  # 3800: the board's full area minus the ring (the hole's 400 px stay board's)


def test_cutouts_still_leak_a_hole_onto_a_host_of_the_same_class(store, make_clip, tmp_path):
    """Known remaining limit: a holed object lying on a larger object of its OWN class still carries the host's
    pixels through its hole. The semantic PNG only records a class per pixel, not an instance, so once two
    objects share a class it cannot tell them apart in the hole; and per-class canvases are still built from
    COCO's outer-contour (hole-filled) polygons, so the same-class canvas fills the hole in too. Resolving per
    class (this change) fixes a DIFFERENT class showing through a hole; it cannot fix this one, since the export
    itself no longer distinguishes the two objects once they share a class."""
    project, clip = make_clip(n_frames=1, classes=("part",))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=0)]
    store.write_clip(clip)
    board = np.zeros((60, 120), bool)
    board[5:55, 5:105] = True  # 50x100 = 5000 px, covers the washer's whole bounding box
    washer = np.zeros((60, 120), bool)
    washer[10:50, 10:50] = True  # 40x40 outer = 1600 px
    washer[20:40, 20:40] = False  # 20x20 hole = 400 px -> the washer itself is 1600 - 400 = 1200 px
    store.write_mask(clip.id, 1, 0, washer)
    store.write_mask(clip.id, 2, 0, board)
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "coco", "masks"]
    src = _export(store, project, clip, formats, data_root)

    cutouts, _sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None)
    assert mode == "per_object"
    areas = sorted(int(c.mask.sum()) for c in cutouts)
    # both objects are class 0, so they can't be told apart by class_id here: the washer's cutout is its full
    # 1600 px outer square (its hole wrongly filled in with the board's own pixels) and the board keeps only
    # what the washer's (filled) footprint left it, 5000 - 1600 = 3400 -- not the correct 1200 / 3800 split.
    assert areas == [1600, 3400]


def test_cutouts_fall_back_to_class_regions_without_coco(store, make_clip, tmp_path):
    project, clip = _multi(store, make_clip)
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "masks"]
    src = _export(store, project, clip, formats, data_root)
    cutouts, _sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None)
    assert mode == "per_class_region"
    assert len(cutouts) == 2 * _train_images(src)
    # the two touching "a" objects come out as one 20x45 region; "b" keeps only what the mask PNG shows of it
    assert sorted((c.class_id, int(c.mask.sum())) for c in cutouts[:2]) == [(0, 900), (1, 375)]


def test_the_derived_manifest_records_how_cutouts_were_made(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    with_coco = _export(store, project, clip, ["yolo_seg", "coco", "masks"], data_root)
    without = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    per_object = run_augment(data_root, with_coco["export_id"], AugmentOptions(per_image=1, seed=0))
    fallback = run_augment(data_root, without["export_id"], AugmentOptions(per_image=1, seed=0))
    assert per_object["cutouts"] == "per_object" and fallback["cutouts"] == "per_class_region"
    assert json.loads((Path(per_object["path"]) / "export.json").read_text())["augment"]["cutouts"] == "per_object"
    assert json.loads((Path(fallback["path"]) / "export.json").read_text())["augment"]["cutouts"] == \
        "per_class_region"


def _lit_source(stem: str) -> str:
    """`lit_<seed>_<source stem>_<k>` -> the source stem."""
    return stem.split("_", 2)[2].rsplit("_", 1)[0]


def test_lighting_variants_copy_each_train_frame_with_its_labels(store, make_clip, tmp_path):
    project, clip = _multi(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "yolo_det", "coco", "masks"], data_root)
    _make_pool(data_root)
    src_dir = Path(src["path"])
    before = _snapshot(src_dir)

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=4, lighting=2))
    new_dir = Path(result["path"])
    manifest = json.loads((new_dir / "export.json").read_text())
    src_manifest = json.loads((src_dir / "export.json").read_text())
    real_train = sorted(p.stem for p in (src_dir / "yolo_seg" / "images" / "train").glob("*.jpg"))

    lit = sorted(p.stem for p in (new_dir / "yolo_seg" / "images" / "train").glob("lit_*.jpg"))
    assert len(lit) == 2 * len(real_train) == manifest["augment"]["lighting_images"]
    assert manifest["augment"]["lighting_variants"] == 2
    assert {_lit_source(s) for s in lit} == set(real_train)
    for stem in lit:
        source = _lit_source(stem)
        for fmt in ("yolo_seg", "yolo_det"):
            rel = Path(fmt) / "labels" / "train"
            assert (new_dir / rel / f"{stem}.txt").read_bytes() == (src_dir / rel / f"{source}.txt").read_bytes()
        rel = Path("masks") / "masks" / "train"
        assert (new_dir / rel / f"{stem}.png").read_bytes() == (src_dir / rel / f"{source}.png").read_bytes()
        relit = cv2.imread(str(new_dir / "images" / f"{stem}.jpg"))
        original = cv2.imread(str(src_dir / "images" / f"{source}.jpg"))
        assert relit.shape == original.shape and not np.array_equal(relit, original)
        for fmt in ("yolo_seg", "yolo_det", "coco", "masks"):
            assert (new_dir / fmt / "images" / "train" / f"{stem}.jpg").is_file()

    from pycocotools.coco import COCO

    coco = COCO(str(new_dir / "coco" / "annotations" / "instances_train.json"))
    by_name = {Path(im["file_name"]).stem: im["id"] for im in coco.dataset["images"]}
    assert len(by_name) == len(coco.dataset["images"])  # no duplicate file names
    ids = [a["id"] for a in coco.dataset["annotations"]]
    assert len(ids) == len(set(ids))
    for stem in lit:
        mine = coco.loadAnns(coco.getAnnIds(imgIds=[by_name[stem]]))
        theirs = coco.loadAnns(coco.getAnnIds(imgIds=[by_name[_lit_source(stem)]]))
        assert [(a["category_id"], a["segmentation"]) for a in mine] == \
            [(a["category_id"], a["segmentation"]) for a in theirs]

    syn = manifest["augment"]["synthetic_images"]
    assert manifest["counts"]["train"]["images"] == len(real_train) + syn + len(lit)
    assert manifest["counts"]["val"] == src_manifest["counts"]["val"]
    for entry, src_entry in zip(manifest["formats"], src_manifest["formats"], strict=True):
        assert entry["images"]["train"] == src_entry["images"]["train"] + syn + len(lit)
        assert entry["images"]["val"] == src_entry["images"]["val"]
        assert entry["annotations"]["val"] == src_entry["annotations"]["val"]
    yolo = next(f for f in manifest["formats"] if f["name"] == "yolo_seg")
    lines = sum(len(p.read_text().splitlines()) for p in (new_dir / "yolo_seg" / "labels" / "train").glob("*.txt"))
    assert yolo["annotations"]["train"] == lines
    coco_entry = next(f for f in manifest["formats"] if f["name"] == "coco")
    assert coco_entry["annotations"]["train"] == len(coco.dataset["annotations"])
    assert _snapshot(src_dir) == before


def test_a_lighting_only_copy_needs_no_background_pool(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "coco", "masks"], data_root)

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=0, lighting=1))
    new_dir = Path(result["path"])
    names = sorted(p.name for p in (new_dir / "yolo_seg" / "images" / "train").glob("*.jpg"))
    assert names and not any(n.startswith("syn_") for n in names)
    assert sum(n.startswith("lit_") for n in names) == len(names) // 2
    manifest = json.loads((new_dir / "export.json").read_text())
    assert manifest["augment"]["synthetic_images"] == 0 and manifest["augment"]["pool"] is None
    assert manifest["name"] == f"{src['export_id']} +1 lighting"
    assert "-lit1-" in result["export_id"] and "-aug" not in result["export_id"]


def test_lighting_variants_are_reproducible_per_seed(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    opts = AugmentOptions(per_image=0, lighting=1, seed=5)

    def lit_images(path):
        return {p.name: p.read_bytes() for p in (Path(path) / "images").glob("lit_*.jpg")}

    a = lit_images(run_augment(data_root, src["export_id"], opts)["path"])
    b = lit_images(run_augment(data_root, src["export_id"], opts)["path"])
    c = lit_images(run_augment(data_root, src["export_id"], AugmentOptions(per_image=0, lighting=1, seed=6))["path"])
    assert a and a == b and list(a.values()) != list(c.values())


def test_check_augment_lighting_errors(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    with pytest.raises(BadRequest, match="at least one"):
        check_augment(data_root, src["export_id"], AugmentOptions(per_image=0, lighting=0))
    with pytest.raises(BadRequest, match="lighting"):
        check_augment(data_root, src["export_id"], AugmentOptions(per_image=0, lighting=4))
    with pytest.raises(BadRequest, match="pool"):
        check_augment(data_root, src["export_id"], AugmentOptions(per_image=1, lighting=1))  # composites need it
    check_augment(data_root, src["export_id"], AugmentOptions(per_image=0, lighting=3))


def test_a_copy_with_composites_and_lighting_names_both(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=2, lighting=1))
    assert result["name"] == f"{src['export_id']} ×2 augmented +1 lighting"
    assert "-aug2x-lit1-" in result["export_id"]


def _same_cutouts(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.class_id == y.class_id
        assert np.array_equal(x.image, y.image) and np.array_equal(x.mask, y.mask)


@pytest.mark.parametrize("formats", [["yolo_seg", "coco", "masks"], ["yolo_seg", "masks"]])
def test_cutouts_are_identical_for_one_and_two_workers(store, make_clip, tmp_path, formats):
    from vdp.augment import derive

    project, clip = _multi(store, make_clip, n=6)
    src = _export(store, project, clip, formats, tmp_path / "data")
    old = derive.UNIT_STEMS
    derive.UNIT_STEMS = 2  # several units even for a small fixture (the worker processes use the default)
    try:
        ticks = []
        one = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: ticks.append(1), workers=1)
        two = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None, workers=2)
    finally:
        derive.UNIT_STEMS = old
    _same_cutouts(one[0], two[0])
    assert one[1] == two[1] and one[2] == two[2]
    assert len(ticks) == _train_images(src)


def test_a_unit_without_objects_contributes_nothing_and_keeps_order(store, make_clip, tmp_path):
    from vdp.augment import derive

    project, clip = _labelled_at(store, make_clip, 6, [0, 1, 4, 5])  # frames 2, 3 are empty
    formats = ["yolo_seg", "coco", "masks"]
    src = _export(store, project, clip, formats, tmp_path / "data")
    whole = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None, workers=1)
    old = derive.UNIT_STEMS
    derive.UNIT_STEMS = 1
    try:
        split = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None, workers=1)
    finally:
        derive.UNIT_STEMS = old
    _same_cutouts(whole[0], split[0])
    assert whole[1] == split[1]
    assert len(split[0]) < len(split[1])


def test_cancel_is_passed_to_the_unit_runner(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    formats = ["yolo_seg", "masks"]
    src = _export(store, project, clip, formats, tmp_path / "data")
    with pytest.raises(ExportCancelled):
        _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None, workers=1, cancel=lambda: True)


def test_cutout_frames_keep_the_sorted_path_order(tmp_path):
    # sorted paths, not sorted stems: "a-b.png" < "a.png" ('-' < '.') although "a" < "a-b"; composites pick a
    # frame size by index, so the order must stay the one the serial loop used
    masks = tmp_path / "masks" / "masks" / "train"
    masks.mkdir(parents=True)
    for stem, (w, h) in {"a": (10, 20), "a-b": (30, 40)}.items():
        cv2.imwrite(str(masks / f"{stem}.png"), np.zeros((h, w), np.uint8))
    _cutouts, sizes, _mode = _cutouts_and_sizes(tmp_path, ["yolo_seg", "masks"], 0, lambda: None, workers=1)
    assert sizes == [(30, 40), (10, 20)]


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _comparable_manifest(path: Path) -> dict:
    m = json.loads((Path(path) / "export.json").read_text())
    return {k: v for k, v in m.items() if k not in ("export_id", "name", "created_at")}


def test_composites_are_identical_for_one_and_three_workers(store, make_clip, tmp_path, monkeypatch):
    from vdp.augment import derive

    monkeypatch.setattr(derive, "UNIT_COMPOSITES", 2)  # several composite units even for a small fixture
    project, clip = _multi(store, make_clip, n=10)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "yolo_det", "coco", "masks"], data_root)
    _make_pool(data_root)
    opts = AugmentOptions(per_image=2, max_objects=3, seed=4, lighting=1)

    one = run_augment(data_root, src["export_id"], opts, workers=1)
    three = run_augment(data_root, src["export_id"], opts, workers=3)

    a, b = _files(Path(one["path"])), _files(Path(three["path"]))
    a.pop("export.json")
    b.pop("export.json")
    assert any(name.startswith("images/syn_") for name in a)
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []
    assert _comparable_manifest(one["path"]) == _comparable_manifest(three["path"])
    assert not (Path(one["path"]) / ".work").exists()


def test_cancel_mid_composites_on_the_pool_removes_output(store, make_clip, tmp_path, monkeypatch):
    from vdp.augment import derive

    monkeypatch.setattr(derive, "UNIT_COMPOSITES", 1)
    project, clip = _labelled(store, make_clip, n=10)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    seen = {"done": 0}
    stop_at = src["counts"]["train"]["images"] + 3  # past the cutout ticks: inside the composite phase

    def progress(done, total):
        seen["done"] = done

    with pytest.raises(ExportCancelled):
        run_augment(data_root, src["export_id"], AugmentOptions(per_image=3), progress=progress,
                    cancel=lambda: seen["done"] >= stop_at, workers=2)
    assert sorted(p.name for p in (data_root / "exports").iterdir()) == [src["export_id"]]


def test_an_unreadable_background_fails_the_job_and_removes_output(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    for p in (data_root / "backgrounds" / "dtd" / "x").glob("*.jpg"):
        p.write_bytes(b"not a jpeg")

    with pytest.raises(Exception, match="cannot identify image"):
        run_augment(data_root, src["export_id"], AugmentOptions(per_image=3), workers=2)
    assert sorted(p.name for p in (data_root / "exports").iterdir()) == [src["export_id"]]


def test_lighting_copies_are_identical_for_one_and_three_workers(store, make_clip, tmp_path, monkeypatch):
    from vdp.augment import derive

    assert hasattr(derive, "_lighting_unit")
    monkeypatch.setattr(derive, "UNIT_LIGHTING", 2)  # several lighting units even for a small fixture
    project, clip = _multi(store, make_clip, n=10)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "coco", "masks"], data_root)
    opts = AugmentOptions(per_image=0, seed=6, lighting=2)

    one = run_augment(data_root, src["export_id"], opts, workers=1)
    three = run_augment(data_root, src["export_id"], opts, workers=3)

    a, b = _files(Path(one["path"])), _files(Path(three["path"]))
    a.pop("export.json")
    b.pop("export.json")
    assert any(name.startswith("images/lit_") for name in a)
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []
    assert _comparable_manifest(one["path"]) == _comparable_manifest(three["path"])
