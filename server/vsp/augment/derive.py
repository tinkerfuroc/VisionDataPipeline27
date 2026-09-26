"""Derived export: real frames plus synthetic composites on background textures (spec B2).

Real frames are never re-labelled: their images are hard-linked in (as for any export) and their per-format
label artifacts (`yolo_seg`/`yolo_det` label files, `masks` PNGs, COCO images+annotations) are copied
verbatim from the source, for both splits. Only synthetic (composite) frames go through the ordinary export
pipeline (`write_dataset`, i.e. the same exporters every export uses); their output is then merged with the
copied real output (COCO's train JSON gets the synthetic images/annotations appended; everything else lives
in its own per-stem file, so "merge" is just placing more files side by side).

Cutouts for composites are still built from the source's semantic `masks` PNGs (one blob per class, per
train frame) -- fine, since that's new synthetic data, not a real frame's own labels.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from ..export import ExportCancelled, manifest_for, write_dataset
from ..export.base import ExportSummary, place_image
from ..export.naming import MAX_NAME
from ..export.plan import ExportPlan, PlanFrame, PlanObject, classes_from_manifest, options_from_manifest
from ..export.polygons import clean_mask
from ..labeling.errors import BadRequest
from ..train.models import find_export
from .backgrounds import POOLS, list_images
from .composite import Cutout, cutout_from, make_composite


@dataclass(frozen=True)
class AugmentOptions:
    per_image: int = 3
    max_objects: int = 2
    seed: int = 0
    pool: str = "dtd"


def _derived_name(source_name: str, per_image: int) -> str:
    """The source's display name (name, else id) plus ` ×<N> augmented`, truncated so the result never
    exceeds `MAX_NAME`: the suffix is always kept whole, and the source part is shortened (trailing
    whitespace stripped off the cut) to make room for it."""
    suffix = f" ×{per_image} augmented"
    base = source_name
    max_base = MAX_NAME - len(suffix)
    if len(base) > max_base:
        base = base[:max_base].rstrip()
    return f"{base}{suffix}"


def check_augment(data_root: Path, export_id: str, opts: AugmentOptions) -> dict:
    """The source manifest, once it and `opts` check out. Raises `NotFound` / `BadRequest`."""
    info = find_export(data_root, export_id)  # id pattern, existence, and the yolo_seg format
    if info.derived_from is not None:
        raise BadRequest(f"export {export_id} is already an augmented copy; augment the original export")
    if "masks" not in info.formats:
        raise BadRequest(f"export {export_id} has no masks format; re-export with Mask images ticked")
    if opts.pool not in POOLS:
        raise BadRequest(f"background pool {opts.pool} not found")
    if not list_images(data_root, opts.pool):
        raise BadRequest(f"background pool {opts.pool} is empty; download it first")
    if not 1 <= opts.per_image <= 10:
        raise BadRequest("per_image must be between 1 and 10")
    if not 1 <= opts.max_objects <= 3:
        raise BadRequest("max_objects must be between 1 and 3")
    if opts.seed < 0:
        raise BadRequest("seed must be 0 or more")
    manifest = json.loads((info.dir / "export.json").read_text())
    if manifest["counts"]["train"]["annotated"] == 0:
        raise BadRequest(f"export {export_id} has no train-split objects to build composites from")
    return manifest


def _cutouts_and_sizes(
    source_dir: Path, min_area_px: int, tick: Callable[[], None],
) -> tuple[list[Cutout], list[tuple[int, int]]]:
    """Cutouts (per class region) and `(width, height)` of every real train frame, read from the source's
    indexed mask PNGs. Feeds composites only -- never a real frame's own labels."""
    cutouts: list[Cutout] = []
    sizes: list[tuple[int, int]] = []
    masks_dir = source_dir / "masks" / "masks" / "train"
    for mask_path in sorted(masks_dir.glob("*.png")):
        tick()
        indexed = np.array(Image.open(mask_path))
        sizes.append((indexed.shape[1], indexed.shape[0]))
        values = sorted(int(v) for v in np.unique(indexed) if v > 0)
        if not values:
            continue
        image = np.array(Image.open(source_dir / "images" / f"{mask_path.stem}.jpg").convert("RGB"))
        for value in values:
            mask = clean_mask(indexed == value, min_area_px)
            if not mask.any():
                continue
            cut = cutout_from(image, mask, value - 1)
            if cut is not None:
                cutouts.append(cut)
    return cutouts, sizes


def _copy_label(source_dir: Path, out: Path, fmt: str, split: str, stem: str) -> None:
    if fmt in ("yolo_seg", "yolo_det"):
        rel = Path(fmt) / "labels" / split / f"{stem}.txt"
    elif fmt == "masks":
        rel = Path("masks") / "masks" / split / f"{stem}.png"
    else:
        return  # coco: images only here; its annotations are merged as one JSON in _merge_coco
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / rel, dst)


def _merge_real_frames(source_dir: Path, out: Path, formats: list[str], tick: Callable[[], None]) -> bool:
    """Hard-link every real frame's image (both splits, into `out/images/` and each format's own images
    dir) and copy its per-format label artifacts verbatim from the source."""
    linked = True
    for split in ("train", "val"):
        stems = sorted(p.stem for p in (source_dir / "masks" / "masks" / split).glob("*.png"))
        for stem in stems:
            tick()
            src_image = source_dir / "images" / f"{stem}.jpg"
            linked = place_image(src_image, out / "images" / f"{stem}.jpg") and linked
            for fmt in formats:
                linked = place_image(src_image, out / fmt / "images" / split / f"{stem}.jpg") and linked
                _copy_label(source_dir, out, fmt, split, stem)
    return linked


def _merge_coco(source_dir: Path, out: Path) -> None:
    """`instances_train.json` = the source's train JSON with the synthetic-only run's images/annotations
    appended (ids renumbered past the source's own); `instances_val.json` = the source's, copied verbatim."""
    annotations_dir = Path("coco") / "annotations"
    src_train = json.loads((source_dir / annotations_dir / "instances_train.json").read_text())
    synth_path = out / annotations_dir / "instances_train.json"
    synth_train = json.loads(synth_path.read_text())

    next_image_id = max((im["id"] for im in src_train["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in src_train["annotations"]), default=0) + 1
    id_map: dict[int, int] = {}
    images = list(src_train["images"])
    for im in sorted(synth_train["images"], key=lambda im: im["id"]):
        id_map[im["id"]] = next_image_id
        images.append({**im, "id": next_image_id})
        next_image_id += 1
    annotations = list(src_train["annotations"])
    for ann in sorted(synth_train["annotations"], key=lambda a: a["id"]):
        annotations.append({**ann, "id": next_ann_id, "image_id": id_map[ann["image_id"]]})
        next_ann_id += 1

    synth_path.write_text(json.dumps({**src_train, "images": images, "annotations": annotations}))
    val_name = annotations_dir / "instances_val.json"
    shutil.copy2(source_dir / val_name, out / val_name)


def _merged_format_summaries(
    src_formats: list[dict], synth_summaries: list[ExportSummary], real_linked: bool,
) -> list[dict]:
    synth_by_name = {s.name: asdict(s) for s in synth_summaries}
    empty = {"images": {"train": 0, "val": 0}, "annotations": {"train": 0, "val": 0}, "linked": True}
    merged = []
    for src in src_formats:
        synth = synth_by_name.get(src["name"], empty)
        merged.append({
            "name": src["name"],
            "images": {sp: src["images"][sp] + synth["images"][sp] for sp in ("train", "val")},
            "annotations": {sp: src["annotations"][sp] + synth["annotations"][sp] for sp in ("train", "val")},
            "linked": bool(src["linked"]) and bool(synth["linked"]) and real_linked,
        })
    return merged


def run_augment(
    data_root: Path, export_id: str, opts: AugmentOptions,
    progress: Callable[[int, int], None] | None = None, cancel: Callable[[], bool] | None = None,
) -> dict:
    data_root = Path(data_root)
    manifest = check_augment(data_root, export_id, opts)
    source_dir = data_root / "exports" / export_id
    new_id = f"{export_id}-aug{opts.per_image}x-{uuid4().hex[:4]}"
    out = data_root / "exports" / new_id
    out.mkdir()  # outside the try: as run_export, fail without touching (or later deleting) another export
    try:
        formats = [f["name"] for f in manifest["formats"]]
        min_area_px = int(manifest["options"]["min_area_px"])
        real_train_n = manifest["counts"]["train"]["images"]
        real_val_n = manifest["counts"]["val"]["images"]
        total_composites = opts.per_image * real_train_n
        # computed once, up front, from the source manifest alone; never shrunk or grown as the job runs
        merge_ticks = real_train_n + real_val_n
        total = real_train_n + total_composites + total_composites * (1 + len(formats)) + merge_ticks
        state = {"done": 0, "total": max(1, total)}

        def tick() -> None:
            if cancel is not None and cancel():
                raise ExportCancelled
            state["done"] += 1
            if progress is not None:
                progress(min(state["done"], state["total"]), state["total"])

        @lru_cache(maxsize=64)  # per run: freed once run_augment returns, never serves a stale re-download
        def load_background(path: Path) -> np.ndarray:
            return np.array(Image.open(path).convert("RGB"))

        cutouts, sizes = _cutouts_and_sizes(source_dir, min_area_px, tick)
        pool_images = list_images(data_root, opts.pool)

        work = out / ".work"
        work_masks, work_images = work / "masks", work / "images"
        work_masks.mkdir(parents=True)
        work_images.mkdir(parents=True)

        synthetic: list[PlanFrame] = []
        for n in range(total_composites):
            rng = np.random.default_rng([opts.seed, n])
            width, height = sizes[int(rng.integers(len(sizes)))]
            background = load_background(pool_images[int(rng.integers(len(pool_images)))])
            composite = make_composite(rng, width, height, background, cutouts, opts.max_objects, min_area_px)
            tick()
            if not composite.objects:
                continue
            stem = f"syn_{opts.seed}_{n:06d}"
            image_path = work_images / f"{stem}.jpg"
            Image.fromarray(composite.image).save(image_path, quality=90)
            objects = []
            for j, (class_id, mask) in enumerate(composite.objects):
                mask_path = work_masks / f"{stem}_{j}.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                objects.append(PlanObject(j, class_id, mask_path, min_area_px))
            synthetic.append(PlanFrame(
                clip_id="synthetic", frame_idx=len(synthetic), stem=stem, split="train",
                image_path=image_path, width=width, height=height, objects=tuple(objects),
            ))

        classes, class_map = classes_from_manifest(manifest)
        options = options_from_manifest(manifest["options"])
        plan = ExportPlan(
            project_id=manifest["project"]["id"], project_name=manifest["project"]["name"], classes=classes,
            class_map=class_map, options=options, clip_ids=tuple(c["id"] for c in manifest["clips"]),
            clip_names={c["id"]: c["name"] for c in manifest["clips"]}, val_clip_ids=options.val_clip_ids or (),
            frames=tuple(synthetic),  # the exporters run on the synthetic frames only; real ones are copied
        )
        synth_summaries, _synth_counts, synth_linked = write_dataset(
            plan, formats, out, tick, link_images=True,
        )

        real_linked = _merge_real_frames(source_dir, out, formats, tick)
        if "coco" in formats:
            _merge_coco(source_dir, out)
        for fmt in ("yolo_seg", "yolo_det"):
            if fmt in formats:
                shutil.copy2(source_dir / fmt / "data.yaml", out / fmt / "data.yaml")

        counts = {
            "train": {
                "images": real_train_n + len(synthetic),
                "annotated": manifest["counts"]["train"]["annotated"] + len(synthetic),
                "empty": manifest["counts"]["train"]["empty"],
            },
            "val": manifest["counts"]["val"],
        }
        linked = synth_linked and real_linked
        result_manifest = manifest_for(plan, new_id, synth_summaries, counts, linked)
        result_manifest["clips"] = manifest["clips"]
        result_manifest["formats"] = _merged_format_summaries(
            manifest["formats"], synth_summaries, real_linked,
        )
        result_manifest["counts"] = counts
        result_manifest["derived_from"] = export_id
        result_manifest["name"] = _derived_name(manifest.get("name") or export_id, opts.per_image)
        result_manifest["augment"] = {
            "per_image": opts.per_image, "max_objects": opts.max_objects, "seed": opts.seed,
            "pool": opts.pool, "synthetic_images": len(synthetic),
        }
        (out / "export.json").write_text(json.dumps(result_manifest, indent=2))
        shutil.rmtree(work)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {
        "export_id": new_id, "path": str(out), "counts": counts, "synthetic": len(synthetic),
        "name": result_manifest["name"],
    }
