"""Derived export: real frames plus synthetic composites on background textures (spec B2).

Reads a source export's manifest and masks (not the store), builds an `ExportPlan` that mixes the source's
real frames with freshly-composited synthetic ones, and lets the existing exporters (`write_dataset`) write
every format the source had. val is the source's val frames, unchanged; train gains `per_image` composites
per real train frame.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from ..export import ExportCancelled, manifest_for, write_dataset
from ..export.base import read_mask
from ..export.plan import ExportOptions, ExportPlan, PlanFrame, PlanObject
from ..labeling.errors import BadRequest
from ..store import NotFound
from ..store.models import ClassDef
from .backgrounds import list_images
from .composite import Cutout, cutout_from, make_composite


@dataclass(frozen=True)
class AugmentOptions:
    per_image: int = 3
    max_objects: int = 2
    seed: int = 0
    pool: str = "dtd"


@lru_cache(maxsize=64)
def _load_background(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def check_augment(data_root: Path, export_id: str, opts: AugmentOptions) -> dict:
    """The source manifest, once it and `opts` check out. Raises `NotFound` / `BadRequest`."""
    manifest_path = Path(data_root) / "exports" / export_id / "export.json"
    if not manifest_path.is_file():
        raise NotFound(f"export {export_id} not found")
    manifest = json.loads(manifest_path.read_text())
    if not any(f.get("name") == "masks" for f in manifest.get("formats", [])):
        raise BadRequest(f"export {export_id} has no masks format; re-export with Mask images ticked")
    if not list_images(data_root, opts.pool):
        raise BadRequest(f"background pool {opts.pool} is empty; download it first")
    if not 1 <= opts.per_image <= 10:
        raise BadRequest("per_image must be between 1 and 10")
    if not 1 <= opts.max_objects <= 3:
        raise BadRequest("max_objects must be between 1 and 3")
    if opts.seed < 0:
        raise BadRequest("seed must be 0 or more")
    return manifest


def _real_frames(source_dir: Path, min_area_px: int, work_masks: Path) -> list[PlanFrame]:
    """Every source frame, real objects rebuilt from the exported indexed masks (one binary PNG per class
    present), in `train`, `val` order, sorted by stem within each split."""
    frames: list[PlanFrame] = []
    frame_idx = 0
    for split in ("train", "val"):
        masks_dir = source_dir / "masks" / "masks" / split
        for mask_path in sorted(masks_dir.glob("*.png")):
            stem = mask_path.stem
            indexed = np.array(Image.open(mask_path))
            objects = []
            for value in sorted(int(v) for v in np.unique(indexed) if v > 0):
                obj_path = work_masks / f"{stem}_{value - 1}.png"
                Image.fromarray(((indexed == value) * 255).astype(np.uint8)).save(obj_path)
                objects.append(PlanObject(value - 1, value - 1, obj_path, min_area_px))
            frames.append(PlanFrame(
                clip_id=stem.split("_")[0], frame_idx=frame_idx, stem=stem, split=split,
                image_path=source_dir / "images" / f"{stem}.jpg",
                width=indexed.shape[1], height=indexed.shape[0], objects=tuple(objects),
            ))
            frame_idx += 1
    return frames


def _cutouts(train_frames: list[PlanFrame]) -> list[Cutout]:
    out: list[Cutout] = []
    for frame in train_frames:
        if not frame.objects:
            continue
        image = np.array(Image.open(frame.image_path).convert("RGB"))
        for obj in frame.objects:
            cut = cutout_from(image, read_mask(obj), obj.class_id)
            if cut is not None:
                out.append(cut)
    return out


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
        work = out / ".work"
        work_masks, work_images = work / "masks", work / "images"
        work_masks.mkdir(parents=True)
        work_images.mkdir(parents=True)
        min_area_px = int(manifest["options"]["min_area_px"])

        real = _real_frames(source_dir, min_area_px, work_masks)
        train_frames = [f for f in real if f.split == "train"]
        cutouts = _cutouts(train_frames)
        if not cutouts:
            raise BadRequest(f"export {export_id} has no train-split objects to build composites from")

        pool_images = list_images(data_root, opts.pool)
        total = opts.per_image * len(train_frames)
        state = {"done": 0, "total": max(1, total)}

        def tick() -> None:
            if cancel is not None and cancel():
                raise ExportCancelled
            state["done"] += 1
            if progress is not None:
                progress(min(state["done"], state["total"]), state["total"])

        synthetic: list[PlanFrame] = []
        for n in range(total):
            rng = np.random.default_rng([opts.seed, n])
            frame = train_frames[int(rng.integers(len(train_frames)))]
            background = _load_background(pool_images[int(rng.integers(len(pool_images)))])
            composite = make_composite(
                rng, frame.width, frame.height, background, cutouts, opts.max_objects, min_area_px,
            )
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
                clip_id="synthetic", frame_idx=len(real) + len(synthetic), stem=stem, split="train",
                image_path=image_path, width=frame.width, height=frame.height, objects=tuple(objects),
            ))

        classes = tuple(ClassDef(id=c["id"], name=c["name"], color=c["color"]) for c in manifest["classes"])
        val_clip_ids = tuple(manifest["options"]["val_clip_ids"] or ())
        options = ExportOptions(**{**manifest["options"], "val_clip_ids": val_clip_ids})
        plan = ExportPlan(
            project_id=manifest["project"]["id"], project_name=manifest["project"]["name"], classes=classes,
            options=options, clip_ids=tuple(c["id"] for c in manifest["clips"]),
            clip_names={c["id"]: c["name"] for c in manifest["clips"]}, val_clip_ids=val_clip_ids,
            frames=tuple(real + synthetic),
        )
        formats = [f["name"] for f in manifest["formats"]]
        state["total"] = state["done"] + len(plan.frames) * (1 + len(formats))
        summaries, counts, linked = write_dataset(plan, formats, out, tick, link_images=True)
        result_manifest = manifest_for(plan, new_id, summaries, counts, linked)
        result_manifest["clips"] = manifest["clips"]
        result_manifest["derived_from"] = export_id
        result_manifest["augment"] = {
            "per_image": opts.per_image, "max_objects": opts.max_objects, "seed": opts.seed,
            "pool": opts.pool, "synthetic_images": len(synthetic),
        }
        (out / "export.json").write_text(json.dumps(result_manifest, indent=2))
        shutil.rmtree(work)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {"export_id": new_id, "path": str(out), "counts": counts, "synthetic": len(synthetic)}
