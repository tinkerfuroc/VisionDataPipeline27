"""Run one export: plan, copy each image once, let every chosen exporter serialise the plan, write the manifest."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from ..store import ProjectStore, now_iso
from .base import ExportSummary, place_image, shared_image
from .naming import clean_name, reserve_export_dir
from .plan import ExportOptions, ExportPlan, build_plan, check_request


class ExportCancelled(Exception):
    pass


def _counts(plan: ExportPlan) -> dict:
    out = {}
    for split in ("train", "val"):
        frames = plan.of_split(split)
        annotated = sum(1 for f in frames if f.objects)
        out[split] = {"images": len(frames), "annotated": annotated, "empty": len(frames) - annotated}
    return out


def write_dataset(
    plan: ExportPlan, formats: list[str], out: Path, tick: Callable[[], None], *, link_images: bool = False,
) -> tuple[list[ExportSummary], dict, bool]:
    """Place each frame image at `out/images/<stem>.jpg` (copied, or hard-linked when `link_images`), then
    let every chosen exporter serialise the plan. `run_export` places store frames by copy; a derived
    export places frames it already owns (in another export, or freshly written composites) by hard link."""
    from . import EXPORTERS  # here, not at module level: the package __init__ imports this module

    (out / "images").mkdir()
    for frame in plan.frames:
        if link_images:
            place_image(frame.image_path, shared_image(out, frame))
        else:
            # a copy: store frames are never linked outside
            shutil.copy2(frame.image_path, shared_image(out, frame))
        tick()
    summaries = [EXPORTERS[name].write(plan, out, tick) for name in formats]
    counts = _counts(plan)
    linked = all(s.linked for s in summaries)
    return summaries, counts, linked


def manifest_for(
    plan: ExportPlan, export_id: str, summaries: list[ExportSummary], counts: dict, linked: bool,
    name: str | None = None,
) -> dict:
    """The manifest `run_export` writes to `export.json` (and a derived export starts its own manifest
    from)."""
    return {
        "export_id": export_id, "name": name, "created_at": now_iso(),
        "project": {"id": plan.project_id, "name": plan.project_name},
        "options": {**asdict(plan.options), "val_clip_ids": list(plan.val_clip_ids)},
        "clips": [
            {"id": cid, "name": plan.clip_names[cid],
             "train": sum(1 for f in plan.frames if f.clip_id == cid and f.split == "train"),
             "val": sum(1 for f in plan.frames if f.clip_id == cid and f.split == "val")}
            for cid in plan.clip_ids
        ],
        "classes": [
            {"id": c.id, "name": c.name, "color": c.color, "yolo_id": c.id, "coco_id": c.id + 1,
             "mask_value": c.id + 1}
            for c in sorted(plan.classes, key=lambda c: c.id)
        ],
        "formats": [asdict(s) for s in summaries],
        "counts": counts,
        # false: the filesystem refused hardlinks, so images were copied per format
        "images_hardlinked": linked,
    }


def run_export(
    store: ProjectStore, project_id: str, clip_ids: list[str], formats: list[str], options: ExportOptions,
    exports_root: Path, progress: Callable[[int, int], None] | None = None, cancel: Callable[[], bool] | None = None,
    name: str | None = None,
) -> dict:
    check_request(store, project_id, clip_ids, formats, options)
    name = clean_name(name)
    n_candidates = sum(
        len([i for i in range(c.frame_count) if i not in set(c.deleted_frames)][:: options.stride])
        for c in (store.read_clip(cid) for cid in clip_ids)
    )
    state = {"done": 0, "total": max(1, n_candidates * (2 + len(formats)))}

    def tick() -> None:
        if cancel is not None and cancel():
            raise ExportCancelled
        state["done"] += 1
        if progress is not None:
            progress(min(state["done"], state["total"]), state["total"])

    export_id, out = reserve_export_dir(exports_root, name)
    try:
        plan = build_plan(store, project_id, clip_ids, options, tick=tick)
        # include_empty=False may have dropped frames: shrink the total so the bar still ends at 100%
        state["total"] = state["done"] + len(plan.frames) * (1 + len(formats))
        summaries, counts, linked = write_dataset(plan, formats, out, tick)
        manifest = manifest_for(plan, export_id, summaries, counts, linked, name=name)
        (out / "export.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # never leave a half-written dataset that looks complete
        raise
    return {
        "export_id": export_id, "path": str(out), "formats": list(formats), "counts": counts, "linked": linked,
        "name": name,
    }
