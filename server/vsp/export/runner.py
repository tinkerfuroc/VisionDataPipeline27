"""Run one export: plan, copy each image once, let every chosen exporter serialise the plan, write the manifest."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from ..store import ProjectStore, now_iso
from .base import shared_image
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


def run_export(
    store: ProjectStore, project_id: str, clip_ids: list[str], formats: list[str], options: ExportOptions,
    exports_root: Path, progress: Callable[[int, int], None] | None = None, cancel: Callable[[], bool] | None = None,
) -> dict:
    from . import EXPORTERS  # here, not at module level: the package __init__ imports this module

    check_request(store, project_id, clip_ids, formats, options)
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

    exports_root = Path(exports_root)
    exports_root.mkdir(parents=True, exist_ok=True)
    export_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = exports_root / export_id
    try:
        plan = build_plan(store, project_id, clip_ids, options, tick=tick)
        # include_empty=False may have dropped frames: shrink the total so the bar still ends at 100%
        state["total"] = state["done"] + len(plan.frames) * (1 + len(formats))
        (out / "images").mkdir(parents=True)
        for frame in plan.frames:
            shutil.copy2(frame.image_path, shared_image(out, frame))  # a copy: store frames are never linked outside
            tick()
        summaries = [EXPORTERS[name].write(plan, out, tick) for name in formats]
        counts = _counts(plan)
        linked = all(s.linked for s in summaries)
        manifest = {
            "export_id": export_id, "created_at": now_iso(),
            "project": {"id": plan.project_id, "name": plan.project_name},
            "options": {**asdict(plan.options), "val_clip_ids": list(plan.val_clip_ids)},
            "clips": [
                {"id": cid, "name": plan.clip_names[cid],
                 "train": sum(1 for f in plan.frames if f.clip_id == cid and f.split == "train"),
                 "val": sum(1 for f in plan.frames if f.clip_id == cid and f.split == "val")}
                for cid in plan.clip_ids
            ],
            "classes": [
                {"id": c.id, "name": c.name, "color": c.color, "yolo_id": c.id, "coco_id": c.id + 1, "mask_value": c.id + 1}
                for c in sorted(plan.classes, key=lambda c: c.id)
            ],
            "formats": [asdict(s) for s in summaries],
            "counts": counts,
            "images_hardlinked": linked,  # false: the filesystem refused hardlinks, so images were copied per format
        }
        (out / "export.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # never leave a half-written dataset that looks complete
        raise
    return {"export_id": export_id, "path": str(out), "formats": list(formats), "counts": counts, "linked": linked}
