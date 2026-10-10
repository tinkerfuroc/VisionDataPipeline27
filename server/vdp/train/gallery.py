"""Validation predictions as a gallery: frame order, per-frame mask IoU (cached), and the Missed /
Worst filters."""

from __future__ import annotations

from pathlib import Path

from ..store import NotFound
from .results import mask_iou, read_ground_truth
from .runs import atomic_write_json, read_json

FILTERS = ("all", "missed", "worst")


def load_gallery(run_dir: Path, labels_dir: Path) -> dict:
    preds_path = run_dir / "predictions.json"
    preds = read_json(preds_path)
    if preds is None:
        raise NotFound("this run has no predictions yet")
    cache_path = run_dir / "predictions_iou.json"
    cache = None
    if cache_path.is_file() and cache_path.stat().st_mtime >= preds_path.stat().st_mtime:
        cache = read_json(cache_path)
    fresh: dict[str, float | None] = {}
    items = []
    for index, im in enumerate(sorted(preds["images"], key=lambda im: im["name"])):
        gt = read_ground_truth(labels_dir / f"{Path(im['name']).stem}.txt")
        if cache is not None and im["name"] in cache:
            iou = cache[im["name"]]
        else:
            iou = mask_iou(im["instances"], gt, im["width"], im["height"])
        fresh[im["name"]] = iou
        items.append({"index": index, "name": im["name"], "width": im["width"],
                      "height": im["height"], "predictions": im["instances"],
                      "ground_truth": gt, "iou": iou})
    if cache != fresh:
        atomic_write_json(cache_path, fresh)
    return {"conf": preds.get("conf"), "items": items}


def filter_items(items: list[dict], name: str) -> list[dict]:
    if name == "missed":
        return [it for it in items if it["ground_truth"] and not it["predictions"]]
    if name == "worst":
        scored = [it for it in items if it["ground_truth"] and it["iou"] is not None]
        return sorted(scored, key=lambda it: (it["iou"], it["index"]))
    return list(items)
