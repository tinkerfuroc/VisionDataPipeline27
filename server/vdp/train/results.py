"""Reading a run: Ultralytics' results.csv, the best epoch, ground-truth labels and the worker log."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..export.polygons import polygons_to_mask

MASK_MAP = "metrics/mAP50-95(M)"
MASK_MAP50 = "metrics/mAP50(M)"
BOX_MAP = "metrics/mAP50-95(B)"
BOX_MAP50 = "metrics/mAP50(B)"


def read_metrics(run_dir: Path) -> list[dict]:
    """One dict per epoch row; header names are stripped (Ultralytics pads them), values are
    floats, epoch an int."""
    try:
        text = (Path(run_dir) / "train" / "results.csv").read_text()
    except FileNotFoundError:
        return []
    rows = []
    for raw in csv.DictReader(text.splitlines()):
        row: dict = {}
        for key, value in raw.items():
            if key is None or value is None:
                continue
            try:
                row[key.strip()] = float(value.strip())
            except ValueError:
                continue
        if "epoch" in row:
            row["epoch"] = int(row["epoch"])
            rows.append(row)
    return rows


def best_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(rows, key=lambda r: (r.get(MASK_MAP, -1.0), r["epoch"]))


def summarize(rows: list[dict]) -> dict | None:
    best = best_row(rows)
    if best is None:
        return None
    return {
        "epoch": best["epoch"], "mask_map50": best.get(MASK_MAP50),
        "mask_map50_95": best.get(MASK_MAP),
        "box_map50": best.get(BOX_MAP50), "box_map50_95": best.get(BOX_MAP),
        "train_seconds": rows[-1].get("time"),
    }


FINAL_VAL = "final_val.json"  # in the run dir: best.pt on the whole val split, for a run validated on a subset


def run_summary(run_dir: Path) -> dict | None:
    """`summarize` of the run's results.csv; for a run validated per epoch on a fixed val subset, its mAPs are those
    of the final evaluation of best.pt on the whole val split (FINAL_VAL, with `val_images` = how many images)."""
    summary = summarize(read_metrics(run_dir))
    try:
        final = json.loads((Path(run_dir) / FINAL_VAL).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return summary
    if summary is None:
        return None
    metrics = final.get("metrics", {})
    summary.update(
        mask_map50=metrics.get(MASK_MAP50), mask_map50_95=metrics.get(MASK_MAP), box_map50=metrics.get(BOX_MAP50),
        box_map50_95=metrics.get(BOX_MAP), val_images=final.get("images"),
    )
    return summary


def read_ground_truth(label_path: Path) -> list[dict]:
    """YOLO-seg label lines (one contour per line, as vdp exports them) as preview instances."""
    try:
        text = Path(label_path).read_text()
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least three points
            continue
        out.append(
            {"class_id": int(parts[0]), "polygons": [[float(v) for v in parts[1:]]]}
        )
    return out


def tail(path: Path, n: int) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])


def _union_mask(instances: list[dict], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), bool)
    for inst in instances:
        for flat in inst["polygons"]:
            points = [(flat[i] * width, flat[i + 1] * height) for i in range(0, len(flat) - 1, 2)]
            if len(points) >= 3:
                canvas |= polygons_to_mask([points], width, height)
    return canvas


def mask_iou(predictions: list[dict], ground_truth: list[dict], width: int, height: int) -> float | None:
    """IoU of the union of predicted polygons against the union of labelled ones; None when both
    are empty."""
    pred = _union_mask(predictions, width, height)
    gt = _union_mask(ground_truth, width, height)
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return None
    return round(int(np.logical_and(pred, gt).sum()) / union, 4)
