"""Reading a run: Ultralytics' results.csv, the best epoch, ground-truth labels and the worker log."""

from __future__ import annotations

import csv
from pathlib import Path

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


def read_ground_truth(label_path: Path) -> list[dict]:
    """YOLO-seg label lines (one contour per line, as vsp exports them) as preview instances."""
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
