"""Helpers shared by the training tests: a minimal yolo_seg export, the fake worker command, polling."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

FAKE_WORKER = [sys.executable, str(Path(__file__).with_name("fake_train_worker.py"))]
EXPORT_ID = "20260101-000000-abcdef"
SQUARE = "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"


def make_export(data_root: Path, export_id: str = EXPORT_ID, n_train: int = 2, n_val: int = 2) -> Path:
    """An export directory shaped like vsp's exporter output, with one class and a square on every image."""
    d = data_root / "exports" / export_id
    y = d / "yolo_seg"
    for split, n in (("train", n_train), ("val", n_val)):
        (y / "images" / split).mkdir(parents=True, exist_ok=True)
        (y / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((64, 96, 3), 90, np.uint8)
            cv2.rectangle(img, (24, 16), (72, 48), (200, 200, 200), -1)
            assert cv2.imwrite(str(y / "images" / split / f"clip_{i:06d}.jpg"), img)
            (y / "labels" / split / f"clip_{i:06d}.txt").write_text(SQUARE + "\n")
    (y / "data.yaml").write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: "Module"\n')
    manifest = {
        "export_id": export_id, "created_at": "2026-01-01T00:00:00+00:00",
        "project": {"id": "p1", "name": "Bench"},
        "classes": [{"id": 0, "name": "Module", "color": "#e6194b", "yolo_id": 0, "coco_id": 1, "mask_value": 1}],
        "formats": [{"name": "yolo_seg", "images": {"train": n_train, "val": n_val},
                     "annotations": {"train": n_train, "val": n_val}, "linked": True}],
    }
    (d / "export.json").write_text(json.dumps(manifest))
    return d


def wait_for(predicate, timeout: float = 15.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")
