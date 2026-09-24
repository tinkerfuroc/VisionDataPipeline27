"""Stands in for `python -m vsp.train.worker` in the scheduler, API and CLI tests.

Behaviour is chosen by run.json's params.epochs:
  1..12  write that many results.csv rows, last.pt, best.pt and predictions.json; exit 0
  13     write result.json {"error": ...}; exit 1
  14     print a line and exit 3 without result.json (a crash)
  99     write one row and last.pt, then wait for SIGTERM and exit 2 (a cancellable long run)
`--onnx` writes train/weights/best.onnx and exits 0.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

HEADER = (
    "                  epoch,                 time,      train/box_loss,      train/seg_loss,"
    "      train/cls_loss,"
    "metrics/mAP50(B),metrics/mAP50-95(B),metrics/mAP50(M),metrics/mAP50-95(M),val/box_loss,"
    "val/seg_loss,val/cls_loss"
)


def row(e: int) -> str:
    m = e / 10
    return (
        f"{e},{e * 2.5},{1 - m},{1 - m},{1 - m},{m + 0.1},{m},{m + 0.05},{m - 0.05},{1 - m},{1 - m},{1 - m}"
    )


def main() -> int:
    run_dir = Path(sys.argv[1])
    run = json.loads((run_dir / "run.json").read_text())
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    if "--onnx" in sys.argv:
        (weights / "best.onnx").write_bytes(b"onnx")
        return 0
    epochs = run["params"]["epochs"]
    print(f"fake worker gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} cwd={os.getcwd()} "
          f"autoinstall={os.environ.get('YOLO_AUTOINSTALL')} resume={run['resume']}", flush=True)
    (run_dir / "progress.json").write_text(
        json.dumps({"phase": "train", "epoch": 1, "epochs": epochs, "batch": 3, "batches": 10}))
    if epochs == 13:
        (run_dir / "result.json").write_text(json.dumps({"error": "OutOfMemoryError: CUDA out of memory"}))
        return 1
    if epochs == 14:
        print("Segmentation fault in a dataloader worker", flush=True)
        return 3
    n = 1 if epochs == 99 else epochs
    rows = "".join(row(e) + "\n" for e in range(1, n + 1))
    (run_dir / "train" / "results.csv").write_text(HEADER + "\n" + rows)
    (weights / "last.pt").write_bytes(b"last")
    if epochs == 99:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(2))
        while True:
            time.sleep(0.05)
    (weights / "best.pt").write_bytes(b"best")
    val = Path(run["export_dir"]) / "yolo_seg" / "images" / "val"
    first = min(p.name for p in val.iterdir())
    instance = {"class_id": 0, "score": 0.9, "polygons": [[0.2, 0.2, 0.8, 0.2, 0.8, 0.8]]}
    (run_dir / "predictions.json").write_text(json.dumps(
        {"conf": 0.25, "images": [{"name": first, "width": 96, "height": 64, "instances": [instance]}]}))
    (run_dir / "result.json").write_text(json.dumps({"best": None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
