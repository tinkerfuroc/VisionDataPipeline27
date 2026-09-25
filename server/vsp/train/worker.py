"""Training subprocess: `python -m vsp.train.worker <run_dir> [--onnx]` (spec: worker.py).

Started by the scheduler with CUDA_VISIBLE_DEVICES set to one GPU (so Ultralytics always uses
device=0) and with cwd = data/train_weights/, where Ultralytics downloads and caches the
pretrained checkpoints. It never writes run.json; it reports through progress.json, result.json
and its exit code (0 done, 2 cancelled, else error).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import yaml

from .models import train_overrides
from .results import read_metrics, summarize
from .runs import atomic_write_json, best_weights, last_weights, onnx_weights, read_json

# Set before any `from ultralytics import ...` below: Ultralytics imports albumentations itself (for its
# own built-in transforms) as a side effect, so this must be set at worker import time, not only inside
# `photometric_transforms`, or its first import still reaches out to PyPI for the version check.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

CANCELLED_EXIT = 2
PREDICT_CONF = 0.25
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def write_data_yaml(yolo_dir: Path, out: Path) -> Path:
    """The export's data.yaml says `path: .`, which Ultralytics resolves against the CWD; pin
    it absolute."""
    data = yaml.safe_load((yolo_dir / "data.yaml").read_text())
    data["path"] = str(Path(yolo_dir).resolve())
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return out


class ProgressWriter:
    """progress.json, rewritten atomically; `force=False` writes at most once per
    `min_interval` seconds."""

    def __init__(
        self, path: Path, min_interval: float = 1.0, clock=time.monotonic
    ) -> None:
        self.path = path
        self.min_interval = min_interval
        self.clock = clock
        self.state = {"phase": "setup", "epoch": 0, "epochs": 0, "batch": 0, "batches": 0}
        self._last: float | None = None

    def update(self, force: bool = False, **fields) -> None:
        self.state.update(fields)
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.min_interval:
            return
        self._last = now
        atomic_write_json(self.path, self.state)


def instances_from_result(result) -> list[dict]:
    """An Ultralytics Results object as preview instances (normalised polygons, rounded to
    5 places)."""
    if result.masks is None or result.boxes is None:
        return []
    out = []
    for poly, cls, conf in zip(
        result.masks.xyn, result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        if len(poly) < 3:
            continue
        flat = [round(float(v), 5) for v in poly.reshape(-1)]
        out.append(
            {"class_id": int(cls), "score": round(float(conf), 4), "polygons": [flat]}
        )
    return out


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def training_finished(ckpt: dict) -> bool:
    """True when an Ultralytics checkpoint (`last.pt`, loaded as a dict) records a training run
    that already ran to completion. Ultralytics stamps a finished checkpoint's `epoch` as -1 (and
    strips the optimizer state), which is also why `model.train(resume=True)` on it fails with
    "nothing to resume"; a checkpoint from a cancelled/interrupted mid-run has `epoch >= 0`."""
    return ckpt.get("epoch", 0) == -1


def _resume_already_finished(run_dir: Path) -> bool:
    path = last_weights(run_dir)
    if not path.is_file():
        return False
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return training_finished(ckpt)


def photometric_transforms() -> list:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # never phone home for a version check
    import albumentations as A

    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.MotionBlur(blur_limit=(3, 9), p=0.2),
        A.GaussianBlur(sigma_limit=(0.3, 1.5), p=0.1),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.CLAHE(p=0.1),
    ]


def _quiet_ultralytics():
    from ultralytics import settings

    if settings.get("sync", True):  # write the settings file once, not on every run
        settings.update({"sync": False})


def _attach_progress(model, progress: ProgressWriter) -> None:
    def epoch_start(trainer) -> None:
        progress.update(
            force=True,
            phase="train",
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            batch=0,
            batches=len(trainer.train_loader),
        )

    def batch_end(trainer) -> None:
        progress.update(batch=progress.state["batch"] + 1)

    def val_start(validator) -> None:
        progress.update(force=True, phase="val")

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_val_start", val_start)


def _predict(run_dir: Path, run: dict, progress: ProgressWriter) -> None:
    from ultralytics import YOLO

    images_dir = Path(run["export_dir"]) / "yolo_seg" / "images" / "val"
    names = sorted(
        p.name for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    model = YOLO(str(best_weights(run_dir)))
    progress.update(force=True, phase="predict", batch=0, batches=len(names))
    items = []
    for i, name in enumerate(names):
        r = model.predict(
            str(images_dir / name),
            imgsz=run["params"]["imgsz"],
            conf=PREDICT_CONF,
            retina_masks=True,
            device=0,
            verbose=False,
        )[0]
        h, w = r.orig_shape
        items.append(
            {"name": name, "width": int(w), "height": int(h), "instances":
             instances_from_result(r)}
        )
        progress.update(batch=i + 1)
    atomic_write_json(
        run_dir / "predictions.json", {"conf": PREDICT_CONF, "images": items}
    )


def train(run_dir: Path) -> int:
    run = read_json(run_dir / "run.json")
    progress = ProgressWriter(run_dir / "progress.json")
    signal.signal(signal.SIGTERM, _raise_interrupt)
    (run_dir / "result.json").unlink(missing_ok=True)
    try:
        from ultralytics import YOLO

        _quiet_ultralytics()
        if run.get("resume") and best_weights(run_dir).is_file() and _resume_already_finished(run_dir):
            pass  # training already ran to completion (cancelled/interrupted mid-prediction); just predict
        elif run.get("resume"):
            model = YOLO(str(last_weights(run_dir)))
            _attach_progress(model, progress)
            model.train(resume=True)
        else:
            data = write_data_yaml(
                Path(run["export_dir"]) / "yolo_seg", run_dir / "data.yaml"
            )
            p = run["params"]
            model = YOLO(f"{run['model']}.pt")
            _attach_progress(model, progress)
            overrides, photometric = train_overrides(p)
            if photometric:
                overrides["augmentations"] = photometric_transforms()
            model.train(
                data=str(data),
                epochs=p["epochs"],
                imgsz=p["imgsz"],
                batch=p["batch"],
                patience=p["patience"],
                workers=p["workers"],
                seed=p["seed"],
                project=str(run_dir),
                name="train",
                exist_ok=True,
                device=0,
                plots=True,
                **overrides,
            )
        _predict(run_dir, run, progress)
    except KeyboardInterrupt:
        atomic_write_json(run_dir / "result.json", {"cancelled": True})
        return CANCELLED_EXIT
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        atomic_write_json(run_dir / "result.json", {"error": f"{type(e).__name__}: {e}"})
        return 1
    atomic_write_json(run_dir / "result.json", {"best": summarize(read_metrics(run_dir))})
    return 0


def export_onnx(run_dir: Path) -> int:
    run = read_json(run_dir / "run.json")
    try:
        from ultralytics import YOLO

        _quiet_ultralytics()
        YOLO(str(best_weights(run_dir))).export(
            format="onnx", imgsz=run["params"]["imgsz"], dynamic=False, simplify=True,
            device=0,
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0 if onnx_weights(run_dir).is_file() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vsp.train.worker")
    parser.add_argument("run_dir")
    parser.add_argument(
        "--onnx", action="store_true", help="export best.pt to ONNX instead of training"
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    return export_onnx(run_dir) if args.onnx else train(run_dir)


if __name__ == "__main__":
    sys.exit(main())
