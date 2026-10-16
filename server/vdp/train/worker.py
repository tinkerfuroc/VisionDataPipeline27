"""Training subprocess: `python -m vdp.train.worker <run_dir> [--onnx]` (spec: worker.py).

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

from .models import model_task, run_data_dir, train_overrides
from .results import FINAL_VAL, SPEED, run_summary
from .runs import atomic_write_json, best_weights, last_weights, onnx_weights, read_json
from .sampled_trainer import sampled_trainer
from .sampling import (
    IMAGE_SUFFIXES,
    ULTRALYTICS_CLOSE_MOSAIC,
    VAL_SUBSET_DIR,
    choose_val_subset,
    read_image_classes,
    write_val_subset,
)
from .warm_cache import drop_unloadable_caches

# Set before any `from ultralytics import ...` below: Ultralytics imports albumentations itself (for its
# own built-in transforms) as a side effect, so this must be set at worker import time, not only inside
# `photometric_transforms`, or its first import still reaches out to PyPI for the version check.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Imported here, not only where the transforms are built: it registers `Lighting` with albumentations, which
# Ultralytics needs to rebuild the run's saved transforms with `A.from_dict`, on Resume too.
from .lighting_aug import Lighting

CANCELLED_EXIT = 2
PREDICT_CONF = 0.25
_IMAGE_SUFFIXES = IMAGE_SUFFIXES


def write_data_yaml(yolo_dir: Path, out: Path) -> Path:
    """The export's data.yaml says `path: .`, which Ultralytics resolves against the CWD; pin
    it absolute."""
    data = yaml.safe_load((yolo_dir / "data.yaml").read_text())
    data["path"] = str(Path(yolo_dir).resolve())
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return out


def full_val_dir(data_yaml: Path) -> Path:
    """The whole val split of the export a run's data.yaml trains on (its `path`), resolved as Ultralytics resolves
    it -- the same path strings as the export's own label cache, so that cache is reused."""
    root = Path(yaml.safe_load(Path(data_yaml).read_text())["path"])
    return (root / yaml.safe_load((root / "data.yaml").read_text())["val"]).resolve()


def prepare_data(run_dir: Path, run: dict, *, rewrite: bool) -> tuple[Path, str | None]:
    """The run's data.yaml (written from its export when `rewrite` or missing) and, for a run validated per epoch on
    a subset (`sampling.val_images_per_epoch`), the subset: chosen from the export's val labels with the run's seed,
    linked into the run dir, and set as data.yaml's `val`. Returns (data.yaml, the export's whole val images dir for
    the final evaluation, or None without a subset). Deterministic, so a resume finds the same subset (kept as is)."""
    data = run_dir / "data.yaml"
    if rewrite or not data.is_file():
        write_data_yaml(run_data_dir(run), data)
    size = (run.get("sampling") or {}).get("val_images_per_epoch")
    if not size:
        return data, None
    full_val = full_val_dir(data)
    names = choose_val_subset(read_image_classes(full_val), size, run["params"].get("seed", 0))
    subset = write_val_subset(full_val, names, run_dir / VAL_SUBSET_DIR)
    content = yaml.safe_load(data.read_text())
    if content["val"] != str(subset):
        content["val"] = str(subset)
        data.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
    return data, str(full_val)


def _sampling_args(run: dict, run_dir: Path, full_val: str | None, *, resume: bool) -> dict:
    """Extra `model.train` arguments for sampled epochs: the trainer, and the scaled schedule (on a resume only
    close_mosaic, the one Ultralytics lets a resume change; warmup comes back from the checkpoint). None of them for
    a run without sampling, which trains exactly as before.

    A resume or Continue of a run that asked for sampled epochs always passes close_mosaic: its own plan's, or
    Ultralytics' 10 when that plan has whole train epochs (e.g. continued onto an export no bigger than the asked
    size) -- the checkpoint's own value may be one scaled for the source's export."""
    plan = run.get("sampling") or {}
    per_epoch = plan.get("train_images_per_epoch")
    args: dict = {}
    if per_epoch or full_val is not None:
        args["trainer"] = sampled_trainer(
            train_images_per_epoch=per_epoch, seed=run["params"].get("seed", 0), full_val=full_val,
            final_val_path=run_dir / FINAL_VAL, task=model_task(run.get("model", "")),
        )
    if per_epoch:
        args["close_mosaic"] = plan["close_mosaic"]
        if not resume:
            args["warmup_epochs"] = plan["warmup_epochs"]
    elif resume and run["params"].get("train_images_per_epoch"):
        args["close_mosaic"] = plan.get("close_mosaic", ULTRALYTICS_CLOSE_MOSAIC)
    return args


def _final_full_val(run_dir: Path, run: dict) -> None:
    """"Mark complete" of a run validated on a subset: evaluate best.pt on the export's whole val split, as a
    finished run's final evaluation does, so its reported metrics are the full-val ones."""
    from ultralytics import YOLO

    data, full_val = prepare_data(run_dir, run, rewrite=False)
    if full_val is None:
        return
    full_data = write_data_yaml(Path(yaml.safe_load(data.read_text())["path"]), run_dir / "data_full.yaml")
    p = run["params"]
    metrics = YOLO(str(best_weights(run_dir))).val(
        data=str(full_data), split="val", imgsz=p["imgsz"], batch=p["batch"] * 2, device=0, plots=False,
        project=str(run_dir / "train"), name="final_val", exist_ok=True, verbose=False,
    )
    values = {k: float(v) for k, v in metrics.results_dict.items() if k != "fitness"}
    images = sum(1 for f in Path(full_val).iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES)
    atomic_write_json(run_dir / FINAL_VAL, {"images": images, "metrics": values})


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
    5 places). A detection result (no masks) gives its boxes: `box` [x1, y1, x2, y2] plus the
    same rectangle as the polygon, so the gallery's IoU and filters work unchanged."""
    if result.boxes is None:
        return []
    if result.masks is None:
        return [_box_instance(xyxy, cls, conf) for xyxy, cls, conf in zip(
            result.boxes.xyxyn.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]
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


def _box_instance(xyxy, cls, conf) -> dict:
    x1, y1, x2, y2 = (round(float(v), 5) for v in xyxy)
    return {"class_id": int(cls), "score": round(float(conf), 4), "box": [x1, y1, x2, y2],
            "polygons": [[x1, y1, x2, y1, x2, y2, x1, y2]]}


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


def augmentation_transforms(*, photometric: bool, lighting: bool) -> list | None:
    """The run's custom albumentations list, or None to keep Ultralytics' own defaults."""
    transforms = photometric_transforms() if photometric else []
    if lighting:
        transforms.append(Lighting())
    return transforms or None


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


def _attach_speed(model, path: Path, clock=time.monotonic) -> None:
    """Measure each epoch's training (images / seconds from epoch start to end, validation excluded) and each
    validation pass, and keep the latest figures in `path` (the Train page's epoch-time estimate reads them)."""
    started: dict = {}
    speed: dict = {}

    def write() -> None:
        atomic_write_json(path, speed)

    def epoch_start(trainer) -> None:
        started["train"] = clock()

    def epoch_end(trainer) -> None:
        seconds = clock() - started.get("train", clock())
        images = len(getattr(trainer.train_loader, "sampler", ()) or ())
        if seconds > 0 and images:
            speed.update(train_images_per_s=round(images / seconds, 2), batch=trainer.batch_size,
                         imgsz=trainer.args.imgsz)
            write()

    def val_start(validator) -> None:
        started["val"] = clock()

    def val_end(validator) -> None:
        seconds = clock() - started.get("val", clock())
        images = len(getattr(getattr(validator, "dataloader", None), "dataset", ()) or ())
        if seconds > 0 and images:
            speed["val_images_per_s"] = round(images / seconds, 2)
            write()

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_val_start", val_start)
    model.add_callback("on_val_end", val_end)


# The same tests Ultralytics' trainer uses to decide an exception is a GPU memory error.
_OOM_MARKERS = ("CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_INTERNAL_ERROR", "unable to find an engine")


def _is_memory_error(e: BaseException) -> bool:
    text = str(e)
    return "out of memory" in text.lower() or any(m in text for m in _OOM_MARKERS)


def oom_message(e: BaseException, *, gpu: str | None, batch, imgsz) -> str | None:
    """A readable error for a GPU out-of-memory failure, or None when `e` is not one. A CUDA "invalid
    argument" counts only when it was raised while handling an OOM (its context chain holds one)."""
    memory = _is_memory_error(e)
    if not memory and "invalid argument" in str(e):
        cause = e.__cause__ or e.__context__
        while cause is not None and not memory:
            memory = _is_memory_error(cause)
            cause = cause.__cause__ or cause.__context__
    if not memory:
        return None
    where = f" on cuda:{gpu}" if gpu else ""
    return (
        f"Out of GPU memory{where} at batch {batch} (image size {imgsz}). Lower the batch size or image size, "
        "train on another GPU, or tick 'Pause SAM3 while training' if SAM3 shares this GPU."
    )


def _predict_oom_message(gpu: str | None) -> str:
    where = f" on cuda:{gpu}" if gpu else ""
    return (
        f"Out of GPU memory{where} while predicting on the val images after training. Free some memory on that "
        "GPU and Resume the run to retry the predictions."
    )


def _attach_fail_on_oom(model, seen: dict) -> None:
    """Make an out-of-memory error end the run instead of letting Ultralytics silently halve the batch:
    in the first epoch on one GPU it retries with half the batch while `trainer._oom_retries < 3`, and it
    resets that counter just before `on_train_epoch_start` (new and resumed runs alike), so using the
    retries up here makes the first OOM raise. Also notes the batch and image size for the message."""

    def epoch_start(trainer) -> None:
        trainer._oom_retries = 3
        seen["batch"] = trainer.batch_size
        seen["imgsz"] = trainer.args.imgsz

    model.add_callback("on_train_epoch_start", epoch_start)


def _predict(run_dir: Path, run: dict, progress: ProgressWriter) -> None:
    from ultralytics import YOLO

    images_dir = run_data_dir(run) / "images" / "val"
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
            retina_masks=model_task(run["model"]) == "segment",
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
    params = run.get("params", {})
    seen = {"batch": params.get("batch"), "imgsz": params.get("imgsz")}  # updated from the live trainer
    stage = "setup"  # "train" inside model.train(), "predict" after it: picks the out-of-memory message
    try:
        from ultralytics import YOLO

        _quiet_ultralytics()
        if run.get("finish"):
            # "Mark complete": finish the run with its best epoch so far instead of training
            # further. Ultralytics does this same stripping at the end of a normal run; it
            # rewrites best.pt in place. last.pt is untouched.
            from ultralytics.utils.torch_utils import strip_optimizer

            strip_optimizer(str(best_weights(run_dir)))
            if (run.get("sampling") or {}).get("val_images_per_epoch"):
                stage = "predict"  # an out-of-memory error here is not the training batch's fault either
                _final_full_val(run_dir, run)
        elif run.get("resume") and best_weights(run_dir).is_file() and _resume_already_finished(run_dir):
            # training already ran to completion (cancelled/interrupted mid-prediction); just predict -- after the
            # final full-val evaluation, if a run validated on a subset was stopped before that finished
            if (run.get("sampling") or {}).get("val_images_per_epoch") and not (run_dir / FINAL_VAL).is_file():
                stage = "predict"  # an out-of-memory error here is not the training batch's fault
                _final_full_val(run_dir, run)
        elif run.get("resume") or run.get("start_mode") == "continue":
            # Resume from this run's own last.pt, always pinning `save_dir` and `data` to this run. A
            # "Continue on new data" run starts from a last.pt copied from its source, whose embedded
            # save_dir and data still point at the *source* run until this run saves a checkpoint of its
            # own -- unpinned, its first launch (or a Resume after a crash before that first save) would
            # train on the source's data and write into the source run's directory, which must never
            # happen. For every other run the pins equal what its own checkpoint already says.
            data, full_val = prepare_data(
                run_dir, run, rewrite=run.get("start_mode") == "continue" and not run.get("resume"),
            )
            model = YOLO(str(last_weights(run_dir)))
            _attach_progress(model, progress)
            _attach_fail_on_oom(model, seen)
            _attach_speed(model, run_dir / SPEED)
            drop_unloadable_caches(Path(yaml.safe_load(data.read_text())["path"]))  # the export it trains on
            drop_unloadable_caches(run_dir / VAL_SUBSET_DIR)  # the subset's own label cache, if it has one
            (run_dir / FINAL_VAL).unlink(missing_ok=True)
            stage = "train"
            model.train(resume=True, data=str(data), save_dir=str(run_dir / "train"),
                        **_sampling_args(run, run_dir, full_val, resume=True))
        else:
            data, full_val = prepare_data(run_dir, run, rewrite=True)
            p = run["params"]
            # "Fine-tune on new data" starts from the source run's best.pt (copied into this run's
            # dir as init.pt) instead of the pretrained checkpoint; otherwise identical.
            weights = str(run_dir / "init.pt") if run.get("start_mode") == "finetune" else f"{run['model']}.pt"
            model = YOLO(weights)
            _attach_progress(model, progress)
            _attach_fail_on_oom(model, seen)
            _attach_speed(model, run_dir / SPEED)
            overrides, photometric, lighting = train_overrides(p)
            transforms = augmentation_transforms(photometric=photometric, lighting=lighting)
            if transforms is not None:
                overrides["augmentations"] = transforms
            drop_unloadable_caches(run_data_dir(run))
            drop_unloadable_caches(run_dir / VAL_SUBSET_DIR)
            (run_dir / FINAL_VAL).unlink(missing_ok=True)
            overrides.update(_sampling_args(run, run_dir, full_val, resume=False))
            stage = "train"
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
        stage = "predict"
        _predict(run_dir, run, progress)
    except KeyboardInterrupt:
        atomic_write_json(run_dir / "result.json", {"cancelled": True})
        return CANCELLED_EXIT
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        # CUDA_VISIBLE_DEVICES is the one physical GPU the scheduler gave this worker (it trains as device 0)
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        message = oom_message(e, gpu=gpu, **seen) if stage in ("train", "predict") else None
        if message is not None and stage == "predict":
            message = _predict_oom_message(gpu)  # not the training batch's fault
        atomic_write_json(run_dir / "result.json", {"error": message or f"{type(e).__name__}: {e}"})
        return 1
    atomic_write_json(run_dir / "result.json", {"best": run_summary(run_dir)})
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
    parser = argparse.ArgumentParser(prog="python -m vdp.train.worker")
    parser.add_argument("run_dir")
    parser.add_argument(
        "--onnx", action="store_true", help="export best.pt to ONNX instead of training"
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    return export_onnx(run_dir) if args.onnx else train(run_dir)


if __name__ == "__main__":
    sys.exit(main())
