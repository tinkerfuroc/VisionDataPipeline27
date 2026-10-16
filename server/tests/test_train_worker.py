from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from train_fixtures import make_export
from vdp.train import worker
from vdp.train.sampling import epoch_plan
from vdp.train.worker import (
    ProgressWriter,
    augmentation_transforms,
    instances_from_result,
    photometric_transforms,
    training_finished,
    write_data_yaml,
)


def test_worker_import_disables_albumentations_update_check():
    # Set at module import time (already happened by the time this test runs), not only inside
    # photometric_transforms, so Ultralytics' own `from ultralytics import YOLO` never triggers the
    # PyPI version check either.
    assert os.environ.get("NO_ALBUMENTATIONS_UPDATE") == "1"


def test_write_data_yaml_makes_path_absolute(tmp_path):
    yolo = make_export(tmp_path) / "yolo_seg"
    out = write_data_yaml(yolo, tmp_path / "data.yaml")
    data = yaml.safe_load(out.read_text())
    assert data["path"] == str(yolo.resolve())
    assert data["train"] == "images/train" and data["names"] == {0: "Module"}


def test_progress_writer_throttles_but_force_always_writes(tmp_path):
    t = [0.0]
    w = ProgressWriter(tmp_path / "p.json", min_interval=1.0, clock=lambda: t[0])
    w.update(force=True, phase="train", epoch=1, epochs=10, batches=100)
    w.update(batch=1)
    assert json.loads((tmp_path / "p.json").read_text())["batch"] == 0  # throttled
    t[0] = 1.5
    w.update(batch=2)
    assert json.loads((tmp_path / "p.json").read_text())["batch"] == 2
    w.update(force=True, phase="val")
    assert json.loads((tmp_path / "p.json").read_text())["phase"] == "val"


def test_training_finished_reads_ultralytics_epoch_marker():
    """Ultralytics stamps a finished checkpoint's `epoch` as -1 (and strips it for resume); a
    mid-training checkpoint (cancelled or interrupted) keeps its epoch number. No ultralytics
    import needed: `training_finished` takes the already-unpickled checkpoint dict."""
    assert training_finished({"epoch": -1}) is True
    assert training_finished({"epoch": 0}) is False
    assert training_finished({"epoch": 41}) is False
    assert training_finished({}) is False  # no epoch key: treat as unfinished, not a crash


def test_instances_from_result():
    poly = np.array([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]], dtype=np.float32)
    tiny = np.array([[0.1, 0.2], [0.3, 0.2]], dtype=np.float32)
    result = SimpleNamespace(
        masks=SimpleNamespace(xyn=[poly, tiny]),
        boxes=SimpleNamespace(cls=np.array([0.0, 0.0]), conf=np.array([0.91234, 0.5])),
    )
    out = instances_from_result(result)
    assert len(out) == 1
    assert out[0]["class_id"] == 0 and out[0]["score"] == 0.9123
    assert out[0]["polygons"][0] == [0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
    assert instances_from_result(SimpleNamespace(masks=None, boxes=None)) == []


def test_photometric_transforms():
    import albumentations as A

    ts = photometric_transforms()
    assert [type(t) for t in ts] == [A.RandomBrightnessContrast, A.MotionBlur, A.GaussianBlur,
                                     A.GaussNoise, A.ImageCompression, A.CLAHE]
    assert [t.p for t in ts] == [0.5, 0.2, 0.1, 0.2, 0.3, 0.1]


def test_augmentation_transforms_combine_photometric_and_lighting():
    from vdp.train.lighting_aug import Lighting

    assert augmentation_transforms(photometric=False, lighting=False) is None
    assert [type(t) for t in augmentation_transforms(photometric=False, lighting=True)] == [Lighting]
    both = augmentation_transforms(photometric=True, lighting=True)
    assert len(both) == len(photometric_transforms()) + 1 and isinstance(both[-1], Lighting)


def test_lighting_transform_round_trips_through_albumentations_serialization():
    """Ultralytics stores custom transforms as `A.to_dict` dicts in the run args (and checkpoints) and
    rebuilds them with `A.from_dict`, also on Resume; the transform must survive that unchanged."""
    import albumentations as A
    from vdp.train.lighting_aug import Lighting

    t = Lighting(low_share=0.6, bgr=True, p=0.3)
    back = A.from_dict(A.to_dict(t))
    assert isinstance(back, Lighting)
    assert (back.low_share, back.bgr, back.p) == (0.6, True, 0.3)


def test_lighting_transform_changes_pixels_only():
    import albumentations as A
    from vdp.train.lighting_aug import Lighting

    img = np.full((64, 64, 3), 128, np.uint8)
    compose = A.Compose([Lighting(p=1.0)], seed=0)
    outs = [compose(image=img)["image"] for _ in range(20)]
    assert all(o.shape == img.shape and o.dtype == np.uint8 for o in outs)
    means = [o.mean() for o in outs]
    assert min(means) < 110 and max(means) > 150  # both kinds occur
    assert issubclass(Lighting, A.ImageOnlyTransform)


def _install_fake_ultralytics(monkeypatch, *, strip_optimizer):
    """A minimal `ultralytics` stand-in, just enough for `worker.train`'s imports to succeed
    without a real install: `YOLO` is never called on the "finish" path, and `settings` only
    needs `.get`/`.update` for `_quiet_ultralytics`."""
    fake = types.ModuleType("ultralytics")
    fake.__path__ = []
    fake.YOLO = object()  # not callable: calling it would fail loudly, proving it was not used
    fake.settings = SimpleNamespace(get=lambda *a: False, update=lambda *a: None)
    fake_utils = types.ModuleType("ultralytics.utils")
    fake_utils.__path__ = []
    fake_torch_utils = types.ModuleType("ultralytics.utils.torch_utils")
    fake_torch_utils.strip_optimizer = strip_optimizer
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", fake_torch_utils)


def test_train_finish_branch_skips_training_strips_best_and_predicts(tmp_path, monkeypatch):
    """worker.train()'s "Mark complete" path (run.json's `finish`): never touches Ultralytics'
    model.train, strips best.pt's optimizer state, runs predictions, and reports success -- just
    like a normal run's own final result.json, minus the training."""
    run_dir = tmp_path / "run"
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best-with-optimizer")
    (weights / "last.pt").write_bytes(b"last")
    (run_dir / "run.json").write_text(json.dumps({"finish": True, "resume": False}))

    stripped = []
    _install_fake_ultralytics(monkeypatch, strip_optimizer=lambda path: stripped.append(path))
    predicted = []
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: predicted.append((rd, r)))

    code = worker.train(run_dir)

    assert code == 0
    assert stripped == [str(weights / "best.pt")]
    assert predicted == [(run_dir, {"finish": True, "resume": False})]
    assert (weights / "last.pt").read_bytes() == b"last"  # untouched
    result = json.loads((run_dir / "result.json").read_text())
    assert "best" in result  # {"best": summarize(read_metrics(run_dir))}, same as a normal run


def _install_fake_yolo(monkeypatch, train=None):
    """A minimal `YOLO` stand-in that records the checkpoint path it was constructed with, the
    callbacks added to it and the keyword arguments `.train()` was called with, instead of actually
    training anything. `train(callbacks)`, if given, runs inside `.train()` (e.g. to raise)."""
    calls: dict = {"callbacks": {}}

    class FakeModel:
        def __init__(self, path):
            calls["init_path"] = path

        def add_callback(self, event, fn):
            calls["callbacks"].setdefault(event, []).append(fn)

        def train(self, **kwargs):
            calls["train_kwargs"] = kwargs
            if train is not None:
                train(calls["callbacks"])

    fake = types.ModuleType("ultralytics")
    fake.__path__ = []
    fake.YOLO = FakeModel
    fake.settings = SimpleNamespace(get=lambda *a: False, update=lambda *a: None)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    return calls


def test_worker_continue_first_launch_targets_the_new_run_dir_never_the_source(tmp_path, monkeypatch):
    """The very first launch of a "Continue on new data" run (`start_mode` "continue", `resume`
    still False -- it has not trained a single epoch under this run id yet) must pass `data` and
    `save_dir` explicitly: the copied `last.pt` embeds the *source* run's save_dir, and without an
    override Ultralytics would resume training into the source run's own directory."""
    run_dir = tmp_path / "run"
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"last")
    export_dir = make_export(tmp_path)
    run = {
        "export_dir": str(export_dir), "model": "yolo26s-seg", "resume": False, "finish": False,
        "start_mode": "continue", "source_run": "some-source-id",
        "params": {"epochs": 10, "imgsz": 640, "batch": 4, "patience": 30, "workers": 0, "seed": 0},
    }
    (run_dir / "run.json").write_text(json.dumps(run))

    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    code = worker.train(run_dir)

    assert code == 0
    assert calls["init_path"] == str(weights / "last.pt")
    assert calls["train_kwargs"] == {
        "resume": True, "data": str(run_dir / "data.yaml"), "save_dir": str(run_dir / "train"),
    }
    assert (run_dir / "data.yaml").is_file()


def test_worker_resume_of_a_stopped_continue_run_stays_pinned_to_its_own_run(tmp_path, monkeypatch):
    """A stopped "continue" run resumes pinned to its own `data.yaml` and `save_dir`. Its last.pt may
    still be the copy from the source run (it crashed before saving one of its own), whose embedded
    save_dir/data point at the source; unpinned, Ultralytics would train on the source's data and
    write into the source run's directory."""
    run_dir = tmp_path / "run"
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"last")
    (run_dir / "data.yaml").write_text("path: /new/export/yolo_seg\n")
    run = {"export_dir": "unused", "model": "yolo26s-seg", "resume": True, "finish": False,
           "start_mode": "continue", "params": {"epochs": 10}}
    (run_dir / "run.json").write_text(json.dumps(run))

    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    code = worker.train(run_dir)

    assert code == 0
    assert calls["init_path"] == str(weights / "last.pt")
    assert calls["train_kwargs"] == {
        "resume": True, "data": str(run_dir / "data.yaml"), "save_dir": str(run_dir / "train"),
    }
    assert (run_dir / "data.yaml").read_text() == "path: /new/export/yolo_seg\n"  # kept, not rewritten


def test_worker_finetune_loads_init_pt_instead_of_pretrained(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "init.pt").write_bytes(b"best")
    export_dir = make_export(tmp_path)
    run = {
        "export_dir": str(export_dir), "model": "yolo26s-seg", "resume": False, "finish": False,
        "start_mode": "finetune", "source_run": "some-source-id",
        "params": {"epochs": 7, "imgsz": 640, "batch": 4, "patience": 30, "workers": 0, "seed": 0,
                   "augment": "default"},
    }
    (run_dir / "run.json").write_text(json.dumps(run))

    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    code = worker.train(run_dir)

    assert code == 0
    assert calls["init_path"] == str(run_dir / "init.pt")
    assert calls["train_kwargs"]["data"] == str(run_dir / "data.yaml")
    assert calls["train_kwargs"]["epochs"] == 7


def test_worker_plain_new_run_still_loads_the_pretrained_checkpoint(tmp_path, monkeypatch):
    """No `start_mode` at all (an ordinary run, as today): still `YOLO(f"{model}.pt")`, not
    `init.pt`."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    export_dir = make_export(tmp_path)
    run = {
        "export_dir": str(export_dir), "model": "yolo26s-seg", "resume": False, "finish": False,
        "params": {"epochs": 7, "imgsz": 640, "batch": 4, "patience": 30, "workers": 0, "seed": 0,
                   "augment": "default"},
    }
    (run_dir / "run.json").write_text(json.dumps(run))

    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    code = worker.train(run_dir)

    assert code == 0
    assert calls["init_path"] == "yolo26s-seg.pt"


def test_train_finish_branch_cancel_reports_cancelled(tmp_path, monkeypatch):
    """A cancel (SIGTERM, here simulated by the predict step raising KeyboardInterrupt) during a
    "finish" job is reported exactly like cancelling any other run."""
    run_dir = tmp_path / "run"
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (run_dir / "run.json").write_text(json.dumps({"finish": True, "resume": False}))

    _install_fake_ultralytics(monkeypatch, strip_optimizer=lambda path: None)

    def _cancel(rd, r, p):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "_predict", _cancel)

    code = worker.train(run_dir)

    assert code == worker.CANCELLED_EXIT
    assert json.loads((run_dir / "result.json").read_text()) == {"cancelled": True}


def _run_json(tmp_path, **extra) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "export_dir": str(make_export(tmp_path)), "model": "yolo26s-seg", "resume": False, "finish": False,
        "params": {"epochs": 7, "imgsz": 960, "batch": 12, "patience": 30, "workers": 0, "seed": 0,
                   "augment": "default"},
        **extra,
    }
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir


def _first_epoch(callbacks, trainer) -> None:
    """What Ultralytics' `_do_train` does before the first batch: reset the OOM counter, then run
    `on_train_epoch_start`."""
    trainer._oom_retries = 0
    for fn in callbacks["on_train_epoch_start"]:
        fn(trainer)


def _trainer(batch=12, imgsz=960):
    return SimpleNamespace(
        _oom_retries=0, epoch=0, epochs=7, train_loader=[0] * 5, batch_size=batch, args=SimpleNamespace(imgsz=imgsz),
    )


def test_fail_on_oom_callback_uses_up_ultralytics_retries():
    """Ultralytics halves the batch and retries (up to 3 times) on an OOM in the first epoch while
    `trainer._oom_retries < 3`; it resets the counter just before `on_train_epoch_start`, so setting
    it there makes the first OOM raise."""
    calls: dict = {}
    model = SimpleNamespace(add_callback=lambda event, fn: calls.setdefault(event, []).append(fn))
    worker._attach_fail_on_oom(model, {})
    trainer = _trainer()
    _first_epoch(calls, trainer)
    assert trainer._oom_retries >= 3


@pytest.mark.parametrize(
    "extra", [{}, {"resume": True}, {"start_mode": "continue"}, {"start_mode": "finetune"}],
)
def test_new_and_resumed_runs_both_disable_oom_auto_reduce(tmp_path, monkeypatch, extra):
    run_dir = _run_json(tmp_path, **extra)
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"last")
    (run_dir / "init.pt").write_bytes(b"best")
    seen = []

    def train(callbacks):
        trainer = _trainer()
        _first_epoch(callbacks, trainer)
        seen.append(trainer._oom_retries)

    _install_fake_yolo(monkeypatch, train)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)
    assert worker.train(run_dir) == 0
    assert seen == [3]


def _oom_run(tmp_path, monkeypatch, exc: BaseException) -> dict:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    run_dir = _run_json(tmp_path)

    def train(callbacks):
        _first_epoch(callbacks, _trainer(batch=12, imgsz=960))
        raise exc

    _install_fake_yolo(monkeypatch, train)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)
    assert worker.train(run_dir) == 1
    return json.loads((run_dir / "result.json").read_text())


def test_worker_reports_a_first_batch_oom_readably(tmp_path, monkeypatch):
    result = _oom_run(tmp_path, monkeypatch, RuntimeError(
        "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.65 GiB total capacity; ...)"
    ))
    assert result == {"error": (
        "Out of GPU memory on cuda:1 at batch 12 (image size 960). Lower the batch size or image size, train on "
        "another GPU, or tick 'Pause SAM3 while training' if SAM3 shares this GPU."
    )}


def test_worker_keeps_other_errors_verbatim(tmp_path, monkeypatch):
    result = _oom_run(tmp_path, monkeypatch, ValueError("bad label file"))
    assert result == {"error": "ValueError: bad label file"}


def test_oom_message_matches_ultralytics_memory_errors():
    kw = {"gpu": "0", "batch": 8, "imgsz": 640}
    for text in (
        "CUDA out of memory. Tried to allocate 20.00 MiB",
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
        "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR",
        "GET was unable to find an engine to execute this computation",
    ):
        assert worker.oom_message(RuntimeError(text), **kw).startswith(
            "Out of GPU memory on cuda:0 at batch 8 (image size 640)."
        ), text
    assert worker.oom_message(RuntimeError("CUDA error: invalid argument"), **kw) is None
    assert worker.oom_message(ValueError("shapes do not match"), **kw) is None


def test_oom_message_maps_invalid_argument_only_right_after_an_oom():
    try:
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        except RuntimeError:
            raise RuntimeError("CUDA error: invalid argument")  # implicit context: raised while handling the OOM
    except RuntimeError as e:
        chained = e
    assert worker.oom_message(chained, gpu="2", batch=4, imgsz=1280).startswith(
        "Out of GPU memory on cuda:2 at batch 4 (image size 1280)."
    )


def test_oom_message_without_a_known_gpu():
    msg = worker.oom_message(RuntimeError("out of memory"), gpu=None, batch=8, imgsz=640)
    assert msg.startswith("Out of GPU memory at batch 8 (image size 640).")


def test_an_oom_while_predicting_after_training_does_not_blame_the_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    run_dir = _run_json(tmp_path)
    _install_fake_yolo(monkeypatch, lambda callbacks: _first_epoch(callbacks, _trainer()))

    def predict(rd, r, p):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(worker, "_predict", predict)
    assert worker.train(run_dir) == 1
    error = json.loads((run_dir / "result.json").read_text())["error"]
    assert error.startswith("Out of GPU memory on cuda:1 while predicting on the val images after training.")
    assert "batch" not in error


@pytest.mark.parametrize("extra", [{}, {"resume": True}])
def test_worker_deletes_an_unloadable_label_cache_before_training(tmp_path, monkeypatch, extra):
    """A cache truncated by a run stopped mid-save would crash Ultralytics' np.load; deleted, it just rescans."""
    run_dir = _run_json(tmp_path, **extra)
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"last")
    labels = Path(json.loads((run_dir / "run.json").read_text())["export_dir"]) / "yolo_seg" / "labels"
    (labels / "train.cache").write_bytes(b"\x93NUMPY truncated")
    with open(labels / "val.cache", "wb") as f:
        np.save(f, {"labels": [], "version": "1"})
    present = []
    _install_fake_yolo(monkeypatch, lambda callbacks: present.extend(sorted(p.name for p in labels.glob("*.cache"))))
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    assert worker.train(run_dir) == 0
    assert present == ["val.cache"]


def _sampled_run(tmp_path, **extra) -> Path:
    """A run (export of 12 train / 6 val images) asking for 4 train and 3 val images per epoch, with the `sampling`
    plan RunStore.create would store for it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {"epochs": 10, "imgsz": 640, "batch": 2, "patience": 30, "workers": 0, "seed": 5, "augment": "default",
              "train_images_per_epoch": 4, "val_images_per_epoch": 3}
    run = {
        "export_dir": str(make_export(tmp_path, n_train=12, n_val=6)), "model": "yolo26s-seg", "resume": False,
        "finish": False, "params": params, "sampling": epoch_plan(params, 12, 6), **extra,
    }
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir


def _fake_sampled_trainer(monkeypatch) -> list[dict]:
    made: list[dict] = []

    def fake(**kwargs):
        made.append(kwargs)
        return "SampledTrainer"

    monkeypatch.setattr(worker, "sampled_trainer", fake)
    return made


def test_new_sampled_run_scales_the_schedule_and_validates_on_a_fixed_subset(tmp_path, monkeypatch):
    run_dir = _sampled_run(tmp_path)
    made = _fake_sampled_trainer(monkeypatch)
    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    assert worker.train(run_dir) == 0

    kw = calls["train_kwargs"]
    assert kw["trainer"] == "SampledTrainer"
    assert (kw["warmup_epochs"], kw["close_mosaic"]) == (3.0, 10)  # x3: warmup 9 capped at 3 (30 %), mosaic 30
    data = yaml.safe_load((run_dir / "data.yaml").read_text())
    subset = run_dir / "val_subset" / "images" / "val"
    assert data["val"] == str(subset) and data["train"] == "images/train"
    assert len(list(subset.iterdir())) == 3
    full_val = str((Path(data["path"]) / "images" / "val").resolve())
    assert made == [{"train_images_per_epoch": 4, "seed": 5, "full_val": full_val,
                     "final_val_path": run_dir / "final_val.json"}]


def test_resumed_sampled_run_keeps_its_subset_and_trainer(tmp_path, monkeypatch):
    run_dir = _sampled_run(tmp_path, resume=True)
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"last")
    made = _fake_sampled_trainer(monkeypatch)
    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)
    assert worker.train(run_dir) == 0  # its first launch never happened: data.yaml and the subset are made now
    first = sorted(p.name for p in (run_dir / "val_subset" / "images" / "val").iterdir())

    assert worker.train(run_dir) == 0

    assert sorted(p.name for p in (run_dir / "val_subset" / "images" / "val").iterdir()) == first
    assert calls["train_kwargs"] == {
        "resume": True, "data": str(run_dir / "data.yaml"), "save_dir": str(run_dir / "train"),
        "trainer": "SampledTrainer", "close_mosaic": 10,
    }
    assert len(made) == 2 and made[1]["train_images_per_epoch"] == 4


def test_val_only_sampling_keeps_whole_train_epochs(tmp_path, monkeypatch):
    run_dir = _sampled_run(tmp_path)
    run = json.loads((run_dir / "run.json").read_text())
    run["params"]["train_images_per_epoch"] = None
    run["sampling"] = epoch_plan(run["params"], 12, 6)
    (run_dir / "run.json").write_text(json.dumps(run))
    made = _fake_sampled_trainer(monkeypatch)
    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    assert worker.train(run_dir) == 0

    assert "warmup_epochs" not in calls["train_kwargs"] and "close_mosaic" not in calls["train_kwargs"]
    assert made[0]["train_images_per_epoch"] is None and made[0]["full_val"] is not None


def test_a_run_without_sampling_uses_ultralytics_own_trainer(tmp_path, monkeypatch):
    run_dir = _run_json(tmp_path)
    made = _fake_sampled_trainer(monkeypatch)
    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)
    assert worker.train(run_dir) == 0
    assert made == [] and "trainer" not in calls["train_kwargs"]
    assert "val_subset" not in [p.name for p in run_dir.iterdir()]


def test_result_reports_the_final_full_validation(tmp_path, monkeypatch):
    run_dir = _sampled_run(tmp_path)
    _fake_sampled_trainer(monkeypatch)
    (run_dir / "final_val.json").write_text('{"images": 1, "metrics": {}}')  # stale, from an earlier attempt

    def train(callbacks):  # what Ultralytics leaves behind: results.csv rows on the subset, then final_val.json
        assert not (run_dir / "final_val.json").exists()
        (run_dir / "train").mkdir(parents=True, exist_ok=True)
        (run_dir / "train" / "results.csv").write_text("epoch,time,metrics/mAP50-95(M)\n1,5.0,0.9\n")
        (run_dir / "final_val.json").write_text(json.dumps({"images": 6, "metrics": {"metrics/mAP50-95(M)": 0.4}}))

    _install_fake_yolo(monkeypatch, train)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)
    assert worker.train(run_dir) == 0
    best = json.loads((run_dir / "result.json").read_text())["best"]
    assert best["mask_map50_95"] == 0.4 and best["val_images"] == 6 and best["epoch"] == 1


def test_mark_complete_of_a_subset_run_validates_best_on_the_full_val_split(tmp_path, monkeypatch):
    run_dir = _sampled_run(tmp_path, finish=True)
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (run_dir / "train" / "results.csv").write_text("epoch,time,metrics/mAP50-95(M)\n1,5.0,0.9\n")
    _install_fake_ultralytics(monkeypatch, strip_optimizer=lambda path: None)
    validated: list = []
    sys.modules["ultralytics"].YOLO = _fake_val_model(validated)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)

    assert worker.train(run_dir) == 0

    (path, kwargs), = validated
    assert path == str(weights / "best.pt")
    full = yaml.safe_load(Path(kwargs["data"]).read_text())
    assert full["val"] == "images/val" and Path(full["path"]).is_absolute()
    assert kwargs["split"] == "val" and kwargs["imgsz"] == 640
    assert json.loads((run_dir / "final_val.json").read_text()) == {
        "images": 6, "metrics": {"metrics/mAP50-95(M)": 0.3},
    }
    assert json.loads((run_dir / "result.json").read_text())["best"]["mask_map50_95"] == 0.3


def test_speed_callbacks_measure_train_and_val_images_per_second(tmp_path):
    calls: dict = {}
    model = SimpleNamespace(add_callback=lambda event, fn: calls.setdefault(event, []).append(fn))
    t = [100.0]
    worker._attach_speed(model, tmp_path / "speed.json", clock=lambda: t[0])
    trainer = SimpleNamespace(train_loader=SimpleNamespace(sampler=list(range(40))), batch_size=8,
                              args=SimpleNamespace(imgsz=640))
    validator = SimpleNamespace(dataloader=SimpleNamespace(dataset=list(range(10))))
    for event, obj, advance in (("on_train_epoch_start", trainer, 0), ("on_train_epoch_end", trainer, 4.0),
                                ("on_val_start", validator, 1.0), ("on_val_end", validator, 2.0)):
        t[0] += advance
        for fn in calls[event]:
            fn(obj)
    assert json.loads((tmp_path / "speed.json").read_text()) == {
        "train_images_per_s": 10.0, "val_images_per_s": 5.0, "batch": 8, "imgsz": 640,
    }


def _fake_val_model(validated: list):
    class FakeModel:
        def __init__(self, path):
            self.path = path

        def val(self, **kwargs):
            validated.append((self.path, kwargs))
            return SimpleNamespace(results_dict={"metrics/mAP50-95(M)": 0.3, "fitness": 0.3})

    return FakeModel


@pytest.mark.parametrize("has_final", [False, True])
def test_resume_of_a_finished_subset_run_runs_a_missing_final_full_validation(tmp_path, monkeypatch, has_final):
    """Killed during Ultralytics' final_eval (last.pt already stamped finished): the resume only predicts, but first
    evaluates best.pt on the whole val split when final_val.json is missing."""
    run_dir = _sampled_run(tmp_path, resume=True)
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (run_dir / "train" / "results.csv").write_text("epoch,time,metrics/mAP50-95(M)\n1,5.0,0.9\n")
    if has_final:
        (run_dir / "final_val.json").write_text(json.dumps({"images": 6, "metrics": {"metrics/mAP50-95(M)": 0.5}}))
    monkeypatch.setattr(worker, "_resume_already_finished", lambda rd: True)
    validated: list = []
    _install_fake_ultralytics(monkeypatch, strip_optimizer=lambda path: None)
    sys.modules["ultralytics"].YOLO = _fake_val_model(validated)
    predicted = []
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: predicted.append(rd))

    assert worker.train(run_dir) == 0

    assert len(validated) == (0 if has_final else 1) and predicted == [run_dir]
    best = json.loads((run_dir / "result.json").read_text())["best"]
    assert best["mask_map50_95"] == (0.5 if has_final else 0.3) and best["val_images"] == 6


def test_resume_onto_whole_train_epochs_resets_close_mosaic(tmp_path, monkeypatch):
    """A Continue of a sampled run onto an export no bigger than its images per epoch trains whole epochs: it must
    not keep the source checkpoint's scaled close_mosaic."""
    run_dir = _sampled_run(tmp_path, resume=True)
    run = json.loads((run_dir / "run.json").read_text())
    run["params"]["train_images_per_epoch"] = 40  # more than the 12 train images: whole epochs
    run["sampling"] = epoch_plan(run["params"], 12, 6)
    (run_dir / "run.json").write_text(json.dumps(run))
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"last")
    _fake_sampled_trainer(monkeypatch)
    calls = _install_fake_yolo(monkeypatch)
    monkeypatch.setattr(worker, "_predict", lambda rd, r, p: None)
    assert worker.train(run_dir) == 0
    assert calls["train_kwargs"]["close_mosaic"] == 10 and "warmup_epochs" not in calls["train_kwargs"]
