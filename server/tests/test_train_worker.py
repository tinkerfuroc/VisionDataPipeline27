from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import yaml
from train_fixtures import make_export
from vdp.train import worker
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


def _install_fake_yolo(monkeypatch):
    """A minimal `YOLO` stand-in that records the checkpoint path it was constructed with and the
    keyword arguments `.train()` was called with, instead of actually training anything."""
    calls: dict = {}

    class FakeModel:
        def __init__(self, path):
            calls["init_path"] = path

        def add_callback(self, *_a, **_k):
            pass

        def train(self, **kwargs):
            calls["train_kwargs"] = kwargs

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
