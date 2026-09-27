from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import yaml
from train_fixtures import make_export
from vsp.train import worker
from vsp.train.worker import (
    ProgressWriter,
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
