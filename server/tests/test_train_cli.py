from __future__ import annotations

import argparse
import json
import time

import vsp.cli
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export
from vsp.config import Settings
from vsp.train.scheduler import Gpu


def _settings(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    return s


def _args(**kw):
    base = {"list": False, "export": EXPORT_ID, "model": None, "gpu": 1, "epochs": 2, "imgsz": 640,
            "batch": 8, "patience": 30, "workers": 0, "seed": 0, "augment": "default"}
    return argparse.Namespace(**{**base, **kw})


def test_list_exports(tmp_path, capsys):
    rc = vsp.cli._train(_settings(tmp_path), _args(list=True), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER)
    assert rc == 0 and EXPORT_ID in capsys.readouterr().out


def test_trains_models_in_order_and_prints_table(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(model=["yolo26s-seg", "yolo11s-seg"]), gpus=[Gpu(0, "A", 1), Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    out = capsys.readouterr().out
    assert rc == 0
    assert "yolo26s-seg" in out and "yolo11s-seg" in out and "epoch   2/2" in out
    assert "mask map50-95" in out.lower()
    runs = sorted((s.project.data_root / "runs").iterdir())
    assert len(runs) == 2


def test_failed_run_returns_nonzero(tmp_path, capsys):
    rc = vsp.cli._train(_settings(tmp_path), _args(model=["yolo26s-seg"], epochs=13), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    assert rc == 1 and "OutOfMemoryError" in capsys.readouterr().out


def test_default_gpu_picks_around_sam3(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(vsp.cli, "cuda_capabilities", lambda: [(8, 0), (7, 5)])
    monkeypatch.setattr(vsp.cli, "pick_device", lambda requested, caps: "cuda:0")
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(gpu=None, model=["yolo26s-seg"]), gpus=[Gpu(0, "A", 1), Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    capsys.readouterr()
    assert rc == 0
    (run_dir,) = (s.project.data_root / "runs").iterdir()
    run = json.loads((run_dir / "run.json").read_text())
    assert run["gpu"] == 1


def test_augment_strong(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(model=["yolo26s-seg"], augment="strong"), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    assert rc == 0
    (run_dir,) = (s.project.data_root / "runs").iterdir()
    run = json.loads((run_dir / "run.json").read_text())
    assert run["params"]["augment"] == "strong"


def test_default_gpu_falls_back_when_pick_device_raises(tmp_path, capsys, monkeypatch):
    def _raise(requested, caps):
        raise ValueError(f"sam3.device must be 'auto' or 'cuda:N', got {requested!r}")

    monkeypatch.setattr(vsp.cli, "cuda_capabilities", list)
    monkeypatch.setattr(vsp.cli, "pick_device", _raise)
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(gpu=None, model=["yolo26s-seg"]), gpus=[Gpu(0, "A", 1), Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    capsys.readouterr()
    assert rc == 0
    (run_dir,) = (s.project.data_root / "runs").iterdir()
    run = json.loads((run_dir / "run.json").read_text())
    assert run["gpu"] == 0


def test_terminating_signal_cancels_the_run(tmp_path, capsys, monkeypatch):
    """SIGTERM/SIGHUP take the same cancel path as Ctrl-C. Driving it here without sending a real
    signal: `_train` does `import time` locally, so its `time.sleep` is this same `time` module's
    `sleep`; making that raise KeyboardInterrupt, once the run is actually running, behaves exactly
    like the raise a signal handler would perform."""
    s = _settings(tmp_path)
    real_sleep = time.sleep

    def run_status() -> str | None:
        runs = list((s.project.data_root / "runs").glob("*/run.json"))
        return json.loads(runs[0].read_text())["status"] if runs else None

    raised = {"done": False}

    def fake_sleep(seconds):
        if not raised["done"] and run_status() == "running":
            raised["done"] = True
            raise KeyboardInterrupt
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    rc = vsp.cli._train(s, _args(model=["yolo26s-seg"], epochs=99, gpu=0), gpus=[Gpu(0, "A", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.02)
    assert rc == 1
    assert raised["done"] is True
    assert run_status() == "cancelled"
    assert "cancelling this command's runs" in capsys.readouterr().out


def test_bad_input(tmp_path, capsys):
    s = _settings(tmp_path)
    assert vsp.cli._train(s, _args(export=None), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER) == 2
    assert vsp.cli._train(s, _args(imgsz=1000), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER) == 2
    assert vsp.cli._train(s, _args(), gpus=[], worker_cmd=FAKE_WORKER) == 1
