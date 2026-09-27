from __future__ import annotations

import argparse
import json
import os
import sys
import time

import vsp.cli
import vsp.train.scheduler
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export, wait_for
from vsp.config import Settings
from vsp.train.models import TrainParams, find_export
from vsp.train.scheduler import Gpu, TrainScheduler


class _EmptyNvml:
    """No NVML devices at all: every free-memory/clock lookup falls back deterministically,
    regardless of whatever real NVIDIA hardware this test happens to run on."""

    def nvmlInit(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return 0


def _no_nvml(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _EmptyNvml())
    monkeypatch.setattr(vsp.train.scheduler, "_nvml_ready", False)


def _settings(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    return s


def _args(**kw):
    base = {"list": False, "export": EXPORT_ID, "model": None, "from_run": None, "mode": None, "gpu": 1,
            "epochs": 2, "imgsz": 640, "batch": 8, "patience": 30, "workers": 0, "seed": 0, "augment": "default"}
    return argparse.Namespace(**{**base, **kw})


def test_list_exports(tmp_path, capsys):
    rc = vsp.cli._train(_settings(tmp_path), _args(list=True), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER)
    out = capsys.readouterr().out
    assert rc == 0 and EXPORT_ID in out
    assert f"{EXPORT_ID}  -  Bench" in out


def test_list_exports_shows_the_name(tmp_path, capsys):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root, name="Cups v2")
    rc = vsp.cli._train(s, _args(list=True), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER)
    out = capsys.readouterr().out
    assert rc == 0
    assert f"{EXPORT_ID}  Cups v2  Bench" in out


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


def test_default_gpu_picks_the_fastest_gpu_even_though_its_index_is_0(tmp_path, capsys, monkeypatch):
    """SAM3 no longer matters to the CLI default either: cuda:0 (the fastest GPU here, by SM count)
    is picked even though the old rule ("the first GPU SAM3 is not on") would have skipped it."""
    _no_nvml(monkeypatch)
    fast = Gpu(0, "RTX 3080 Ti", 12 * 2**30, capability=(8, 6), sm_count=80)
    slow = Gpu(1, "RTX 2080 Ti", 11 * 2**30, capability=(7, 5), sm_count=68)
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(gpu=None, model=["yolo26s-seg"]), gpus=[fast, slow],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    out = capsys.readouterr().out
    assert rc == 0
    (run_dir,) = (s.project.data_root / "runs").iterdir()
    run = json.loads((run_dir / "run.json").read_text())
    assert run["gpu"] == 0
    assert "on cuda:0 (RTX 3080 Ti, fastest idle GPU)" in out


def test_default_gpu_skips_a_gpu_busy_in_another_process(tmp_path, capsys, monkeypatch):
    _no_nvml(monkeypatch)
    fast = Gpu(0, "RTX 3080 Ti", 12 * 2**30, capability=(8, 6), sm_count=80)
    slow = Gpu(1, "RTX 2080 Ti", 11 * 2**30, capability=(7, 5), sm_count=68)
    s = _settings(tmp_path)
    export = find_export(s.project.data_root, EXPORT_ID)
    # A `vsp train` in another terminal is already training on the fastest GPU.
    pre = TrainScheduler(s.project.data_root, [fast, slow])
    foreign = pre.runs.create(export, "yolo11s-seg", 0, TrainParams(epochs=2), owner_pid=os.getppid())
    pre.runs.update(foreign["id"], status="running")
    rc = vsp.cli._train(s, _args(gpu=None, model=["yolo26s-seg"]), gpus=[fast, slow],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    out = capsys.readouterr().out
    assert rc == 0
    runs = sorted((s.project.data_root / "runs").iterdir())
    new_run = next(r for r in runs if r.name != foreign["id"])
    run = json.loads((new_run / "run.json").read_text())
    assert run["gpu"] == 1
    assert "on cuda:1 (RTX 2080 Ti, fastest idle GPU)" in out


def test_augment_strong(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(model=["yolo26s-seg"], augment="strong"), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    assert rc == 0
    (run_dir,) = (s.project.data_root / "runs").iterdir()
    run = json.loads((run_dir / "run.json").read_text())
    assert run["params"]["augment"] == "strong"


def test_default_gpu_all_busy_falls_back_to_the_fastest_gpu(tmp_path, capsys, monkeypatch):
    _no_nvml(monkeypatch)
    fast = Gpu(0, "RTX 3080 Ti", 12 * 2**30, capability=(8, 6), sm_count=80)
    slow = Gpu(1, "RTX 2080 Ti", 11 * 2**30, capability=(7, 5), sm_count=68)
    s = _settings(tmp_path)
    export = find_export(s.project.data_root, EXPORT_ID)
    pre = TrainScheduler(s.project.data_root, [fast, slow])
    for gpu in (0, 1):
        r = pre.runs.create(export, "yolo11s-seg", gpu, TrainParams(epochs=2), owner_pid=os.getppid())
        pre.runs.update(r["id"], status="running")
    rc = vsp.cli._train(s, _args(gpu=None, model=["yolo26s-seg"]), gpus=[fast, slow],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    out = capsys.readouterr().out
    assert rc == 0
    runs = [json.loads((d / "run.json").read_text()) for d in (s.project.data_root / "runs").iterdir()]
    (run,) = [r for r in runs if r["model"] == "yolo26s-seg"]
    assert run["gpu"] == 0  # still the fastest, even with every GPU busy
    assert "on cuda:0 (RTX 3080 Ti, fastest GPU, all busy)" in out


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


def test_from_run_model_is_invalid(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(from_run="some-id", mode="continue", model=["yolo26s-seg"]),
                        gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER)
    assert rc == 2 and "--model is invalid with --from-run" in capsys.readouterr().out


def test_from_run_needs_mode(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(from_run="some-id"), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER)
    assert rc == 2 and "--mode is required" in capsys.readouterr().out


def test_from_run_continue_trains_one_run_from_the_source(tmp_path, capsys, monkeypatch):
    """A "continue" run copies the source's own params verbatim -- here epochs=99, the fake
    worker's long-running "cancellable" case (needed to get a resumable/cancelled source in the
    first place) -- so, like `test_terminating_signal_cancels_the_run`, Ctrl-C (simulated the same
    way) is what ends the foreground CLI's wait once the new run is up."""
    s = _settings(tmp_path)
    export = find_export(s.project.data_root, EXPORT_ID)
    sched = TrainScheduler(s.project.data_root, [Gpu(1, "B", 1)], worker_cmd=FAKE_WORKER)
    sched.start(recover=False)
    (rid,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=99))
    wait_for(lambda: sched.runs.read(rid)["status"] == "running")
    wait_for(lambda: (sched.runs.dir(rid) / "train" / "weights" / "last.pt").is_file())
    sched.cancel(rid)
    wait_for(lambda: sched.runs.read(rid)["status"] == "cancelled")
    sched.shutdown(timeout=10)

    real_sleep = time.sleep
    raised = {"done": False}

    def new_run_status() -> str | None:
        others = [p for p in (s.project.data_root / "runs").glob("*/run.json") if p.parent.name != rid]
        return json.loads(others[0].read_text())["status"] if others else None

    def fake_sleep(seconds):
        if not raised["done"] and new_run_status() == "running":
            raised["done"] = True
            raise KeyboardInterrupt
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    rc = vsp.cli._train(s, _args(from_run=rid, mode="continue", gpu=1), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.02)
    out = capsys.readouterr().out
    assert rc == 1 and raised["done"] is True
    runs = [json.loads(p.read_text()) for p in (s.project.data_root / "runs").glob("*/run.json")]
    new = next(r for r in runs if r["id"] != rid)
    assert new["source_run"] == rid and new["start_mode"] == "continue" and new["status"] == "cancelled"
    assert "yolo26s-seg" in out


def test_from_run_finetune_trains_one_run_from_the_source(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(model=["yolo26s-seg"], epochs=2), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    assert rc == 0
    (run_dir,) = (s.project.data_root / "runs").iterdir()
    rid = run_dir.name

    rc = vsp.cli._train(s, _args(from_run=rid, mode="finetune", gpu=1, epochs=3), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    out = capsys.readouterr().out
    assert rc == 0
    runs = [json.loads(p.read_text()) for p in (s.project.data_root / "runs").glob("*/run.json")]
    new = next(r for r in runs if r["id"] != rid)
    assert new["source_run"] == rid and new["start_mode"] == "finetune" and new["params"]["epochs"] == 3
    assert "yolo26s-seg" in out
