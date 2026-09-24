from __future__ import annotations

import os
import subprocess
import sys

import pytest
from train_fixtures import FAKE_WORKER, make_export, wait_for
from vsp.labeling.errors import BadRequest, Conflict
from vsp.train.models import TrainParams, find_export
from vsp.train.scheduler import Gpu, TrainScheduler, default_gpu

GPUS = [Gpu(0, "Fast", 12 * 2**30), Gpu(1, "Slow", 11 * 2**30)]


@pytest.fixture
def env(tmp_path):
    data_root = tmp_path / "data"
    export = find_export(data_root, make_export(data_root).name)
    sched = TrainScheduler(data_root, GPUS, worker_cmd=FAKE_WORKER, kill_after=5.0)
    yield sched, export
    sched.shutdown(timeout=10)


def status(sched, rid):
    return sched.runs.read(rid)["status"]


def until(sched, rid, *wanted):
    return wait_for(lambda: status(sched, rid) in wanted)


def test_default_gpu():
    assert default_gpu(GPUS, 0) == 1
    assert default_gpu(GPUS, None) == 0
    assert default_gpu(GPUS[:1], 0) == 0
    assert default_gpu([], 0) is None


def test_submit_validates(env):
    sched, export = env
    with pytest.raises(BadRequest):
        sched.submit(export, [], TrainParams())
    with pytest.raises(BadRequest):
        sched.submit(export, [("yolov5s-seg", 0)], TrainParams())
    with pytest.raises(BadRequest):
        sched.submit(export, [("yolo26s-seg", 7)], TrainParams())
    assert sched.runs.list() == []


def test_one_run_per_gpu_and_gpus_in_parallel(env):
    sched, export = env
    sched.start()
    a, b = sched.submit(export, [("yolo26s-seg", 0), ("yolo26m-seg", 0)], TrainParams(epochs=99))
    (c,) = sched.submit(export, [("yolo11s-seg", 1)], TrainParams(epochs=3))
    until(sched, a, "running")
    until(sched, c, "done")  # GPU 1 does not wait for GPU 0
    snap = {g["index"]: g for g in sched.snapshot()}
    assert snap[0]["running"] == a and snap[0]["running_kind"] == "train" and snap[0]["queue"] == [b]
    assert status(sched, b) == "queued"
    sched.cancel(a)
    until(sched, a, "cancelled")
    until(sched, b, "running")
    sched.cancel(b)
    until(sched, b, "cancelled")
    done = sched.runs.read(c)
    assert done["best"]["epoch"] == 3 and done["best"]["mask_map50_95"] == pytest.approx(0.25)
    assert done["finished_at"] and done["pid"] is None
    log = (sched.runs.dir(c) / "worker.log").read_text()
    assert "gpu=1" in log and f"cwd={sched.weights_dir}" in log and "autoinstall=False" in log


def test_cancel_queued_run_never_starts(env):
    sched, export = env
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    sched.cancel(rid)  # not started: the scheduler threads are not running yet
    assert status(sched, rid) == "cancelled"
    sched.start()
    assert sched.idle()
    with pytest.raises(Conflict):
        sched.cancel(rid)


def test_error_message_from_result_json_or_log_tail(env):
    sched, export = env
    sched.start()
    (oom,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=13))
    (crash,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=14))
    until(sched, oom, "error")
    until(sched, crash, "error")
    assert sched.runs.read(oom)["error"] == "OutOfMemoryError: CUDA out of memory"
    assert "Segmentation fault" in sched.runs.read(crash)["error"]
    assert not sched.resumable(sched.runs.read(oom))  # no last.pt


def test_resume_after_cancel(env):
    sched, export = env
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=99))
    until(sched, rid, "running")
    wait_for(lambda: (sched.runs.dir(rid) / "train" / "weights" / "last.pt").is_file())
    sched.cancel(rid)
    until(sched, rid, "cancelled")
    assert sched.resumable(sched.runs.read(rid))
    sched.resume(rid)
    until(sched, rid, "running")
    wait_for(lambda: "resume=True" in (sched.runs.dir(rid) / "worker.log").read_text())
    sched.cancel(rid)
    until(sched, rid, "cancelled")
    assert sched.runs.read(rid)["resume"] is False
    (done,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=1))
    until(sched, done, "done")
    with pytest.raises(Conflict):
        sched.resume(done)


def test_onnx_export(env):
    sched, export = env
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    with pytest.raises(Conflict):
        sched.export_onnx(rid)  # still queued or running
    until(sched, rid, "done")
    sched.export_onnx(rid)
    wait_for(lambda: sched.runs.read(rid)["onnx"] == "done")
    assert (sched.runs.dir(rid) / "train" / "weights" / "best.onnx").is_file()
    assert status(sched, rid) == "done"


def test_delete_only_terminal(env):
    sched, export = env
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=99))
    until(sched, rid, "running")
    with pytest.raises(Conflict):
        sched.delete(rid)
    sched.cancel(rid)
    until(sched, rid, "cancelled")
    sched.delete(rid)
    assert not sched.runs.dir(rid).exists()


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_recover_after_restart(env):
    sched, export = env
    dead = _dead_pid()
    (running, queued, foreign) = [
        sched.runs.create(export, "yolo26s-seg", 1, TrainParams(epochs=2), owner_pid=dead)["id"]
        for _ in range(3)
    ]
    sched.runs.update(running, status="running", pid=dead, onnx="running")
    sched.runs.update(foreign, status="running", owner_pid=os.getppid())  # a live CLI owns this one
    sched.start()
    until(sched, queued, "done")
    r = sched.runs.read(running)
    assert r["status"] == "interrupted" and r["onnx"] is None and r["pid"] is None
    assert status(sched, foreign) == "running"


def test_shutdown_interrupts_running_runs(tmp_path):
    data_root = tmp_path / "data"
    export = find_export(data_root, make_export(data_root).name)
    sched = TrainScheduler(data_root, GPUS, worker_cmd=FAKE_WORKER, kill_after=5.0)
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=99))
    (waiting,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    until(sched, rid, "running")
    sched.shutdown(timeout=10)
    assert status(sched, rid) == "interrupted"
    assert status(sched, waiting) == "queued"  # picked up again by the next start()
