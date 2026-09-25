from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from train_fixtures import FAKE_WORKER, make_export, wait_for
from vsp.labeling.errors import BadRequest, Conflict
from vsp.train.models import TrainParams, find_export
from vsp.train.scheduler import Gpu, TrainScheduler, _is_worker_of, default_gpu

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
    time.sleep(0.5)
    assert status(sched, rid) == "cancelled"
    assert not (sched.runs.dir(rid) / "worker.log").exists()


def test_queued_run_cancelled_externally_is_skipped(env):
    """A run that is cancelled on disk (e.g. by another process sharing this run store) while it
    is still sitting in this scheduler's own queue must never be launched."""
    sched, export = env
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    sched.runs.update(rid, status="cancelled")
    sched.start()
    time.sleep(0.5)
    assert status(sched, rid) == "cancelled"
    assert not (sched.runs.dir(rid) / "worker.log").exists()


def test_recover_is_idempotent(env):
    sched, export = env
    run = sched.runs.create(export, "yolo26s-seg", 0, TrainParams(epochs=2), owner_pid=os.getpid())
    sched.recover()
    sched.recover()
    snap = {g["index"]: g for g in sched.snapshot()}
    assert snap[0]["queue"] == [run["id"]]


def test_cancel_queued_run_owned_by_another_live_process(env):
    sched, export = env
    run = sched.runs.create(export, "yolo26s-seg", 0, TrainParams(epochs=2), owner_pid=os.getppid())
    with pytest.raises(Conflict):
        sched.cancel(run["id"])
    assert status(sched, run["id"]) == "queued"


def test_queue_continues_after_error(env):
    sched, export = env
    sched.start()
    (err_id,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=13))
    (ok_id,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    until(sched, err_id, "error")
    until(sched, ok_id, "done")


def test_launch_failure_does_not_kill_gpu_thread(tmp_path):
    data_root = tmp_path / "data"
    export = find_export(data_root, make_export(data_root).name)
    sched = TrainScheduler(data_root, GPUS, worker_cmd=["/nonexistent/binary"], kill_after=5.0)
    sched.start()
    try:
        a, b = sched.submit(export, [("yolo26s-seg", 0), ("yolo26m-seg", 0)], TrainParams(epochs=2))
        until(sched, a, "error")
        until(sched, b, "error")  # the GPU-0 thread survived launching "a" and picked up "b"
        assert "could not start the worker" in sched.runs.read(a)["error"]
        assert "could not start the worker" in sched.runs.read(b)["error"]
    finally:
        sched.shutdown(timeout=10)


def test_finish_train_prefers_done_over_interrupted_during_shutdown(env):
    """A run whose worker exits 0 right as shutdown() fires must be recorded done, not
    interrupted."""
    sched, export = env
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    sched.start()
    until(sched, rid, "done")
    sched.runs.update(rid, status="running", finished_at=None)
    sched._stopping = True
    sched._finish_train(rid, 0, False)
    assert status(sched, rid) == "done"


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


def test_recover_classifies_a_dead_running_run_from_result_json(env):
    """A worker survives an unclean server exit (its own session) and can finish on its own before
    the server restarts; recover() must read result.json rather than blindly marking it
    interrupted."""
    sched, export = env
    dead = _dead_pid()
    done_id, err_id, gone_id = [
        sched.runs.create(export, "yolo26s-seg", 0, TrainParams(epochs=2), owner_pid=dead)["id"]
        for _ in range(3)
    ]
    for rid in (done_id, err_id, gone_id):
        sched.runs.update(rid, status="running", pid=dead)
    (sched.runs.dir(done_id) / "result.json").write_text(
        json.dumps({"best": {"epoch": 2, "mask_map50_95": 0.4}})
    )
    (sched.runs.dir(err_id) / "result.json").write_text(json.dumps({"error": "x"}))
    # gone_id: no result.json at all
    sched.recover()
    done = sched.runs.read(done_id)
    assert done["status"] == "done" and done["best"] == {"epoch": 2, "mask_map50_95": 0.4}
    assert done["pid"] is None and done["finished_at"]
    err = sched.runs.read(err_id)
    assert err["status"] == "error" and err["error"] == "x"
    gone = sched.runs.read(gone_id)
    assert gone["status"] == "interrupted"


def test_recover_keeps_killing_a_confirmed_live_worker(env):
    """A `running` run whose recorded pid is alive and confirmed (via /proc/<pid>/cmdline) to be
    this run's worker is still training; recover() must still kill it and mark it interrupted,
    ignoring any result.json (a stale one from a previous attempt, say)."""
    sched, export = env
    run = sched.runs.create(export, "yolo26s-seg", 0, TrainParams(epochs=2), owner_pid=os.getpid())
    run_dir = sched.runs.dir(run["id"])
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", str(run_dir)], start_new_session=True
    )
    try:
        sched.runs.update(run["id"], status="running", pid=proc.pid)
        (run_dir / "result.json").write_text(json.dumps({"best": {"epoch": 1}}))
        sched.start()
        until(sched, run["id"], "interrupted")
        assert proc.poll() is not None  # killed
    finally:
        proc.wait()


def test_is_worker_of_true_for_the_actual_worker(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", str(run_dir)], start_new_session=True
    )
    try:
        wait_for(lambda: Path(f"/proc/{proc.pid}/cmdline").is_file())
        assert _is_worker_of(proc.pid, run_dir) is True
    finally:
        proc.terminate()
        proc.wait()


def test_recover_does_not_kill_unrelated_process_with_recycled_pid(env):
    """run.json's `pid` may have been recycled by the OS since it was written (a reboot, or pid
    wraparound); recover() must confirm via /proc/<pid>/cmdline that the pid is actually this
    run's worker before sending it SIGKILL."""
    sched, export = env
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        dead = _dead_pid()
        run = sched.runs.create(export, "yolo26s-seg", 0, TrainParams(epochs=2), owner_pid=dead)
        sched.runs.update(run["id"], status="running", pid=proc.pid)
        sched.start()
        until(sched, run["id"], "interrupted")
        assert proc.poll() is None  # still alive: recover() must not have SIGKILLed it
    finally:
        proc.terminate()
        proc.wait()


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
