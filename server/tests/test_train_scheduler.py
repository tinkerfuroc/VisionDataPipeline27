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
from vsp.train.scheduler import (
    Gpu,
    TrainScheduler,
    _is_worker_of,
    cores_per_sm,
    default_gpu,
    estimated_tflops,
    gpu_free_memory,
    gpu_max_clock_ghz,
    rank_gpus,
)

GPUS = [Gpu(0, "Fast", 12 * 2**30), Gpu(1, "Slow", 11 * 2**30)]

# RTX 3080 Ti (cuda:0, SAM3's GPU) and RTX 2080 Ti (cuda:1) as on the real machine this task is
# about: same numbers as the brief (cc, SM count, GB) plus each card's published max SM clock.
GPU_3080TI = Gpu(0, "NVIDIA GeForce RTX 3080 Ti", 12 * 2**30, capability=(8, 6), sm_count=80, uuid="3080ti")
GPU_2080TI = Gpu(1, "NVIDIA GeForce RTX 2080 Ti", 11 * 2**30, capability=(7, 5), sm_count=68, uuid="2080ti")
CLOCKS = {0: 1.665, 1: 1.545}


class _FakeMem:
    def __init__(self, free: int) -> None:
        self.free = free


class FakeNvml:
    """A minimal stand-in for the `pynvml` module, keyed by the same UUIDs as the `Gpu`s passed to
    it, in an order deliberately different from torch's (mirrors NVML's real PCI-order quirk)."""

    NVML_CLOCK_SM = 1

    def __init__(self, devices: dict[str, dict]) -> None:
        # devices: {uuid: {"free": bytes|None (None => raise on GetMemoryInfo), "clock_mhz": int|None}}
        self._order = list(devices)
        self._devices = devices
        self.inited = False

    def nvmlInit(self) -> None:
        self.inited = True

    def nvmlDeviceGetCount(self) -> int:
        return len(self._order)

    def nvmlDeviceGetHandleByIndex(self, i: int):
        return self._order[i]

    def nvmlDeviceGetUUID(self, handle) -> str:
        return f"GPU-{handle}"

    def nvmlDeviceGetMemoryInfo(self, handle):
        free = self._devices[handle]["free"]
        if free is None:
            raise RuntimeError("NVML_ERROR_UNKNOWN")
        return _FakeMem(free)

    def nvmlDeviceGetMaxClockInfo(self, handle, clock_type):
        assert clock_type == self.NVML_CLOCK_SM
        clock_mhz = self._devices[handle]["clock_mhz"]
        if clock_mhz is None:
            raise RuntimeError("NVML_ERROR_UNKNOWN")
        return clock_mhz


@pytest.fixture(autouse=True)
def _reset_nvml_lazy_init(monkeypatch):
    """`_nvml_init()` memoises success across calls; force it to re-run `nvmlInit()` against
    whatever fake (or absent) `pynvml` each test installs, instead of leaking state between tests."""
    monkeypatch.setattr("vsp.train.scheduler._nvml_ready", False)


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


def test_default_gpu_skips_busy_and_falls_back_to_fastest_when_all_busy():
    ranked = rank_gpus([GPU_2080TI, GPU_3080TI], CLOCKS)  # fastest first: 3080 Ti, 2080 Ti
    assert default_gpu(ranked, set()) == 0
    assert default_gpu(ranked, {0}) == 1
    assert default_gpu(ranked, {0, 1}) == 0  # all busy: still the fastest, not None
    assert default_gpu(ranked[:1], set()) == 0
    assert default_gpu([], set()) is None


def test_default_gpu_does_not_avoid_sam3s_gpu():
    """SAM3 lives on cuda:0 (the fastest GPU here); `default_gpu` takes no sam3 argument at all
    any more and must still pick it when it is idle."""
    ranked = rank_gpus([GPU_2080TI, GPU_3080TI], CLOCKS)
    assert default_gpu(ranked, set()) == 0


def test_cores_per_sm_table():
    assert cores_per_sm((7, 5)) == 64  # Turing (2080 Ti)
    assert cores_per_sm((8, 0)) == 64  # Ampere datacenter (A100)
    assert cores_per_sm((8, 6)) == 128  # Ampere consumer (3080 Ti / 3090)
    assert cores_per_sm((8, 9)) == 128  # Ada (4060)
    assert cores_per_sm((12, 0)) == 128  # unlisted/newer: defaults to 128


def test_estimated_tflops_matches_published_fp32_figures():
    # Published FP32 TFLOPS: RTX 3080 Ti ~34.1, RTX 2080 Ti ~13.45, RTX 3090 ~35.6, RTX 4060 ~15.1.
    assert estimated_tflops(80, (8, 6), 1.665) == pytest.approx(34.1, abs=0.05)
    assert estimated_tflops(68, (7, 5), 1.545) == pytest.approx(13.45, abs=0.05)
    assert estimated_tflops(82, (8, 6), 1.695) == pytest.approx(35.58, abs=0.05)
    assert estimated_tflops(24, (8, 9), 2.46) == pytest.approx(15.11, abs=0.05)


def test_rank_gpus_by_estimated_throughput_regardless_of_index_order():
    ranked = rank_gpus([GPU_2080TI, GPU_3080TI], CLOCKS)
    assert [g.index for g in ranked] == [0, 1]
    ranked = rank_gpus([GPU_3080TI, GPU_2080TI], CLOCKS)
    assert [g.index for g in ranked] == [0, 1]


def test_rank_gpus_estimate_beats_capability_alone():
    """An RTX 3090-like card (cc 8.6, 82 SMs, 1.695 GHz) outranks an RTX 4060-like card (cc 8.9, 24
    SMs, 2.46 GHz) even though its compute capability is lower."""
    g3090 = Gpu(0, "RTX 3090", 24 * 2**30, capability=(8, 6), sm_count=82)
    g4060 = Gpu(1, "RTX 4060", 8 * 2**30, capability=(8, 9), sm_count=24)
    ranked = rank_gpus([g4060, g3090], {0: 1.695, 1: 2.46})
    assert [g.index for g in ranked] == [0, 1]


def test_rank_gpus_memory_only_breaks_a_throughput_tie():
    fast_small = Gpu(0, "fast, less VRAM", 8 * 2**30, capability=(8, 6), sm_count=80)
    slow_big = Gpu(1, "slower, more VRAM", 24 * 2**30, capability=(7, 5), sm_count=40)
    ranked = rank_gpus([slow_big, fast_small], {0: 1.7, 1: 1.5})
    assert [g.index for g in ranked] == [0, 1]  # a small GPU with more VRAM does not outrank a faster one

    tied_a = Gpu(0, "tied, less VRAM", 8 * 2**30, capability=(8, 6), sm_count=80)
    tied_b = Gpu(1, "tied, more VRAM", 24 * 2**30, capability=(8, 6), sm_count=80)
    assert [g.index for g in rank_gpus([tied_a, tied_b], {0: 1.665, 1: 1.665})] == [1, 0]


def _install_nvml(monkeypatch, fake) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", fake)


def test_gpu_free_memory_matches_by_uuid_with_swapped_nvml_order():
    """NVML enumerates in PCI order (2080 Ti first here), the reverse of torch's CUDA order; the
    match must follow the UUID, not either index."""
    fake = FakeNvml({"2080ti": {"free": 9 * 2**30, "clock_mhz": 1545}, "3080ti": {"free": 5 * 2**30, "clock_mhz": 1665}})
    with pytest.MonkeyPatch.context() as mp:
        _install_nvml(mp, fake)
        free = gpu_free_memory([GPU_3080TI, GPU_2080TI])
    assert free == {0: 5 * 2**30, 1: 9 * 2**30}


def test_gpu_free_memory_none_on_nvml_failure():
    with pytest.MonkeyPatch.context() as mp:
        _install_nvml(mp, FakeNvml({}))  # no devices at all: neither uuid is found
        free = gpu_free_memory([GPU_3080TI, GPU_2080TI])
    assert free == {0: None, 1: None}


def test_gpu_free_memory_none_when_nvml_is_entirely_unavailable(monkeypatch):
    """`nvmlInit()` itself failing (no driver, no NVML library, ...) must never raise; every GPU
    simply gets no free-memory reading."""
    monkeypatch.setattr("vsp.train.scheduler._nvml_init", lambda: False)
    free = gpu_free_memory([GPU_3080TI, GPU_2080TI])
    assert free == {0: None, 1: None}


def test_gpu_max_clock_ghz_matches_by_uuid():
    fake = FakeNvml({"2080ti": {"free": 1, "clock_mhz": 1545}, "3080ti": {"free": 1, "clock_mhz": 1665}})
    with pytest.MonkeyPatch.context() as mp:
        _install_nvml(mp, fake)
        clocks = gpu_max_clock_ghz([GPU_3080TI, GPU_2080TI])
    assert clocks == pytest.approx({0: 1.665, 1: 1.545})


def test_gpu_max_clock_ghz_falls_back_to_1_5ghz_on_failure():
    """A GPU NVML has no clock reading for (or is not found for at all) still gets a usable
    fallback clock, so ranking degrades to SMs × cores rather than breaking."""
    fake = FakeNvml({"3080ti": {"free": 1, "clock_mhz": None}})  # 2080ti missing entirely
    with pytest.MonkeyPatch.context() as mp:
        _install_nvml(mp, fake)
        clocks = gpu_max_clock_ghz([GPU_3080TI, GPU_2080TI])
    assert clocks == {0: 1.5, 1: 1.5}


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


def test_busy_gpus_from_in_memory_queue_and_running(env):
    sched, export = env
    sched.start()
    (running_id,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=99))
    until(sched, running_id, "running")
    assert 1 in sched.busy_gpus()
    assert 0 not in sched.busy_gpus()
    sched.cancel(running_id)
    until(sched, running_id, "cancelled")
    wait_for(lambda: 1 not in sched.busy_gpus())


def test_busy_gpus_from_a_foreign_live_process(env):
    """A `vsp train` CLI in another terminal, sharing this data root, counts as busy even though
    this scheduler never queued or launched anything for it."""
    sched, export = env
    run = sched.runs.create(export, "yolo26s-seg", 0, TrainParams(epochs=2), owner_pid=os.getppid())
    sched.runs.update(run["id"], status="running", pid=os.getppid())
    assert sched.busy_gpus() == {0}


def test_busy_gpus_ignores_a_stale_running_row_from_a_dead_process(env):
    sched, export = env
    dead = _dead_pid()
    run = sched.runs.create(export, "yolo26s-seg", 0, TrainParams(epochs=2), owner_pid=dead)
    sched.runs.update(run["id"], status="running", pid=dead)
    assert sched.busy_gpus() == set()


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
