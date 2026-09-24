"""Per-GPU training queues (spec: scheduler.py). One worker subprocess per GPU at a time; GPUs run in
parallel.

Locking: one Condition guards the queues, the running table and every run.json write made here. A run
is popped, marked running and launched inside one critical section, so a concurrent cancel always
finds it either queued (and drops it) or running with a process to signal.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ..labeling.errors import BadRequest, Conflict
from .models import ExportInfo, TrainParams, check_model
from .results import read_metrics, summarize, tail
from .runs import TERMINAL, RunStore, last_weights, now_iso, read_json
from .worker import CANCELLED_EXIT

log = logging.getLogger(__name__)

WORKER_CMD = (sys.executable, "-m", "vsp.train.worker")


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory_total: int  # bytes


def detect_gpus() -> list[Gpu]:
    """CUDA devices in PyTorch's order, which is also the order CUDA_VISIBLE_DEVICES indexes."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(Gpu(i, props.name, int(props.total_memory)))
    return gpus


def default_gpu(gpus: list[Gpu], sam3_gpu: int | None) -> int | None:
    """The first GPU SAM3 is not on; SAM3's own GPU only when it is the only one."""
    for g in gpus:
        if g.index != sam3_gpu:
            return g.index
    return gpus[0].index if gpus else None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def _is_worker_of(pid: int, run_dir: Path) -> bool:
    """True only if /proc/<pid>'s command line names this run's directory — guards against a pid
    that has been recycled (a reboot, or pid wraparound) since run.json was last written."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    args = raw.decode(errors="replace").split("\x00")
    return str(run_dir) in args


class TrainScheduler:
    def __init__(
        self, data_root: Path, gpus: list[Gpu], *, worker_cmd=WORKER_CMD, kill_after: float = 30.0
    ) -> None:
        self.data_root = Path(data_root)
        self.runs = RunStore(self.data_root / "runs")
        self.weights_dir = self.data_root / "train_weights"
        self.gpus = list(gpus)
        self._worker_cmd = tuple(worker_cmd)
        self._kill_after = kill_after
        self._cv = threading.Condition()
        self._queues: dict[int, deque[tuple[str, str]]] = {g.index: deque() for g in self.gpus}
        self._running: dict[int, tuple[str, str, subprocess.Popen]] = {}
        self._cancelled: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._stopping = False

    # ---- lifecycle ----

    def start(self, *, recover: bool = True) -> None:
        """`recover=False` is for the CLI: it must not touch runs a server (or another CLI) owns."""
        if recover:
            self.recover()
        for g in self.gpus:
            t = threading.Thread(target=self._loop, args=(g.index,), name=f"train-gpu{g.index}", daemon=True)
            t.start()
            self._threads.append(t)

    def shutdown(self, timeout: float = 60.0) -> None:
        """Stop taking work and SIGTERM running workers; their runs end `interrupted` (resumable)."""
        with self._cv:
            self._stopping = True
            running = list(self._running.values())
            self._cv.notify_all()
        for _, _, proc in running:
            _signal_group(proc.pid, signal.SIGTERM)
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    def recover(self) -> None:
        """Adopt what a previous process left on disk: running → interrupted, queued → queued
        here again. Idempotent: a run already sitting in one of our queues is left alone, so a
        second call (or a call after some runs are already in flight) never double-queues it."""
        me = os.getpid()
        with self._cv:
            for run in sorted(self.runs.list(), key=lambda r: (r["created_at"], r["id"])):
                owner = run.get("owner_pid")
                if owner not in (None, me) and _alive(owner):
                    continue  # another live scheduler (a `vsp train` in a terminal) owns this run
                changes: dict = {}
                if run["status"] == "running":
                    pid = run.get("pid")
                    if _alive(pid) and _is_worker_of(pid, self.runs.dir(run["id"])):
                        _signal_group(pid, signal.SIGKILL)  # an orphan of a killed server
                    changes.update(status="interrupted", pid=None, finished_at=now_iso())
                if run.get("onnx") in ("queued", "running"):
                    changes["onnx"] = None
                if run["status"] == "queued":
                    if run["gpu"] not in self._queues:
                        changes.update(status="error", error=f"GPU {run['gpu']} is not available")
                    elif not self._in_queue(run["id"]):
                        self._queues[run["gpu"]].append((run["id"], "train"))
                        changes["owner_pid"] = me
                if changes:
                    self.runs.update(run["id"], **changes)
            self._cv.notify_all()

    # ---- requests ----

    def submit(self, export: ExportInfo, specs: list[tuple[str, int]], params: TrainParams) -> list[str]:
        if not specs:
            raise BadRequest("choose at least one model")
        for model, gpu in specs:
            check_model(model)
            if gpu not in self._queues:
                raise BadRequest(f"GPU {gpu} does not exist")
        ids = []
        with self._cv:
            for model, gpu in specs:
                run = self.runs.create(export, model, gpu, params, owner_pid=os.getpid())
                self._queues[gpu].append((run["id"], "train"))
                ids.append(run["id"])
            self._cv.notify_all()
        return ids

    def cancel(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if run["status"] == "queued":
                owner = run.get("owner_pid")
                if not self._in_queue(run_id) and owner not in (None, os.getpid()) and _alive(owner):
                    raise Conflict(f"run {run_id} is queued in another process")
                self._drop(run_id)
                self.runs.update(run_id, status="cancelled", finished_at=now_iso())
                return
            if run["status"] != "running":
                raise Conflict(f"run {run_id} is {run['status']}")
            proc = self._proc_of(run_id)
            if proc is None:
                raise Conflict(f"run {run_id} is running in another process")
            self._cancelled.add(run_id)
        _signal_group(proc.pid, signal.SIGTERM)
        timer = threading.Timer(self._kill_after, self._kill_if_alive, args=(proc,))
        timer.daemon = True
        timer.start()

    def resumable(self, run: dict) -> bool:
        weights = last_weights(self.runs.dir(run["id"]))
        return run["status"] in ("cancelled", "interrupted", "error") and weights.is_file()

    def resume(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if not self.resumable(run):
                raise Conflict(f"run {run_id} cannot be resumed")
            if run["gpu"] not in self._queues:
                raise Conflict(f"GPU {run['gpu']} is not available")
            self.runs.update(
                run_id, status="queued", resume=True, error=None, finished_at=None, owner_pid=os.getpid()
            )
            self._queues[run["gpu"]].append((run_id, "train"))
            self._cv.notify_all()

    def export_onnx(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if run["status"] != "done":
                raise Conflict(f"run {run_id} is {run['status']}; only a finished run can be exported")
            if run.get("onnx") in ("queued", "running"):
                raise Conflict(f"an ONNX export of run {run_id} is already {run['onnx']}")
            if run["gpu"] not in self._queues:
                raise Conflict(f"GPU {run['gpu']} is not available")
            self.runs.update(run_id, onnx="queued")
            self._queues[run["gpu"]].append((run_id, "onnx"))
            self._cv.notify_all()

    def delete(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if run["status"] not in TERMINAL or run.get("onnx") in ("queued", "running"):
                raise Conflict(f"run {run_id} is {run['status']}; cancel it first")
            self.runs.delete(run_id)

    def snapshot(self) -> list[dict]:
        with self._cv:
            out = []
            for g in self.gpus:
                running = self._running.get(g.index)
                out.append({
                    "index": g.index, "name": g.name, "memory_total": g.memory_total,
                    "running": running[0] if running else None,
                    "running_kind": running[1] if running else None,
                    "queue": [rid for rid, _ in self._queues[g.index]],
                })
            return out

    def idle(self) -> bool:
        with self._cv:
            return not self._running and not any(self._queues.values())

    # ---- execution ----

    def _drop(self, run_id: str) -> None:
        for q in self._queues.values():
            for item in [i for i in q if i[0] == run_id]:
                q.remove(item)

    def _in_queue(self, run_id: str) -> bool:
        return any(rid == run_id for q in self._queues.values() for rid, _ in q)

    def _should_launch(self, run_id: str, kind: str) -> bool:
        """Re-read run.json right before launching, under the same lock as the pop: guards
        against a concurrent cancel (by this or another process) or a stale queue entry left by
        a second `recover()`."""
        run = self.runs.read(run_id)
        if kind == "train":
            return run["status"] == "queued"
        return run.get("onnx") == "queued"

    def _proc_of(self, run_id: str) -> subprocess.Popen | None:
        for rid, _, proc in self._running.values():
            if rid == run_id:
                return proc
        return None

    def _kill_if_alive(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            _signal_group(proc.pid, signal.SIGKILL)

    def _loop(self, gpu: int) -> None:
        while True:
            with self._cv:
                while not self._stopping and not self._queues[gpu]:
                    self._cv.wait()
                if self._stopping:
                    return
                run_id, kind = self._queues[gpu].popleft()
                if not self._should_launch(run_id, kind):
                    continue
                try:
                    proc = self._launch(run_id, kind, gpu)
                except Exception as e:
                    log.exception("could not start %s for run %s", kind, run_id)
                    self._launch_failed(run_id, kind, e)
                    continue
                self._running[gpu] = (run_id, kind, proc)
            code = proc.wait()
            _signal_group(proc.pid, signal.SIGKILL)  # reap any dataloader children left behind
            with self._cv:
                del self._running[gpu]
                cancelled = run_id in self._cancelled
                self._cancelled.discard(run_id)
                try:
                    if kind == "train":
                        self._finish_train(run_id, code, cancelled)
                    else:
                        self._finish_onnx(run_id, code)
                except Exception:
                    log.exception("could not record the end of %s for run %s", kind, run_id)
                self._cv.notify_all()

    def _launch(self, run_id: str, kind: str, gpu: int) -> subprocess.Popen:
        run_dir = self.runs.dir(run_id)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        if kind == "train":
            (run_dir / "result.json").unlink(missing_ok=True)
            (run_dir / "progress.json").unlink(missing_ok=True)
        env = {
            **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "YOLO_AUTOINSTALL": "False",
            "PYTHONUNBUFFERED": "1",
        }
        args = [*self._worker_cmd, str(run_dir), *(["--onnx"] if kind == "onnx" else [])]
        with open(run_dir / "worker.log", "ab") as log_file:
            proc = subprocess.Popen(
                args, cwd=self.weights_dir, env=env, stdin=subprocess.DEVNULL, stdout=log_file,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        try:
            if kind == "train":
                self.runs.update(run_id, status="running", started_at=now_iso(), finished_at=None,
                                 error=None, pid=proc.pid, owner_pid=os.getpid())
            else:
                self.runs.update(run_id, onnx="running")
        except Exception:
            # the subprocess is already running but untracked; do not leave it orphaned
            _signal_group(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        return proc

    def _launch_failed(self, run_id: str, kind: str, e: Exception) -> None:
        with contextlib.suppress(Exception):
            if kind == "train":
                self.runs.update(
                    run_id, status="error", error=f"could not start the worker: {e}", finished_at=now_iso()
                )
            else:
                self.runs.update(run_id, onnx=f"error: could not start the export: {e}")

    def _finish_train(self, run_id: str, code: int, cancelled: bool) -> None:
        run_dir = self.runs.dir(run_id)
        result = read_json(run_dir / "result.json", {}) or {}
        fields: dict = {"pid": None, "resume": False, "finished_at": now_iso()}
        if code == 0:
            fields.update(status="done", best=result.get("best") or summarize(read_metrics(run_dir)))
        elif self._stopping and not cancelled:
            fields["status"] = "interrupted"
        elif cancelled or code == CANCELLED_EXIT:
            fields["status"] = "cancelled"
        else:
            message = (
                result.get("error") or tail(run_dir / "worker.log", 20) or f"worker exited with code {code}"
            )
            fields.update(status="error", error=message)
        self.runs.update(run_id, **fields)

    def _finish_onnx(self, run_id: str, code: int) -> None:
        if code == 0:
            self.runs.update(run_id, onnx="done")
        elif self._stopping:
            self.runs.update(run_id, onnx=None)
        else:
            detail = tail(self.runs.dir(run_id) / "worker.log", 5) or f"exit code {code}"
            self.runs.update(run_id, onnx=f"error: {detail}")
