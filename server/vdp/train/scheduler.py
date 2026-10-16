"""Per-GPU training queues (spec: scheduler.py). One worker subprocess per GPU at a time; GPUs run in
parallel.

Locking: one Condition guards the queues, the running table and every run.json write made here. A run
is popped, marked running and launched inside one critical section, so a concurrent cancel always
finds it either queued (and drops it) or running with a process to signal.

Hooks (the server's SAM3 pause uses both): `before_queue`, passed to a request, runs under that lock
once the request has been validated, right before its run(s) are queued; if it raises, nothing is
queued. `on_gpu_idle(gpu)` fires outside the lock whenever a GPU is left with nothing queued or
running (and once per GPU at `start()`); it may be stale by the time it runs, so a listener that
cares re-checks `gpu_idle()`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..labeling.errors import BadRequest, Conflict
from .models import ExportInfo, TrainParams, check_model, run_classes
from .results import read_metrics, summarize, tail
from .runs import TERMINAL, RunStore, best_weights, last_weights, now_iso, read_json
from .worker import CANCELLED_EXIT

log = logging.getLogger(__name__)

WORKER_CMD = (sys.executable, "-m", "vdp.train.worker")


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory_total: int  # bytes
    capability: tuple[int, int] = (0, 0)  # (major, minor), e.g. (8, 6) for an Ampere RTX 3080 Ti
    sm_count: int = 0  # torch's multi_processor_count
    uuid: str = ""  # bare uuid (no "GPU-" prefix), as torch reports it; matched against NVML by this


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
        gpus.append(Gpu(
            i, props.name, int(props.total_memory),
            capability=(props.major, props.minor), sm_count=props.multi_processor_count,
            uuid=str(props.uuid),
        ))
    return gpus


# CUDA cores per SM by compute capability major.minor. NVIDIA-only machine assumption (no AMD/Intel
# fallback needed). Kepler/Maxwell (3.x/5.x) are not listed since this pipeline requires CUDA 12+ /
# recent torch builds; an unlisted (older, or newer than anything below) capability defaults to 128,
# which matches every Ampere-and-later consumer/datacenter part currently shipping.
_CORES_PER_SM = {
    (6, 0): 64, (6, 1): 128, (6, 2): 128,  # Pascal
    (7, 0): 64, (7, 2): 64, (7, 5): 64,  # Volta / Turing
    (8, 0): 64,  # Ampere datacenter (A100)
    (8, 6): 128, (8, 7): 128, (8, 9): 128,  # Ampere consumer/workstation, Ada
    (9, 0): 128,  # Hopper
}


def cores_per_sm(capability: tuple[int, int]) -> int:
    return _CORES_PER_SM.get(tuple(capability), 128)


def estimated_tflops(sm_count: int, capability: tuple[int, int], clock_ghz: float) -> float:
    """Rough FP32 TFLOPS estimate used only to rank GPUs by speed: SMs × cores/SM × 2 (FMA) × max
    SM clock (GHz), divided by 1000 to land in TFLOPS (this is the standard back-of-envelope GPU
    TFLOPS formula and matches published FP32 figures, e.g. an RTX 3080 Ti comes out to ~34). Not
    a real benchmark (ignores boost sustain, memory bandwidth, etc.), but a much better speed proxy
    than compute capability or SM count alone (it ranks a wide, lower-clocked card correctly
    against a narrow, high-clocked one, e.g. a 3090 over a 4060 despite the 4060's newer cc)."""
    return sm_count * cores_per_sm(capability) * 2 * clock_ghz / 1000


def rank_gpus(gpus: list[Gpu], clocks: dict[int, float]) -> list[Gpu]:
    """Fastest first, by estimated compute throughput; VRAM only breaks a throughput tie, then index."""
    def key(g: Gpu) -> tuple[float, int, int]:
        tflops = estimated_tflops(g.sm_count, g.capability, clocks.get(g.index, 1.5))
        return (-tflops, -g.memory_total, g.index)
    return sorted(gpus, key=key)


def default_gpu(ranked_gpus: list[Gpu], busy: set[int]) -> int | None:
    """The fastest GPU (already ranked, fastest first) that is not busy; the fastest one if every
    GPU is busy; None without GPUs. SAM3's GPU is not excluded — the caller-visible "shares memory
    with SAM3" warning is enough; it does not disqualify the fastest GPU from being the default."""
    for g in ranked_gpus:
        if g.index not in busy:
            return g.index
    return ranked_gpus[0].index if ranked_gpus else None


_nvml_ready = False


def _nvml_init() -> bool:
    global _nvml_ready
    if _nvml_ready:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 — no driver/NVML/pynvml: never raise, just report unavailable
        return False
    _nvml_ready = True
    return True


def _nvml_uuid_map() -> dict[str, object]:
    """{bare uuid: NVML device handle} for every NVML device (NVML/PCI order, which may differ
    from torch's CUDA order); empty on any failure, so callers always fall back cleanly."""
    if not _nvml_init():
        return {}
    try:
        import pynvml
        out = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            out[uuid.removeprefix("GPU-")] = handle
        return out
    except Exception:  # noqa: BLE001 — any NVML query can fail; fall back to no devices at all
        return {}


def gpu_free_memory(gpus: list[Gpu]) -> dict[int, int | None]:
    """Free VRAM (bytes) per torch index, via NVML, matched by UUID (never by index: NVML's own
    device order need not agree with torch's). None for a GPU NVML fails on, or is not found for."""
    handles = _nvml_uuid_map()
    out: dict[int, int | None] = {}
    for g in gpus:
        handle = handles.get(g.uuid)
        if handle is None:
            out[g.index] = None
            continue
        try:
            import pynvml
            out[g.index] = int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        except Exception:  # noqa: BLE001 — never raise for one GPU's free memory; None instead
            out[g.index] = None
    return out


def gpu_max_clock_ghz(gpus: list[Gpu], *, fallback: float = 1.5) -> dict[int, float]:
    """Max SM clock (GHz) per torch index, via NVML, matched by UUID. `fallback` (1.5 GHz) is used
    whenever NVML cannot supply a clock for a GPU, so ranking still works from SMs × cores alone."""
    handles = _nvml_uuid_map()
    out: dict[int, float] = {}
    for g in gpus:
        handle = handles.get(g.uuid)
        clock_mhz = None
        if handle is not None:
            try:
                import pynvml
                clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except Exception:  # noqa: BLE001 — never raise for one GPU's clock; fall back instead
                clock_mhz = None
        out[g.index] = (clock_mhz / 1000.0) if clock_mhz is not None else fallback
    return out


def _busy_from_runs(runs: list[dict], *, exclude_pid: int | None) -> set[int]:
    """GPU indexes with a run.json at status "running", owned by a live process other than
    `exclude_pid` (that caller's own in-flight runs are tracked in memory instead, see
    `TrainScheduler.busy_gpus`). A running row with no live owner is a stale leftover from a
    process that died before `recover()` could reclassify it — not actually busy."""
    busy = set()
    for run in runs:
        if run["status"] != "running":
            continue
        owner = run.get("owner_pid")
        if owner == exclude_pid:
            continue
        if owner is not None and not _alive(owner):
            continue
        busy.add(run["gpu"])
    return busy


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
        self.on_gpu_idle: Callable[[int], None] | None = None  # see the module docstring

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
                    continue  # another live scheduler (a `vdp train` in a terminal) owns this run
                changes: dict = {}
                if run["status"] == "running":
                    pid = run.get("pid")
                    run_dir = self.runs.dir(run["id"])
                    if _alive(pid) and _is_worker_of(pid, run_dir):
                        _signal_group(pid, signal.SIGKILL)  # an orphan of a killed server, still training
                        changes.update(status="interrupted", pid=None, finished_at=now_iso())
                    else:
                        # The worker already exited (it survives an unclean server exit, running in its
                        # own session) — recover what it reported rather than discarding a finished run.
                        changes.update(self._classify_dead_run(run_dir))
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

    def _classify_dead_run(self, run_dir: Path) -> dict:
        """A `running` run whose worker has already exited (dead pid, or a pid that /proc no longer
        confirms as this run's worker): classify it from result.json the way `_finish_train` would,
        instead of always marking it `interrupted` and losing a run that finished after the server
        died."""
        result = read_json(run_dir / "result.json")
        fields: dict = {"pid": None, "finished_at": now_iso()}
        if result is None:
            fields["status"] = "interrupted"
        elif "best" in result:
            fields.update(status="done", best=result.get("best") or summarize(read_metrics(run_dir)))
        elif "cancelled" in result:
            fields["status"] = "cancelled"
        elif "error" in result:
            fields.update(status="error", error=result["error"])
        else:
            fields["status"] = "interrupted"
        return fields

    # ---- requests ----

    def submit(
        self, export: ExportInfo, specs: list[tuple[str, int]], params: TrainParams, *,
        before_queue: Callable[[], None] | None = None,
    ) -> list[str]:
        if not specs:
            raise BadRequest("choose at least one model")
        for model, gpu in specs:
            check_model(model)
            if gpu not in self._queues:
                raise BadRequest(f"GPU {gpu} does not exist")
        ids = []
        with self._cv:
            if before_queue is not None:
                before_queue()
            try:
                for model, gpu in specs:
                    run = self.runs.create(export, model, gpu, params, owner_pid=os.getpid())
                    self._queues[gpu].append((run["id"], "train"))
                    ids.append(run["id"])
            finally:  # runs queued before a later one failed to be created still start
                self._cv.notify_all()
        return ids

    def submit_from(
        self, source_id: str, export: ExportInfo, mode: str, gpu: int, params: TrainParams | None = None, *,
        before_queue: Callable[[], None] | None = None,
    ) -> str:
        """A new run that starts from `source_id`'s checkpoint on `export` (which must carry the
        same classes, in the same order) instead of pretrained weights, queued like any `submit`
        run; the source run is never modified. "continue" resumes the source's own optimizer,
        LR schedule and training params on the new export -- only for a `resumable` source;
        `params` is ignored for it (the source's own `params` are copied verbatim; honouring a
        handful of trivial overrides such as batch/workers/patience was considered and skipped
        for simplicity -- pass a `params` here and it is silently ignored). "finetune" starts a
        fresh schedule (`params`, or `TrainParams()`) from the source's `best.pt` -- any source
        that has one, finished or not."""
        if mode not in ("continue", "finetune"):
            raise BadRequest(f"unknown mode {mode!r}")
        if gpu not in self._queues:
            raise BadRequest(f"GPU {gpu} does not exist")
        source = self.runs.read(source_id)
        source_dir = self.runs.dir(source_id)
        if mode == "continue":
            if not self.resumable(source):
                raise Conflict(f"run {source_id} cannot be continued (not stopped, or no last.pt)")
        elif not best_weights(source_dir).is_file():
            raise Conflict(f"run {source_id} cannot be fine-tuned (no best.pt)")
        source_classes = run_classes(source_dir, source, self.data_root)
        if source_classes is None:
            raise Conflict(f"run {source_id}'s classes are unknown")
        if tuple(export.classes) != source_classes:
            raise Conflict(
                f"export {export.id}'s classes ({', '.join(export.classes)}) do not match "
                f"run {source_id}'s classes ({', '.join(source_classes)})"
            )
        run_params = TrainParams(**source["params"]) if mode == "continue" else (params or TrainParams())
        with self._cv:
            if before_queue is not None:
                before_queue()
            run = self.runs.create(export, source["model"], gpu, run_params, owner_pid=os.getpid())
            run_dir = self.runs.dir(run["id"])
            extra: dict = {"source_run": source_id, "start_mode": mode}
            if mode == "continue":
                (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
                shutil.copy2(last_weights(source_dir), last_weights(run_dir))
                source_results = source_dir / "train" / "results.csv"
                if source_results.is_file():
                    (run_dir / "train").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_results, run_dir / "train" / "results.csv")
                extra["continued_from_epoch"] = len(read_metrics(source_dir))
            else:
                init_path = run_dir / "init.pt"
                shutil.copy2(best_weights(source_dir), init_path)
                extra["init_weights"] = str(init_path.resolve())
            self.runs.update(run["id"], **extra)
            self._queues[gpu].append((run["id"], "train"))
            self._cv.notify_all()
        return run["id"]

    def cancel(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if run["status"] == "queued":
                owner = run.get("owner_pid")
                if not self._in_queue(run_id) and owner not in (None, os.getpid()) and _alive(owner):
                    raise Conflict(f"run {run_id} is queued in another process")
                self._drop(run_id)
                self.runs.update(run_id, status="cancelled", finished_at=now_iso())
                proc = None
            elif run["status"] != "running":
                raise Conflict(f"run {run_id} is {run['status']}")
            else:
                proc = self._proc_of(run_id)
                if proc is None:
                    raise Conflict(f"run {run_id} is running in another process")
                self._cancelled.add(run_id)
        if proc is None:  # dropped from its queue: no GPU thread wakes up for it, so report here
            self._report_idle(run["gpu"])
            return
        _signal_group(proc.pid, signal.SIGTERM)
        timer = threading.Timer(self._kill_after, self._kill_if_alive, args=(proc,))
        timer.daemon = True
        timer.start()

    def resumable(self, run: dict) -> bool:
        weights = last_weights(self.runs.dir(run["id"]))
        return run["status"] in ("cancelled", "interrupted", "error") and weights.is_file()

    def completable(self, run: dict) -> bool:
        """A cancelled or interrupted run (not `error`: something went wrong, so its `best.pt`
        is not trusted) that already has a `best.pt` -- i.e. it got past at least one validation
        epoch -- can be marked complete with its best epoch so far instead of resumed."""
        weights = best_weights(self.runs.dir(run["id"]))
        return run["status"] in ("cancelled", "interrupted") and weights.is_file()

    def _requeue_stopped(
        self, run_id: str, gpu: int | None, *, eligible, ineligible_message: str, extra_fields: dict,
        before_queue: Callable[[], None] | None,
    ) -> None:
        """Common to `resume` and `complete`: `gpu` moves the run there (validated like
        `submit`'s: an unknown GPU is a BadRequest, the caller's own bad input); omitted, it
        stays on the run's own GPU as before (an unknown GPU there is a Conflict: the run's own
        recorded state has gone stale, not the caller's fault, e.g. the server restarted with
        fewer GPUs). Both queue the run as a normal "train" item -- it shows in the GPU queues,
        can be cancelled, and survives a restart like any queued run."""
        with self._cv:
            run = self.runs.read(run_id)
            if not eligible(run):
                raise Conflict(ineligible_message.format(run_id))
            if gpu is not None:
                if gpu not in self._queues:
                    raise BadRequest(f"GPU {gpu} does not exist")
                target = gpu
            else:
                target = run["gpu"]
                if target not in self._queues:
                    raise Conflict(f"GPU {target} is not available")
            if before_queue is not None:
                before_queue()
            fields: dict = {
                "status": "queued", "error": None, "finished_at": None, "owner_pid": os.getpid(),
                **extra_fields,
            }
            if gpu is not None:
                fields["gpu"] = gpu
            self.runs.update(run_id, **fields)
            self._queues[target].append((run_id, "train"))
            self._cv.notify_all()

    def resume(self, run_id: str, gpu: int | None = None, *, before_queue: Callable[[], None] | None = None) -> None:
        self._requeue_stopped(
            run_id, gpu, eligible=self.resumable, ineligible_message="run {} cannot be resumed",
            extra_fields={"resume": True, "finish": False}, before_queue=before_queue,
        )

    def complete(
        self, run_id: str, gpu: int | None = None, *, before_queue: Callable[[], None] | None = None
    ) -> None:
        """Finishes a stopped run with its best epoch so far instead of training further: the
        worker skips straight to stripping `best.pt`'s optimizer state and running validation
        predictions (see `worker.train`'s `finish` branch)."""
        self._requeue_stopped(
            run_id, gpu, eligible=self.completable,
            ineligible_message="run {} cannot be marked complete",
            extra_fields={"finish": True, "resume": False}, before_queue=before_queue,
        )

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

    def gpu_idle(self, gpu: int) -> bool:
        """Nothing queued or running on `gpu` in this scheduler (another process's runs do not count)."""
        with self._cv:
            return self._idle_locked(gpu)

    def busy_gpus(self) -> set[int]:
        """GPU indexes this scheduler should not default new runs onto: one with a run queued or
        running here, or with a foreign process's run.json (a `vdp train` elsewhere) reporting
        status "running" on it."""
        with self._cv:
            busy = {g for g, q in self._queues.items() if q} | set(self._running)
        return busy | _busy_from_runs(self.runs.list(), exclude_pid=os.getpid())

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

    def _report_idle(self, gpu: int) -> None:
        """Calls `on_gpu_idle(gpu)` (outside the lock) if `gpu` is idle and the scheduler is not stopping."""
        hook = self.on_gpu_idle
        with self._cv:
            report = hook is not None and not self._stopping and self._idle_locked(gpu)
        if report:
            try:
                hook(gpu)
            except Exception:  # a listener must never take a GPU thread down
                log.exception("on_gpu_idle(%d) failed", gpu)

    def _idle_locked(self, gpu: int) -> bool:
        return not self._queues.get(gpu) and gpu not in self._running

    def _loop(self, gpu: int) -> None:
        while True:
            # Once at start, then after every item this thread took (finished, failed to launch, or
            # skipped as stale): so a GPU whose last run just ended, however, is reported idle.
            self._report_idle(gpu)
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
        finish_job = self.runs.read(run_id).get("finish", False)
        # `finish` must never stick: a queued item always trains next time unless something
        # (`complete()`) sets it again, so a later plain Resume does not skip training.
        fields: dict = {"pid": None, "resume": False, "finish": False, "finished_at": now_iso()}
        if code == 0:
            fields.update(status="done", best=result.get("best") or summarize(read_metrics(run_dir)))
            if finish_job:
                fields["completed_early"] = True
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
