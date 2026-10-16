"""VRAM estimates per batch size (POST /api/train/vram-estimate): runs the vram_probe subprocess and caches what it
measured in data/train_weights/vram_estimates.json, one entry per (model, image size, task, objects per image, GPU
name).

The cache keeps the probe's raw numbers; the calibration margin is applied when an answer is reported, so changing
the margin never needs a re-measurement (a change to what the probe measures bumps PROBE_VERSION instead, which
retires older entries). Ultralytics is only ever imported in the probe subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from ..jobs import JobFailed
from .runs import atomic_write_json, now_iso
from .vram_probe import HEADROOM_BYTES, PREFIX, PROBE_INSTANCES

log = logging.getLogger(__name__)

TASK = "segment"  # every model in models.MODELS is a -seg model
CACHE_NAME = "vram_estimates.json"
LOG_NAME = "vram_probe.log"  # the last probe's output, next to the cache in data/train_weights
PROBE_CMD = (sys.executable, "-m", "vdp.train.vram_probe")
# Bump when the probe measures differently: cached entries of another version are ignored (measured anew).
PROBE_VERSION = 2

# Objects per image the probe's synthetic labels carry. With an export: the most objects in any one of its train
# images, times MOSAIC_FACTOR -- Ultralytics' mosaic augmentation tiles four images into one, and this project's
# runs on an export with at most 4 objects per image logged up to ~11.6 objects per training image at their worst
# batch -- rounded up to a power of two (at least DEFAULT_OBJECTS, so small counts share one cache entry) and capped
# at MAX_OBJECTS, Ultralytics' max_det. Without an export: DEFAULT_OBJECTS.
DEFAULT_OBJECTS = PROBE_INSTANCES
MOSAIC_FACTOR = 3
MAX_OBJECTS = 300
# A cached answer with batches that did not fit is measured again when the probe would now get this much more
# memory than it had (other processes freed some).
STALE_SLACK_BYTES = 2**29

# Calibration margin: estimate = probe peak * MARGIN_FACTOR + MARGIN_BYTES.
# Derived 2026-10-16 on this machine's RTX 2080 Ti, yolo26s-seg at 960 px: the probe measured 6.50-6.71 GiB peak
# reserved at batch 8 (2 to 8 objects per image), while this project's past runs with the same model, image size
# and batch logged an Ultralytics GPU_mem (reserved memory) of 6.45-6.73 GiB on that GPU over whole runs and up to
# 6.90 GiB on the 3080 Ti (a run with ~11 objects per image at its worst batch) -- at most ~3% above the probe. The
# factor covers that plus run-to-run variation (object counts, allocator fragmentation over a long run); the
# fixed part covers what a process holds outside PyTorch's allocator, which neither GPU_mem nor the probe counts
# but the GPU's free memory does: the CUDA context and libraries, measured at 0.25 GiB (NVML per-process usage
# minus reserved memory) on the 2080 Ti, rounded up for other GPUs and drivers. Validation (batch x 2, no gradients,
# after Ultralytics empties the cache) peaks lower than training at these settings -- those runs' GPU_mem after
# validation epochs stayed within the first epoch's range -- so the margin needs no allowance for it.
MARGIN_FACTOR = 1.10
MARGIN_BYTES = int(0.4 * 2**30)

# runner(model, imgsz, gpu, objects, on_batch(batch, bytes|None), should_stop()) ->
# {gpu_name, total_bytes, limit_bytes, estimates}, or None when it stopped because should_stop() said so.
ProbeRunner = Callable[[str, int, int, int, Callable[[int, int | None], None], Callable[[], bool]], dict | None]


class ProbeFailed(JobFailed):
    """The probe subprocess failed (its own error message, or how it exited)."""


def with_margin(measured: int | None) -> int | None:
    return None if measured is None else int(measured * MARGIN_FACTOR) + MARGIN_BYTES


def objects_bucket(n: int) -> int:
    """`n` objects per image rounded up to a power of two, at least DEFAULT_OBJECTS, at most MAX_OBJECTS."""
    bucket = DEFAULT_OBJECTS
    while bucket < n:
        bucket *= 2
    return min(bucket, MAX_OBJECTS)


def export_objects(yolo_dir: Path) -> int:
    """The probe's objects per image for a YOLO export folder: the most label lines in any train image, times
    MOSAIC_FACTOR, bucketed (see the constants above)."""
    most = 0
    labels = Path(yolo_dir) / "labels" / "train"
    if labels.is_dir():
        for f in labels.iterdir():
            if f.suffix == ".txt":
                try:
                    most = max(most, sum(1 for line in f.read_text().splitlines() if line.strip()))
                except OSError:
                    continue
    return objects_bucket(most * MOSAIC_FACTOR)


def cache_key(model: str, imgsz: int, objects: int, gpu_name: str) -> str:
    return f"{model}|{imgsz}|{TASK}|{objects}|{gpu_name}"


def _valid_entry(entry) -> bool:
    """A cache entry this version can use: {version, measured: [{batch, bytes}], measured_at, limit_bytes}."""
    if not isinstance(entry, dict) or entry.get("version") != PROBE_VERSION:
        return False
    measured = entry.get("measured")
    if not isinstance(measured, list) or not measured or not isinstance(entry.get("measured_at"), str):
        return False
    if not _is_int(entry.get("limit_bytes")):
        return False
    return all(
        isinstance(e, dict) and _is_int(e.get("batch")) and "bytes" in e and (e["bytes"] is None or _is_int(e["bytes"]))
        for e in measured
    )


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def needs_remeasure(entry: dict, free_bytes: int | None) -> bool:
    """True when some batch did not fit and the probe would now get clearly more memory than it had then."""
    if free_bytes is None or all(e["bytes"] is not None for e in entry["measured"]):
        return False
    return free_bytes - HEADROOM_BYTES > entry["limit_bytes"] + STALE_SLACK_BYTES


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_probe(
    model: str, imgsz: int, gpu: int, objects: int, on_batch: Callable[[int, int | None], None],
    should_stop: Callable[[], bool], *, cwd: Path, cmd=PROBE_CMD, poll: float = 0.5,
) -> dict | None:
    """Run the probe on one GPU (CUDA_VISIBLE_DEVICES, as the train worker), cwd = `cwd` (data/train_weights, where
    the checkpoints are cached), its non-probe output written to `cwd/vram_probe.log`. Kills it (and returns None)
    as soon as `should_stop()` is true, checked at least every `poll` seconds."""
    if should_stop():
        return None
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "YOLO_AUTOINSTALL": "False", "PYTHONUNBUFFERED": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    }
    log_file = open(cwd / LOG_NAME, "w", encoding="utf-8")  # noqa: SIM115 -- closed by the reader thread
    try:
        proc = subprocess.Popen(
            [*cmd, model, str(imgsz), "--objects", str(objects)], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True,
        )
    except OSError:
        log_file.close()
        raise
    events: queue.Queue = queue.Queue()

    def read() -> None:
        try:
            for line in proc.stdout:
                if line.startswith(PREFIX):
                    try:
                        events.put(json.loads(line[len(PREFIX):]))
                        continue
                    except ValueError:
                        pass
                log_file.write(line)
        finally:
            log_file.close()
            events.put(None)

    reader = threading.Thread(target=read, name="vram-probe-reader", daemon=True)
    reader.start()
    start: dict = {}
    try:
        while True:
            if should_stop():
                return None
            try:
                event = events.get(timeout=poll)
            except queue.Empty:
                continue
            if event is None:
                code = proc.wait()
                raise ProbeFailed(f"the VRAM probe exited with code {code} without a result; see {cwd / LOG_NAME}")
            kind = event.get("type")
            if kind == "start":
                start = event
            elif kind == "batch":
                on_batch(event["batch"], event["bytes"])
            elif kind == "error":
                raise ProbeFailed(event["message"])
            elif kind == "done":
                return {"gpu_name": start.get("gpu_name"), "total_bytes": start.get("total_bytes"),
                        "limit_bytes": start.get("limit_bytes"), "estimates": event["estimates"]}
    finally:
        if proc.poll() is None:
            _signal_group(proc.pid, signal.SIGKILL)
        proc.wait()
        reader.join(timeout=5)  # so the log is complete (and closed) once this returns


def default_runner(weights_dir: Path) -> ProbeRunner:
    """The real probe runner; tests/conftest.py replaces this so no test can start a GPU probe by accident."""
    def run(model, imgsz, gpu, objects, on_batch, should_stop):
        return run_probe(model, imgsz, gpu, objects, on_batch, should_stop, cwd=weights_dir)
    return run


class VramEstimator:
    """Measures (through `runner`, the probe subprocess by default) and caches VRAM per batch size. One probe at a
    time; `stop_probe_on(gpu)` (the scheduler's `before_launch` hook) stops one on that GPU and waits for it."""

    def __init__(self, data_root: Path, *, runner: ProbeRunner | None = None) -> None:
        self.weights_dir = Path(data_root) / "train_weights"
        self.path = self.weights_dir / CACHE_NAME
        self._runner = runner
        self._lock = threading.Lock()
        self._objects: dict[Path, int] = {}  # export_objects per export folder (an export never changes)
        self._active: tuple[int, threading.Event, threading.Event] | None = None  # (gpu, stop, finished)

    def objects_for(self, yolo_dir: Path | None) -> int:
        if yolo_dir is None:
            return DEFAULT_OBJECTS
        yolo_dir = Path(yolo_dir)
        if yolo_dir not in self._objects:
            self._objects[yolo_dir] = export_objects(yolo_dir)
        return self._objects[yolo_dir]

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        entries = data.get("entries") if isinstance(data, dict) else None
        return entries if isinstance(entries, dict) else {}

    def cached(self, model: str, imgsz: int, objects: int, gpu_name: str) -> dict | None:
        """The cached entry {version, measured: [{batch, bytes|None}], measured_at, limit_bytes}, or None (also for
        an entry of another probe version or of the wrong shape)."""
        with self._lock:
            entry = self._load().get(cache_key(model, imgsz, objects, gpu_name))
        return entry if _valid_entry(entry) else None

    def measure(
        self, model: str, imgsz: int, objects: int, gpu, *, on_batch: Callable[[int, int | None], None],
        should_stop: Callable[[], bool],
    ) -> dict | None:
        """Run the probe on `gpu` (a scheduler Gpu) and cache the result; None (nothing cached) when stopped, by
        `should_stop()` or by `stop_probe_on`."""
        runner = self._runner or default_runner(self.weights_dir)
        stop, finished = threading.Event(), threading.Event()
        with self._lock:
            self._active = (gpu.index, stop, finished)
        try:
            out = runner(model, imgsz, gpu.index, objects, on_batch, lambda: stop.is_set() or should_stop())
        finally:
            with self._lock:
                self._active = None
            finished.set()
        if out is None or stop.is_set():
            return None
        entry = {"version": PROBE_VERSION,
                 "measured": [{"batch": int(e["batch"]), "bytes": e["bytes"]} for e in out["estimates"]],
                 "measured_at": now_iso(), "limit_bytes": int(out["limit_bytes"])}
        with self._lock:
            entries = self._load()
            entries[cache_key(model, imgsz, objects, gpu.name)] = entry
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, {"entries": entries})
        return entry

    def stop_probe_on(self, gpu: int, timeout: float = 30.0) -> None:
        """Stop a probe running on `gpu` and wait until its process has exited (the runner kills and reaps it)."""
        with self._lock:
            active = self._active
        if active is not None and active[0] == gpu:
            _, stop, finished = active
            stop.set()
            if not finished.wait(timeout):
                log.warning("the VRAM probe on cuda:%d did not stop within %.0f s", gpu, timeout)

    @staticmethod
    def view(entry: dict, model: str, imgsz: int, objects: int, gpu, *, free_bytes: int | None, cached: bool,
             objects_from_export: bool) -> dict:
        """The job result: the entry with the margin applied, plus the GPU's current free memory. A batch that did not
        fit has `bytes: null` and `over_bytes`: it needs more than that (the probe's memory limit, with the margin)."""
        over = with_margin(entry["limit_bytes"])
        return {
            "model": model, "imgsz": imgsz, "gpu": gpu.index, "gpu_name": gpu.name, "total_bytes": gpu.memory_total,
            "free_bytes": free_bytes, "measured_at": entry["measured_at"], "cached": cached,
            "objects_per_image": objects, "objects_from_export": objects_from_export,
            "estimates": [
                {"batch": e["batch"], "bytes": with_margin(e["bytes"]), "over_bytes": over if e["bytes"] is None else None}
                for e in entry["measured"]
            ],
        }
