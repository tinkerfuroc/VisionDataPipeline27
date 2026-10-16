"""VRAM estimates per batch size (POST /api/train/vram-estimate): runs the vram_probe subprocess and caches what it
measured in data/train_weights/vram_estimates.json, one entry per (model, image size, task, GPU name).

The cache keeps the probe's raw numbers; the calibration margin is applied when an answer is reported, so changing
the margin never needs a re-measurement. Ultralytics is only ever imported in the probe subprocess.
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

from .runs import atomic_write_json, now_iso
from .vram_probe import PREFIX

log = logging.getLogger(__name__)

TASK = "segment"  # every model in models.MODELS is a -seg model
CACHE_NAME = "vram_estimates.json"
LOG_NAME = "vram_probe.log"  # the last probe's output, next to the cache in data/train_weights
PROBE_CMD = (sys.executable, "-m", "vdp.train.vram_probe")

# Calibration margin: estimate = probe peak * MARGIN_FACTOR + MARGIN_BYTES.
# Derived 2026-10-16 on this machine's RTX 2080 Ti, yolo26s-seg at 960 px: the probe measured 6.50-6.71 GiB peak
# reserved at batch 8 (2 to 8 objects per image), while this project's past runs with the same model, image size
# and batch logged an Ultralytics GPU_mem (reserved memory) of 6.45-6.73 GiB on that GPU over whole runs and up to
# 6.90 GiB on the 3080 Ti (a run with ~11 objects per image at its worst batch) -- at most ~3% above the probe. The
# factor covers that plus run-to-run variation (object counts, allocator fragmentation over a long run); the
# fixed part covers what a process holds outside PyTorch's allocator, which neither GPU_mem nor the probe counts
# but the GPU's free memory does: the CUDA context and libraries, measured at 0.25 GiB (NVML per-process usage
# minus reserved memory) on the 2080 Ti, rounded up for other GPUs and drivers.
MARGIN_FACTOR = 1.10
MARGIN_BYTES = int(0.4 * 2**30)

# runner(model, imgsz, gpu, on_batch(batch, bytes|None), should_stop()) -> {gpu_name, total_bytes, estimates} or
# None when it stopped because should_stop() said so.
ProbeRunner = Callable[[str, int, int, Callable[[int, int | None], None], Callable[[], bool]], dict | None]


class ProbeFailed(Exception):
    """The probe subprocess failed (its own error message, or how it exited)."""


def with_margin(measured: int | None) -> int | None:
    return None if measured is None else int(measured * MARGIN_FACTOR) + MARGIN_BYTES


def cache_key(model: str, imgsz: int, gpu_name: str) -> str:
    return f"{model}|{imgsz}|{TASK}|{gpu_name}"


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_probe(
    model: str, imgsz: int, gpu: int, on_batch: Callable[[int, int | None], None], should_stop: Callable[[], bool],
    *, cwd: Path, cmd=PROBE_CMD, poll: float = 0.5,
) -> dict | None:
    """Run the probe on one GPU (CUDA_VISIBLE_DEVICES, as the train worker), cwd = `cwd` (data/train_weights, where
    the checkpoints are cached), its non-probe output written to `cwd/vram_probe.log`. Kills it (and returns None)
    as soon as `should_stop()` is true, checked at least every `poll` seconds."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "YOLO_AUTOINSTALL": "False", "PYTHONUNBUFFERED": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    }
    log_file = open(cwd / LOG_NAME, "w", encoding="utf-8")  # noqa: SIM115 -- closed by the reader thread
    try:
        proc = subprocess.Popen(
            [*cmd, model, str(imgsz)], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
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

    threading.Thread(target=read, name="vram-probe-reader", daemon=True).start()
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
                        "estimates": event["estimates"]}
    finally:
        if proc.poll() is None:
            _signal_group(proc.pid, signal.SIGKILL)
        proc.wait()


def default_runner(weights_dir: Path) -> ProbeRunner:
    """The real probe runner; tests/conftest.py replaces this so no test can start a GPU probe by accident."""
    def run(model, imgsz, gpu, on_batch, should_stop):
        return run_probe(model, imgsz, gpu, on_batch, should_stop, cwd=weights_dir)
    return run


class VramEstimator:
    """Measures (through `runner`, the probe subprocess by default) and caches VRAM per batch size."""

    def __init__(self, data_root: Path, *, runner: ProbeRunner | None = None) -> None:
        self.weights_dir = Path(data_root) / "train_weights"
        self.path = self.weights_dir / CACHE_NAME
        self._runner = runner
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        entries = data.get("entries") if isinstance(data, dict) else None
        return entries if isinstance(entries, dict) else {}

    def cached(self, model: str, imgsz: int, gpu_name: str) -> dict | None:
        """The cached entry {measured: [{batch, bytes|None}], measured_at}, or None."""
        with self._lock:
            return self._load().get(cache_key(model, imgsz, gpu_name))

    def measure(
        self, model: str, imgsz: int, gpu, *, on_batch: Callable[[int, int | None], None],
        should_stop: Callable[[], bool],
    ) -> dict | None:
        """Run the probe on `gpu` (a scheduler Gpu) and cache the result; None (nothing cached) when stopped."""
        runner = self._runner or default_runner(self.weights_dir)
        out = runner(model, imgsz, gpu.index, on_batch, should_stop)
        if out is None:
            return None
        entry = {"measured": [{"batch": int(e["batch"]), "bytes": e["bytes"]} for e in out["estimates"]],
                 "measured_at": now_iso()}
        with self._lock:
            entries = self._load()
            entries[cache_key(model, imgsz, gpu.name)] = entry
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, {"entries": entries})
        return entry

    @staticmethod
    def view(entry: dict, model: str, imgsz: int, gpu, *, free_bytes: int | None, cached: bool) -> dict:
        """The job result: the cached/measured entry with the margin applied, plus the GPU's current free memory."""
        return {
            "model": model, "imgsz": imgsz, "gpu": gpu.index, "gpu_name": gpu.name, "total_bytes": gpu.memory_total,
            "free_bytes": free_bytes, "measured_at": entry["measured_at"], "cached": cached,
            "estimates": [{"batch": e["batch"], "bytes": with_margin(e["bytes"])} for e in entry["measured"]],
        }
