"""Background label-cache warm-up: `python -m vdp.train.warm_cache <export_dir>`.

Before its first batch, a training run on a new export spends most of a minute in Ultralytics scanning the train
labels (it opens every image to verify it) and then writes `<yolo dir>/labels/train.cache` (and `val.cache`); a
later run on the same export loads that cache in under a second. The server starts this module right after an
export or augment job finishes, so the scan has usually happened by the time the user trains. It builds the very
datasets training builds, with Ultralytics' own code, so the cache's format and hash are what training checks.

The server side (`LabelCacheWarmer`) only spawns the subprocess: Ultralytics is imported in the child, never in a
server thread. One warm-up runs at a time; a failure is logged to `<export_dir>/warm_cache.log` and ignored.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# export format dir -> Ultralytics task: segmentation trains on yolo_seg, detection on yolo_det when the export has it
# (else on yolo_seg, whose cache serves both: the cache holds boxes and polygons, and its hash ignores the task)
FORMATS = (("yolo_seg", "segment"), ("yolo_det", "detect"))
SPLITS = ("train", "val")
NICE = 10
LOG_NAME = "warm_cache.log"
STOP_TIMEOUT_S = 5.0


def atomic_cache_saver(save: Callable) -> Callable:
    """Wrap Ultralytics' `save_dataset_cache_file(prefix, path, x, version)`, which deletes the old cache and then
    writes the new one in place, so it writes a temporary file and renames it over `path`: a training run starting
    meanwhile then sees either no cache (and scans, as it would anyway) or a whole one, never a half-written one."""

    def save_atomically(prefix: str, path: Path, x: dict, version: str) -> None:
        path = Path(path)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            save(prefix, tmp, x, version)
            if tmp.is_file():
                os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    return save_atomically


def drop_unloadable_caches(yolo_dir: Path) -> list[Path]:
    """Delete `labels/{train,val}.cache` under a YOLO dir when it cannot be loaded (e.g. truncated by a run killed
    mid-save): Ultralytics only rescans on a missing or stale cache and crashes on a corrupt one. Returns what was
    deleted; a missing dir or cache is fine."""
    import numpy as np

    dropped = []
    for split in SPLITS:
        cache = Path(yolo_dir) / "labels" / f"{split}.cache"
        if not cache.is_file():
            continue
        try:
            np.load(str(cache), allow_pickle=True).item()
        except Exception:  # noqa: BLE001 -- anything unloadable is rebuilt by the next scan
            log.warning("deleting unloadable label cache %s", cache)
            cache.unlink(missing_ok=True)
            dropped.append(cache)
    return dropped


def warm(export_dir: Path) -> list[Path]:
    """Scan (or check) the export's yolo_seg (and yolo_det) train and val labels exactly as a training run would, leaving their
    `.cache` files behind (an unloadable one is deleted and rebuilt). Returns the cache files that exist afterwards."""
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # never phone home for a version check
    from ultralytics.data import dataset as ul_dataset
    from ultralytics.data.utils import check_class_names

    original = ul_dataset.save_dataset_cache_file
    ul_dataset.save_dataset_cache_file = atomic_cache_saver(original)
    try:
        return _warm_formats(Path(export_dir), ul_dataset.YOLODataset, check_class_names)
    finally:
        ul_dataset.save_dataset_cache_file = original


def _warm_formats(export_dir: Path, dataset_cls, check_class_names) -> list[Path]:
    caches = []
    for fmt, task in FORMATS:
        yolo_dir = export_dir / fmt
        if not (yolo_dir / "data.yaml").is_file():
            continue
        for stale in yolo_dir.glob("labels/*.cache.*.tmp"):  # left by a warm-up killed mid-save
            stale.unlink(missing_ok=True)
        drop_unloadable_caches(yolo_dir)
        data = yaml.safe_load((yolo_dir / "data.yaml").read_text())
        # As training sees it: the worker pins `path` to the resolved yolo dir (write_data_yaml), and Ultralytics
        # resolves each split against it; the hash covers these exact path strings.
        root = yolo_dir.resolve()
        names = check_class_names(data["names"])
        for split in SPLITS:
            images = (root / data[split]).resolve()
            ds = dataset_cls(
                img_path=str(images), data={"names": names, "channels": data.get("channels", 3)}, task=task,
                augment=False, prefix=f"{fmt} {split}: ",
            )
            cache = Path(ds.label_files[0]).parent.with_suffix(".cache")  # where get_labels() keeps it
            if cache.is_file():
                caches.append(cache)
    return caches


def popen_warmup(export_dir: Path) -> subprocess.Popen | None:
    """Start the warm-up subprocess for one export, its output appended to `<export_dir>/warm_cache.log`."""
    try:
        with open(Path(export_dir) / LOG_NAME, "ab") as log_file:
            return subprocess.Popen(
                [sys.executable, "-m", "vdp.train.warm_cache", str(export_dir)],
                env={**os.environ, "YOLO_AUTOINSTALL": "False", "NO_ALBUMENTATIONS_UPDATE": "1"},
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
            )
    except OSError:
        log.warning("could not start the label-cache warm-up for %s", export_dir, exc_info=True)
        return None


# What LabelCacheWarmer spawns unless given `spawn=`; the test suite swaps it for a no-op (tests/conftest.py) so no
# export or augment job in a test starts a real Ultralytics process.
default_spawn: Callable[[Path], object] = popen_warmup


class LabelCacheWarmer:
    """Runs label-cache warm-ups one at a time in the background. `request()` never blocks and never raises; an
    export already queued or warming is not queued twice. `spawn(export_dir)` returns a Popen-like object (with
    `wait()` and `kill()`) or None; tests inject a fake one."""

    def __init__(
        self, *, enabled: bool, available: Callable[[], bool] = lambda: True,
        spawn: Callable[[Path], object] | None = None,
    ) -> None:
        self.enabled = enabled
        self._available = available
        self._spawn = spawn
        self._lock = threading.Lock()
        self._queue: deque[Path] = deque()
        self._current: Path | None = None
        self._proc = None
        self._closed = False

    def request(self, export_dir: Path) -> bool:
        """Queue a warm-up for this export; False when skipped (disabled, Ultralytics missing, already queued)."""
        try:
            if not self.enabled or not self._available():
                return False
            export_dir = Path(export_dir)
            with self._lock:
                if self._closed or export_dir == self._current or export_dir in self._queue:
                    return False
                self._queue.append(export_dir)
                if self._current is None:
                    self._start_next()
            return True
        except Exception:  # a warm-up is only a speed-up: never fail the caller (an export job)
            log.exception("could not queue the label-cache warm-up for %s", export_dir)
            return False

    def _start_next(self) -> None:
        """With the lock held: start the next queued warm-up, if any (skipping any that fail to start)."""
        while self._queue and not self._closed:
            export_dir = self._queue.popleft()
            spawn = self._spawn or default_spawn
            proc = spawn(export_dir)
            if proc is None:
                continue
            self._current, self._proc = export_dir, proc
            threading.Thread(target=self._wait, args=(proc,), name="label-cache-warmup", daemon=True).start()
            return
        self._current = self._proc = None

    def _wait(self, proc) -> None:
        try:
            proc.wait()
        finally:
            with self._lock:
                if self._proc is proc:
                    self._current = self._proc = None
                    self._start_next()

    def shutdown(self) -> None:
        """Drop the queue and stop a running warm-up: SIGTERM, so it removes a half-saved temp file on the way
        out, then SIGKILL if it has not exited within a few seconds."""
        with self._lock:
            self._closed = True
            self._queue.clear()
            proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=STOP_TIMEOUT_S)
        except Exception:  # already gone, or stuck; either way shutdown goes on
            log.warning("could not stop the label-cache warm-up", exc_info=True)


def _exit_on_sigterm(signum, frame) -> None:
    raise SystemExit(128 + signum)  # unwinds, so a half-written temp cache file is removed (atomic_cache_saver)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vdp.train.warm_cache")
    parser.add_argument("export_dir")
    args = parser.parse_args(argv)
    os.nice(NICE)  # low priority: a training run or the server must not wait for this
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        for cache in warm(Path(args.export_dir)):
            print(f"warm: {cache}", flush=True)
    except Exception:  # noqa: BLE001 -- logged; the warm-up is only a speed-up
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
