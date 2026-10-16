from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from train_fixtures import make_export, wait_for
from vdp.train import warm_cache
from vdp.train.warm_cache import LabelCacheWarmer, atomic_cache_saver
from vdp.train.worker import write_data_yaml


def _with_yolo_det(export_dir: Path) -> Path:
    """Add a yolo_det format (boxes) next to make_export's yolo_seg, as a real export with both has."""
    seg, det = export_dir / "yolo_seg", export_dir / "yolo_det"
    shutil.copytree(seg, det)
    for label in (det / "labels").rglob("*.txt"):
        label.write_text("0 0.5 0.5 0.5 0.5\n")
    return export_dir


def _training_dataset(export_dir: Path, tmp_path: Path, split: str, monkeypatch):
    """The dataset a training run builds: Ultralytics' own data.yaml handling of the worker's data.yaml (its
    font download stubbed out), then YOLODataset for the split, as the segment trainer's build_dataset does."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import dataset as ul_dataset
    from ultralytics.data import utils as ul_utils
    from ultralytics.data.build import build_yolo_dataset

    monkeypatch.setattr(ul_utils, "check_font", lambda *a, **k: None)
    data = ul_utils.check_det_dataset(str(write_data_yaml(export_dir / "yolo_seg", tmp_path / "data.yaml")))

    def no_rescan(self, path):
        raise AssertionError(f"rescanned {path} instead of loading the warmed cache")

    monkeypatch.setattr(ul_dataset.YOLODataset, "cache_labels", no_rescan)
    cfg = get_cfg(overrides={"task": "segment"})
    return build_yolo_dataset(cfg, data[split], 2, data, mode="train" if split == "train" else "val")


def test_warm_writes_the_caches_a_training_run_then_loads_without_rescanning(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    export_dir = _with_yolo_det(make_export(tmp_path / "data", n_train=4, n_val=2))

    caches = warm_cache.warm(export_dir)

    expected = {export_dir.resolve() / fmt / "labels" / f"{split}.cache"
                for fmt in ("yolo_seg", "yolo_det") for split in ("train", "val")}
    assert set(caches) == expected and all(c.is_file() for c in expected)
    assert not list(export_dir.rglob("*.tmp"))
    for split in ("train", "val"):
        assert len(_training_dataset(export_dir, tmp_path, split, monkeypatch).labels) == {"train": 4, "val": 2}[split]


def test_warm_skips_formats_the_export_lacks(tmp_path):
    pytest.importorskip("ultralytics")
    export_dir = make_export(tmp_path / "data", n_train=2, n_val=2)
    assert {c.parent.parent.name for c in warm_cache.warm(export_dir)} == {"yolo_seg"}


def test_atomic_cache_saver_writes_a_temp_file_then_renames_it(tmp_path):
    target = tmp_path / "labels" / "train.cache"
    target.parent.mkdir()
    target.write_bytes(b"old")
    seen = []

    def save(prefix, path, x, version):  # Ultralytics' own: deletes `path`, then writes it in place
        seen.append(Path(path))
        assert target.read_bytes() == b"old"  # the old cache stays whole while the new one is written
        Path(path).write_bytes(b"new")

    atomic_cache_saver(save)("p", target, {}, "1")
    assert seen[0] != target and seen[0].parent == target.parent
    assert target.read_bytes() == b"new" and sorted(target.parent.iterdir()) == [target]


def test_atomic_cache_saver_leaves_no_temp_file_when_ultralytics_skips_the_save(tmp_path):
    atomic_cache_saver(lambda *a: None)("p", tmp_path / "train.cache", {}, "1")
    assert list(tmp_path.iterdir()) == []


def test_main_reports_a_failure_without_raising(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(warm_cache.os, "nice", lambda n: None)

    def boom(export_dir):
        raise FileNotFoundError("export deleted")

    monkeypatch.setattr(warm_cache, "warm", boom)
    assert warm_cache.main([str(tmp_path / "gone")]) == 1
    assert "export deleted" in capsys.readouterr().err


class FakeProc:
    def __init__(self, ignores_sigterm: bool = False) -> None:
        self.done = threading.Event()
        self.signals: list[str] = []
        self.ignores_sigterm = ignores_sigterm

    def wait(self, timeout=None):
        if not self.done.wait(timeout) and timeout is not None:
            raise subprocess.TimeoutExpired("warm_cache", timeout)
        return 0

    def terminate(self) -> None:
        self.signals.append("TERM")
        if not self.ignores_sigterm:
            self.done.set()

    def kill(self) -> None:
        self.signals.append("KILL")
        self.done.set()


class FakeSpawn:
    def __init__(self) -> None:
        self.started: list[Path] = []
        self.procs: list[FakeProc] = []

    def __call__(self, export_dir):
        self.started.append(Path(export_dir))
        self.procs.append(FakeProc())
        return self.procs[-1]


def test_warmer_runs_one_warmup_at_a_time_and_never_queues_an_export_twice(tmp_path):
    spawn = FakeSpawn()
    w = LabelCacheWarmer(enabled=True, spawn=spawn)
    a, b = tmp_path / "a", tmp_path / "b"
    assert w.request(a) is True
    assert w.request(a) is False  # already warming
    assert w.request(b) is True
    assert w.request(b) is False  # already queued
    assert spawn.started == [a]
    spawn.procs[0].done.set()
    wait_for(lambda: spawn.started == [a, b])
    spawn.procs[1].done.set()
    wait_for(lambda: w.request(a))  # finished: a new export of the same dir may be warmed again
    w.shutdown()


def test_warmer_skips_when_disabled_or_ultralytics_is_missing(tmp_path):
    spawn = FakeSpawn()
    assert LabelCacheWarmer(enabled=False, spawn=spawn).request(tmp_path) is False
    assert LabelCacheWarmer(enabled=True, available=lambda: False, spawn=spawn).request(tmp_path) is False
    assert spawn.started == []


def test_warmer_moves_on_when_a_warmup_cannot_start(tmp_path):
    started = []

    def spawn(export_dir):
        started.append(export_dir)
        return None if export_dir.name == "a" else FakeProc()

    w = LabelCacheWarmer(enabled=True, spawn=spawn)
    w.request(tmp_path / "a")
    w.request(tmp_path / "b")
    assert started == [tmp_path / "a", tmp_path / "b"]  # a's failure did not leave the queue stuck
    w.shutdown()


def test_warmer_never_raises_into_the_caller(tmp_path):
    def spawn(export_dir):
        raise RuntimeError("fork failed")

    assert LabelCacheWarmer(enabled=True, spawn=spawn).request(tmp_path) is False


def test_warmer_shutdown_kills_the_running_warmup_and_drops_the_queue(tmp_path):
    spawn = FakeSpawn()
    w = LabelCacheWarmer(enabled=True, spawn=spawn)
    w.request(tmp_path / "a")
    w.request(tmp_path / "b")
    w.shutdown()
    assert spawn.procs[0].signals == ["TERM"]
    assert spawn.started == [tmp_path / "a"]
    assert w.request(tmp_path / "c") is False


def test_warmer_shutdown_kills_a_warmup_that_ignores_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(warm_cache, "STOP_TIMEOUT_S", 0.05)
    proc = FakeProc(ignores_sigterm=True)
    w = LabelCacheWarmer(enabled=True, spawn=lambda d: proc)
    w.request(tmp_path)
    w.shutdown()
    assert proc.signals == ["TERM", "KILL"]


def test_warm_removes_temp_files_left_by_a_killed_warmup(tmp_path):
    pytest.importorskip("ultralytics")
    export_dir = make_export(tmp_path / "data", n_train=2, n_val=2)
    stale = export_dir / "yolo_seg" / "labels" / "train.cache.12345.tmp"
    stale.write_bytes(b"half")
    warm_cache.warm(export_dir)
    assert not stale.exists()


def test_a_sigterm_unwinds_the_warmup_so_its_temp_file_is_removed(tmp_path):
    target = tmp_path / "train.cache"

    def save(prefix, path, x, version):
        Path(path).write_bytes(b"half")
        warm_cache._exit_on_sigterm(15, None)  # SIGTERM arrives mid-save

    with pytest.raises(SystemExit):
        atomic_cache_saver(save)("p", target, {}, "1")
    assert list(tmp_path.iterdir()) == []


def test_real_warmup_subprocess_writes_the_caches(tmp_path):
    """The real spawn path: `python -m vdp.train.warm_cache` in a low-priority subprocess, logging to the export."""
    pytest.importorskip("ultralytics")
    export_dir = make_export(tmp_path / "data", n_train=3, n_val=2)
    w = LabelCacheWarmer(enabled=True, spawn=warm_cache.popen_warmup)  # the real one, not the suite's no-op
    assert w.request(export_dir)
    caches = [export_dir / "yolo_seg" / "labels" / f"{s}.cache" for s in ("train", "val")]
    wait_for(lambda: w._current is None and all(c.is_file() for c in caches), timeout=120, interval=0.2)
    assert "warm:" in (export_dir / "warm_cache.log").read_text()
