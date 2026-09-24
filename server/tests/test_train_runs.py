from __future__ import annotations

import os
from datetime import datetime

import pytest
from train_fixtures import EXPORT_ID, make_export
from vsp.store import NotFound
from vsp.train.models import TrainParams, find_export
from vsp.train.runs import RunStore, atomic_write_json, new_run_id, read_json


def test_new_run_id_shape():
    rid = new_run_id("yolo26s-seg", datetime(2026, 9, 24, 12, 3, 1))  # noqa: DTZ001
    assert rid.startswith("20260924-120301-yolo26s-seg-") and len(rid.split("-")[-1]) == 4


def test_atomic_json_roundtrip(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert read_json(tmp_path / "a.json") == {"x": 1}
    assert read_json(tmp_path / "missing.json", {}) == {}
    (tmp_path / "bad.json").write_text("{")
    assert read_json(tmp_path / "bad.json") is None
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_json_cleanup_on_os_replace_error(tmp_path, monkeypatch):
    """atomic_write_json cleans up .tmp file when os.replace raises."""
    import vsp.train.runs

    def raise_os_error(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(vsp.train.runs.os, "replace", raise_os_error)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_json(tmp_path / "test.json", {"x": 1})
    assert not list(tmp_path.glob("*.tmp"))


def test_create_read_update_list_delete(tmp_path):
    export = find_export(tmp_path, make_export(tmp_path).name)
    store = RunStore(tmp_path / "runs")
    run = store.create(export, "yolo26s-seg", 1, TrainParams(epochs=5), owner_pid=os.getpid())
    assert run["status"] == "queued" and run["gpu"] == 1 and run["params"]["epochs"] == 5
    assert run["export_id"] == EXPORT_ID and run["export_dir"] == str(export.dir)
    assert run["project_name"] == "Bench" and run["resume"] is False and run["best"] is None
    assert store.read(run["id"]) == run
    assert store.update(run["id"], status="running")["status"] == "running"
    second = store.create(export, "yolo11m-seg", 0, TrainParams(), owner_pid=os.getpid())
    assert next(r["id"] for r in store.list()) in (run["id"], second["id"])
    assert {r["id"] for r in store.list()} == {run["id"], second["id"]}
    store.delete(run["id"])
    with pytest.raises(NotFound):
        store.read(run["id"])
    with pytest.raises(NotFound):
        store.dir("../../etc")
    assert RunStore(tmp_path / "nothing").list() == []
