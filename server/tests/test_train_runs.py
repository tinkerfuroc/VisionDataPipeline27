from __future__ import annotations

import json
import os
from datetime import datetime

import pytest
from train_fixtures import EXPORT_ID, make_export
from vdp.store import NotFound
from vdp.train.models import TrainParams, find_export
from vdp.train.runs import RunStore, atomic_write_json, new_run_id, read_json


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
    import vdp.train.runs

    def raise_os_error(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(vdp.train.runs.os, "replace", raise_os_error)
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


def test_list_skips_malformed_run_json(tmp_path):
    export = find_export(tmp_path, make_export(tmp_path).name)
    store = RunStore(tmp_path / "runs")
    good = store.create(export, "yolo26s-seg", 0, TrainParams(epochs=5), owner_pid=os.getpid())

    missing_field = store.root / "missing-field"
    missing_field.mkdir()
    (missing_field / "run.json").write_text(json.dumps({
        "id": "20260101-000000-yolo26s-seg-aaaa", "status": "queued", "gpu": 0, "model": "yolo26s-seg",
        "params": {},
        # no created_at
    }))

    bad_id = store.root / "bad-id"
    bad_id.mkdir()
    (bad_id / "run.json").write_text(json.dumps({
        "id": "not-a-valid-run-id", "created_at": "2026-01-01T00:00:00+00:00", "status": "queued",
        "gpu": 0, "model": "yolo26s-seg", "params": {},
    }))

    assert [r["id"] for r in store.list()] == [good["id"]]


def test_create_records_the_epoch_plan_for_the_export(tmp_path):
    export = find_export(tmp_path, make_export(tmp_path, n_train=40, n_val=10).name)
    store = RunStore(tmp_path / "runs")
    params = TrainParams(epochs=50, batch=4, train_images_per_epoch=10, val_images_per_epoch=4)
    run = store.create(export, "yolo26s-seg", 0, params, owner_pid=os.getpid())
    assert run["sampling"] == {
        "train_images": 40, "train_images_per_epoch": 10, "val_images": 10, "val_images_per_epoch": 4,
        "warmup_epochs": 12.0, "close_mosaic": 40,
    }
    assert run["params"]["train_images_per_epoch"] == 10  # what was asked, as asked
    plain = store.create(export, "yolo26s-seg", 0, TrainParams(), owner_pid=os.getpid())
    assert plain["sampling"] is None
