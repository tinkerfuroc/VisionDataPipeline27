from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine

# Helpers copied from test_api_core.py rather than imported: importing `client` (a fixture) alongside test
# functions that take `client` as a parameter trips ruff's F811 (redefinition) in this repo's configuration.


def _settings(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    s.ingest.allowed_roots = [str(tmp_path)]
    return s


def wait_job(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error", "cancelled"):
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {snap}")


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        yield c


def _frames_dir(tmp_path, n=5):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(n):
        assert cv2.imwrite(str(src / f"f_{i}.png"), np.full((60, 120, 3), 90, np.uint8))
    return src


def _project(client, classes=("part",)):
    p = client.post("/api/projects", json={"name": "Board"}).json()
    for name in classes:
        assert client.post(f"/api/projects/{p['id']}/classes", json={"name": name}).status_code == 201
    return p["id"]


def _clip_with_masks(client, tmp_path, project_id):
    started = client.post(f"/api/projects/{project_id}/clips/import-frames", json={"path": str(_frames_dir(tmp_path, n=6)), "fps": 10})
    clip_id = wait_job(client, started.json()["job_id"])["result"]["clip_id"]
    accepted = client.post(f"/api/clips/{clip_id}/accept", json={"frame": 0, "prompt": {"type": "clicks", "points": [[30, 30, 1]]}, "class_id": 0})
    assert accepted.status_code == 200, accepted.text
    job = client.post(f"/api/clips/{clip_id}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]
    assert wait_job(client, job)["status"] == "done"
    return clip_id


def test_export_runs_as_a_job_and_reports_where_it_wrote(client, tmp_path):
    project_id = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project_id)
    started = client.post(f"/api/projects/{project_id}/export", json={"clips": [clip_id], "formats": ["yolo_det", "coco"]})
    assert started.status_code == 202, started.text
    snap = wait_job(client, started.json()["job_id"])
    assert snap["status"] == "done" and snap["kind"] == "export"
    out = Path(snap["result"]["path"])
    assert out.parent.name == "exports" and (out / "coco" / "annotations" / "instances_train.json").is_file()
    assert json.loads((out / "export.json").read_text())["options"]["val_fraction"] == 0.2


def test_export_with_a_name_produces_a_slugged_export_id(client, tmp_path):
    project_id = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project_id)
    started = client.post(f"/api/projects/{project_id}/export",
                          json={"clips": [clip_id], "formats": ["yolo_seg"], "name": "Cups v2"})
    assert started.status_code == 202, started.text
    snap = wait_job(client, started.json()["job_id"])
    assert snap["result"]["export_id"] == "cups-v2" and snap["result"]["name"] == "Cups v2"


def test_export_rejects_an_overlong_name_before_the_job_starts(client, tmp_path):
    project_id = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project_id)
    started = client.post(f"/api/projects/{project_id}/export",
                          json={"clips": [clip_id], "formats": ["yolo_seg"], "name": "a" * 81})
    assert started.status_code == 400


def test_export_request_errors(client, tmp_path):
    project_id = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project_id)
    url = f"/api/projects/{project_id}/export"
    assert client.post(url, json={"clips": [clip_id], "formats": ["voc"]}).status_code == 400
    assert client.post(url, json={"clips": [], "formats": ["coco"]}).status_code == 400
    assert client.post(url, json={"clips": ["nope"], "formats": ["coco"]}).status_code == 404
    assert client.post("/api/projects/nope/export", json={"clips": [clip_id], "formats": ["coco"]}).status_code == 404
    assert client.post(url, json={"clips": [clip_id], "formats": ["coco"], "options": {"val_fraction": 1.5}}).status_code == 422
    body = client.post(url, json={"clips": [clip_id], "formats": ["coco"], "options": {"split_mode": "per_clip"}})
    assert body.status_code == 400 and "at least two clips" in body.json()["detail"]


class _SlowEngine(FakeEngine):
    def propagate(self, frames, seed_mask, cancel_cb=None):
        for item in super().propagate(frames, seed_mask, cancel_cb):
            time.sleep(0.03)
            yield item


def test_export_is_refused_while_a_propagation_runs(tmp_path):
    app = create_app(_settings(tmp_path), engine=_SlowEngine(), load_engine=False)
    with TestClient(app) as slow:
        project_id = _project(slow)
        started = slow.post(f"/api/projects/{project_id}/clips/import-frames", json={"path": str(_frames_dir(tmp_path, n=40)), "fps": 10})
        clip_id = wait_job(slow, started.json()["job_id"])["result"]["clip_id"]
        click = {"type": "clicks", "points": [[30, 30, 1]]}
        assert slow.post(f"/api/clips/{clip_id}/accept", json={"frame": 0, "prompt": click, "class_id": 0}).status_code == 200
        job_id = slow.post(f"/api/clips/{clip_id}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]
        busy = slow.post(f"/api/projects/{project_id}/export", json={"clips": [clip_id], "formats": ["coco"]})
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id
        slow.post(f"/api/jobs/{job_id}/cancel")
        wait_job(slow, job_id)


def test_a_failed_export_is_a_job_error_and_leaves_no_directory(client, tmp_path, monkeypatch):
    project_id = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project_id)

    def boom(self, plan, out_dir, tick):
        raise RuntimeError("disk full")

    monkeypatch.setattr("vsp.export.coco.Coco.write", boom)
    started = client.post(f"/api/projects/{project_id}/export", json={"clips": [clip_id], "formats": ["yolo_det", "coco"]})
    snap = wait_job(client, started.json()["job_id"])
    assert snap["status"] == "error" and "disk full" in snap["error"]
    assert list((tmp_path / "data" / "exports").iterdir()) == []


def test_a_clip_cannot_be_deleted_while_an_export_reads_it(client, tmp_path, monkeypatch):
    import threading

    project_id = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project_id)
    entered, release = threading.Event(), threading.Event()
    from vsp.export.coco import Coco

    real = Coco.write

    def gated(self, plan, out_dir, tick):
        entered.set()
        assert release.wait(10)
        return real(self, plan, out_dir, tick)

    monkeypatch.setattr(Coco, "write", gated)
    job_id = client.post(f"/api/projects/{project_id}/export", json={"clips": [clip_id], "formats": ["coco"]}).json()["job_id"]
    assert entered.wait(10)
    refused = client.delete(f"/api/clips/{clip_id}")
    assert refused.status_code == 409 and refused.json()["job_id"] == job_id
    release.set()
    assert wait_job(client, job_id)["status"] == "done"
    assert client.delete(f"/api/clips/{clip_id}").status_code == 204
