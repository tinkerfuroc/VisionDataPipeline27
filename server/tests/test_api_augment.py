from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from vdp.api.app import create_app
from vdp.config import Settings
from vdp.engine.fake import FakeEngine
from vdp.export import ExportOptions, run_export
from vdp.store.models import LabelObject


def _settings(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    return s


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        yield c


def wait_job(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error", "cancelled"):
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {snap}")


def test_list_fetch_and_unknown_pool(client, server, monkeypatch):
    from vdp.augment import backgrounds

    # POOLS["dtd"] is the same dict object imported into both vdp.augment.backgrounds and
    # vdp.api.augment, so mutating one key on it (auto-reverted by monkeypatch) is visible to both,
    # unlike replacing the whole POOLS mapping (which would only rebind this module's local name).
    monkeypatch.setitem(backgrounds.POOLS["dtd"], "url", server)

    listed = client.get("/api/backgrounds").json()
    assert listed == [{"name": "dtd", "label": "DTD textures", "images": 0,
                       "size_bytes": backgrounds.POOLS["dtd"]["size"], "fetching": None}]

    r = client.post("/api/backgrounds/dtd/fetch")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    snap = wait_job(client, job_id)
    assert snap["status"] == "done" and snap["result"] == {"pool": "dtd", "images": 3, "skipped": False}

    listed = client.get("/api/backgrounds").json()
    assert listed[0]["images"] == 3

    assert client.post("/api/backgrounds/does-not-exist/fetch").status_code == 404


def _labelled(store, make_clip, n=6):
    project, clip = make_clip(n_frames=n)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(n):
        store.write_mask(clip.id, 1, i, mask)
    return project, clip


def _make_pool(data_root, n=3, seed=0):
    d = data_root / "backgrounds" / "dtd" / "x"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(d / f"bg_{i}.jpg"), img)


def test_augment_route_creates_derived_export_and_serves_samples(client, store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    exports_root = data_root / "exports"
    src = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), exports_root)
    _make_pool(data_root)

    started = client.post(f"/api/exports/{src['export_id']}/augment", json={"per_image": 1, "seed": 2})
    assert started.status_code == 202, started.text
    snap = wait_job(client, started.json()["job_id"])
    assert snap["status"] == "done" and snap["kind"] == "augment"
    new_id = snap["result"]["export_id"]

    exports = client.get("/api/exports").json()
    entry = next(e for e in exports if e["id"] == new_id)
    assert entry["derived_from"] == src["export_id"]

    listed = client.get(f"/api/exports/{new_id}/samples", params={"limit": 3}).json()
    assert len(listed) == 3
    for item in listed:
        assert item["name"].startswith("syn_") and item["ground_truth"]

    name = listed[0]["name"]
    img = client.get(f"/api/exports/{new_id}/images/train/{name}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    assert client.get(f"/api/exports/{new_id}/images/test/x.jpg").status_code == 404


def test_augment_route_makes_a_lighting_only_copy_and_samples_its_relit_frames(client, store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), data_root / "exports")

    started = client.post(f"/api/exports/{src['export_id']}/augment", json={"per_image": 0, "lighting": 2})
    assert started.status_code == 202, started.text
    snap = wait_job(client, started.json()["job_id"])
    assert snap["status"] == "done", snap
    new_id = snap["result"]["export_id"]

    listed = client.get(f"/api/exports/{new_id}/samples", params={"limit": 48}).json()
    assert listed and all(item["name"].startswith("lit_") for item in listed)

    bad = client.post(f"/api/exports/{src['export_id']}/augment", json={"per_image": 0, "lighting": 0})
    assert bad.status_code == 400
    assert client.post(f"/api/exports/{src['export_id']}/augment", json={"lighting": 4}).status_code == 422


def test_augment_requires_masks_format(client, store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = run_export(store, project.id, [clip.id], ["yolo_seg"], ExportOptions(), data_root / "exports")
    assert client.post(f"/api/exports/{src['export_id']}/augment", json={}).status_code == 400


def test_augment_unknown_pool_is_bad_request(client, store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    exports_root = data_root / "exports"
    src = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), exports_root)
    _make_pool(data_root)
    r = client.post(f"/api/exports/{src['export_id']}/augment", json={"pool": "does-not-exist"})
    assert r.status_code == 400


def test_augment_and_fetch_share_one_exclusive_key(client, store, make_clip, server, tmp_path, monkeypatch):
    from vdp.augment import backgrounds

    monkeypatch.setitem(backgrounds.POOLS["dtd"], "url", server)
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    exports_root = data_root / "exports"
    src = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), exports_root)
    _make_pool(data_root)

    started = client.post(f"/api/exports/{src['export_id']}/augment", json={"per_image": 1, "seed": 2})
    assert started.status_code == 202, started.text
    r = client.post("/api/backgrounds/dtd/fetch")
    assert r.status_code == 409
    assert r.json()["job_id"] == started.json()["job_id"]
    wait_job(client, started.json()["job_id"])


def test_server_shutdown_cancels_a_running_augment_job_and_waits_for_it(tmp_path):
    app = create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    seen = []

    def slow_augment(job):  # stands in for run_augment: stops at the next unit once cancelled
        while not job.cancel_requested():
            time.sleep(0.01)
        time.sleep(0.3)  # the running units finish and the half-written export is removed
        seen.append("cleaned up")
        return {"cancelled": True}

    with TestClient(app):
        job = app.state.vdp.jobs.start("augment", slow_augment, exclusive="augment-data")
        start = time.monotonic()
    assert time.monotonic() - start < 30
    assert job.status == "cancelled" and seen == ["cleaned up"]


def test_augment_limits_are_validated_and_accepted(client, store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), data_root / "exports")
    _make_pool(data_root)
    url = f"/api/exports/{src['export_id']}/augment"
    real = json.loads((data_root / "exports" / src["export_id"] / "export.json").read_text())["counts"]["train"]["images"]

    r = client.post(url, json={"per_image": 1, "max_train_images": real})
    assert r.status_code == 400 and "already has" in r.json()["detail"]
    assert client.post(url, json={"per_image": 1, "time_limit_s": 5}).status_code == 422
    assert client.post(url, json={"per_image": 1, "max_train_images": 0}).status_code == 422
    assert client.post(url, json={"per_image": 1, "max_train_images": 10_000_001}).status_code == 422

    ok = client.post(url, json={"per_image": 1, "max_train_images": real + 2, "time_limit_s": 60})
    assert ok.status_code == 202, ok.text
    snap = wait_job(client, ok.json()["job_id"])
    assert snap["status"] == "done", snap

    # the export list carries the derived copy's limits block, so the web can say how the copy was cut
    listed = {e["id"]: e for e in client.get("/api/exports").json()}
    assert listed[snap["result"]["export_id"]]["augment_limits"] == snap["result"]["limits"]
    assert listed[snap["result"]["export_id"]]["augment_limits"]["stopped_by"] == "images"
    assert listed[src["export_id"]]["augment_limits"] is None
