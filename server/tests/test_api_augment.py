from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine


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
    from vsp.augment import backgrounds

    # POOLS["dtd"] is the same dict object imported into both vsp.augment.backgrounds and
    # vsp.api.augment, so mutating one key on it (auto-reverted by monkeypatch) is visible to both,
    # unlike replacing the whole POOLS mapping (which would only rebind this module's local name).
    monkeypatch.setitem(backgrounds.POOLS["dtd"], "url", server)

    listed = client.get("/api/backgrounds").json()
    assert listed == [{"name": "dtd", "label": "DTD textures", "images": 0,
                       "size_bytes": backgrounds.POOLS["dtd"]["size"], "fetching": None}]

    r = client.post("/api/backgrounds/dtd/fetch")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    snap = wait_job(client, job_id)
    assert snap["status"] == "done" and snap["result"] == {"pool": "dtd", "images": 2, "skipped": False}

    listed = client.get("/api/backgrounds").json()
    assert listed[0]["images"] == 2

    assert client.post("/api/backgrounds/does-not-exist/fetch").status_code == 404
