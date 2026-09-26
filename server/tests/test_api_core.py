import threading
import time

import cv2
import numpy as np
import pytest
import vsp.api.clips as clips_mod
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine
from vsp.engine.types import EngineNotLoaded
from vsp.ingest import FrameDirSource


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


def test_health_with_a_loaded_engine(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["engine"] == {"loaded": True, "device": None, "error": None}
    assert body["upload"] == {"max_bytes": 8 * 2**30}


def test_health_reports_a_background_load_failure(tmp_path):
    class Broken:
        device = None
        load_error = None

        def is_loaded(self):
            return False

        def load(self):
            self.load_error = "no CUDA device"
            raise EngineNotLoaded(self.load_error)

    with TestClient(create_app(_settings(tmp_path), engine=Broken(), load_engine=True)) as c:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and c.get("/api/health").json()["engine"]["error"] is None:
            time.sleep(0.02)
        assert c.get("/api/health").json()["engine"] == {"loaded": False, "device": None, "error": "no CUDA device"}


def test_projects_and_classes(client):
    assert client.get("/api/projects").json() == []
    pid = _project(client, classes=("connector", "capacitor"))
    got = client.get(f"/api/projects/{pid}").json()
    assert [c["name"] for c in got["classes"]] == ["connector", "capacitor"]
    assert [p["id"] for p in client.get("/api/projects").json()] == [pid]
    dup = client.post(f"/api/projects/{pid}/classes", json={"name": "connector"})
    assert dup.status_code == 400 and "already exists" in dup.json()["detail"]
    assert client.get("/api/projects/nope-000000").status_code == 404
    assert client.get("/api/projects/..%2Fx").status_code in (400, 404)


def test_import_frames_then_browse_the_clip(client, tmp_path):
    pid = _project(client)
    src = _frames_dir(tmp_path)
    r = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(src), "fps": 12, "stride": 2})
    assert r.status_code == 202
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done"
    clip_id = snap["result"]["clip_id"]

    listing = client.get(f"/api/projects/{pid}/clips").json()
    assert [c["id"] for c in listing] == [clip_id]
    assert listing[0]["frame_count"] == 3 and listing[0]["labeled_frames"] == 0

    detail = client.get(f"/api/clips/{clip_id}").json()
    assert detail["clip"]["fps"] == 12 and detail["states"] == ["empty"] * 3
    assert [c["name"] for c in detail["classes"]] == ["part"]

    frame = client.get(f"/api/clips/{clip_id}/frames/0")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/clips/{clip_id}/thumbs/0").status_code == 200
    assert client.get(f"/api/clips/{clip_id}/frames/3").status_code == 404
    assert client.get(f"/api/clips/{clip_id}/objects/1/masks/0").status_code == 404
    # the un-scoped mask route no longer exists
    assert client.get(f"/api/clips/{clip_id}/masks/0").status_code == 404

    assert client.delete(f"/api/clips/{clip_id}").status_code == 204
    assert client.get(f"/api/clips/{clip_id}").status_code == 404


def test_import_frames_is_confined_to_allowed_roots(client, tmp_path):
    pid = _project(client)
    outside = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": "/etc"})
    assert outside.status_code == 400 and "outside" in outside.json()["detail"]
    missing = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(tmp_path / "nope")})
    assert missing.status_code == 400
    assert client.post("/api/projects/nope-000000/clips/import-frames", json={"path": str(tmp_path)}).status_code == 404


def test_the_multipart_import_video_route_is_gone(client, tmp_path):
    """Uploads are one raw-body route now (see test_api_upload.py); a second import path must not creep back."""
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-video", files={"file": ("x.mp4", b"x", "video/mp4")})
    assert r.status_code in (404, 405)


def test_import_job_cancelled_mid_way_ends_cancelled_with_no_clip(client, tmp_path, monkeypatch):
    """Deterministic, no sleeps: a gated FrameDirSource pauses after the 2nd frame so the test can
    issue the cancel before the importer's next cancel-check, then release the gate."""
    entered = threading.Event()
    release = threading.Event()
    real_source_cls = FrameDirSource

    class GatedSource:
        type = "frame_dir"

        def __init__(self, path, fps=30.0):
            self._inner = real_source_cls(path, fps=fps)
            self.name = self._inner.name
            self.fps = self._inner.fps

        def frames(self):
            for i, fr in enumerate(self._inner.frames()):
                yield fr
                if i == 1:  # after 2 frames have been written
                    entered.set()
                    assert release.wait(5)

    monkeypatch.setattr(clips_mod, "FrameDirSource", GatedSource)

    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(_frames_dir(tmp_path, n=30))})
    job_id = r.json()["job_id"]
    assert entered.wait(5)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    release.set()

    snap = wait_job(client, job_id)
    assert snap["status"] == "cancelled"
    assert "clip_id" not in snap["result"]
    assert client.get(f"/api/projects/{pid}/clips").json() == []


def test_job_endpoints_and_websocket_replay(client, tmp_path):
    missing = client.get("/api/jobs/nope")
    assert missing.status_code == 404 and missing.json()["detail"] == "job nope not found"
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(_frames_dir(tmp_path))})
    job_id = r.json()["job_id"]
    wait_job(client, job_id)
    events = []
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        try:
            while True:
                events.append(ws.receive_json())
        except WebSocketDisconnect:
            pass
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[-1]["type"] == "done" and events[0]["type"] == "progress"
