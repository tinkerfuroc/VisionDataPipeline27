import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine
from vsp.engine.types import EngineNotLoaded


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


def test_import_video_upload(client, tmp_path):
    pid = _project(client)
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (120, 60))
    for _ in range(10):
        writer.write(np.full((60, 120, 3), 90, np.uint8))
    writer.release()
    with path.open("rb") as fh:
        r = client.post(
            f"/api/projects/{pid}/clips/import-video", files={"file": ("bench.avi", fh, "video/avi")}, data={"stride": "2"}
        )
    assert r.status_code == 202
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done"
    detail = client.get(f"/api/clips/{snap['result']['clip_id']}").json()
    assert detail["clip"]["frame_count"] == 5 and detail["clip"]["source"] == {"type": "video", "name": "bench.avi"}
    assert list((tmp_path / "data" / "tmp").iterdir()) == []  # the upload was removed after import


def test_a_bad_upload_becomes_a_job_error(client):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-video", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "error" and "cannot open" in snap["error"]


def test_job_endpoints_and_websocket_replay(client, tmp_path):
    assert client.get("/api/jobs/nope").status_code == 404
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
