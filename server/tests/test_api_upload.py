from __future__ import annotations

import io
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api_core import _project, wait_job
from vdp.api.app import create_app
from vdp.config import Settings
from vdp.engine.fake import FakeEngine


def _settings(tmp_path, **ingest):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    s.ingest.allowed_roots = [str(tmp_path)]
    for key, value in ingest.items():
        setattr(s.ingest, key, value)
    return s


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        yield c


def _png(value: int = 90) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((60, 120, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _video_bytes(path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (120, 60))
    for _ in range(10):
        writer.write(np.full((60, 120, 3), 90, np.uint8))
    writer.release()
    return path.read_bytes()


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_a_video_upload_becomes_a_clip(client, tmp_path):
    pid = _project(client)
    body = _video_bytes(tmp_path / "v.avi")
    r = client.post(f"/api/projects/{pid}/clips/upload?name=bench.avi&stride=2", content=body)
    assert r.status_code == 202, r.text
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done", snap
    detail = client.get(f"/api/clips/{snap['result']['clip_id']}").json()
    assert detail["clip"]["frame_count"] == 5
    assert detail["clip"]["source"] == {"type": "video", "name": "bench.avi"}
    assert list((tmp_path / "data" / "uploads").iterdir()) == []  # removed once the job ended


def test_a_zip_upload_becomes_a_frame_directory_clip(client, tmp_path):
    pid = _project(client)
    body = _zip_bytes({
        "run/frame_2.png": _png(2), "run/frame_10.png": _png(3), "run/frame_1.png": _png(1), "run/notes.txt": b"x",
    })
    r = client.post(f"/api/projects/{pid}/clips/upload?name=run.zip&fps=12&stride=2", content=body)
    assert r.status_code == 202, r.text
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done", snap
    detail = client.get(f"/api/clips/{snap['result']['clip_id']}").json()
    assert detail["clip"]["source"] == {"type": "frame_dir", "name": "run"}
    assert detail["clip"]["fps"] == 12 and detail["clip"]["frame_count"] == 2
    assert detail["clip"]["original_names"] == {"0": "run/frame_1.png", "1": "run/frame_10.png"}


def test_an_unsupported_name_is_refused_before_anything_is_written(client, tmp_path):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/upload?name=notes.txt", content=b"hello")
    assert r.status_code == 400 and "not a video" in r.json()["detail"]
    assert not (tmp_path / "data" / "uploads").exists()
    assert client.post("/api/projects/nope-000000/clips/upload?name=x.mp4", content=b"x").status_code == 404


def test_a_declared_body_over_the_limit_is_refused_with_413(tmp_path):
    app = create_app(_settings(tmp_path, max_upload_bytes=10), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        pid = _project(c)
        r = c.post(f"/api/projects/{pid}/clips/upload?name=x.mp4", content=b"x" * 100)  # Content-Length: 100
        assert r.status_code == 413 and "limit is 10" in r.json()["detail"]
        assert not (tmp_path / "data" / "uploads").exists()


def test_a_chunked_body_over_the_limit_is_refused_while_it_streams(tmp_path):
    """No Content-Length: the cap has to fire from `save_stream`. TestClient sends a generator body chunked."""
    app = create_app(_settings(tmp_path, max_upload_bytes=10), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        pid = _project(c)

        def body():
            yield b"x" * 8
            yield b"x" * 8

        r = c.post(f"/api/projects/{pid}/clips/upload?name=x.mp4", content=body())
        assert r.status_code == 413
        assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_an_unreadable_upload_becomes_a_job_error_and_cleans_up(client, tmp_path):
    pid = _project(client)
    bad_video = client.post(f"/api/projects/{pid}/clips/upload?name=x.mp4", content=b"not a video")
    snap = wait_job(client, bad_video.json()["job_id"])
    assert snap["status"] == "error" and "cannot open" in snap["error"]
    bad_zip = client.post(f"/api/projects/{pid}/clips/upload?name=x.zip", content=b"not a zip")
    snap = wait_job(client, bad_zip.json()["job_id"])
    assert snap["status"] == "error" and "cannot read" in snap["error"]
    assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_a_cancelled_upload_import_removes_the_temp_file(client, tmp_path):
    pid = _project(client)
    body = _zip_bytes({f"f{i}.png": _png() for i in range(40)})
    started = client.post(f"/api/projects/{pid}/clips/upload?name=run.zip", content=body).json()["job_id"]
    client.delete(f"/api/jobs/{started}")  # may land before or after the job starts; both are fine
    snap = wait_job(client, started)
    assert snap["status"] in ("done", "cancelled")
    assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_stale_uploads_are_swept_when_the_app_is_built(tmp_path):
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    litter = uploads / "old_upload.mp4"
    litter.write_bytes(b"x")
    create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    assert not litter.exists()  # a killed server's partial upload never survives the next start
