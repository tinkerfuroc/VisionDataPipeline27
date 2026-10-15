from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_api_core import _project, wait_job
from test_api_upload import _png, _settings, _video_bytes, _zip_bytes
from vdp.api.app import create_app
from vdp.auth import AuthStore
from vdp.engine.fake import FakeEngine


def _app(tmp_path, **ingest):
    return create_app(_settings(tmp_path, **ingest), engine=FakeEngine(), load_engine=False)


@pytest.fixture
def client(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        yield c


def _create(client, pid, name, size, **extra):
    return client.post(f"/api/projects/{pid}/uploads", json={"name": name, "size": size, **extra})


def _send_all(client, upload_id, body, chunk=7):
    last = None
    for offset in range(0, len(body), chunk):
        last = client.put(f"/api/uploads/{upload_id}?offset={offset}", content=body[offset:offset + chunk])
        assert last.status_code == 200, last.text
    return last


def test_a_video_sent_in_chunks_becomes_a_clip(client, tmp_path):
    pid = _project(client)
    body = _video_bytes(tmp_path / "v.avi")
    r = _create(client, pid, "IMG_0412.MOV", len(body), stride=2)  # iPhone names are upper case
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["received"] == 0 and created["chunk_size"] == 8 * 2**20
    last = _send_all(client, created["upload_id"], body)
    assert last.json()["received"] == len(body) and "job_id" in last.json()
    snap = wait_job(client, last.json()["job_id"])
    assert snap["status"] == "done", snap
    clip = client.get(f"/api/clips/{snap['result']['clip_id']}").json()
    assert clip["clip"]["frame_count"] == 5  # 10 frames, stride 2
    assert client.get(f"/api/uploads/{created['upload_id']}").status_code == 404  # cleaned up after the import
    assert list((tmp_path / "data" / "uploads" / "resumable").glob("*")) == []


def test_a_zip_works_the_same_way(client):
    pid = _project(client)
    body = _zip_bytes({"a/0001.png": _png(), "a/0002.png": _png()})
    upload_id = _create(client, pid, "frames.zip", len(body), fps=10).json()["upload_id"]
    snap = wait_job(client, _send_all(client, upload_id, body, chunk=50).json()["job_id"])
    assert snap["status"] == "done" and snap["result"]["frame_count"] == 2


def test_progress_list_and_offset_rules(client):
    pid = _project(client)
    upload_id = _create(client, pid, "v.mp4", 10).json()["upload_id"]
    assert client.put(f"/api/uploads/{upload_id}?offset=0", content=b"abcd").json() == {"received": 4}
    assert client.put(f"/api/uploads/{upload_id}?offset=0", content=b"abcd").json() == {"received": 4}  # retry
    gap = client.put(f"/api/uploads/{upload_id}?offset=6", content=b"gh")
    assert gap.status_code == 409 and gap.json()["received"] == 4
    assert client.get(f"/api/uploads/{upload_id}").json()["received"] == 4
    listed = client.get(f"/api/projects/{pid}/uploads").json()
    assert [u["upload_id"] for u in listed] == [upload_id] and listed[0]["name"] == "v.mp4"
    assert client.put(f"/api/uploads/{upload_id}?offset=4", content=b"").status_code == 400
    assert client.put(f"/api/uploads/{upload_id}?offset=4", content=b"x" * 7).status_code == 413  # past size


def test_create_validation(client, tmp_path):
    pid = _project(client)
    assert _create(client, pid, "notes.txt", 10).status_code == 400
    assert _create(client, pid, "v.mp4", 0).status_code == 422
    assert _create(client, pid, "v.mp4", 10, stride=0).status_code == 422
    assert _create(client, pid, "v.mp4", 10, fps=0).status_code == 422
    assert _create(client, "nope", "v.mp4", 10).status_code == 404


def test_create_over_the_cap_is_413(tmp_path):
    with TestClient(_app(tmp_path, max_upload_bytes=100)) as client:
        pid = _project(client)
        assert _create(client, pid, "v.mp4", 101).status_code == 413


def test_unknown_ids_are_404(client):
    assert client.get("/api/uploads/" + "a" * 32).status_code == 404
    assert client.put("/api/uploads/" + "a" * 32 + "?offset=0", content=b"x").status_code == 404
    assert client.delete("/api/uploads/nothex").status_code == 404


def test_a_chunk_over_the_chunk_size_is_413(client):
    pid = _project(client)
    upload_id = _create(client, pid, "v.mp4", 9 * 2**20).json()["upload_id"]
    r = client.put(f"/api/uploads/{upload_id}?offset=0", content=b"x" * (8 * 2**20 + 1))
    assert r.status_code == 413
    assert client.get(f"/api/uploads/{upload_id}").json()["received"] == 0


def test_delete_cancels_an_unfinished_upload(client):
    pid = _project(client)
    upload_id = _create(client, pid, "v.mp4", 10).json()["upload_id"]
    assert client.delete(f"/api/uploads/{upload_id}").status_code == 204
    assert client.get(f"/api/projects/{pid}/uploads").json() == []


def test_a_repeated_final_chunk_returns_the_same_job(client, tmp_path):
    pid = _project(client)
    body = _video_bytes(tmp_path / "v.avi")
    upload_id = _create(client, pid, "v.avi", len(body)).json()["upload_id"]
    client.put(f"/api/uploads/{upload_id}?offset=0", content=body[:-5])
    first = client.put(f"/api/uploads/{upload_id}?offset={len(body) - 5}", content=body[-5:]).json()
    again = client.put(f"/api/uploads/{upload_id}?offset={len(body) - 5}", content=body[-5:])
    if again.status_code == 200:  # the record still exists while the import runs
        assert again.json()["job_id"] == first["job_id"]
    else:  # the import already finished and removed it
        assert again.status_code == 404
    assert wait_job(client, first["job_id"])["status"] == "done"


def test_unfinished_uploads_survive_a_restart(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        pid = _project(client)
        upload_id = _create(client, pid, "v.mp4", 10).json()["upload_id"]
        client.put(f"/api/uploads/{upload_id}?offset=0", content=b"abcd")
    with TestClient(_app(tmp_path)) as client:
        assert client.get(f"/api/uploads/{upload_id}").json()["received"] == 4


def test_the_phone_link_redirects_to_the_upload_page(client):
    r = client.get("/upload?project=abc", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/#/upload?project=abc"
    assert client.get("/upload", follow_redirects=False).headers["location"] == "/#/upload"


def test_the_login_gate_covers_the_new_routes(tmp_path):
    settings = _settings(tmp_path)
    AuthStore(settings.project.data_root).set_password("pw")
    with TestClient(create_app(settings, engine=FakeEngine(), load_engine=False)) as client:
        assert client.post("/api/projects/p/uploads", json={"name": "v.mp4", "size": 1}).status_code == 401
        assert client.put("/api/uploads/" + "a" * 32 + "?offset=0", content=b"x").status_code == 401
        r = client.get("/upload?project=p", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login?next=%2Fupload")
