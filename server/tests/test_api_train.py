from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export, wait_for
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine
from vsp.train.scheduler import Gpu, TrainScheduler


@pytest.fixture
def client(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    sched = TrainScheduler(s.project.data_root, [Gpu(0, "Fast", 12 * 2**30), Gpu(1, "Slow", 11 * 2**30)],
                           worker_cmd=FAKE_WORKER, kill_after=5.0)
    app = create_app(s, engine=FakeEngine(), load_engine=False, train=sched)
    with TestClient(app) as c:
        yield c


def start(client, *, epochs=2, model="yolo26s-seg", gpu=1):
    r = client.post("/api/train/runs", json={
        "export_id": EXPORT_ID, "runs": [{"model": model, "gpu": gpu}],
        "params": {"epochs": epochs, "imgsz": 640},
    })
    assert r.status_code == 201, r.text
    return r.json()["run_ids"][0]


def until(client, rid, *statuses):
    return wait_for(lambda: (v := client.get(f"/api/train/runs/{rid}").json())["status"] in statuses and v)


def test_status_and_exports(client):
    st = client.get("/api/train/status").json()
    assert st["available"] is True and st["error"] is None
    assert [g["index"] for g in st["gpus"]] == [0, 1] and st["gpus"][0]["queue"] == []
    assert "yolo11m-seg" in st["models"] and st["default_models"][0] == "yolo26s-seg"
    assert st["defaults"]["imgsz"] == 960
    assert st["aug_presets"]["strong"]["degrees"] == 15
    assert st["defaults"]["augment"] == "default"
    exports = client.get("/api/exports").json()
    assert exports == [{"id": EXPORT_ID, "project_name": "Bench", "created_at": "2026-01-01T00:00:00+00:00",
                        "classes": ["Module"], "train_images": 2, "val_images": 2}]


def test_run_lifecycle_metrics_predictions_weights(client):
    rid = start(client)
    view = until(client, rid, "done")
    assert view["epochs_done"] == 2 and len(view["metrics"]) == 2 and view["metrics"][1]["epoch"] == 2
    assert view["best"]["epoch"] == 2 and view["has_best"] and not view["has_onnx"]
    assert "pid" not in view and "export_dir" not in view and "owner_pid" not in view
    listed = client.get("/api/train/runs").json()
    assert [r["id"] for r in listed] == [rid] and "metrics" not in listed[0]

    page = client.get(f"/api/train/runs/{rid}/predictions?offset=0&limit=5").json()
    assert page["total"] == 1
    item = page["items"][0]
    assert item["name"] == "clip_000000.jpg" and item["predictions"][0]["score"] == 0.9
    assert item["ground_truth"][0]["polygons"][0][:2] == [0.25, 0.25]
    img = client.get(f"/api/exports/{EXPORT_ID}/images/val/clip_000000.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/exports/{EXPORT_ID}/images/val/..%2Fexport.json").status_code == 404

    w = client.get(f"/api/train/runs/{rid}/weights/best.pt")
    assert w.status_code == 200 and w.content == b"best"
    assert client.get(f"/api/train/runs/{rid}/weights/best.onnx").status_code == 404
    assert client.get(f"/api/train/runs/{rid}/weights/args.yaml").status_code == 404
    assert client.post(f"/api/train/runs/{rid}/onnx").status_code == 200
    wait_for(lambda: client.get(f"/api/train/runs/{rid}").json()["has_onnx"])
    assert client.get(f"/api/train/runs/{rid}/weights/best.onnx").content == b"onnx"
    assert "fake worker" in client.get(f"/api/train/runs/{rid}/log").text

    assert client.delete(f"/api/train/runs/{rid}").status_code == 204
    assert client.get(f"/api/train/runs/{rid}").status_code == 404


def test_cancel_resume_and_conflicts(client, tmp_path):
    # The fake worker (epochs=99) writes last.pt and only then installs its SIGTERM handler; cancelling
    # before last.pt exists can kill it via the default SIGTERM action before that point, so wait for
    # last.pt first (mirrors test_train_scheduler.py::test_resume_after_cancel's guard).
    def last_pt():
        return (tmp_path / "data" / "runs" / rid / "train" / "weights" / "last.pt").is_file()

    rid = start(client, epochs=99)
    running = until(client, rid, "running")
    assert running["progress"] is None or running["progress"]["epoch"] == 1
    gpu = next(g for g in client.get("/api/train/status").json()["gpus"] if g["index"] == 1)
    assert gpu["running"] == rid
    wait_for(last_pt)
    assert client.delete(f"/api/train/runs/{rid}").status_code == 409
    assert client.post(f"/api/train/runs/{rid}/onnx").status_code == 409
    assert client.get(f"/api/train/runs/{rid}/predictions").status_code == 404
    assert client.post(f"/api/train/runs/{rid}/cancel").status_code == 200
    cancelled = until(client, rid, "cancelled")
    assert cancelled["resumable"] is True
    assert client.post(f"/api/train/runs/{rid}/cancel").status_code == 409
    assert client.post(f"/api/train/runs/{rid}/resume").json()["status"] in ("queued", "running")
    until(client, rid, "running")
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")


def test_augment_presets(client):
    r = client.post("/api/train/runs", json={
        "export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 1}],
        "params": {"epochs": 2, "imgsz": 640, "augment": "strong"},
    })
    assert r.status_code == 201
    rid = r.json()["run_ids"][0]
    run = client.get(f"/api/train/runs/{rid}").json()
    assert run["params"]["augment"] == "strong"


def test_bad_requests(client):
    body = {"export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 0}]}
    bad_model = {**body, "runs": [{"model": "yolov8s-seg", "gpu": 0}]}
    bad_gpu = {**body, "runs": [{"model": "yolo26s-seg", "gpu": 5}]}
    assert client.post("/api/train/runs", json=bad_model).status_code == 400
    assert client.post("/api/train/runs", json=bad_gpu).status_code == 400
    assert client.post("/api/train/runs", json={**body, "runs": []}).status_code == 400
    assert client.post("/api/train/runs", json={**body, "export_id": "missing"}).status_code == 404
    assert client.post("/api/train/runs", json={**body, "params": {"imgsz": 1000}}).status_code == 422
    assert client.post("/api/train/runs", json={**body, "params": {"augment": "custom"}}).status_code == 422
    assert client.get("/api/train/runs/not-a-run").status_code == 404
    assert client.get("/api/exports/missing/images/val/a.jpg").status_code == 404
    assert client.get("/api/train/runs/20260101-000000-yolo26s-seg-face/log").status_code == 404


def test_unavailable_without_ultralytics(client, monkeypatch):
    import vsp.api.train as train_api

    monkeypatch.setattr(train_api, "find_spec", lambda name: None)
    st = client.get("/api/train/status").json()
    assert st["available"] is False and "uv sync" in st["error"]
    body = {"export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 0}]}
    assert client.post("/api/train/runs", json=body).status_code == 503


def test_predictions_filters_and_counts(client):
    rid = start(client)
    until(client, rid, "done")
    page = client.get(f"/api/train/runs/{rid}/predictions?filter=all").json()
    assert page["counts"]["all"] == 1 and page["counts"]["missed"] == 0 and page["counts"]["worst"] == 1
    item = page["items"][0]
    assert item["index"] == 0 and 0.0 < item["iou"] <= 1.0
    assert client.get(f"/api/train/runs/{rid}/predictions?filter=missed").json()["total"] == 0
    assert client.get(f"/api/train/runs/{rid}/predictions?filter=nope").status_code == 422
