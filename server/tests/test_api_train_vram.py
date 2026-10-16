from __future__ import annotations

import sys
import threading

import pytest
from fastapi.testclient import TestClient
from test_api_train import _FakeNvml, start, until
from train_fixtures import FAKE_WORKER, make_export, wait_for
from vdp.api import train as train_api
from vdp.api.app import create_app
from vdp.config import Settings
from vdp.engine.fake import FakeEngine
from vdp.train import scheduler as scheduler_mod
from vdp.train.scheduler import Gpu, TrainScheduler
from vdp.train.vram import VramEstimator, with_margin

GIB = 2**30
MEASURED = [{"batch": 1, "bytes": 1 * GIB}, {"batch": 2, "bytes": 2 * GIB}, {"batch": 4, "bytes": 4 * GIB},
            {"batch": 8, "bytes": None}, {"batch": 12, "bytes": None}]


class FakeRunner:
    """Stands in for the probe subprocess. `gate`, when set, holds the probe after its first batch."""

    def __init__(self, measured=MEASURED) -> None:
        self.calls: list[tuple] = []
        self.measured = measured
        self.gate: threading.Event | None = None
        self.stopped = False

    def __call__(self, model, imgsz, gpu, on_batch, should_stop):
        self.calls.append((model, imgsz, gpu))
        for i, e in enumerate(self.measured):
            on_batch(e["batch"], e["bytes"])
            if i == 0 and self.gate is not None:
                while not self.gate.wait(0.02):
                    if should_stop():
                        self.stopped = True
                        return None
        return {"gpu_name": "ignored: the scheduler's GPU name is the key", "total_bytes": 0,
                "estimates": self.measured}


@pytest.fixture(autouse=True)
def _nvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml({"u0": {"free": 9 * GIB, "clock_mhz": 1800}}))
    monkeypatch.setattr(scheduler_mod, "_nvml_ready", False)


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def client(tmp_path, runner):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    gpus = [Gpu(0, "RTX Fast", 12 * GIB, uuid="u0"), Gpu(1, "RTX Slow", 11 * GIB, uuid="u1")]
    sched = TrainScheduler(s.project.data_root, gpus, worker_cmd=FAKE_WORKER, kill_after=5.0)
    app = create_app(s, engine=FakeEngine(), load_engine=False, train=sched,
                     vram=VramEstimator(s.project.data_root, runner=runner))
    with TestClient(app) as c:
        yield c


def estimate(client, **body):
    return client.post("/api/train/vram-estimate", json={"model": "yolo26s-seg", "imgsz": 960, "gpu": 0, **body})


def finished(client, job_id):
    return wait_for(lambda: (j := client.get(f"/api/jobs/{job_id}").json())["status"] in ("done", "error",
                                                                                         "cancelled") and j)


def test_measures_and_returns_estimates_with_the_margin(client, runner, tmp_path):
    r = estimate(client)
    assert r.status_code == 202, r.text
    job = finished(client, r.json()["job_id"])
    assert job["status"] == "done" and job["kind"] == "vram_estimate"
    res = job["result"]
    assert res == {
        "model": "yolo26s-seg", "imgsz": 960, "gpu": 0, "gpu_name": "RTX Fast", "total_bytes": 12 * GIB,
        "free_bytes": 9 * GIB, "measured_at": res["measured_at"], "cached": False,
        "estimates": [{"batch": e["batch"], "bytes": with_margin(e["bytes"])} for e in MEASURED],
    }
    assert runner.calls == [("yolo26s-seg", 960, 0)]
    assert (tmp_path / "data" / "train_weights" / "vram_estimates.json").is_file()


def test_reports_progress_per_batch(client):
    job_id = estimate(client).json()["job_id"]
    finished(client, job_id)
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        events = []
        while not events or events[-1]["type"] not in ("done", "error"):
            events.append(ws.receive_json())
    progress = [e for e in events if e["type"] == "progress"]
    assert [(e["done"], e["total"], e["batch"]) for e in progress] == [
        (1, 5, 1), (2, 5, 2), (3, 5, 4), (4, 5, 8), (5, 5, 12)]


def test_a_cached_answer_returns_without_running_the_probe(client, runner):
    first = finished(client, estimate(client).json()["job_id"])["result"]
    second = finished(client, estimate(client).json()["job_id"])
    assert second["status"] == "done" and len(runner.calls) == 1
    assert second["result"] == {**first, "cached": True}
    # other image size, other model or other GPU: measured anew
    finished(client, estimate(client, imgsz=640).json()["job_id"])
    finished(client, estimate(client, gpu=1).json()["job_id"])
    assert runner.calls[1:] == [("yolo26s-seg", 640, 0), ("yolo26s-seg", 960, 1)]


def test_remeasure_ignores_the_cache(client, runner):
    finished(client, estimate(client).json()["job_id"])
    job = finished(client, estimate(client, remeasure=True).json()["job_id"])
    assert job["result"]["cached"] is False and len(runner.calls) == 2


def test_free_memory_is_read_fresh_even_for_a_cached_answer(client, monkeypatch):
    finished(client, estimate(client).json()["job_id"])
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml({"u0": {"free": 3 * GIB, "clock_mhz": 1800}}))
    monkeypatch.setattr(scheduler_mod, "_nvml_ready", False)
    job = finished(client, estimate(client).json()["job_id"])
    assert job["result"]["cached"] is True and job["result"]["free_bytes"] == 3 * GIB


@pytest.mark.parametrize("body, status, detail", [
    ({"model": "yolov5s"}, 400, "unknown model"),
    ({"imgsz": 300}, 422, "greater than or equal to 320"),
    ({"imgsz": 2048}, 422, "less than or equal to 1920"),
    ({"imgsz": 650}, 422, "multiple of 32"),
    ({"gpu": 7}, 400, "GPU 7"),
])
def test_validation(client, runner, body, status, detail):
    r = estimate(client, **body)
    assert r.status_code == status and detail in r.text
    assert runner.calls == []


def test_refuses_while_a_training_run_is_on_that_gpu(client, runner):
    rid = start(client, epochs=99, gpu=0)
    until(client, rid, "running")
    r = estimate(client)
    assert r.status_code == 409 and "training" in r.json()["detail"] and "cuda:0" in r.json()["detail"]
    assert runner.calls == []
    assert estimate(client, gpu=1).status_code == 202  # the other GPU is free
    client.post(f"/api/train/runs/{rid}/cancel")


def test_refuses_even_a_cached_answer_while_the_gpu_trains(client, runner):
    """The cached numbers would be right, but the button is disabled then anyway; one rule is simpler."""
    finished(client, estimate(client).json()["job_id"])
    rid = start(client, epochs=99, gpu=0)
    until(client, rid, "running")
    assert estimate(client).status_code == 409
    client.post(f"/api/train/runs/{rid}/cancel")


def test_stops_the_probe_when_training_starts_on_that_gpu(client, runner):
    runner.gate = threading.Event()
    job_id = estimate(client).json()["job_id"]
    wait_for(lambda: runner.calls)
    rid = start(client, epochs=99, gpu=0)
    job = finished(client, job_id)
    assert job["status"] == "error" and "training run was queued" in job["error"] and runner.stopped
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")
    wait_for(lambda: client.app.state.vdp.train.gpu_idle(0))
    runner.gate = None
    assert finished(client, estimate(client).json()["job_id"])["result"]["cached"] is False  # nothing was cached


def test_cancel_stops_the_probe(client, runner):
    runner.gate = threading.Event()
    job_id = estimate(client).json()["job_id"]
    wait_for(lambda: runner.calls)
    client.delete(f"/api/jobs/{job_id}")
    job = finished(client, job_id)
    assert job["status"] == "cancelled" and runner.stopped


def test_one_measurement_at_a_time(client, runner):
    runner.gate = threading.Event()
    first = estimate(client).json()["job_id"]
    wait_for(lambda: runner.calls)
    r = estimate(client, gpu=1)
    assert r.status_code == 409 and r.json()["job_id"] == first
    runner.gate.set()
    finished(client, first)


def test_a_failed_probe_is_a_job_error(client, runner):
    def broken(*args):
        from vdp.train.vram import ProbeFailed
        raise ProbeFailed("FileNotFoundError: yolo26s-seg.pt")

    client.app.state.vdp.vram._runner = broken
    job = finished(client, estimate(client).json()["job_id"])
    assert job["status"] == "error" and "yolo26s-seg.pt" in job["error"]


def test_ultralytics_missing_is_503(client, runner, monkeypatch):
    monkeypatch.setattr(train_api, "ultralytics_error", lambda: "ultralytics is not installed")
    r = estimate(client)
    assert r.status_code == 503 and "ultralytics" in r.json()["detail"] and runner.calls == []
