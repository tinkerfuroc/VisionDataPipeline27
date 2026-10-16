"""Pausing SAM3 for training: `pause_sam3` on the train routes, the automatic reload once SAM3's GPU has
no training left, `POST /api/engine/load`, and what the rest of the API reports while it is paused."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient
from test_api_core import _settings
from test_api_labeling import CLICK, _clip
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export, wait_for
from vdp.api.app import create_app
from vdp.api.deps import GPU
from vdp.engine.fake import FakeEngine
from vdp.train.scheduler import Gpu, TrainScheduler

SAM3_GPU = 1
VRAM = 3 * 2**30


@pytest.fixture
def env(tmp_path):
    s = _settings(tmp_path)
    make_export(s.project.data_root)
    sched = TrainScheduler(s.project.data_root, [Gpu(0, "Fast", 12 * 2**30), Gpu(1, "Slow", 11 * 2**30)],
                           worker_cmd=FAKE_WORKER, kill_after=5.0)
    engine = FakeEngine(device=f"cuda:{SAM3_GPU}", vram_bytes=VRAM)
    with TestClient(create_app(s, engine=engine, load_engine=False, train=sched)) as c:
        yield c, engine


def start(client, *gpus, epochs=99, pause=True, model="yolo26s-seg"):
    return client.post("/api/train/runs", json={
        "export_id": EXPORT_ID, "runs": [{"model": model, "gpu": g} for g in gpus],
        "params": {"epochs": epochs, "imgsz": 640}, "pause_sam3": pause,
    })


def until(client, rid, *statuses):
    return wait_for(lambda: (v := client.get(f"/api/train/runs/{rid}").json())["status"] in statuses and v)


def engine_view(client):
    return client.get("/api/health").json()["engine"]


def test_pause_releases_sam3_and_reloads_it_once_the_last_run_on_its_gpu_ends(env):
    client, engine = env
    st = client.get("/api/train/status").json()
    assert st["sam3_gpu"] == SAM3_GPU and st["sam3_vram_bytes"] == VRAM
    assert engine_view(client)["paused"] is False

    r = start(client, SAM3_GPU, SAM3_GPU)
    assert r.status_code == 201, r.text
    first, second = r.json()["run_ids"]
    assert engine.release_calls == 1 and not engine.is_loaded()
    assert engine_view(client) == {"loaded": False, "device": "cuda:1", "error": None, "paused": True}
    assert client.get("/api/train/status").json()["sam3_gpu"] == SAM3_GPU  # where it will reload

    until(client, first, "running")
    client.post(f"/api/train/runs/{first}/cancel")
    until(client, first, "cancelled")
    until(client, second, "running")
    assert engine_view(client)["paused"] is True  # the second run still trains on SAM3's GPU
    client.post(f"/api/train/runs/{second}/cancel")
    until(client, second, "cancelled")
    wait_for(lambda: engine.is_loaded())
    assert engine_view(client)["paused"] is False and engine.load_calls == 1


def test_auto_reload_also_follows_a_run_that_ends_in_an_error(env):
    client, engine = env
    (rid,) = start(client, SAM3_GPU, epochs=13).json()["run_ids"]  # the fake worker fails at once
    until(client, rid, "error")
    wait_for(lambda: engine.is_loaded())
    assert engine_view(client)["paused"] is False


def test_pause_is_refused_while_a_propagation_runs(env):
    client, engine = env
    release = threading.Event()
    job = client.app.state.vdp.jobs.start("propagate", lambda job: release.wait(5) and None, exclusive=GPU)
    try:
        r = start(client, SAM3_GPU)
        assert r.status_code == 409 and r.json()["job_id"] == job.id
        assert engine.is_loaded() and engine.release_calls == 0
        assert client.get("/api/train/runs").json() == []  # nothing started
        assert engine_view(client)["paused"] is False
    finally:
        release.set()
        job.join(5)


def test_pause_is_ignored_off_sam3s_gpu_or_without_a_loaded_engine(env):
    client, engine = env
    assert start(client, 0, epochs=1).status_code == 201
    assert engine.is_loaded() and engine.release_calls == 0 and engine_view(client)["paused"] is False

    engine.release()  # e.g. it failed to load: there is nothing to free
    assert start(client, SAM3_GPU, epochs=1).status_code == 201
    assert engine.release_calls == 1 and engine_view(client)["paused"] is False


def test_an_invalid_paused_start_leaves_sam3_loaded(env):
    client, engine = env
    assert start(client, SAM3_GPU, model="yolov5s-seg").status_code == 400
    assert start(client, 7).status_code == 400
    assert engine.is_loaded() and engine.release_calls == 0


def test_pause_on_from_checkpoint_and_resume(env, tmp_path):
    client, engine = env
    (src,) = start(client, 0, epochs=2, pause=False).json()["run_ids"]
    until(client, src, "done")
    r = client.post(f"/api/train/runs/{src}/from-checkpoint", json={
        "export_id": EXPORT_ID, "mode": "finetune", "gpu": SAM3_GPU, "pause_sam3": True,
    })
    assert r.status_code == 201, r.text
    assert not engine.is_loaded() and engine_view(client)["paused"] is True
    until(client, r.json()["run_id"], "done")
    wait_for(lambda: engine.is_loaded())

    (rid,) = start(client, 0, pause=False).json()["run_ids"]
    until(client, rid, "running")
    wait_for(lambda: (tmp_path / "data" / "runs" / rid / "train" / "weights" / "last.pt").is_file())
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")
    resumed = client.post(f"/api/train/runs/{rid}/resume", json={"gpu": SAM3_GPU, "pause_sam3": True})
    assert resumed.status_code == 200, resumed.text
    assert not engine.is_loaded() and engine_view(client)["paused"] is True
    until(client, rid, "running")
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")
    wait_for(lambda: engine.is_loaded())


def test_labeling_while_paused_names_the_pause(env, tmp_path):
    client, engine = env
    cid = _clip(client, tmp_path, n=4)
    (rid,) = start(client, SAM3_GPU).json()["run_ids"]
    r = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK, "object_id": None})
    assert r.status_code == 503 and r.json()["detail"] == "SAM3 is paused for training"
    until(client, rid, "running")
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")
    wait_for(lambda: engine.is_loaded())


def test_manual_load_clears_the_pause_and_the_run_ending_does_not_load_again(env):
    client, engine = env
    (rid,) = start(client, SAM3_GPU).json()["run_ids"]
    r = client.post("/api/engine/load")
    assert r.status_code == 202
    wait_for(lambda: engine.is_loaded())
    assert engine_view(client)["paused"] is False and engine.load_calls == 1
    assert client.post("/api/engine/load").status_code == 202  # already loaded: a no-op
    until(client, rid, "running")
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")
    wait_for(lambda: client.app.state.vdp.train.gpu_idle(SAM3_GPU))
    time.sleep(0.2)  # give an (unwanted) idle-triggered reload the chance to happen
    assert engine.load_calls == 1


def test_manual_load_is_a_noop_while_the_engine_is_already_loading(tmp_path):
    gate = threading.Event()

    class SlowLoad(FakeEngine):
        def load(self):
            gate.wait(5)
            super().load()

    engine = SlowLoad()
    engine.release()  # start out unloaded, like the real engine
    with TestClient(create_app(_settings(tmp_path), engine=engine, load_engine=True)) as client:
        assert client.post("/api/engine/load").status_code == 202  # the startup load is still running
        gate.set()
        wait_for(lambda: engine.is_loaded())
        assert engine.load_calls == 1
