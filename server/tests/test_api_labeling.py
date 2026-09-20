import base64
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect
from test_api_core import _frames_dir, _project, _settings, wait_job
from vsp.api.app import create_app
from vsp.engine.fake import FakeEngine

CLICK = {"type": "clicks", "points": [[30, 30, 1]]}


class SlowEngine(FakeEngine):
    def propagate(self, frames, seed_mask, cancel_cb=None):
        for item in super().propagate(frames, seed_mask, cancel_cb):
            time.sleep(0.03)
            yield item


def _client(tmp_path, engine):
    return TestClient(create_app(_settings(tmp_path), engine=engine, load_engine=False))


def _clip(client, tmp_path, n=20):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(_frames_dir(tmp_path, n))})
    return wait_job(client, r.json()["job_id"])["result"]["clip_id"]


def _mask(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")) > 127


def test_full_labeling_flow(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)

        cand = client.post(f"/api/clips/{cid}/prompt", json={"frame": 5, "prompt": CLICK}).json()["candidates"]
        assert len(cand) == 1 and cand[0]["area"] == 400 and _mask(cand[0]["mask_png_b64"])[30, 30]
        assert client.get(f"/api/clips/{cid}").json()["states"][5] == "empty"  # preview persisted nothing

        first = client.post(f"/api/clips/{cid}/accept", json={"frame": 5, "prompt": CLICK, "class_id": 0})
        assert first.status_code == 200 and first.json()["states"][5] == "keyframe"
        assert client.get(f"/api/clips/{cid}/masks/5").headers["content-type"] == "image/png"

        job = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 5, "direction": "both"})
        assert job.status_code == 202
        snap = wait_job(client, job.json()["job_id"])
        assert snap["status"] == "done" and snap["result"]["processed"] == 19 and snap["result"]["lost_at"] is None
        assert snap["result"]["lost_by_direction"] == {}
        states = client.get(f"/api/clips/{cid}").json()["states"]
        assert states.count("masked") == 19 and states[5] == "keyframe"

        fixed = client.post(
            f"/api/clips/{cid}/accept",
            json={"frame": 12, "prompt": {"type": "clicks", "points": [[44, 30, 1], [40, 22, 0]]}},
        ).json()
        kf = next(k for k in fixed["clip"]["objects"][0]["keyframes"] if k["frame_idx"] == 12)
        assert "box_from_mask" in kf["prompt"]["payload"]  # a correction, not a fresh prompt
        again = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 12, "direction": "forward"})
        assert wait_job(client, again.json()["job_id"])["result"]["processed"] == 7  # 13..19

        assert client.post(f"/api/clips/{cid}/frames/3/delete").json()["states"][3] == "deleted"
        assert client.post(f"/api/clips/{cid}/frames/3/restore").json()["states"][3] == "masked"
        assert client.post(f"/api/clips/{cid}/prune", json={"from": 18}).json()["states"][18:] == ["deleted"] * 2
        cleared = client.post(f"/api/clips/{cid}/clear-range", json={"start": 10, "end": 14}).json()
        assert cleared["states"][10:15] == ["empty", "empty", "keyframe", "empty", "empty"]


def test_validation_errors_are_400_not_failed_jobs(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=6)
        assert client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).status_code == 400
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "class_id": 0})
        nk = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 3, "direction": "forward"})
        assert nk.status_code == 400 and "not a keyframe" in nk.json()["detail"]
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK, "class_id": 9}).status_code == 400
        tiny = {"type": "box", "box": [1, 1, 4, 4]}
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": tiny}).json() == {"candidates": []}
        no = client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": tiny})
        assert no.status_code == 400 and "no usable mask" in no.json()["detail"]
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": {"type": "clicks"}}).status_code == 422
        assert client.post(f"/api/clips/{cid}/frames/99/delete").status_code == 400
        assert client.post("/api/clips/nope/prompt", json={"frame": 0, "prompt": CLICK}).status_code == 404


def test_gpu_is_exclusive_and_jobs_can_be_cancelled_and_streamed(tmp_path):
    with _client(tmp_path, SlowEngine()) as client:
        cid = _clip(client, tmp_path, n=40)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "class_id": 0})
        job_id = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]

        busy = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"})
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK}).status_code == 409
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK}).status_code == 409
        assert client.post(f"/api/clips/{cid}/frames/30/delete").status_code == 200  # flagging a frame is safe mid-run
        blocked = client.post(f"/api/clips/{cid}/clear-range", json={"start": 1, "end": 3})
        assert blocked.status_code == 409 and blocked.json()["job_id"] == job_id  # it mutates what the runner writes

        events = []
        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            try:
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["type"] == "frame" and event["frame"] == 5:
                        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
            except WebSocketDisconnect:
                pass
        assert [e["seq"] for e in events] == list(range(len(events)))
        assert events[0]["type"] == "progress" and events[-1]["type"] == "cancelled"
        snap = wait_job(client, job_id)
        assert snap["status"] == "cancelled" and 5 <= snap["result"]["processed"] < 39
        landed = client.get(f"/api/clips/{cid}").json()["states"].count("masked")
        assert landed == snap["result"]["written"]  # cancel keeps what landed
        ok = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK})
        assert ok.status_code == 200  # GPU is free again


def test_engine_not_loaded_is_503_but_browsing_still_works(tmp_path):
    class NotReady(FakeEngine):
        load_error = "no CUDA device with compute capability >= 8.0"

        def is_loaded(self):
            return False

    with _client(tmp_path, NotReady()) as client:
        cid = _clip(client, tmp_path, n=4)  # import needs no engine
        r = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK})
        assert r.status_code == 503 and "compute capability" in r.json()["detail"]
        assert client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).status_code == 503
        assert client.get(f"/api/clips/{cid}").status_code == 200
        assert client.post(f"/api/clips/{cid}/frames/1/delete").status_code == 200


@pytest.mark.parametrize("body", [{"from_frame": 0, "direction": "sideways"}, {"direction": "forward"}])
def test_propagate_body_validation(tmp_path, body):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=4)
        assert client.post(f"/api/clips/{cid}/propagate", json=body).status_code == 422
