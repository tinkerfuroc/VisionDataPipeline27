import base64
import io
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect
from test_api_core import _frames_dir, _project, _settings, wait_job
from vdp.api.app import create_app
from vdp.engine.fake import FakeEngine
from vdp.store.project import encode_mask_png

CLICK = {"type": "clicks", "points": [[30, 30, 1]]}


class SlowEngine(FakeEngine):
    def propagate(self, frames, seeds, cancel_cb=None):
        for item in super().propagate(frames, seeds, cancel_cb):
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

        cand = client.post(f"/api/clips/{cid}/prompt", json={"frame": 5, "prompt": CLICK, "object_id": None}).json()["candidates"]
        assert len(cand) == 1 and cand[0]["area"] == 400 and _mask(cand[0]["mask_png_b64"])[30, 30]
        assert client.get(f"/api/clips/{cid}").json()["states"][5] == "empty"  # preview persisted nothing

        first = client.post(f"/api/clips/{cid}/accept", json={"frame": 5, "prompt": CLICK, "object_id": None, "class_id": 0})
        assert first.status_code == 200 and first.json()["states"][5] == "keyframe"
        assert client.get(f"/api/clips/{cid}/objects/1/masks/5").headers["content-type"] == "image/png"

        job = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 5, "direction": "both"})
        assert job.status_code == 202
        snap = wait_job(client, job.json()["job_id"])
        assert snap["status"] == "done" and snap["result"]["processed"] == 19 and snap["result"]["lost_at"] is None
        assert snap["result"]["lost_by_direction"] == {}
        states = client.get(f"/api/clips/{cid}").json()["states"]
        assert states.count("masked") == 19 and states[5] == "keyframe"

        fixed = client.post(
            f"/api/clips/{cid}/accept",
            json={"frame": 12, "prompt": {"type": "clicks", "points": [[44, 30, 1], [40, 22, 0]]}, "object_id": 1},
        ).json()
        kf = next(k for k in fixed["clip"]["objects"][0]["keyframes"] if k["frame_idx"] == 12)
        assert "box_from_mask" in kf["prompt"]["payload"]  # a correction, not a fresh prompt
        again = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 12, "direction": "forward"})
        assert wait_job(client, again.json()["job_id"])["result"]["processed"] == 7  # 13..19

        assert client.post(f"/api/clips/{cid}/frames/3/delete").json()["states"][3] == "deleted"
        assert client.post(f"/api/clips/{cid}/frames/3/restore").json()["states"][3] == "masked"
        assert client.post(f"/api/clips/{cid}/prune", json={"from": 18}).json()["states"][18:] == ["deleted"] * 2
        cleared = client.post(f"/api/clips/{cid}/clear-range", json={"start": 10, "end": 14}).json()
        assert cleared["states"][10:15] == ["empty"] * 5  # the keyframe at 12 goes too
        assert [k["frame_idx"] for k in cleared["clip"]["objects"][0]["keyframes"]] == [5]


def test_validation_errors_are_400_not_failed_jobs(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=6)
        assert client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).status_code == 400
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        nk = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 3, "direction": "forward"})
        assert nk.status_code == 400 and "frame 3 has no mask" in nk.json()["detail"]
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK, "object_id": None, "class_id": 9}).status_code == 400
        tiny = {"type": "box", "box": [1, 1, 4, 4]}
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": tiny, "object_id": None}).json() == {"candidates": []}
        no = client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": tiny, "object_id": 1})
        assert no.status_code == 400 and "no usable mask" in no.json()["detail"]
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": {"type": "clicks"}, "object_id": None}).status_code == 422
        assert client.post(f"/api/clips/{cid}/frames/99/delete").status_code == 400
        assert client.post("/api/clips/nope/prompt", json={"frame": 0, "prompt": CLICK, "object_id": None}).status_code == 404


def test_gpu_is_exclusive_and_jobs_can_be_cancelled_and_streamed(tmp_path):
    with _client(tmp_path, SlowEngine()) as client:
        cid = _clip(client, tmp_path, n=40)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        job_id = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]

        busy = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"})
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK, "object_id": None}).status_code == 409
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK, "object_id": 1}).status_code == 409
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
        ok = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK, "object_id": None})
        assert ok.status_code == 200  # GPU is free again


def test_an_in_flight_prompt_blocks_propagate_atomically(tmp_path):
    """The GPU exclusivity check must be atomic with the engine call: a propagate that arrives while a
    prompt/accept/clear-range call is mid-flight must not slip in ahead of it (or behind it, silently)."""

    class GatedEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.release.set()  # open by default so the setup accept below does not block

        def predict_prompt(self, *args, **kwargs):
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("gate was never released")
            return super().predict_prompt(*args, **kwargs)

    eng = GatedEngine()
    with _client(tmp_path, eng) as client:
        cid = _clip(client, tmp_path)
        setup = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        assert setup.status_code == 200

        eng.entered.clear()
        eng.release.clear()  # close the gate
        results: dict[str, object] = {}

        def _prompt(key):
            results[key] = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK, "object_id": None})

        t1 = threading.Thread(target=_prompt, args=("a",))
        t1.start()
        assert eng.entered.wait(5)  # t1 is inside predict_prompt, holding the GPU lease

        try:
            blocked = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"})
            assert blocked.status_code == 409 and blocked.json()["job_id"] == "interactive"

            t2 = threading.Thread(target=_prompt, args=("b",))
            t2.start()  # a second interactive call queues on the gate; it must not bounce with 409
        finally:
            eng.release.set()  # a failing assertion above must not leave the helper thread(s) blocked
        t1.join(5)
        t2.join(5)
        assert not t1.is_alive() and not t2.is_alive(), "helper thread(s) did not finish -- TestClient may not " \
            "support concurrent requests from threads in this Starlette version"
        assert results["a"].status_code == 200 and results["b"].status_code == 200

        again = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"})
        assert again.status_code == 202  # the GPU is free again once every lease has been released
        assert wait_job(client, again.json()["job_id"])["status"] == "done"


def test_mask_response_is_not_cached_but_frames_are(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})

        mask_resp = client.get(f"/api/clips/{cid}/objects/1/masks/0")
        assert mask_resp.status_code == 200 and mask_resp.headers.get("cache-control") == "no-cache"

        frame_resp = client.get(f"/api/clips/{cid}/frames/0")
        assert frame_resp.status_code == 200 and "cache-control" not in frame_resp.headers


def test_prompt_normalises_inverted_box_and_rejects_out_of_frame_point(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        inverted = {"type": "box", "box": [80, 30, 40, 10]}
        ok = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": inverted, "object_id": None})
        assert ok.status_code == 200 and len(ok.json()["candidates"]) == 1

        bad_point = {"type": "clicks", "points": [[-1, 5, 1]]}
        bad = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": bad_point, "object_id": None})
        assert bad.status_code == 400 and isinstance(bad.json()["detail"], str)


def test_delete_clip_is_blocked_while_a_gpu_job_runs(tmp_path):
    with _client(tmp_path, SlowEngine()) as client:
        cid = _clip(client, tmp_path, n=40)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        job_id = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]

        busy = client.delete(f"/api/clips/{cid}")
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id
        assert client.get(f"/api/clips/{cid}").status_code == 200

        wait_job(client, job_id)
        ok = client.delete(f"/api/clips/{cid}")
        assert ok.status_code == 204
        assert client.get(f"/api/clips/{cid}").status_code == 404


def test_engine_not_loaded_is_503_but_browsing_still_works(tmp_path):
    class NotReady(FakeEngine):
        load_error = "no CUDA device with compute capability >= 8.0"

        def is_loaded(self):
            return False

    with _client(tmp_path, NotReady()) as client:
        cid = _clip(client, tmp_path, n=4)  # import needs no engine
        r = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK, "object_id": None})
        assert r.status_code == 503 and "compute capability" in r.json()["detail"]
        assert client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).status_code == 503
        assert client.get(f"/api/clips/{cid}").status_code == 200
        assert client.post(f"/api/clips/{cid}/frames/1/delete").status_code == 200


@pytest.mark.parametrize("body", [{"from_frame": 0, "direction": "sideways"}, {"direction": "forward"}])
def test_propagate_body_validation(tmp_path, body):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=4)
        assert client.post(f"/api/clips/{cid}/propagate", json=body).status_code == 422


def test_paint_route_writes_a_keyframe_and_is_refused_while_a_propagation_runs(tmp_path):
    with _client(tmp_path, SlowEngine()) as client:
        cid = _clip(client, tmp_path, n=40)
        info = client.get(f"/api/clips/{cid}").json()["clip"]
        m = np.zeros((info["height"], info["width"]), dtype=bool)
        m[5:15, 5:25] = True
        b64 = base64.b64encode(encode_mask_png(m)).decode("ascii")
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 2, "mask_png_b64": b64, "object_id": None}).status_code == 400
        r = client.post(f"/api/clips/{cid}/mask", json={"frame": 2, "mask_png_b64": b64, "object_id": None, "class_id": 0})
        assert r.status_code == 200 and r.json()["states"][2] == "keyframe"
        assert r.json()["clip"]["objects"][0]["keyframes"][0]["prompt"] == {"type": "paint", "payload": {}}
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 2, "mask_png_b64": "nope", "object_id": 1}).status_code == 400
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 99, "mask_png_b64": b64, "object_id": 1}).status_code == 400
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": -1, "mask_png_b64": b64, "object_id": 1}).status_code == 422
        job_id = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 2, "direction": "forward"}).json()["job_id"]
        busy = client.post(f"/api/clips/{cid}/mask", json={"frame": 3, "mask_png_b64": b64, "object_id": 1})
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id  # it replaces what the runner writes
        wait_job(client, job_id)


def test_objects_over_http(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        r = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        assert r.status_code == 200 and r.json()["object_ids"] == [1]
        r = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": {"type": "clicks", "points": [[90, 30, 1]]}, "object_id": None, "class_id": 0})
        assert r.json()["object_ids"] == [2]
        detail = r.json()
        assert set(detail["object_states"]) == {"1", "2"} and detail["object_states"]["2"][0] == "keyframe"
        assert client.get(f"/api/clips/{cid}/objects/2/masks/0").headers["content-type"] == "image/png"
        assert client.get(f"/api/clips/{cid}/objects/9/masks/0").status_code == 404
        r = client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK, "object_id": 1, "class_id": 0})
        assert r.status_code == 400
        assert client.patch(f"/api/clips/{cid}/objects/2", json={"class_id": 0}).status_code == 200
        r = client.delete(f"/api/clips/{cid}/objects/2")
        assert r.status_code == 200 and [o["id"] for o in r.json()["clip"]["objects"]] == [1]


def test_prompt_preview_uses_the_given_object(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        # object 1: an ordinary accept at (30, 30)
        r1 = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        assert r1.status_code == 200 and r1.json()["object_ids"] == [1]

        # object 2: a hand-painted mask spanning x 70..100
        info = client.get(f"/api/clips/{cid}").json()["clip"]
        m = np.zeros((info["height"], info["width"]), dtype=bool)
        m[20:40, 70:100] = True
        b64 = base64.b64encode(encode_mask_png(m)).decode("ascii")
        r2 = client.post(f"/api/clips/{cid}/mask", json={"frame": 0, "mask_png_b64": b64, "object_id": None, "class_id": 0})
        assert r2.status_code == 200 and r2.json()["object_id"] == 2

        click_80 = {"type": "clicks", "points": [[80, 30, 1]]}
        with_obj = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": click_80, "object_id": 2})
        assert with_obj.status_code == 200
        assert with_obj.json()["candidates"][0]["bbox"][2] > 95  # derived from object 2's existing mask

        without_obj = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": click_80, "object_id": None})
        assert without_obj.status_code == 200
        assert without_obj.json()["candidates"][0]["bbox"][2] <= 95  # a fresh click prediction, no object to derive from


def test_prompt_with_an_unknown_object_id_is_404(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        r = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK, "object_id": 5})
        assert r.status_code == 404


def test_propagate_objects_body_and_events(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=10)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        client.post(
            f"/api/clips/{cid}/accept",
            json={"frame": 0, "prompt": {"type": "clicks", "points": [[90, 30, 1]]}, "object_id": None, "class_id": 0},
        )

        empty = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward", "objects": []})
        assert empty.status_code == 422

        r = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward", "objects": [2]})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        job = wait_job(client, job_id)
        assert job["status"] == "done" and job["result"]["objects"] == [2]

        states = client.get(f"/api/clips/{cid}").json()["object_states"]
        assert "masked" in states["2"]  # only object 2 was propagated
        assert set(states["1"]) <= {"empty", "keyframe"}

        unknown = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward", "objects": [9]})
        assert unknown.status_code == 404

        no_mask = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 5, "direction": "forward", "objects": [1]})
        assert no_mask.status_code == 400

        events = []
        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            try:
                while True:
                    events.append(ws.receive_json())
            except WebSocketDisconnect:
                pass
        frame_events = [e for e in events if e["type"] == "frame"]
        assert frame_events
        assert all("objects" in e and "has_mask" not in e for e in frame_events)


def test_delete_object_is_refused_while_a_propagation_or_an_export_runs(tmp_path, monkeypatch):
    with _client(tmp_path, SlowEngine()) as client:
        cid = _clip(client, tmp_path, n=40)
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0}).status_code == 200
        job_id = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]

        busy = client.delete(f"/api/clips/{cid}/objects/1")
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id

        patched = client.patch(f"/api/clips/{cid}/objects/1", json={"class_id": 0})
        assert patched.status_code == 200  # PATCH takes the clip lock only, not the GPU lease

        wait_job(client, job_id)

        project_id = client.get(f"/api/clips/{cid}").json()["clip"]["project_id"]
        entered, release = threading.Event(), threading.Event()
        from vdp.export.coco import Coco

        real = Coco.write

        def gated(self, plan, out_dir, tick):
            entered.set()
            assert release.wait(10)
            return real(self, plan, out_dir, tick)

        monkeypatch.setattr(Coco, "write", gated)
        export_job = client.post(f"/api/projects/{project_id}/export", json={"clips": [cid], "formats": ["coco"]}).json()["job_id"]
        assert entered.wait(10)

        refused = client.delete(f"/api/clips/{cid}/objects/1")
        assert refused.status_code == 409 and refused.json()["job_id"] == export_job

        release.set()
        assert wait_job(client, export_job)["status"] == "done"

        ok = client.delete(f"/api/clips/{cid}/objects/1")
        assert ok.status_code == 200 and ok.json()["clip"]["objects"] == []


def test_clear_range_by_object(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=10)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 3, "prompt": CLICK, "object_id": None, "class_id": 0})
        client.post(
            f"/api/clips/{cid}/accept",
            json={"frame": 3, "prompt": {"type": "clicks", "points": [[90, 30, 1]]}, "object_id": None, "class_id": 0},
        )

        r = client.post(f"/api/clips/{cid}/clear-range", json={"start": 0, "end": 5, "objects": [2]})
        assert r.status_code == 200
        object_states = r.json()["object_states"]
        assert object_states["1"][3] == "keyframe" and object_states["2"][3] == "empty"

        empty = client.post(f"/api/clips/{cid}/clear-range", json={"start": 0, "end": 5, "objects": []})
        assert empty.status_code == 422


def test_bodies_must_name_the_object(tmp_path):
    """object_id is required (null = a new object): a body without it is a 422 that writes nothing, and the old
    first-object mask route is gone."""
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        ok = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        assert ok.status_code == 200
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK}).status_code == 422
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 3, "prompt": CLICK}).status_code == 422
        info = client.get(f"/api/clips/{cid}").json()["clip"]
        m = np.zeros((info["height"], info["width"]), dtype=bool)
        m[5:15, 5:25] = True
        b64 = base64.b64encode(encode_mask_png(m)).decode("ascii")
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 3, "mask_png_b64": b64}).status_code == 422
        assert client.get(f"/api/clips/{cid}").json()["states"][3] == "empty"  # nothing was written
        assert client.get(f"/api/clips/{cid}/masks/0").status_code == 404
        assert client.get(f"/api/clips/{cid}/objects/1/masks/0").status_code == 200


def test_new_object_accept_with_an_unknown_class_writes_nothing(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        before = client.get(f"/api/clips/{cid}").json()
        bad = client.post(f"/api/clips/{cid}/accept", json={"frame": 3, "prompt": CLICK, "object_id": None, "class_id": 9})
        assert bad.status_code == 400
        after = client.get(f"/api/clips/{cid}").json()
        assert after["states"] == before["states"] and after["clip"]["objects"] == before["clip"]["objects"]


def test_frame_range_delete_and_restore_routes(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=10)
        r = client.post(f"/api/clips/{cid}/frames/delete-range", json={"start": 2, "end": 5})
        assert r.status_code == 200 and r.json()["clip"]["deleted_frames"] == [2, 3, 4, 5]
        assert r.json()["states"][3] == "deleted"
        r = client.post(f"/api/clips/{cid}/frames/restore-range", json={"start": 4, "end": 9})
        assert r.json()["clip"]["deleted_frames"] == [2, 3]
        assert client.post(f"/api/clips/{cid}/frames/delete-range", json={"start": 5, "end": 2}).status_code == 400
        assert client.post(f"/api/clips/{cid}/frames/delete-range", json={"start": 0, "end": 10}).status_code == 400
        assert client.post("/api/clips/nope/frames/delete-range", json={"start": 0, "end": 1}).status_code == 404
