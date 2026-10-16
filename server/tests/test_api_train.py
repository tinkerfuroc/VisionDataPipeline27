from __future__ import annotations

import json
import sys

import pytest
from fastapi.testclient import TestClient
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export, wait_for
from vdp.api.app import create_app
from vdp.config import Settings
from vdp.engine.fake import FakeEngine
from vdp.train import scheduler as scheduler_mod
from vdp.train.scheduler import Gpu, TrainScheduler


class _FakeNvml:
    """{uuid: {"free": bytes, "clock_mhz": int}}-driven stand-in for `pynvml`."""

    NVML_CLOCK_SM = 1

    def __init__(self, devices: dict[str, dict]) -> None:
        self._order = list(devices)
        self._devices = devices

    def nvmlInit(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return len(self._order)

    def nvmlDeviceGetHandleByIndex(self, i: int):
        return self._order[i]

    def nvmlDeviceGetUUID(self, handle) -> str:
        return f"GPU-{handle}"

    def nvmlDeviceGetMemoryInfo(self, handle):
        class _Mem:
            free = self._devices[handle]["free"]
        return _Mem()

    def nvmlDeviceGetMaxClockInfo(self, handle, clock_type):
        return self._devices[handle]["clock_mhz"]


@pytest.fixture(autouse=True)
def _no_real_nvml(monkeypatch):
    """No fake `pynvml` installed here: with an empty NVML (no devices at all), every lookup falls
    back deterministically no matter what real NVIDIA hardware this happens to run on."""
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml({}))
    monkeypatch.setattr(scheduler_mod, "_nvml_ready", False)


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
    # "Fast" (12 GiB) and "Slow" (11 GiB) both have the fixture's default (0, 0) capability / 0 SMs,
    # so they tie on estimated throughput; VRAM breaks the tie and "Fast" (already index 0) stays first.
    assert [g["index"] for g in st["gpus"]] == [0, 1] and st["gpus"][0]["queue"] == []
    assert st["default_gpu"] == 0
    g0 = st["gpus"][0]
    assert g0["capability"] == "0.0" and g0["sm_count"] == 0 and g0["memory_free"] is None
    assert isinstance(g0["tflops"], (int, float))
    assert "yolo11m-seg" in st["models"] and st["default_models"][0] == "yolo26s-seg"
    assert st["defaults"]["imgsz"] == 960
    assert st["aug_presets"]["strong"]["degrees"] == 15
    assert st["defaults"]["augment"] == "default"
    exports = client.get("/api/exports").json()
    assert exports == [{"id": EXPORT_ID, "project_name": "Bench", "created_at": "2026-01-01T00:00:00+00:00",
                        "classes": ["Module"], "train_images": 2, "val_images": 2,
                        "formats": ["yolo_seg"], "derived_from": None, "name": None,
                        "augment_limits": None}]


def test_status_orders_gpus_best_first_regardless_of_index(tmp_path):
    """The fastest GPU by estimated throughput is listed first even when it does not sit at the
    lowest index (mirrors the real machine: cuda:0 is the slower card there)."""
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    slower_at_0 = Gpu(0, "RTX 2080 Ti", 11 * 2**30, capability=(7, 5), sm_count=68)
    faster_at_1 = Gpu(1, "RTX 3080 Ti", 12 * 2**30, capability=(8, 6), sm_count=80)
    sched = TrainScheduler(s.project.data_root, [slower_at_0, faster_at_1], worker_cmd=FAKE_WORKER)
    app = create_app(s, engine=FakeEngine(), load_engine=False, train=sched)
    with TestClient(app) as c:
        st = c.get("/api/train/status").json()
    assert [g["index"] for g in st["gpus"]] == [1, 0]
    assert st["default_gpu"] == 1
    assert st["gpus"][0]["capability"] == "8.6" and st["gpus"][0]["sm_count"] == 80
    assert st["gpus"][0]["tflops"] > st["gpus"][1]["tflops"]


def test_default_gpu_skips_a_gpu_with_a_run_on_it(client):
    rid = start(client, epochs=99, gpu=0)  # occupies cuda:0, the tie-broken default here
    until(client, rid, "running")
    st = client.get("/api/train/status").json()
    assert st["default_gpu"] == 1
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")


def test_status_reports_free_vram_via_nvml(tmp_path, monkeypatch):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    gpus = [Gpu(0, "A", 12 * 2**30, uuid="uuid-a"), Gpu(1, "B", 11 * 2**30, uuid="uuid-b")]
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml({
        "uuid-b": {"free": 9 * 2**30, "clock_mhz": 1545}, "uuid-a": {"free": 5 * 2**30, "clock_mhz": 1665},
    }))
    monkeypatch.setattr(scheduler_mod, "_nvml_ready", False)
    sched = TrainScheduler(s.project.data_root, gpus, worker_cmd=FAKE_WORKER)
    app = create_app(s, engine=FakeEngine(), load_engine=False, train=sched)
    with TestClient(app) as c:
        st = c.get("/api/train/status").json()
    assert {g["index"]: g["memory_free"] for g in st["gpus"]} == {0: 5 * 2**30, 1: 9 * 2**30}


def test_rename_export(client):
    renamed = client.patch(f"/api/exports/{EXPORT_ID}", json={"name": "Better"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Better"
    exports = client.get("/api/exports").json()
    assert exports[0]["name"] == "Better"

    assert client.patch("/api/exports/nope", json={"name": "x"}).status_code == 404
    assert client.patch(f"/api/exports/{EXPORT_ID}", json={"name": "a" * 81}).status_code == 400

    assert client.patch(f"/api/exports/{EXPORT_ID}", json={}).status_code == 422

    cleared = client.patch(f"/api/exports/{EXPORT_ID}", json={"name": None})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["name"] is None


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
    # best.onnx appears before the export job has finished; deleting the run needs the job done too
    wait_for(lambda: (v := client.get(f"/api/train/runs/{rid}").json())["has_onnx"]
             and v["onnx"] not in ("queued", "running"))
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


def test_resume_on_a_chosen_gpu(client, tmp_path):
    def last_pt():
        return (tmp_path / "data" / "runs" / rid / "train" / "weights" / "last.pt").is_file()

    rid = start(client, epochs=99, gpu=1)
    until(client, rid, "running")
    wait_for(last_pt)
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")

    resumed = client.post(f"/api/train/runs/{rid}/resume", json={"gpu": 0})
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["gpu"] == 0
    running = until(client, rid, "running")
    assert running["gpu"] == 0
    gpu0 = next(g for g in client.get("/api/train/status").json()["gpus"] if g["index"] == 0)
    assert gpu0["running"] == rid
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")

    assert client.post(f"/api/train/runs/{rid}/resume", json={"gpu": 5}).status_code == 400
    assert client.post(f"/api/train/runs/{rid}/resume", json={}).json()["gpu"] == 0  # keeps its current GPU
    until(client, rid, "running")
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")


def test_complete_on_a_chosen_gpu(client, tmp_path):
    def weights_dir():
        return tmp_path / "data" / "runs" / rid / "train" / "weights"

    def last_pt():
        return (weights_dir() / "last.pt").is_file()

    rid = start(client, epochs=99, gpu=1)
    until(client, rid, "running")
    wait_for(last_pt)
    client.post(f"/api/train/runs/{rid}/cancel")
    cancelled = until(client, rid, "cancelled")
    assert cancelled["resumable"] is True and cancelled["completable"] is False  # no best.pt yet

    # As if the run had already gotten past its first validation epoch before being cancelled.
    (weights_dir() / "best.pt").write_bytes(b"best")
    assert client.get(f"/api/train/runs/{rid}").json()["completable"] is True

    assert client.post(f"/api/train/runs/{rid}/complete", json={"gpu": 7}).status_code == 400
    run = client.get(f"/api/train/runs/{rid}").json()
    assert run["status"] == "cancelled" and run["gpu"] == 1  # unchanged by the rejected request

    completed = client.post(f"/api/train/runs/{rid}/complete", json={"gpu": 0})
    assert completed.status_code == 200, completed.text
    assert completed.json()["gpu"] == 0
    done = until(client, rid, "done")
    assert done["completed_early"] is True and done["has_best"] is True and done["resumable"] is False

    assert client.post(f"/api/train/runs/{rid}/complete").status_code == 409  # already done


def test_complete_conflicts(client):
    rid = start(client, epochs=99, gpu=1)
    until(client, rid, "running")
    assert client.post(f"/api/train/runs/{rid}/complete").status_code == 409  # still running


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
    import vdp.api.train as train_api

    monkeypatch.setattr(train_api, "find_spec", lambda name: None)
    st = client.get("/api/train/status").json()
    assert st["available"] is False and "uv sync" in st["error"]
    body = {"export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 0}]}
    assert client.post("/api/train/runs", json=body).status_code == 503


def test_run_view_carries_classes_and_start_mode_fields(client):
    rid = start(client, epochs=2)
    view = until(client, rid, "done")
    assert view["classes"] == ["Module"]
    assert view["source_run"] is None and view["start_mode"] is None and view["continued_from_epoch"] is None
    assert view["continuable"] == view["resumable"] and view["finetunable"] == view["has_best"]
    assert view["finetunable"] is True


def test_from_checkpoint_continue_happy_path(client, tmp_path):
    rid = start(client, epochs=99, gpu=1)
    until(client, rid, "running")
    wait_for(lambda: (tmp_path / "data" / "runs" / rid / "train" / "weights" / "last.pt").is_file())
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")

    make_export(tmp_path / "data", "20260201-000000-newexp")
    r = client.post(f"/api/train/runs/{rid}/from-checkpoint",
                    json={"export_id": "20260201-000000-newexp", "mode": "continue", "gpu": 0})
    assert r.status_code == 201, r.text
    new_id = r.json()["run_id"]
    assert new_id != rid
    new = client.get(f"/api/train/runs/{new_id}").json()
    assert new["source_run"] == rid and new["start_mode"] == "continue"
    assert new["export_id"] == "20260201-000000-newexp"
    until(client, new_id, "running")
    client.post(f"/api/train/runs/{new_id}/cancel")
    until(client, new_id, "cancelled")


def test_from_checkpoint_finetune_happy_path(client, tmp_path):
    rid = start(client, epochs=2, gpu=1)
    until(client, rid, "done")

    make_export(tmp_path / "data", "20260201-000000-newexp")
    r = client.post(f"/api/train/runs/{rid}/from-checkpoint", json={
        "export_id": "20260201-000000-newexp", "mode": "finetune", "gpu": 0,
        "params": {"epochs": 5, "imgsz": 640},
    })
    assert r.status_code == 201, r.text
    new_id = r.json()["run_id"]
    new = client.get(f"/api/train/runs/{new_id}").json()
    assert new["source_run"] == rid and new["start_mode"] == "finetune"
    assert new["params"]["epochs"] == 5
    until(client, new_id, "done")


def test_from_checkpoint_errors(client, tmp_path):
    rid = start(client, epochs=2, gpu=1)
    until(client, rid, "done")
    make_export(tmp_path / "data", "20260201-000000-widget", classes=("Widget",))

    # unknown run / export
    assert client.post("/api/train/runs/nope/from-checkpoint",
                       json={"export_id": EXPORT_ID, "mode": "finetune", "gpu": 0}).status_code == 404
    assert client.post(f"/api/train/runs/{rid}/from-checkpoint",
                       json={"export_id": "missing", "mode": "finetune", "gpu": 0}).status_code == 404
    # class mismatch
    mismatch = client.post(f"/api/train/runs/{rid}/from-checkpoint",
                           json={"export_id": "20260201-000000-widget", "mode": "finetune", "gpu": 0})
    assert mismatch.status_code == 409 and "Widget" in mismatch.text and "Module" in mismatch.text
    # ineligible: a done run cannot be "continued"
    assert client.post(f"/api/train/runs/{rid}/from-checkpoint",
                       json={"export_id": EXPORT_ID, "mode": "continue", "gpu": 0}).status_code == 409
    # bad gpu
    assert client.post(f"/api/train/runs/{rid}/from-checkpoint",
                       json={"export_id": EXPORT_ID, "mode": "finetune", "gpu": 7}).status_code == 400
    # bad mode (422: not one of the Literal's two values)
    assert client.post(f"/api/train/runs/{rid}/from-checkpoint",
                       json={"export_id": EXPORT_ID, "mode": "nonsense", "gpu": 0}).status_code == 422


def test_predictions_filters_and_counts(client):
    rid = start(client)
    until(client, rid, "done")
    page = client.get(f"/api/train/runs/{rid}/predictions?filter=all").json()
    assert page["counts"]["all"] == 1 and page["counts"]["missed"] == 0 and page["counts"]["worst"] == 1
    item = page["items"][0]
    assert item["index"] == 0 and 0.0 < item["iou"] <= 1.0
    assert client.get(f"/api/train/runs/{rid}/predictions?filter=missed").json()["total"] == 0
    assert client.get(f"/api/train/runs/{rid}/predictions?filter=nope").status_code == 422


def test_sampled_epoch_params_and_the_runs_measured_speed(client, tmp_path):
    params = {"epochs": 2, "imgsz": 640, "batch": 1, "train_images_per_epoch": 1, "val_images_per_epoch": 1}
    r = client.post("/api/train/runs", json={"export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 0}],
                                             "params": params})
    assert r.status_code == 201, r.text
    rid = r.json()["run_ids"][0]
    view = until(client, rid, "done")
    assert view["params"]["train_images_per_epoch"] == 1 and view["params"]["val_images_per_epoch"] == 1
    assert view["sampling"] == {"train_images": 2, "train_images_per_epoch": 1, "val_images": 2,
                                "val_images_per_epoch": 1, "warmup_epochs": 1.0, "close_mosaic": 2,
                                "val_images_chosen": None}  # the fake worker links no subset
    subset = tmp_path / "data" / "runs" / rid / "val_subset" / "images" / "val"
    subset.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):  # one more than asked, to cover every class
        (subset / name).write_bytes(b"x")
    assert client.get(f"/api/train/runs/{rid}").json()["sampling"]["val_images_chosen"] == 2
    assert view["speed"] is None
    speed = {"train_images_per_s": 65.0, "val_images_per_s": 80.0, "batch": 1, "imgsz": 640}
    (tmp_path / "data" / "runs" / rid / "speed.json").write_text(json.dumps(speed))
    assert client.get("/api/train/runs").json()[0]["speed"] == speed
    assert client.get("/api/train/status").json()["defaults"]["train_images_per_epoch"] is None
    bad = client.post("/api/train/runs", json={"export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 0}],
                                               "params": {**params, "batch": 4, "train_images_per_epoch": 2}})
    assert bad.status_code == 422


def test_a_detection_run_reports_its_task_and_box_ground_truth(client, tmp_path):
    make_export(tmp_path / "data", "20260201-000000-boxes", det=True)
    r = client.post("/api/train/runs", json={"export_id": "20260201-000000-boxes",
                                             "runs": [{"model": "yolo26n", "gpu": 1}], "params": {"epochs": 2}})
    rid = r.json()["run_ids"][0]
    view = until(client, rid, "done")
    assert (view["task"], view["data_format"]) == ("detect", "yolo_det")
    item = client.get(f"/api/train/runs/{rid}/predictions").json()["items"][0]
    assert item["ground_truth"] == [{"class_id": 0, "box": [0.25, 0.25, 0.75, 0.75],
                                     "polygons": [[0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75]]}]
    seg = until(client, start(client), "done")
    assert (seg["task"], seg["data_format"]) == ("segment", "yolo_seg")
    assert "yolo26n" in client.get("/api/train/status").json()["models"]


def test_from_checkpoint_refuses_to_cross_boxes_and_masks(client):
    rid = start(client, epochs=2, gpu=1)
    until(client, rid, "done")
    r = client.post(f"/api/train/runs/{rid}/from-checkpoint",
                    json={"export_id": EXPORT_ID, "mode": "finetune", "gpu": 0, "model": "yolo26s"})
    assert r.status_code == 400 and "a box model can't fine-tune a mask model's run" in r.text
    r = client.post(f"/api/train/runs/{rid}/from-checkpoint",
                    json={"export_id": EXPORT_ID, "mode": "finetune", "gpu": 0, "model": "yolo26s-seg"})
    assert r.status_code == 201, r.text
