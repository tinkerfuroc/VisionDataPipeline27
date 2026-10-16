from __future__ import annotations

import json
import sys
import textwrap
import threading
import time

import pytest
from train_fixtures import make_export
from vdp.train import vram
from vdp.train.scheduler import Gpu
from vdp.train.vram import (
    MARGIN_BYTES,
    MARGIN_FACTOR,
    PROBE_VERSION,
    ProbeFailed,
    VramEstimator,
    export_objects,
    needs_remeasure,
    objects_bucket,
    run_probe,
    with_margin,
)

G = 2**30
GPU = Gpu(1, "NVIDIA GeForce RTX 2080 Ti", 11 * G)
MEASURED = [{"batch": 1, "bytes": G}, {"batch": 2, "bytes": 2 * G}, {"batch": 4, "bytes": 4 * G},
            {"batch": 8, "bytes": None}, {"batch": 12, "bytes": None}]
LIMIT = 6 * G


def _runner(calls: list, measured=MEASURED):
    def run(model, imgsz, gpu, objects, on_batch, should_stop):
        calls.append((model, imgsz, gpu, objects))
        for e in measured:
            on_batch(e["batch"], e["bytes"])
        return {"gpu_name": GPU.name, "total_bytes": GPU.memory_total, "limit_bytes": LIMIT, "estimates": measured}
    return run


def _measure(est, model="yolo26s-seg", imgsz=960, objects=8, gpu=GPU, **kw):
    kw.setdefault("on_batch", lambda b, n: None)
    kw.setdefault("should_stop", lambda: False)
    return est.measure(model, imgsz, objects, gpu, **kw)


def test_with_margin_scales_and_adds_the_fixed_overhead():
    assert with_margin(None) is None
    assert with_margin(4 * G) == int(4 * G * MARGIN_FACTOR) + MARGIN_BYTES
    assert 1.0 < MARGIN_FACTOR < 1.3 and 0 < MARGIN_BYTES < G


def test_measure_caches_raw_numbers_and_reports_them_with_the_margin(tmp_path):
    calls, progress = [], []
    est = VramEstimator(tmp_path / "data", runner=_runner(calls))
    assert est.cached("yolo26s-seg", 960, 16, GPU.name) is None

    result = _measure(est, objects=16, on_batch=lambda b, n: progress.append(b))

    assert calls == [("yolo26s-seg", 960, 1, 16)] and progress == [1, 2, 4, 8, 12]
    entry = est.cached("yolo26s-seg", 960, 16, GPU.name)
    assert entry == {"version": PROBE_VERSION, "measured": MEASURED, "measured_at": result["measured_at"],
                     "limit_bytes": LIMIT}
    saved = json.loads((tmp_path / "data" / "train_weights" / "vram_estimates.json").read_text())
    assert list(saved["entries"]) == ["yolo26s-seg|960|segment|16|NVIDIA GeForce RTX 2080 Ti"]

    view = est.view(entry, "yolo26s-seg", 960, 16, GPU, free_bytes=5 * G, cached=False, objects_from_export=True)
    over = with_margin(LIMIT)
    assert view == {
        "model": "yolo26s-seg", "imgsz": 960, "gpu": 1, "gpu_name": GPU.name, "total_bytes": GPU.memory_total,
        "free_bytes": 5 * G, "measured_at": entry["measured_at"], "cached": False,
        "objects_per_image": 16, "objects_from_export": True,
        "estimates": [
            {"batch": 1, "bytes": with_margin(G), "over_bytes": None},
            {"batch": 2, "bytes": with_margin(2 * G), "over_bytes": None},
            {"batch": 4, "bytes": with_margin(4 * G), "over_bytes": None},
            {"batch": 8, "bytes": None, "over_bytes": over},
            {"batch": 12, "bytes": None, "over_bytes": over},
        ],
    }


def test_the_cache_is_per_model_imgsz_objects_and_gpu_name(tmp_path):
    est = VramEstimator(tmp_path / "data", runner=_runner([]))
    _measure(est)
    assert est.cached("yolo26s-seg", 960, 8, GPU.name) is not None
    assert est.cached("yolo26s-seg", 640, 8, GPU.name) is None
    assert est.cached("yolo26m-seg", 960, 8, GPU.name) is None
    assert est.cached("yolo26s-seg", 960, 16, GPU.name) is None
    assert est.cached("yolo26s-seg", 960, 8, "NVIDIA GeForce RTX 3080 Ti") is None
    # a second instance (a restarted server) reads the same file
    assert VramEstimator(tmp_path / "data").cached("yolo26s-seg", 960, 8, GPU.name) is not None


def test_an_unreadable_cache_file_counts_as_empty(tmp_path):
    path = tmp_path / "data" / "train_weights" / "vram_estimates.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    est = VramEstimator(tmp_path / "data", runner=_runner([]))
    assert est.cached("yolo26s-seg", 960, 8, GPU.name) is None
    _measure(est)
    assert est.cached("yolo26s-seg", 960, 8, GPU.name) is not None


@pytest.mark.parametrize("entry", [
    "not a dict",
    {"measured": MEASURED, "measured_at": "t", "limit_bytes": LIMIT},  # an older probe version
    {"version": PROBE_VERSION + 1, "measured": MEASURED, "measured_at": "t", "limit_bytes": LIMIT},
    {"version": PROBE_VERSION, "measured": [], "measured_at": "t", "limit_bytes": LIMIT},
    {"version": PROBE_VERSION, "measured": [{"batch": "1", "bytes": 5}], "measured_at": "t", "limit_bytes": LIMIT},
    {"version": PROBE_VERSION, "measured": [{"batch": 1, "bytes": "5"}], "measured_at": "t", "limit_bytes": LIMIT},
    {"version": PROBE_VERSION, "measured": [{"batch": 1}], "measured_at": "t", "limit_bytes": LIMIT},
    {"version": PROBE_VERSION, "measured": MEASURED, "measured_at": None, "limit_bytes": LIMIT},
    {"version": PROBE_VERSION, "measured": MEASURED, "measured_at": "t"},
])
def test_a_bad_or_outdated_cache_entry_is_a_miss(tmp_path, entry):
    path = tmp_path / "data" / "train_weights" / "vram_estimates.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"entries": {f"yolo26s-seg|960|segment|8|{GPU.name}": entry}}))
    assert VramEstimator(tmp_path / "data").cached("yolo26s-seg", 960, 8, GPU.name) is None


def test_a_stopped_measurement_is_not_cached(tmp_path):
    est = VramEstimator(tmp_path / "data", runner=lambda *a: None)
    assert _measure(est, should_stop=lambda: True) is None
    assert est.cached("yolo26s-seg", 960, 8, GPU.name) is None


def test_stop_probe_on_stops_the_probe_on_that_gpu_and_waits_for_it(tmp_path):
    started, exited = threading.Event(), threading.Event()

    def runner(model, imgsz, gpu, objects, on_batch, should_stop):
        started.set()
        while not should_stop():
            time.sleep(0.01)
        time.sleep(0.2)  # killing and reaping the process
        exited.set()

    est = VramEstimator(tmp_path / "data", runner=runner)
    out = []
    t = threading.Thread(target=lambda: out.append(_measure(est)))
    t.start()
    assert started.wait(5)
    est.stop_probe_on(0)  # another GPU: no effect
    assert not exited.is_set()
    est.stop_probe_on(1)
    assert exited.is_set()  # returned only once the probe was gone
    t.join(5)
    assert out == [None] and est.cached("yolo26s-seg", 960, 8, GPU.name) is None
    est.stop_probe_on(1)  # nothing running: a no-op


def test_needs_remeasure_only_when_something_did_not_fit_and_more_memory_is_free_now():
    entry = {"measured": MEASURED, "limit_bytes": LIMIT}
    headroom = vram.HEADROOM_BYTES
    assert not needs_remeasure(entry, None)
    assert not needs_remeasure(entry, LIMIT + headroom)  # the probe would get what it had
    assert not needs_remeasure(entry, LIMIT + headroom + vram.STALE_SLACK_BYTES)
    assert needs_remeasure(entry, LIMIT + headroom + vram.STALE_SLACK_BYTES + 1)
    all_fit = {"measured": MEASURED[:3], "limit_bytes": LIMIT}
    assert not needs_remeasure(all_fit, 100 * G)


def test_objects_bucket():
    assert [objects_bucket(n) for n in (0, 1, 8, 9, 12, 16, 17, 200, 299, 1000)] == [
        8, 8, 8, 16, 16, 16, 32, 256, 300, 300]


def test_export_objects_takes_the_busiest_train_image_times_the_mosaic_factor(tmp_path):
    export = make_export(tmp_path / "data", n_train=3, n_val=1)
    labels = export / "yolo_seg" / "labels" / "train"
    square = (labels / "clip_000000.txt").read_text().strip()
    (labels / "clip_000001.txt").write_text("\n".join([square] * 5) + "\n\n")  # 5 objects, a blank line
    (labels / "clip_000002.txt").write_text("")
    assert export_objects(export / "yolo_seg") == objects_bucket(5 * vram.MOSAIC_FACTOR) == 16
    est = VramEstimator(tmp_path / "data")
    assert est.objects_for(export / "yolo_seg") == 16 and est.objects_for(None) == vram.DEFAULT_OBJECTS


# ---- run_probe: the subprocess side, with a stand-in probe script ----

def _fake_probe(tmp_path, body: str) -> list[str]:
    script = tmp_path / "fake_probe.py"
    script.write_text(textwrap.dedent(f"""
        import json, os, sys, time
        P = {vram.PREFIX!r}
        def say(**e):
            print(P + json.dumps(e), flush=True)
    """) + textwrap.dedent(body))
    return [sys.executable, str(script)]


def test_run_probe_streams_batches_and_returns_the_result(tmp_path):
    cmd = _fake_probe(tmp_path, """
        print("Ultralytics noise", flush=True)
        say(type="start", gpu_name=":".join([os.environ["CUDA_VISIBLE_DEVICES"], *sys.argv[1:]]),
            total_bytes=100, limit_bytes=50)
        say(type="batch", batch=1, bytes=10)
        say(type="batch", batch=2, bytes=None)
        say(type="done", estimates=[{"batch": 1, "bytes": 10}, {"batch": 2, "bytes": None}])
    """)
    seen = []
    out = run_probe("yolo26n-seg", 320, 1, 16, lambda b, n: seen.append((b, n)), lambda: False,
                    cwd=tmp_path / "w", cmd=cmd)
    assert out == {"gpu_name": "1:yolo26n-seg:320:--objects:16", "total_bytes": 100, "limit_bytes": 50,
                   "estimates": [{"batch": 1, "bytes": 10}, {"batch": 2, "bytes": None}]}
    assert seen == [(1, 10), (2, None)]
    assert "Ultralytics noise" in (tmp_path / "w" / vram.LOG_NAME).read_text()


def test_run_probe_raises_the_probes_own_error(tmp_path):
    cmd = _fake_probe(tmp_path, """
        say(type="error", message="FileNotFoundError: yolo26n-seg.pt")
        sys.exit(1)
    """)
    with pytest.raises(ProbeFailed, match="FileNotFoundError: yolo26n-seg.pt"):
        run_probe("yolo26n-seg", 320, 0, 8, lambda b, n: None, lambda: False, cwd=tmp_path / "w", cmd=cmd)


def test_run_probe_raises_when_the_probe_dies_without_an_answer(tmp_path):
    cmd = _fake_probe(tmp_path, "sys.exit(3)\n")
    with pytest.raises(ProbeFailed, match="exited with code 3"):
        run_probe("yolo26n-seg", 320, 0, 8, lambda b, n: None, lambda: False, cwd=tmp_path / "w", cmd=cmd)


def test_run_probe_kills_the_probe_when_asked_to_stop(tmp_path):
    cmd = _fake_probe(tmp_path, """
        say(type="start", gpu_name="g", total_bytes=1)
        say(type="batch", batch=1, bytes=10)
        time.sleep(60)
    """)
    seen = []
    started = time.monotonic()
    out = run_probe("yolo26n-seg", 320, 0, 8, lambda b, n: seen.append(b), lambda: bool(seen),
                    cwd=tmp_path / "w", cmd=cmd, poll=0.05)
    assert out is None and seen == [1] and time.monotonic() - started < 10


def test_the_cache_key_names_the_models_task():
    from vdp.train.vram import cache_key

    assert cache_key("yolo26s-seg", 960, 8, "GPU") == "yolo26s-seg|960|segment|8|GPU"  # as before: still cached
    assert cache_key("yolo26s", 960, 8, "GPU") == "yolo26s|960|detect|8|GPU"
