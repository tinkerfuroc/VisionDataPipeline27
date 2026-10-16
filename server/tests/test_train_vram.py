from __future__ import annotations

import json
import sys
import textwrap
import time

import pytest
from vdp.train import vram
from vdp.train.scheduler import Gpu
from vdp.train.vram import MARGIN_BYTES, MARGIN_FACTOR, ProbeFailed, VramEstimator, run_probe, with_margin

GPU = Gpu(1, "NVIDIA GeForce RTX 2080 Ti", 11 * 2**30)
MEASURED = [{"batch": 1, "bytes": 2**30}, {"batch": 2, "bytes": 2 * 2**30}, {"batch": 4, "bytes": 4 * 2**30},
            {"batch": 8, "bytes": None}, {"batch": 12, "bytes": None}]


def _runner(calls: list, measured=MEASURED):
    def run(model, imgsz, gpu, on_batch, should_stop):
        calls.append((model, imgsz, gpu))
        for e in measured:
            on_batch(e["batch"], e["bytes"])
        return {"gpu_name": GPU.name, "total_bytes": GPU.memory_total, "estimates": measured}
    return run


def test_with_margin_scales_and_adds_the_fixed_overhead():
    assert with_margin(None) is None
    assert with_margin(4 * 2**30) == int(4 * 2**30 * MARGIN_FACTOR) + MARGIN_BYTES
    assert 1.0 < MARGIN_FACTOR < 1.3 and 0 < MARGIN_BYTES < 2**30


def test_measure_caches_raw_numbers_and_reports_them_with_the_margin(tmp_path):
    calls, progress = [], []
    est = VramEstimator(tmp_path / "data", runner=_runner(calls))
    assert est.cached("yolo26s-seg", 960, GPU.name) is None

    result = est.measure("yolo26s-seg", 960, GPU, on_batch=lambda b, n: progress.append(b), should_stop=lambda: False)

    assert calls == [("yolo26s-seg", 960, 1)] and progress == [1, 2, 4, 8, 12]
    entry = est.cached("yolo26s-seg", 960, GPU.name)
    assert entry["measured"] == MEASURED and entry["measured_at"] == result["measured_at"]
    saved = json.loads((tmp_path / "data" / "train_weights" / "vram_estimates.json").read_text())
    assert list(saved["entries"]) == ["yolo26s-seg|960|segment|NVIDIA GeForce RTX 2080 Ti"]

    view = est.view(entry, "yolo26s-seg", 960, GPU, free_bytes=5 * 2**30, cached=False)
    assert view == {
        "model": "yolo26s-seg", "imgsz": 960, "gpu": 1, "gpu_name": GPU.name, "total_bytes": GPU.memory_total,
        "free_bytes": 5 * 2**30, "measured_at": entry["measured_at"], "cached": False,
        "estimates": [{"batch": e["batch"], "bytes": with_margin(e["bytes"])} for e in MEASURED],
    }


def test_the_cache_is_per_model_imgsz_and_gpu_name(tmp_path):
    est = VramEstimator(tmp_path / "data", runner=_runner([]))
    est.measure("yolo26s-seg", 960, GPU, on_batch=lambda b, n: None, should_stop=lambda: False)
    assert est.cached("yolo26s-seg", 960, GPU.name) is not None
    assert est.cached("yolo26s-seg", 640, GPU.name) is None
    assert est.cached("yolo26m-seg", 960, GPU.name) is None
    assert est.cached("yolo26s-seg", 960, "NVIDIA GeForce RTX 3080 Ti") is None
    # a second instance (a restarted server) reads the same file
    assert VramEstimator(tmp_path / "data").cached("yolo26s-seg", 960, GPU.name) is not None


def test_an_unreadable_cache_file_counts_as_empty(tmp_path):
    path = tmp_path / "data" / "train_weights" / "vram_estimates.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    est = VramEstimator(tmp_path / "data", runner=_runner([]))
    assert est.cached("yolo26s-seg", 960, GPU.name) is None
    est.measure("yolo26s-seg", 960, GPU, on_batch=lambda b, n: None, should_stop=lambda: False)
    assert est.cached("yolo26s-seg", 960, GPU.name) is not None


def test_a_stopped_measurement_is_not_cached(tmp_path):
    est = VramEstimator(tmp_path / "data", runner=lambda *a: None)
    assert est.measure("yolo26s-seg", 960, GPU, on_batch=lambda b, n: None, should_stop=lambda: True) is None
    assert est.cached("yolo26s-seg", 960, GPU.name) is None


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
        say(type="start", gpu_name=os.environ["CUDA_VISIBLE_DEVICES"] + ":" + sys.argv[1] + ":" + sys.argv[2],
            total_bytes=100)
        say(type="batch", batch=1, bytes=10)
        say(type="batch", batch=2, bytes=None)
        say(type="done", estimates=[{"batch": 1, "bytes": 10}, {"batch": 2, "bytes": None}])
    """)
    seen = []
    out = run_probe("yolo26n-seg", 320, 1, lambda b, n: seen.append((b, n)), lambda: False,
                    cwd=tmp_path / "w", cmd=cmd)
    assert out == {"gpu_name": "1:yolo26n-seg:320", "total_bytes": 100,
                   "estimates": [{"batch": 1, "bytes": 10}, {"batch": 2, "bytes": None}]}
    assert seen == [(1, 10), (2, None)]
    assert "Ultralytics noise" in (tmp_path / "w" / vram.LOG_NAME).read_text()


def test_run_probe_raises_the_probes_own_error(tmp_path):
    cmd = _fake_probe(tmp_path, """
        say(type="error", message="FileNotFoundError: yolo26n-seg.pt")
        sys.exit(1)
    """)
    with pytest.raises(ProbeFailed, match="FileNotFoundError: yolo26n-seg.pt"):
        run_probe("yolo26n-seg", 320, 0, lambda b, n: None, lambda: False, cwd=tmp_path / "w", cmd=cmd)


def test_run_probe_raises_when_the_probe_dies_without_an_answer(tmp_path):
    cmd = _fake_probe(tmp_path, "sys.exit(3)\n")
    with pytest.raises(ProbeFailed, match="exited with code 3"):
        run_probe("yolo26n-seg", 320, 0, lambda b, n: None, lambda: False, cwd=tmp_path / "w", cmd=cmd)


def test_run_probe_kills_the_probe_when_asked_to_stop(tmp_path):
    cmd = _fake_probe(tmp_path, """
        say(type="start", gpu_name="g", total_bytes=1)
        say(type="batch", batch=1, bytes=10)
        time.sleep(60)
    """)
    seen = []
    started = time.monotonic()
    out = run_probe("yolo26n-seg", 320, 0, lambda b, n: seen.append(b), lambda: bool(seen),
                    cwd=tmp_path / "w", cmd=cmd, poll=0.05)
    assert out is None and seen == [1] and time.monotonic() - started < 10
