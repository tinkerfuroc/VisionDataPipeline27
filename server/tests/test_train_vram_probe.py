from __future__ import annotations

import json

import pytest
from vdp.train import vram_probe
from vdp.train.vram_probe import BATCHES, measure_batches


def _step_ooming_at(limit: int, calls: list[int]):
    def step(batch: int) -> int:
        calls.append(batch)
        if batch >= limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return batch * 1000
    return step


def test_measure_batches_runs_every_size_in_order():
    calls: list[int] = []
    assert measure_batches(_step_ooming_at(99, calls)) == [
        {"batch": b, "bytes": b * 1000} for b in BATCHES
    ]
    assert calls == [1, 2, 4, 8, 12]


def test_measure_batches_stops_at_the_first_oom_and_marks_it_and_larger_as_not_fitting():
    calls: list[int] = []
    seen = []
    out = measure_batches(_step_ooming_at(8, calls), on_result=lambda b, n: seen.append((b, n)))
    assert out == [{"batch": 1, "bytes": 1000}, {"batch": 2, "bytes": 2000}, {"batch": 4, "bytes": 4000},
                   {"batch": 8, "bytes": None}, {"batch": 12, "bytes": None}]
    assert calls == [1, 2, 4, 8]  # 12 is never tried once 8 did not fit
    assert seen == [(1, 1000), (2, 2000), (4, 4000), (8, None), (12, None)]


def test_measure_batches_counts_a_cuda_backend_memory_error_as_not_fitting():
    def step(batch: int) -> int:
        if batch == 2:
            raise RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
        return 5
    assert [e["bytes"] for e in measure_batches(step)] == [5, None, None, None, None]


def test_measure_batches_lets_other_errors_through():
    def step(batch: int) -> int:
        raise ValueError("bad weights")
    with pytest.raises(ValueError, match="bad weights"):
        measure_batches(step)


def test_main_prints_progress_and_result_lines(monkeypatch, capsys):
    monkeypatch.setattr(vram_probe, "_cuda_device", lambda: ("NVIDIA Test GPU", 12 * 2**30))
    monkeypatch.setattr(vram_probe, "build_step", lambda model, imgsz: _step_ooming_at(12, []))
    monkeypatch.setattr(vram_probe.os, "nice", lambda n: None)

    assert vram_probe.main(["yolo26n-seg", "640"]) == 0

    lines = [json.loads(line.removeprefix(vram_probe.PREFIX)) for line in capsys.readouterr().out.splitlines()
             if line.startswith(vram_probe.PREFIX)]
    assert lines[0] == {"type": "start", "gpu_name": "NVIDIA Test GPU", "total_bytes": 12 * 2**30}
    assert [x["batch"] for x in lines if x["type"] == "batch"] == [1, 2, 4, 8, 12]
    assert lines[-1] == {"type": "done", "estimates": [
        {"batch": 1, "bytes": 1000}, {"batch": 2, "bytes": 2000}, {"batch": 4, "bytes": 4000},
        {"batch": 8, "bytes": 8000}, {"batch": 12, "bytes": None}]}


def test_main_reports_a_failure_as_an_error_line(monkeypatch, capsys):
    monkeypatch.setattr(vram_probe, "_cuda_device", lambda: ("NVIDIA Test GPU", 2**30))

    def broken(model, imgsz):
        raise FileNotFoundError("yolo26n-seg.pt")

    monkeypatch.setattr(vram_probe, "build_step", broken)
    monkeypatch.setattr(vram_probe.os, "nice", lambda n: None)

    assert vram_probe.main(["yolo26n-seg", "640"]) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert json.loads(last.removeprefix(vram_probe.PREFIX)) == {
        "type": "error", "message": "FileNotFoundError: yolo26n-seg.pt"}


def test_synthetic_batch_matches_what_the_segment_trainer_feeds_the_loss():
    torch = pytest.importorskip("torch")
    b = vram_probe.synthetic_batch(3, 320, device="cpu", classes=2, instances=4)
    assert b["img"].shape == (3, 3, 320, 320) and b["img"].dtype == torch.float32
    assert b["batch_idx"].tolist() == [0.0] * 4 + [1.0] * 4 + [2.0] * 4
    assert b["cls"].shape == (12, 1) and b["bboxes"].shape == (12, 4)
    assert 0 < b["bboxes"].min() and b["bboxes"].max() < 1
    # overlap_mask=True (the default): one (H/4, W/4) map per image holding instance index + 1
    assert b["masks"].shape == (3, 80, 80) and b["masks"].max() == 4
    assert b["sem_masks"].shape == (3, 80, 80)
