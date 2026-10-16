"""A real VRAM probe, small (yolo11n-seg at 320 px; downloads yolo11n-seg.pt once into the tmp data root)."""

from __future__ import annotations

from itertools import pairwise

import pytest
from vdp.train import vram
from vdp.train.scheduler import detect_gpus
from vdp.train.vram_probe import BATCHES

pytestmark = pytest.mark.gpu


def test_real_probe_measures_growing_memory_per_batch(tmp_path):
    pytest.importorskip("ultralytics")
    gpus = detect_gpus()  # initialises CUDA: not at collection time (see test_train_gpu.py)
    if not gpus:
        pytest.skip("needs CUDA")
    gpu = gpus[-1]
    seen = []
    out = vram.run_probe("yolo11n-seg", 320, gpu.index, lambda b, n: seen.append(b), lambda: False,
                         cwd=tmp_path / "train_weights")
    log = (tmp_path / "train_weights" / vram.LOG_NAME).read_text()[-3000:]
    assert out is not None, log
    assert out["gpu_name"] == gpu.name and seen == list(BATCHES)
    sizes = [e["bytes"] for e in out["estimates"] if e["bytes"] is not None]
    assert len(sizes) >= 3, (out, log)  # 1, 2 and 4 fit on any GPU worth training on
    assert all(a < b for a, b in pairwise(sizes)), out
