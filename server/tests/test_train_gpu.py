"""Real Ultralytics training on a tiny synthetic export (downloads yolo11n-seg.pt once into the tmp data root)."""

from __future__ import annotations

import pytest
from train_fixtures import make_export, wait_for
from vsp.train.models import TrainParams, find_export
from vsp.train.scheduler import TrainScheduler, detect_gpus

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not detect_gpus(), reason="needs CUDA")
def test_real_worker_trains_one_epoch(tmp_path):
    pytest.importorskip("ultralytics")
    data_root = tmp_path / "data"
    export = find_export(data_root, make_export(data_root, n_train=16, n_val=4).name)
    gpus = detect_gpus()
    sched = TrainScheduler(data_root, gpus)
    sched.start()
    try:
        (rid,) = sched.submit(export, [("yolo11n-seg", gpus[-1].index)],
                              TrainParams(epochs=1, imgsz=320, batch=4, workers=0))
        wait_for(lambda: sched.runs.read(rid)["status"] in ("done", "error", "cancelled"),
                timeout=600, interval=1)
        run = sched.runs.read(rid)
        assert run["status"] == "done", (sched.runs.dir(rid) / "worker.log").read_text()[-3000:]
        d = sched.runs.dir(rid)
        assert (d / "train" / "weights" / "best.pt").is_file() and (d / "predictions.json").is_file()
        assert run["best"]["epoch"] == 1
    finally:
        sched.shutdown()
