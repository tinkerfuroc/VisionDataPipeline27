from __future__ import annotations

import argparse

import vsp.cli
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export
from vsp.config import Settings
from vsp.train.scheduler import Gpu


def _settings(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    return s


def _args(**kw):
    base = {"list": False, "export": EXPORT_ID, "model": None, "gpu": 1, "epochs": 2, "imgsz": 640,
            "batch": 8, "patience": 30, "workers": 0, "seed": 0}
    return argparse.Namespace(**{**base, **kw})


def test_list_exports(tmp_path, capsys):
    rc = vsp.cli._train(_settings(tmp_path), _args(list=True), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER)
    assert rc == 0 and EXPORT_ID in capsys.readouterr().out


def test_trains_models_in_order_and_prints_table(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(model=["yolo26s-seg", "yolo11s-seg"]), gpus=[Gpu(0, "A", 1), Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    out = capsys.readouterr().out
    assert rc == 0
    assert "yolo26s-seg" in out and "yolo11s-seg" in out and "epoch   2/2" in out
    assert "mask map50-95" in out.lower()
    runs = sorted((s.project.data_root / "runs").iterdir())
    assert len(runs) == 2


def test_failed_run_returns_nonzero(tmp_path, capsys):
    rc = vsp.cli._train(_settings(tmp_path), _args(model=["yolo26s-seg"], epochs=13), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    assert rc == 1 and "OutOfMemoryError" in capsys.readouterr().out


def test_bad_input(tmp_path, capsys):
    s = _settings(tmp_path)
    assert vsp.cli._train(s, _args(export=None), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER) == 2
    assert vsp.cli._train(s, _args(imgsz=1000), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER) == 2
    assert vsp.cli._train(s, _args(), gpus=[], worker_cmd=FAKE_WORKER) == 1
