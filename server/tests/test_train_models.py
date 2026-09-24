from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from train_fixtures import EXPORT_ID, make_export
from vsp.labeling.errors import BadRequest
from vsp.store import NotFound
from vsp.train.models import DEFAULT_MODELS, MODELS, TrainParams, check_model, find_export, list_exports


def test_model_whitelist():
    assert "yolo26s-seg" in MODELS and "yolo11x-seg" in MODELS and len(MODELS) == 10
    assert DEFAULT_MODELS == ("yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg")
    check_model("yolo26m-seg")
    with pytest.raises(BadRequest):
        check_model("yolov8s-seg")


def test_params_defaults_and_bounds():
    p = TrainParams()
    assert (p.epochs, p.imgsz, p.batch, p.patience, p.workers, p.seed) == (100, 960, 8, 30, 8, 0)
    for bad in ({"imgsz": 1000}, {"imgsz": 288}, {"epochs": 0}, {"batch": 65}, {"workers": 17}, {"x": 1}):
        with pytest.raises(ValidationError):
            TrainParams(**bad)


def test_find_export(tmp_path):
    make_export(tmp_path, n_train=3, n_val=2)
    info = find_export(tmp_path, EXPORT_ID)
    assert info.project_name == "Bench" and info.classes == ("Module",)
    assert (info.train_images, info.val_images) == (3, 2)
    assert info.yolo_dir == tmp_path / "exports" / EXPORT_ID / "yolo_seg"
    assert info.to_json()["id"] == EXPORT_ID and "dir" not in info.to_json()


def test_find_export_errors(tmp_path):
    with pytest.raises(NotFound):
        find_export(tmp_path, "nope")
    with pytest.raises(NotFound):
        find_export(tmp_path, "../etc")
    d = make_export(tmp_path)
    m = json.loads((d / "export.json").read_text())
    m["formats"] = [{"name": "coco", "images": {"train": 1, "val": 1}}]
    (d / "export.json").write_text(json.dumps(m))
    with pytest.raises(BadRequest):
        find_export(tmp_path, EXPORT_ID)


def test_list_exports_newest_first_and_skips_incomplete(tmp_path):
    make_export(tmp_path, "20260101-000000-aaaaaa")
    make_export(tmp_path, "20260102-000000-bbbbbb")
    (tmp_path / "exports" / "20260103-000000-cccccc").mkdir()  # an export still being written: no manifest
    assert [e.id for e in list_exports(tmp_path)] == ["20260102-000000-bbbbbb", "20260101-000000-aaaaaa"]
    assert list_exports(tmp_path / "missing") == []
