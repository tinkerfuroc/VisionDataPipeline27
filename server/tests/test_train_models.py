from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from train_fixtures import EXPORT_ID, make_export
from vsp.labeling.errors import BadRequest
from vsp.store import NotFound
from vsp.train.models import (
    DEFAULT_MODELS,
    MODELS,
    STRONG_AUG,
    AugParams,
    TrainParams,
    check_model,
    find_export,
    list_exports,
    rename_export,
    run_classes,
    train_overrides,
)


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
    assert info.name is None and info.to_json()["name"] is None


def test_find_export_with_a_name(tmp_path):
    make_export(tmp_path, name="Cups v2")
    info = find_export(tmp_path, EXPORT_ID)
    assert info.name == "Cups v2" and info.to_json()["name"] == "Cups v2"


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


def test_list_exports_orders_by_created_at_then_id_descending(tmp_path):
    # folder-name order (alphabetical) deliberately differs from created_at order
    make_export(tmp_path, "aaa-export", created_at="2026-01-01T00:00:00+00:00")
    make_export(tmp_path, "zzz-export", created_at="2026-01-03T00:00:00+00:00")
    make_export(tmp_path, "mmm-export-1", created_at="2026-01-02T00:00:00+00:00")
    make_export(tmp_path, "mmm-export-2", created_at="2026-01-02T00:00:00+00:00")
    assert [e.id for e in list_exports(tmp_path)] == [
        "zzz-export", "mmm-export-2", "mmm-export-1", "aaa-export",
    ]


def test_rename_export_sets_and_clears_the_name(tmp_path):
    make_export(tmp_path)
    before = json.loads((tmp_path / "exports" / EXPORT_ID / "export.json").read_text())

    info = rename_export(tmp_path, EXPORT_ID, "Cups v2")
    assert info.name == "Cups v2"
    after = json.loads((tmp_path / "exports" / EXPORT_ID / "export.json").read_text())
    assert {k: v for k, v in after.items() if k != "name"} == {k: v for k, v in before.items() if k != "name"}

    cleared = rename_export(tmp_path, EXPORT_ID, None)
    assert cleared.name is None
    cleared_manifest = json.loads((tmp_path / "exports" / EXPORT_ID / "export.json").read_text())
    assert cleared_manifest["name"] is None


def test_rename_export_rejects_an_overlong_name(tmp_path):
    make_export(tmp_path)
    with pytest.raises(BadRequest, match="at most 80"):
        rename_export(tmp_path, EXPORT_ID, "a" * 81)


def test_aug_presets_and_overrides():
    assert train_overrides({"epochs": 5}) == ({}, False)  # old run.json
    assert train_overrides(TrainParams().model_dump()) == ({}, False)
    kw, photo = train_overrides(TrainParams(augment="strong").model_dump())
    assert photo is True and kw["degrees"] == 15 and kw["copy_paste"] == 0.3
    assert kw["copy_paste_mode"] == "flip"
    expected = {**STRONG_AUG.model_dump(exclude={"photometric"}), "copy_paste_mode": "flip"}
    assert "photometric" not in kw and kw == expected
    custom = TrainParams(augment="custom",
                         aug=AugParams(degrees=5, photometric=False)).model_dump()
    kw, photo = train_overrides(custom)
    assert kw["degrees"] == 5 and kw["fliplr"] == 0.5 and photo is False


def test_run_classes_from_data_yaml_dict_names_ordered_by_yolo_id(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "data.yaml").write_text("path: .\nnames:\n  1: Bar\n  0: Foo\n")
    assert run_classes(run_dir, {"export_id": "nope"}, tmp_path) == ("Foo", "Bar")


def test_run_classes_from_data_yaml_list_names(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "data.yaml").write_text("path: .\nnames:\n  - Foo\n  - Bar\n")
    assert run_classes(run_dir, {"export_id": "nope"}, tmp_path) == ("Foo", "Bar")


def test_run_classes_falls_back_to_export_without_a_data_yaml(tmp_path):
    make_export(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert run_classes(run_dir, {"export_id": EXPORT_ID}, tmp_path) == ("Module",)


def test_run_classes_none_when_neither_data_yaml_nor_export_is_available(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert run_classes(run_dir, {"export_id": "missing"}, tmp_path) is None


def test_custom_needs_values_and_bounds():
    with pytest.raises(ValidationError):
        TrainParams(augment="custom")
    for bad in ({"degrees": 181}, {"perspective": 0.01}, {"mixup": 1.5}, {"nope": 1}):
        with pytest.raises(ValidationError):
            AugParams(**bad)
