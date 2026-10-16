from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from train_fixtures import EXPORT_ID, make_export
from vdp.labeling.errors import BadRequest
from vdp.store import NotFound
from vdp.train.models import (
    DEFAULT_MODELS,
    MODELS,
    STRONG_AUG,
    AugParams,
    TrainParams,
    check_model,
    default_lr0,
    find_export,
    list_exports,
    model_task,
    optim_train_args,
    rename_export,
    resolve_optim,
    run_classes,
    run_data_dir,
    run_optim,
    train_overrides,
    with_legacy_optim,
)


def test_model_whitelist():
    assert "yolo26s-seg" in MODELS and "yolo11x-seg" in MODELS and len(MODELS) == 15
    assert MODELS[10:] == ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")  # detection: YOLO26 only
    assert DEFAULT_MODELS == ("yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg")
    check_model("yolo26m-seg")
    check_model("yolo26s")
    for bad in ("yolov8s-seg", "yolo11s", "yolo26s-pose"):
        with pytest.raises(BadRequest):
            check_model(bad)


def test_model_task():
    assert model_task("yolo26s-seg") == "segment" and model_task("yolo11x-seg") == "segment"
    assert model_task("yolo26n") == "detect" and model_task("yolo26x") == "detect"


def test_export_train_format_prefers_yolo_det_for_detection(tmp_path):
    seg_only = find_export(tmp_path, make_export(tmp_path, "seg-only").name)
    both = find_export(tmp_path, make_export(tmp_path, "both", det=True).name)
    assert both.formats == ("yolo_seg", "yolo_det")
    assert both.train_format("detect") == "yolo_det" and both.train_format("segment") == "yolo_seg"
    assert seg_only.train_format("detect") == "yolo_seg"  # boxes from the polygons (Ultralytics does this)
    assert seg_only.data_dir("detect") == seg_only.dir / "yolo_seg"
    assert both.data_dir("detect") == both.dir / "yolo_det"
    # listed in the manifest but its data.yaml is missing (a half-written export): not used
    (both.dir / "yolo_det" / "data.yaml").unlink()
    assert both.train_format("detect") == "yolo_seg"


def test_run_data_dir_defaults_to_yolo_seg_for_older_runs(tmp_path):
    assert run_data_dir({"export_dir": str(tmp_path)}) == tmp_path / "yolo_seg"
    assert run_data_dir({"export_dir": str(tmp_path), "data_format": "yolo_det"}) == tmp_path / "yolo_det"


def test_params_defaults_and_bounds():
    p = TrainParams()
    assert (p.epochs, p.imgsz, p.batch, p.patience, p.workers, p.seed) == (100, 960, 8, 30, 8, 0)
    for bad in ({"imgsz": 1000}, {"imgsz": 288}, {"epochs": 0}, {"batch": 65}, {"workers": 17}, {"x": 1}):
        with pytest.raises(ValidationError):
            TrainParams(**bad)


def test_optimizer_params_defaults_and_bounds():
    p = TrainParams()
    assert (p.optimizer, p.lr0, p.warmup_passes, p.cos_lr) == ("AdamW", None, 1.0, False)
    assert TrainParams(optimizer="SGD", lr0=0.1, warmup_passes=0, cos_lr=True).lr0 == 0.1
    assert TrainParams(lr0=1e-6, warmup_passes=10).warmup_passes == 10
    for bad in ({"optimizer": "MuSGD"}, {"optimizer": "adamw"}, {"lr0": 0}, {"lr0": 0.2}, {"lr0": 1e-7},
                {"warmup_passes": -1}, {"warmup_passes": 11}):
        with pytest.raises(ValidationError):
            TrainParams(**bad)


def test_default_lr0_per_optimizer_and_class_count():
    assert default_lr0("AdamW", 12) == 0.000625  # Ultralytics' own fit, round(0.002 * 5 / (4 + nc), 6)
    assert default_lr0("AdamW", 1) == 0.001  # 0.002 clamped down
    assert default_lr0("AdamW", 40) == 0.0003  # 0.000227 clamped up
    assert default_lr0("SGD", 12) == 0.003
    assert default_lr0("auto", 12) is None  # Ultralytics picks it


def test_older_params_read_as_the_settings_they_trained_with():
    old = {"epochs": 50, "batch": 4}
    assert with_legacy_optim(old) == {"epochs": 50, "batch": 4, "optimizer": "auto", "lr0": None,
                                      "warmup_passes": 3.0, "cos_lr": False}
    assert with_legacy_optim({"optimizer": "SGD", "warmup_passes": 1.0})["optimizer"] == "SGD"  # kept when present


def test_resolve_optim_for_whole_epochs():
    p = TrainParams(epochs=50).model_dump()
    assert resolve_optim(p, 12, None) == {"optimizer": "AdamW", "lr0": 0.000625, "warmup_epochs": 1.0, "cos_lr": False}
    p = TrainParams(epochs=50, optimizer="SGD", lr0=0.01, warmup_passes=2.5, cos_lr=True).model_dump()
    assert resolve_optim(p, 12, None) == {"optimizer": "SGD", "lr0": 0.01, "warmup_epochs": 2.5, "cos_lr": True}
    p = TrainParams(epochs=50, optimizer="auto", lr0=0.01).model_dump()
    assert resolve_optim(p, 12, None)["lr0"] is None  # ignored: Ultralytics picks optimizer and lr0
    p = TrainParams(epochs=3, warmup_passes=5).model_dump()
    assert resolve_optim(p, 12, None)["warmup_epochs"] == 2.0  # Ultralytics never warms up through the last epoch


def test_resolve_optim_takes_a_sampled_runs_warmup_from_its_plan():
    plan = {"train_images": 40, "train_images_per_epoch": 10, "warmup_epochs": 4.0, "close_mosaic": 40}
    assert resolve_optim(TrainParams().model_dump(), 12, plan)["warmup_epochs"] == 4.0
    val_only = {"train_images": 40, "train_images_per_epoch": None, "val_images_per_epoch": 4}
    assert resolve_optim(TrainParams(epochs=50).model_dump(), 12, val_only)["warmup_epochs"] == 1.0


def test_run_optim_of_an_older_run_is_what_ultralytics_did():
    assert run_optim({"params": {"epochs": 50}, "sampling": None}) == {
        "optimizer": "auto", "lr0": None, "warmup_epochs": 3.0, "cos_lr": False}
    sampled = {"params": {"epochs": 50}, "sampling": {"train_images_per_epoch": 10, "warmup_epochs": 12.0}}
    assert run_optim(sampled)["warmup_epochs"] == 12.0
    stored = {"optimizer": "SGD", "lr0": 0.003, "warmup_epochs": 1.0, "cos_lr": False}
    assert run_optim({"params": {}, "optim": stored}) == stored


def test_optim_train_args():
    adamw = {"optimizer": "AdamW", "lr0": 0.000625, "warmup_epochs": 1.0, "cos_lr": False}
    # no 0.1 bias lr at the start of the warm-up for AdamW -- what Ultralytics' own auto pick does for it
    assert optim_train_args(adamw) == {"optimizer": "AdamW", "lr0": 0.000625, "warmup_epochs": 1.0, "cos_lr": False,
                                       "warmup_bias_lr": 0.0}
    sgd = {"optimizer": "SGD", "lr0": 0.003, "warmup_epochs": 0.0, "cos_lr": True}
    assert optim_train_args(sgd) == {"optimizer": "SGD", "lr0": 0.003, "warmup_epochs": 0.0, "cos_lr": True}
    auto = {"optimizer": "auto", "lr0": None, "warmup_epochs": 3.0, "cos_lr": False}
    assert optim_train_args(auto) == {"optimizer": "auto", "warmup_epochs": 3.0, "cos_lr": False}


def test_optimizer_names_are_ultralytics_own():
    pytest.importorskip("ultralytics")
    import inspect

    from ultralytics.engine.trainer import BaseTrainer

    source = inspect.getsource(BaseTrainer.build_optimizer)
    assert '"AdamW"' in source and '"SGD"' in source and '"auto"' in source


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
    assert train_overrides({"epochs": 5}) == ({}, False, False)  # old run.json
    assert train_overrides(TrainParams().model_dump()) == ({}, False, False)
    kw, photo, lighting = train_overrides(TrainParams(augment="strong").model_dump())
    assert photo is True and lighting is True and kw["degrees"] == 15 and kw["copy_paste"] == 0.3
    assert kw["copy_paste_mode"] == "flip"
    expected = {**STRONG_AUG.model_dump(exclude={"photometric", "lighting"}), "copy_paste_mode": "flip"}
    assert "photometric" not in kw and "lighting" not in kw and kw == expected
    custom = TrainParams(augment="custom",
                         aug=AugParams(degrees=5, photometric=False, lighting=True)).model_dump()
    kw, photo, lighting = train_overrides(custom)
    assert kw["degrees"] == 5 and kw["fliplr"] == 0.5 and photo is False and lighting is True


def test_custom_aug_from_before_lighting_reads_as_off():
    old = TrainParams(augment="custom", aug=AugParams(degrees=5)).model_dump()
    del old["aug"]["lighting"]
    assert train_overrides(old)[2] is False


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


def test_images_per_epoch_params_are_optional_and_bounded():
    p = TrainParams()
    assert p.train_images_per_epoch is None and p.val_images_per_epoch is None  # all, as before
    p = TrainParams(batch=8, train_images_per_epoch=8, val_images_per_epoch=1)
    assert (p.train_images_per_epoch, p.val_images_per_epoch) == (8, 1)
    assert TrainParams(train_images_per_epoch=10**9).train_images_per_epoch == 10**9  # clamped at run start
    for bad in ({"batch": 8, "train_images_per_epoch": 7}, {"train_images_per_epoch": 0}, {"val_images_per_epoch": 0}):
        with pytest.raises(ValidationError):
            TrainParams(**bad)
    old = {"epochs": 10, "imgsz": 640, "batch": 4, "patience": 30, "workers": 0, "seed": 0}  # a run from before
    assert TrainParams(**old).train_images_per_epoch is None
