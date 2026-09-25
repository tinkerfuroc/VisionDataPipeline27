from __future__ import annotations

import json
import os

import pytest
from vsp.store import NotFound
from vsp.train.gallery import filter_items, load_gallery

FULL = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
LEFT = [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0]


def _setup(tmp_path):
    run_dir, labels = tmp_path / "run", tmp_path / "labels"
    run_dir.mkdir()
    labels.mkdir()
    inst = lambda poly: {"class_id": 0, "score": 0.9, "polygons": [poly]}
    images = [  # deliberately not in name order
        {"name": "c.jpg", "width": 10, "height": 10, "instances": [inst(LEFT)]},   # iou 0.5
        {"name": "a.jpg", "width": 10, "height": 10, "instances": [inst(FULL)]},   # iou 1.0
        {"name": "b.jpg", "width": 10, "height": 10, "instances": []},             # missed
        {"name": "d.jpg", "width": 10, "height": 10, "instances": []},             # no GT, no pred
    ]
    (run_dir / "predictions.json").write_text(json.dumps({"conf": 0.25, "images": images}))
    for stem in ("a", "b", "c"):
        (labels / f"{stem}.txt").write_text("0 " + " ".join(str(v) for v in FULL) + "\n")
    return run_dir, labels


def test_items_in_name_order_with_iou(tmp_path):
    run_dir, labels = _setup(tmp_path)
    g = load_gallery(run_dir, labels)
    assert g["conf"] == 0.25
    assert [(i["index"], i["name"], i["iou"]) for i in g["items"]] == [
        (0, "a.jpg", 1.0), (1, "b.jpg", 0.0), (2, "c.jpg", 0.6), (3, "d.jpg", None)]


def test_filters(tmp_path):
    items = load_gallery(*_setup(tmp_path))["items"]
    assert [i["name"] for i in filter_items(items, "all")] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert [i["name"] for i in filter_items(items, "missed")] == ["b.jpg"]
    assert [i["name"] for i in filter_items(items, "worst")] == ["b.jpg", "c.jpg", "a.jpg"]


def test_cache_written_and_rebuilt_when_predictions_change(tmp_path):
    run_dir, labels = _setup(tmp_path)
    load_gallery(run_dir, labels)
    cache = run_dir / "predictions_iou.json"
    assert json.loads(cache.read_text())["c.jpg"] == 0.6
    preds = json.loads((run_dir / "predictions.json").read_text())
    preds["images"][0]["instances"] = []          # c.jpg now missed
    (run_dir / "predictions.json").write_text(json.dumps(preds))
    later = cache.stat().st_mtime + 5
    os.utime(run_dir / "predictions.json", (later, later))
    assert {i["name"]: i["iou"] for i in load_gallery(run_dir, labels)["items"]}["c.jpg"] == 0.0


def test_missing_predictions(tmp_path):
    with pytest.raises(NotFound):
        load_gallery(tmp_path, tmp_path)
