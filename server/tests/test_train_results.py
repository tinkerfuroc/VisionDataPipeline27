from __future__ import annotations

import json

import pytest
from vdp.train.results import best_row, read_ground_truth, read_metrics, run_summary, summarize, tail

# Ultralytics right-aligns header names with spaces; YOLO26 adds l1/sem losses.
CSV = (
    "                  epoch,                 time,  train/box_loss,  train/seg_loss,  train/l1_loss,"
    "  metrics/mAP50(B),  metrics/mAP50-95(B),  metrics/mAP50(M),  metrics/mAP50-95(M)\n"
    "1,4.8,1.4,5.0,0.03,0.10,0.05,0.08,0.04\n"
    "2,9.1,1.2,4.1,0.02,0.60,0.40,0.55,0.35\n"
    "3,13.5,1.1,3.9,0.02,0.62,0.41,0.57,0.35\n"
)


def _run_dir(tmp_path, text=CSV):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "results.csv").write_text(text)
    return tmp_path


def test_read_metrics_strips_headers_and_types(tmp_path):
    rows = read_metrics(_run_dir(tmp_path))
    assert len(rows) == 3
    assert rows[0]["epoch"] == 1 and isinstance(rows[0]["epoch"], int)
    assert rows[1]["metrics/mAP50-95(M)"] == pytest.approx(0.35)
    assert rows[0]["train/l1_loss"] == pytest.approx(0.03)


def test_read_metrics_missing_and_partial(tmp_path):
    assert read_metrics(tmp_path) == []
    rows = read_metrics(_run_dir(tmp_path, CSV + "4,17"))  # a row caught mid-write
    assert [r["epoch"] for r in rows] == [1, 2, 3, 4]
    assert "metrics/mAP50-95(M)" not in rows[3]


def test_best_row_prefers_later_epoch_on_ties(tmp_path):
    rows = read_metrics(_run_dir(tmp_path))
    assert best_row(rows)["epoch"] == 3
    assert best_row([]) is None


def test_summarize(tmp_path):
    s = summarize(read_metrics(_run_dir(tmp_path)))
    assert s == {
        "epoch": 3, "mask_map50": pytest.approx(0.57),
        "mask_map50_95": pytest.approx(0.35),
        "box_map50": pytest.approx(0.62), "box_map50_95": pytest.approx(0.41),
        "train_seconds": pytest.approx(13.5),
    }
    assert summarize([]) is None


def test_run_summary_takes_the_final_full_validation_when_there_is_one(tmp_path):
    """A run validated per epoch on a val subset: its best epoch is picked on the subset, but the numbers shown are
    the final evaluation of best.pt on the whole val split (final_val.json, written by the worker)."""
    run_dir = _run_dir(tmp_path)
    assert run_summary(run_dir) == summarize(read_metrics(run_dir))  # no final_val.json: as before
    (run_dir / "final_val.json").write_text(json.dumps({"images": 1643, "metrics": {
        "metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.5, "metrics/mAP50(M)": 0.65, "metrics/mAP50-95(M)": 0.45,
    }}))
    assert run_summary(run_dir) == {
        "epoch": 3, "mask_map50": 0.65, "mask_map50_95": 0.45, "box_map50": 0.7, "box_map50_95": 0.5,
        "train_seconds": pytest.approx(13.5), "val_images": 1643,
    }
    assert run_summary(tmp_path / "missing") is None


def test_read_ground_truth_one_instance_per_line(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n\n0 0.1 0.2\n")
    gt = read_ground_truth(p)
    assert len(gt) == 2  # the 1-point line is not a polygon
    assert gt[0] == {"class_id": 0, "polygons": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]]}
    assert read_ground_truth(tmp_path / "missing.txt") == []


def test_tail(tmp_path):
    p = tmp_path / "log"
    p.write_text("".join(f"line {i}\n" for i in range(50)))
    assert tail(p, 3) == "line 47\nline 48\nline 49"
    assert tail(tmp_path / "missing", 3) == ""


from vdp.train.results import mask_iou

SQ = {"class_id": 0, "polygons": [[0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0]]}      # left half
FULL = {"class_id": 0, "polygons": [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]}    # whole image


def test_mask_iou_cases():
    assert mask_iou([], [], 10, 10) is None
    assert mask_iou([SQ], [], 10, 10) == 0.0
    assert mask_iou([], [SQ], 10, 10) == 0.0
    assert mask_iou([SQ], [SQ], 10, 10) == 1.0
    assert mask_iou([SQ], [FULL], 10, 10) == 0.6


def test_a_detection_runs_best_epoch_and_summary_use_box_map(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "results.csv").write_text(
        "epoch,time,metrics/mAP50(B),metrics/mAP50-95(B)\n1,5,0.6,0.4\n2,9,0.7,0.5\n3,12,0.65,0.45\n")
    s = run_summary(tmp_path)
    assert s["epoch"] == 2 and s["box_map50_95"] == 0.5 and s["box_map50"] == 0.7
    assert s["mask_map50_95"] is None and s["mask_map50"] is None
