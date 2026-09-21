from __future__ import annotations

import json

import numpy as np
import pytest
from vsp.export import ExportCancelled, ExportOptions, run_export
from vsp.store.models import LabelObject


def _labelled(store, make_clip, n=6):
    project, clip = make_clip(n_frames=n)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(n):
        store.write_mask(clip.id, 1, i, mask)
    return project, clip


def test_run_export_writes_every_format_a_manifest_and_reports_progress(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    seen = []
    result = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), tmp_path / "exports",
                        progress=lambda done, total: seen.append((done, total)))
    out = tmp_path / "exports" / result["export_id"]
    assert result["path"] == str(out) and result["formats"] == ["yolo_seg", "masks"]
    assert result["counts"] == {"train": {"images": 4, "annotated": 4, "empty": 0}, "val": {"images": 2, "annotated": 2, "empty": 0}}
    assert len(list((out / "images").glob("*.jpg"))) == 6
    assert (out / "yolo_seg" / "data.yaml").is_file() and (out / "masks" / "masks" / "val").is_dir()
    assert not (out / "coco").exists()
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["options"]["split_mode"] == "contiguous" and manifest["options"]["min_area_px"] == 64
    assert manifest["clips"] == [{"id": clip.id, "name": clip.source.name, "train": 4, "val": 2}]
    assert manifest["classes"] == [{"id": 0, "name": "part", "color": manifest["classes"][0]["color"], "yolo_id": 0, "coco_id": 1, "mask_value": 1}]
    assert seen[-1] == (6 * 4, 6 * 4) and [d for d, _ in seen] == sorted(d for d, _ in seen)
    # the store was only read: no stray file, and the export holds copies, not links to store frames
    import os
    assert os.stat(store.frame_path(clip.id, 0)).st_nlink == 1


def test_a_cancelled_export_leaves_nothing_behind(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 8

    with pytest.raises(ExportCancelled):
        run_export(store, project.id, [clip.id], ["coco"], ExportOptions(), tmp_path / "exports", cancel=cancel)
    assert list((tmp_path / "exports").iterdir()) == []
