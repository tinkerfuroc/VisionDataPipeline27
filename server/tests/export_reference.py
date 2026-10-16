"""The byte-identity reference for `write_dataset`: a verbatim copy of the per-format loop it replaced (place every
image, then let each exporter class serialise the whole plan). It calls the real exporter classes. Also the rich
multi-clip project (`rich_project`) the plan and runner tests share."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from vdp.export import EXPORTERS, ExportOptions
from vdp.export.base import ExportSummary, place_image, shared_image
from vdp.export.plan import ExportPlan, PlanFrame
from vdp.ingest import FrameDirSource, import_clip
from vdp.labeling.regions import apply_to_image, crop_to
from vdp.store.models import LabelObject, RegionRange

JPEG_QUALITY = 95


def _write_region_image(frame: PlanFrame, dst: Path) -> None:
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read {frame.image_path}")
    out = crop_to(apply_to_image(image, frame.region), frame.region)
    if not cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f"cannot write {dst}")


def _counts(plan: ExportPlan) -> dict:
    out = {}
    for split in ("train", "val"):
        frames = plan.of_split(split)
        annotated = sum(1 for f in frames if f.objects)
        out[split] = {"images": len(frames), "annotated": annotated, "empty": len(frames) - annotated}
    return out


def ref_write_dataset(
    plan: ExportPlan, formats: list[str], out: Path, tick: Callable[[], None], *, link_images: bool = False,
) -> tuple[list[ExportSummary], dict, bool]:
    (out / "images").mkdir()
    for frame in plan.frames:
        if link_images:
            place_image(frame.image_path, shared_image(out, frame))
        elif frame.region is not None:
            _write_region_image(frame, shared_image(out, frame))  # greyed and cropped: its own image
        else:
            # a copy: store frames are never linked outside
            shutil.copy2(frame.image_path, shared_image(out, frame))
        tick()
    summaries = [EXPORTERS[name].write(plan, out, tick) for name in formats]
    counts = _counts(plan)
    linked = all(s.linked for s in summaries)
    return summaries, counts, linked


# --- a rich multi-clip project, shared by the plan and runner tests ---------------------------------------------

def _blobs(*boxes, holes=()):
    mask = np.zeros((60, 120), bool)
    for y0, y1, x0, x1 in boxes:
        mask[y0:y1, x0:x1] = True
    for y0, y1, x0, x1 in holes:
        mask[y0:y1, x0:x1] = False
    return mask


def _extra_clip(store, project_id, tmp_path, n):
    src = tmp_path / f"more_{uuid.uuid4().hex[:6]}"
    src.mkdir()
    for i in range(n):
        assert cv2.imwrite(str(src / f"{i:04d}.png"), np.full((60, 120, 3), 40 + 10 * i, np.uint8))
    return import_clip(store, project_id, FrameDirSource(src))


def rich_project(store, make_clip, tmp_path):
    """3 clips, 3 classes (overlapping, two-blob and holed masks), a noise-only mask, empty frames, a region on part
    of the first clip. Export with class_ids=(0, 2): the bowl is left out."""
    project, first = make_clip(n_frames=8, classes=("cup", "bowl", "plate"))
    second = _extra_clip(store, project.id, tmp_path, 6)
    third = _extra_clip(store, project.id, tmp_path, 5)
    for clip in (first, second, third):
        clip = store.read_clip(clip.id)
        clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=1), LabelObject(id=3, class_id=2)]
        store.write_clip(clip)
        for i in range(clip.frame_count):
            if i in (5, 6) and clip.id == first.id:
                continue  # empty frames
            store.write_mask(clip.id, 1, i, _blobs((5 + i, 35 + i, 10, 50), (45, 55, 70, 90), holes=((15, 20, 20, 30),)))
            store.write_mask(clip.id, 2, i, _blobs((20, 50, 30 + i, 80)))  # the bowl, overlapping both
            if i % 2 == 0:  # the plate overlaps the cup on even frames
                store.write_mask(clip.id, 3, i, _blobs((25, 45, 40, 100)))
    # one frame where the cup's mask is nothing but noise (every blob < 64 px)
    store.write_mask(second.id, 1, 3, _blobs((2, 6, 2, 6), (30, 34, 60, 64), (50, 57, 100, 108)))
    first = store.read_clip(first.id)
    first.regions = [RegionRange(start=1, end=3, crop=(5, 2, 110, 58), blocks=[(60, 30, 90, 50)])]
    store.write_clip(first)
    return project, [first.id, second.id, third.id], ExportOptions(class_ids=(0, 2))
