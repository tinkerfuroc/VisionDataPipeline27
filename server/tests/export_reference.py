"""The byte-identity reference for `write_dataset`: a verbatim copy of the per-format loop it replaced (place every
image, then let each exporter class serialise the whole plan). It calls the real exporter classes."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
from vdp.export import EXPORTERS
from vdp.export.base import ExportSummary, place_image, shared_image
from vdp.export.plan import ExportPlan, PlanFrame
from vdp.labeling.regions import apply_to_image, crop_to

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
