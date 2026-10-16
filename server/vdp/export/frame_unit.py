"""One export frame, every format: the work unit of `write_dataset`'s pool. Module-level and picklable; imports
only export/labeling/store helpers so a spawn worker starts in well under a second."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..labeling.regions import apply_to_image, crop_to
from .base import place_image, read_mask, shared_image
from .coco import coco_annotations
from .mask_png import write_semantic_png
from .plan import PlanFrame
from .yolo import yolo_det_lines, yolo_seg_lines

JPEG_QUALITY = 95
YOLO_FORMATS = ("yolo_seg", "yolo_det")


@dataclass(frozen=True)
class FrameJob:
    out: Path
    formats: tuple[str, ...]
    palette: tuple[int, ...] | None  # masks format only
    link_images: bool


@dataclass
class FrameResult:
    annotations: dict[str, int]  # per format
    linked: dict[str, bool]  # per format: its image was hard-linked to the shared one
    overlap_px: int
    coco: list[dict]  # coco_annotations of the frame, no ids ([] without the coco format)


def write_frames(unit: tuple[FrameJob, list[PlanFrame]]) -> list[FrameResult]:
    job, frames = unit
    return [_write_frame(job, f) for f in frames]


def _write_region_image(frame: PlanFrame, dst: Path) -> None:
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read {frame.image_path}")
    out = crop_to(apply_to_image(image, frame.region), frame.region)
    if not cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f"cannot write {dst}")


def _write_frame(job: FrameJob, frame: PlanFrame) -> FrameResult:
    """The frame's shared image, then each format's image link and label file, as its exporter's `write` would."""
    out = job.out
    shared = shared_image(out, frame)
    if job.link_images:
        place_image(frame.image_path, shared)
    elif frame.region is not None:
        _write_region_image(frame, shared)  # greyed and cropped: its own image
    else:
        shutil.copy2(frame.image_path, shared)  # a copy: store frames are never linked outside
    objects = [(o.class_id, read_mask(o)) for o in frame.objects]
    annotations: dict[str, int] = {}
    linked: dict[str, bool] = {}
    overlap_px = 0
    coco: list[dict] = []
    for name in job.formats:
        root = out / name
        linked[name] = place_image(shared, root / "images" / frame.split / f"{frame.stem}.jpg")
        if name in YOLO_FORMATS:
            lines_of = yolo_seg_lines if name == "yolo_seg" else yolo_det_lines
            lines = lines_of(objects, frame.width, frame.height)
            # an empty label file is YOLO's way of saying "background image"
            (root / "labels" / frame.split / f"{frame.stem}.txt").write_text("".join(f"{ln}\n" for ln in lines))
            annotations[name] = len(lines)
        elif name == "masks":
            assert job.palette is not None
            overlap_px = write_semantic_png(
                objects, frame.height, frame.width, list(job.palette),
                root / "masks" / frame.split / f"{frame.stem}.png",
            )
            annotations[name] = len(frame.objects)
        elif name == "coco":
            coco = coco_annotations(objects)
            annotations[name] = len(coco)
        else:
            raise ValueError(f"unknown format {name!r}")
    return FrameResult(annotations=annotations, linked=linked, overlap_px=overlap_px, coco=coco)
