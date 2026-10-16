"""YOLO-seg (normalised polygons, one line per contour) and YOLO-det (normalised cx cy w h of the mask's box)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan
from .polygons import mask_bbox, mask_to_polygons


def data_yaml(classes) -> str:
    # json.dumps: a quoted JSON string is a valid YAML scalar, so names with ':' or '#' survive the round trip
    names = "\n".join(f"  {c.id}: {json.dumps(c.name)}" for c in sorted(classes, key=lambda c: c.id))
    return f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n"


def yolo_seg_lines(objects: Sequence[tuple[int, np.ndarray]], width: int, height: int) -> list[str]:
    """`(export class id, cleaned mask)` pairs -> one normalised-polygon line per contour."""
    out = []
    for class_id, mask in objects:
        for polygon in mask_to_polygons(mask):
            coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in polygon)
            out.append(f"{class_id} {coords}")
    return out


def yolo_det_lines(objects: Sequence[tuple[int, np.ndarray]], width: int, height: int) -> list[str]:
    """`(export class id, cleaned mask)` pairs -> one normalised `cx cy w h` line per non-empty mask."""
    out = []
    for class_id, mask in objects:
        box = mask_bbox(mask)
        if box is None:
            continue
        x, y, w, h = box
        out.append(f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")
    return out


class _Yolo:
    name = ""

    def lines(self, width: int, height: int, objects: Sequence[tuple[int, np.ndarray]]) -> list[str]:
        raise NotImplementedError

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary:
        root = out_dir / self.name
        summary = ExportSummary(self.name)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for frame in plan.frames:
            linked = place_image(shared_image(out_dir, frame), root / "images" / frame.split / f"{frame.stem}.jpg")
            summary.linked = summary.linked and linked
            lines = self.lines(frame.width, frame.height, [(o.class_id, read_mask(o)) for o in frame.objects])
            # an empty label file is YOLO's way of saying "background image"
            (root / "labels" / frame.split / f"{frame.stem}.txt").write_text("".join(f"{ln}\n" for ln in lines))
            summary.images[frame.split] += 1
            summary.annotations[frame.split] += len(lines)
            tick()
        (root / "data.yaml").write_text(data_yaml(plan.classes))
        return summary


class YoloSeg(_Yolo):
    name = "yolo_seg"

    def lines(self, width, height, objects):
        return yolo_seg_lines(objects, width, height)


class YoloDet(_Yolo):
    name = "yolo_det"

    def lines(self, width, height, objects):
        return yolo_det_lines(objects, width, height)
