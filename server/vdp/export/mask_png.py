"""Semantic masks: 8-bit indexed PNG, 0 = background, export class id + 1 = the class, palette from the class
colours. One value per pixel: objects are painted largest first, so a smaller object lying on a larger one stays
visible; the pixels two objects share are counted."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..labeling.errors import BadRequest
from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan


def palette_for(classes) -> list[int]:
    palette = [0, 0, 0] * 256
    for c in classes:
        if 0 <= c.id + 1 <= 255:
            rgb = c.color.lstrip("#")
            palette[(c.id + 1) * 3 : (c.id + 1) * 3 + 3] = [int(rgb[i : i + 2], 16) for i in (0, 2, 4)]
    return palette


@dataclass
class MaskSummary(ExportSummary):
    overlap_px: int = 0  # pixels claimed by more than one object, over every frame of both splits


def paint(objects: Sequence[tuple[int, np.ndarray]], height: int, width: int) -> tuple[np.ndarray, int]:
    """`(value, mask)` pairs -> the semantic canvas and the number of pixels claimed by more than one object.
    Largest area first (ties keep the given order), so the smaller object wins where two overlap."""
    canvas = np.zeros((height, width), np.uint8)
    claims = np.zeros((height, width), np.uint16)
    for value, mask in sorted(objects, key=lambda vm: -int(vm[1].sum())):  # sorted() is stable
        canvas[mask] = value
        claims += mask
    return canvas, int((claims > 1).sum())


def write_semantic_png(objects: Sequence[tuple[int, np.ndarray]], height: int, width: int, palette: list[int],
                       path: Path) -> int:
    """Paint `(export class id, cleaned mask)` pairs (value = class id + 1) into an indexed PNG at `path`;
    returns the number of overlapping pixels."""
    canvas, overlap = paint([(class_id + 1, mask) for class_id, mask in objects], height, width)
    img = Image.fromarray(canvas)  # mode "L"; putpalette turns it into an indexed ("P") image in place
    img.putpalette(palette)
    img.save(path)
    return overlap


class MaskPng:
    name = "masks"

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> MaskSummary:
        if any(c.id + 1 > 255 for c in plan.classes):
            raise BadRequest("the mask image format holds at most 255 classes (8-bit pixels, 0 is the background)")
        root = out_dir / self.name
        summary = MaskSummary(self.name)
        palette = palette_for(plan.classes)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for frame in plan.frames:
            linked = place_image(shared_image(out_dir, frame), root / "images" / frame.split / f"{frame.stem}.jpg")
            summary.linked = summary.linked and linked
            objects = [(obj.class_id, read_mask(obj)) for obj in frame.objects]
            summary.overlap_px += write_semantic_png(
                objects, frame.height, frame.width, palette, root / "masks" / frame.split / f"{frame.stem}.png",
            )
            summary.images[frame.split] += 1
            summary.annotations[frame.split] += len(frame.objects)
            tick()
        return summary
