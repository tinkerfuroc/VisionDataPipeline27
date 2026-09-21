"""Semantic masks: 8-bit indexed PNG, 0 = background, class_id + 1 = the class, palette from the class colours."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan


def _palette(plan: ExportPlan) -> list[int]:
    palette = [0, 0, 0] * 256
    for c in plan.classes:
        if 0 <= c.id + 1 <= 255:
            rgb = c.color.lstrip("#")
            palette[(c.id + 1) * 3 : (c.id + 1) * 3 + 3] = [int(rgb[i : i + 2], 16) for i in (0, 2, 4)]
    return palette


class MaskPng:
    name = "masks"

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary:
        root = out_dir / self.name
        summary = ExportSummary(self.name)
        palette = _palette(plan)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for frame in plan.frames:
            linked = place_image(shared_image(out_dir, frame), root / "images" / frame.split / f"{frame.stem}.jpg")
            summary.linked = summary.linked and linked
            canvas = np.zeros((frame.height, frame.width), np.uint8)
            for obj in frame.objects:
                canvas[read_mask(obj)] = obj.class_id + 1
            img = Image.fromarray(canvas)  # mode "L"; putpalette turns it into an indexed ("P") image in place
            img.putpalette(palette)
            img.save(root / "masks" / frame.split / f"{frame.stem}.png")
            summary.images[frame.split] += 1
            summary.annotations[frame.split] += len(frame.objects)
            tick()
        return summary
