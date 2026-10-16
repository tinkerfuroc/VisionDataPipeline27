"""COCO instances JSON: polygon segmentation, bbox, area, iscrowd 0, one category per project class.
Category ids are class_id + 1 because many COCO consumers reserve 0 for background."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan
from .polygons import mask_bbox, mask_to_polygons


def coco_annotations(objects: Sequence[tuple[int, np.ndarray]]) -> list[dict]:
    """`(export class id, cleaned mask)` pairs -> COCO annotation dicts without `id` / `image_id`."""
    out = []
    for class_id, mask in objects:
        polygons = mask_to_polygons(mask)
        box = mask_bbox(mask)
        if not polygons or box is None:
            continue
        out.append({
            "category_id": class_id + 1,
            "segmentation": [[round(v, 2) for point in polygon for v in point] for polygon in polygons],
            "bbox": list(box), "area": float(mask.sum()), "iscrowd": 0,
        })
    return out


class Coco:
    name = "coco"

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary:
        root = out_dir / self.name
        summary = ExportSummary(self.name)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        categories = [{"id": c.id + 1, "name": c.name, "supercategory": "object"} for c in sorted(plan.classes, key=lambda c: c.id)]
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            images, annotations = [], []
            for frame in plan.of_split(split):
                linked = place_image(shared_image(out_dir, frame), root / "images" / split / f"{frame.stem}.jpg")
                summary.linked = summary.linked and linked
                image_id = len(images) + 1
                images.append({"id": image_id, "file_name": f"{frame.stem}.jpg", "width": frame.width, "height": frame.height})
                for ann in coco_annotations([(obj.class_id, read_mask(obj)) for obj in frame.objects]):
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id, **ann})
                tick()
            summary.images[split] = len(images)
            summary.annotations[split] = len(annotations)
            body = {"info": {"description": plan.project_name}, "licenses": [], "images": images,
                    "annotations": annotations, "categories": categories}
            (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(body))
        return summary
