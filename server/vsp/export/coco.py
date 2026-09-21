"""COCO instances JSON: polygon segmentation, bbox, area, iscrowd 0, one category per project class.
Category ids are class_id + 1 because many COCO consumers reserve 0 for background."""

from __future__ import annotations

import json
from pathlib import Path

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan
from .polygons import mask_bbox, mask_to_polygons

# pycocotools fills polygons in continuous coordinates; see mask_to_polygons
COCO_EDGE_SCALE = 2  # measured: IoU 1.0 through pycocotools on a 36 px disc (0.93 at scale 1)


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
                for obj in frame.objects:
                    mask = read_mask(obj)
                    polygons = mask_to_polygons(mask, edge_scale=COCO_EDGE_SCALE)
                    box = mask_bbox(mask)
                    if not polygons or box is None:
                        continue
                    annotations.append({
                        "id": len(annotations) + 1, "image_id": image_id, "category_id": obj.class_id + 1,
                        "segmentation": [[round(v, 2) for point in polygon for v in point] for polygon in polygons],
                        "bbox": list(box), "area": float(mask.sum()), "iscrowd": 0,
                    })
                tick()
            summary.images[split] = len(images)
            summary.annotations[split] = len(annotations)
            body = {"info": {"description": plan.project_name}, "licenses": [], "images": images,
                    "annotations": annotations, "categories": categories}
            (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(body))
        return summary
