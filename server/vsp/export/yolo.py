"""YOLO-seg (normalised polygons, one line per contour) and YOLO-det (normalised cx cy w h of the mask's box)."""

from __future__ import annotations

from pathlib import Path

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan, PlanFrame
from .polygons import mask_bbox, mask_to_polygons


def _data_yaml(plan: ExportPlan) -> str:
    names = "\n".join(f"  {c.id}: {c.name}" for c in sorted(plan.classes, key=lambda c: c.id))
    return f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n"


class _Yolo:
    name = ""

    def lines(self, frame: PlanFrame) -> list[str]:
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
            lines = self.lines(frame)
            # an empty label file is YOLO's way of saying "background image"
            (root / "labels" / frame.split / f"{frame.stem}.txt").write_text("".join(f"{ln}\n" for ln in lines))
            summary.images[frame.split] += 1
            summary.annotations[frame.split] += len(lines)
            tick()
        (root / "data.yaml").write_text(_data_yaml(plan))
        return summary


class YoloSeg(_Yolo):
    name = "yolo_seg"

    def lines(self, frame: PlanFrame) -> list[str]:
        out = []
        for obj in frame.objects:
            for polygon in mask_to_polygons(read_mask(obj)):
                coords = " ".join(f"{x / frame.width:.6f} {y / frame.height:.6f}" for x, y in polygon)
                out.append(f"{obj.class_id} {coords}")
        return out


class YoloDet(_Yolo):
    name = "yolo_det"

    def lines(self, frame: PlanFrame) -> list[str]:
        out = []
        for obj in frame.objects:
            box = mask_bbox(read_mask(obj))
            if box is None:
                continue
            x, y, w, h = box
            out.append(
                f"{obj.class_id} {(x + w / 2) / frame.width:.6f} {(y + h / 2) / frame.height:.6f} "
                f"{w / frame.width:.6f} {h / frame.height:.6f}"
            )
        return out
