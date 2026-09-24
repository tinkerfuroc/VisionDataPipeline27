"""Dataset export. Reads from the store only; one ExportPlan, many Exporters (spec §8).
Adding a format = one module with a class that has `name` and `write`, registered here."""

from .base import Exporter, ExportSummary
from .coco import Coco
from .mask_png import MaskPng
from .plan import FORMATS, ExportOptions, ExportPlan, build_plan, check_request
from .yolo import YoloDet, YoloSeg

EXPORTERS: dict[str, Exporter] = {e.name: e for e in (YoloSeg(), YoloDet(), Coco(), MaskPng())}
assert tuple(EXPORTERS) == FORMATS

from .runner import ExportCancelled, manifest_for, run_export, write_dataset

__all__ = [
    "EXPORTERS", "FORMATS", "ExportCancelled", "ExportOptions", "ExportPlan", "ExportSummary", "Exporter",
    "build_plan", "check_request", "manifest_for", "run_export", "write_dataset",
]
