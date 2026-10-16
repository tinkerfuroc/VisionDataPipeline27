"""Dataset export. Reads from the store only; one ExportPlan, written one frame at a time (spec §8).
`run_export` never calls `Exporter.write`: the classes are the per-format reference the tests compare against.
Adding a format = a module with a class that has `name` and `write` (registered here, and in `plan.FORMATS`), plus
its per-frame branch in `frame_unit._write_frame`, its directories in `runner._make_dirs`, and its per-split files
in `runner.write_dataset`."""

from .base import Exporter, ExportSummary
from .coco import Coco
from .mask_png import MaskPng
from .naming import clean_name, reserve_export_dir, slug
from .plan import (
    FORMATS,
    ExportOptions,
    ExportPlan,
    build_plan,
    check_request,
    classes_from_manifest,
    options_from_manifest,
    select_classes,
)
from .yolo import YoloDet, YoloSeg

EXPORTERS: dict[str, Exporter] = {e.name: e for e in (YoloSeg(), YoloDet(), Coco(), MaskPng())}
assert tuple(EXPORTERS) == FORMATS

from .runner import ExportCancelled, manifest_for, run_export, write_dataset

__all__ = [
    "EXPORTERS", "FORMATS", "ExportCancelled", "ExportOptions", "ExportPlan", "ExportSummary", "Exporter",
    "build_plan", "check_request", "classes_from_manifest", "clean_name", "manifest_for", "options_from_manifest",
    "reserve_export_dir", "run_export", "select_classes", "slug", "write_dataset",
]
