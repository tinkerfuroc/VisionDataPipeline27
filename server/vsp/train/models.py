"""Training request validation: the model whitelist, hyper-parameters and yolo_seg export lookup."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..export.naming import clean_name
from ..labeling.errors import BadRequest
from ..store import NotFound

FAMILIES = ("yolo26", "yolo11")
SIZES = ("n", "s", "m", "l", "x")
MODELS = tuple(f"{family}{size}-seg" for family in FAMILIES for size in SIZES)
DEFAULT_MODELS = ("yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg")
_EXPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AugParams(BaseModel):
    """Ultralytics augmentation hyper-parameters we expose; defaults are Ultralytics' own."""

    model_config = ConfigDict(extra="forbid")

    degrees: float = Field(default=0.0, ge=0, le=180)
    shear: float = Field(default=0.0, ge=0, le=45)
    perspective: float = Field(default=0.0, ge=0, le=0.001)
    flipud: float = Field(default=0.0, ge=0, le=1)
    fliplr: float = Field(default=0.5, ge=0, le=1)
    scale: float = Field(default=0.5, ge=0, le=0.9)
    translate: float = Field(default=0.1, ge=0, le=0.9)
    hsv_h: float = Field(default=0.015, ge=0, le=0.1)
    hsv_s: float = Field(default=0.7, ge=0, le=1)
    hsv_v: float = Field(default=0.4, ge=0, le=1)
    mosaic: float = Field(default=1.0, ge=0, le=1)
    mixup: float = Field(default=0.0, ge=0, le=1)
    copy_paste: float = Field(default=0.0, ge=0, le=1)
    photometric: bool = False


STRONG_AUG = AugParams(degrees=15, shear=5, perspective=0.0005, flipud=0.5, scale=0.6,
                       translate=0.2, hsv_h=0.02, hsv_v=0.5, mixup=0.1, copy_paste=0.3,
                       photometric=True)
AUG_PRESETS = {"default": AugParams(), "strong": STRONG_AUG}


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=960, ge=320, le=1920)
    batch: int = Field(default=8, ge=1, le=64)
    patience: int = Field(default=30, ge=0, le=1000)
    workers: int = Field(default=8, ge=0, le=16)
    seed: int = Field(default=0, ge=0)
    augment: Literal["default", "strong", "custom"] = "default"
    aug: AugParams | None = None

    @field_validator("imgsz")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        if v % 32:
            raise ValueError("imgsz must be a multiple of 32")
        return v

    @model_validator(mode="after")
    def _custom_needs_values(self) -> TrainParams:
        if self.augment == "custom" and self.aug is None:
            raise ValueError("augment 'custom' needs aug values")
        return self


@dataclass(frozen=True)
class ExportInfo:
    id: str
    dir: Path
    project_name: str
    created_at: str
    classes: tuple[str, ...]
    train_images: int
    val_images: int
    formats: tuple[str, ...] = ()
    derived_from: str | None = None
    name: str | None = None

    @property
    def yolo_dir(self) -> Path:
        return self.dir / "yolo_seg"

    def to_json(self) -> dict:
        return {
            "id": self.id, "project_name": self.project_name, "created_at": self.created_at,
            "classes": list(self.classes), "train_images": self.train_images, "val_images": self.val_images,
            "formats": list(self.formats), "derived_from": self.derived_from, "name": self.name,
        }


def check_model(model: str) -> None:
    if model not in MODELS:
        raise BadRequest(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")


def find_export(data_root: Path, export_id: str) -> ExportInfo:
    if not _EXPORT_ID_RE.match(export_id):
        raise NotFound(f"export {export_id} not found")
    d = Path(data_root) / "exports" / export_id
    manifest_path = d / "export.json"
    if not manifest_path.is_file():
        raise NotFound(f"export {export_id} not found")
    manifest = json.loads(manifest_path.read_text())
    fmt = next((f for f in manifest.get("formats", []) if f.get("name") == "yolo_seg"), None)
    if fmt is None or not (d / "yolo_seg" / "data.yaml").is_file():
        raise BadRequest(f"export {export_id} has no yolo_seg dataset")
    classes = tuple(c["name"] for c in sorted(manifest.get("classes", []), key=lambda c: c["yolo_id"]))
    formats = tuple(f["name"] for f in manifest.get("formats", []))
    return ExportInfo(
        id=export_id, dir=d, project_name=manifest.get("project", {}).get("name", ""),
        created_at=manifest.get("created_at", ""), classes=classes,
        train_images=int(fmt["images"]["train"]), val_images=int(fmt["images"]["val"]),
        formats=formats, derived_from=manifest.get("derived_from"), name=manifest.get("name"),
    )


def list_exports(data_root: Path) -> list[ExportInfo]:
    """Every export with a yolo_seg dataset, newest first by the manifest's `created_at` (then id)."""
    root = Path(data_root) / "exports"
    if not root.is_dir():
        return []
    out = []
    for d in (p for p in root.iterdir() if p.is_dir()):
        try:
            out.append(find_export(data_root, d.name))
        except (NotFound, BadRequest, KeyError, ValueError):  # half-written, not yolo_seg, or unreadable
            continue
    return sorted(out, key=lambda e: (e.created_at, e.id), reverse=True)


# One process-wide lock, deliberately not one per export: renames are rare, so a single lock serialising all
# of them (rather than the bookkeeping of a lock per export id) is the simpler choice and costs nothing in practice.
_RENAME_LOCK = threading.Lock()


def rename_export(data_root: Path, export_id: str, name: str | None) -> ExportInfo:
    """Set (or clear, with None/blank) an export's display name. The folder id never changes, so runs keep
    working. Goes through `find_export`, so this works for any export with a yolo_seg dataset (the same set
    listed on the Train page), and 404s / 400s exactly as `find_export` would."""
    from .runs import atomic_write_json

    name = clean_name(name)
    info = find_export(data_root, export_id)  # 404 / not-yolo_seg checks
    with _RENAME_LOCK:
        path = info.dir / "export.json"
        manifest = json.loads(path.read_text())
        manifest["name"] = name
        atomic_write_json(path, manifest)
    return find_export(data_root, export_id)


def train_overrides(params: dict) -> tuple[dict, bool]:
    """Ultralytics keyword arguments for a run's params (a run.json dict; older runs have no
    augment keys)."""
    mode = params.get("augment", "default")
    if mode == "strong":
        aug = STRONG_AUG
    elif mode == "custom":
        aug = AugParams(**params["aug"])
    else:
        return {}, False
    values = aug.model_dump()
    photometric = values.pop("photometric")
    return {**values, "copy_paste_mode": "flip"}, photometric
