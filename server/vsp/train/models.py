"""Training request validation: the model whitelist, hyper-parameters and yolo_seg export lookup."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..labeling.errors import BadRequest
from ..store import NotFound

FAMILIES = ("yolo26", "yolo11")
SIZES = ("n", "s", "m", "l", "x")
MODELS = tuple(f"{family}{size}-seg" for family in FAMILIES for size in SIZES)
DEFAULT_MODELS = ("yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg")
_EXPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=960, ge=320, le=1920)
    batch: int = Field(default=8, ge=1, le=64)
    patience: int = Field(default=30, ge=0, le=1000)
    workers: int = Field(default=8, ge=0, le=16)
    seed: int = Field(default=0, ge=0)

    @field_validator("imgsz")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        if v % 32:
            raise ValueError("imgsz must be a multiple of 32")
        return v


@dataclass(frozen=True)
class ExportInfo:
    id: str
    dir: Path
    project_name: str
    created_at: str
    classes: tuple[str, ...]
    train_images: int
    val_images: int

    @property
    def yolo_dir(self) -> Path:
        return self.dir / "yolo_seg"

    def to_json(self) -> dict:
        return {
            "id": self.id, "project_name": self.project_name, "created_at": self.created_at,
            "classes": list(self.classes), "train_images": self.train_images, "val_images": self.val_images,
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
    return ExportInfo(
        id=export_id, dir=d, project_name=manifest.get("project", {}).get("name", ""),
        created_at=manifest.get("created_at", ""), classes=classes,
        train_images=int(fmt["images"]["train"]), val_images=int(fmt["images"]["val"]),
    )


def list_exports(data_root: Path) -> list[ExportInfo]:
    """Every export with a yolo_seg dataset, newest first (export ids start with a timestamp)."""
    root = Path(data_root) / "exports"
    if not root.is_dir():
        return []
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        try:
            out.append(find_export(data_root, d.name))
        except (NotFound, BadRequest, KeyError, ValueError):  # half-written, not yolo_seg, or unreadable
            continue
    return out
