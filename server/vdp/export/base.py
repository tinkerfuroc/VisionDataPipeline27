"""What every exporter shares: the protocol, image placement, mask loading."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ..labeling.regions import apply_to_mask, crop_to
from .plan import ExportPlan, PlanFrame, PlanObject
from .polygons import clean_mask

Tick = Callable[[], None]


@dataclass
class ExportSummary:
    name: str
    images: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0})
    annotations: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0})
    linked: bool = True  # False as soon as one image had to be copied instead of hardlinked


class Exporter(Protocol):
    name: str

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary: ...


def shared_image(out_dir: Path, frame: PlanFrame) -> Path:
    """The export's own copy of a frame; format directories link to it, never to the store."""
    return out_dir / "images" / f"{frame.stem}.jpg"


def place_image(src: Path, dst: Path) -> bool:
    """Hardlink `src` at `dst`, copying when the filesystem refuses. Returns True when linked."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


def read_mask(obj: PlanObject) -> np.ndarray:
    """The object as every format sees it: the stored mask cut to its frame's region (excluded area cleared, then
    cropped) with its noise blobs removed (see `clean_mask`)."""
    mask = np.array(Image.open(obj.mask_path).convert("L")) > 127
    if obj.region is not None:
        mask = crop_to(apply_to_mask(mask, obj.region), obj.region)
    return clean_mask(mask, obj.min_area_px)
