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

from .plan import ExportPlan, PlanFrame, PlanObject

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
    return np.array(Image.open(obj.mask_path).convert("L")) > 127
