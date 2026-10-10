from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np


class IngestError(Exception):
    pass


@dataclass(frozen=True)
class SourceFrame:
    image: np.ndarray  # BGR uint8
    original_name: str | None
    src_path: Path | None


class FrameSource(Protocol):
    type: Literal["video", "frame_dir"]
    name: str
    fps: float

    def frames(self) -> Iterator[SourceFrame]: ...
