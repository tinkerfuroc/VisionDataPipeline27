"""Directory of numbered frames. Idea ported from the template's FolderSource; adds natural sort + validation."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import cv2

from .source import IngestError, SourceFrame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def natural_key(name: str) -> list:
    """'frame_2.png' < 'frame_10.png'. re.split alternates text/digits, so types align by position."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def resolve_allowed(path: str | Path, allowed_roots: list[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    for root in allowed_roots:
        r = Path(root).expanduser().resolve()
        if resolved == r or r in resolved.parents:
            return resolved
    raise IngestError(f"{resolved} is outside the allowed import roots")


class FrameDirSource:
    type = "frame_dir"

    def __init__(self, path: Path, fps: float = 30.0) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise IngestError(f"{self.path} is not a directory")
        self.files = sorted(
            (f for f in self.path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda f: natural_key(f.name),
        )
        if not self.files:
            raise IngestError(f"no .jpg/.jpeg/.png files in {self.path}")
        self.name = self.path.name
        self.fps = float(fps)

    def frames(self) -> Iterator[SourceFrame]:
        for f in self.files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                raise IngestError(f"unreadable image: {f.name}")
            yield SourceFrame(image=img, original_name=f.name, src_path=f)
