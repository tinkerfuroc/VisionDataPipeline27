"""Video decoding through OpenCV's bundled FFmpeg; no system ffmpeg needed."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from .source import IngestError, SourceFrame


class VideoFileSource:
    type = "video"

    def __init__(self, path: Path, name: str | None = None) -> None:
        self.path = Path(path)
        self.name = name or self.path.name
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise IngestError(f"cannot open video: {self.name}")
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        self.fps = float(fps) if fps and fps > 0 else 30.0

    def frames(self) -> Iterator[SourceFrame]:
        cap = cv2.VideoCapture(str(self.path))  # rotation metadata is auto-applied by default
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield SourceFrame(image=img, original_name=None, src_path=None)
        finally:
            cap.release()
