"""A .zip of images as a frame source. Entries are read straight out of the archive; nothing is extracted."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .frame_dir import IMAGE_SUFFIXES, natural_key
from .source import IngestError, SourceFrame

MAX_ENTRIES = 100_000
MAX_BYTES = 64 * 2**30


def _select(infos: list[zipfile.ZipInfo], max_entries: int, max_bytes: int) -> list[zipfile.ZipInfo]:
    """Image entries only, in natural order; refuses an archive that holds more than the configured limits.

    Names are checked even though nothing is ever extracted: an archive carrying `../` or an absolute path is
    malformed, and refusing its entries outright keeps that true if extraction is ever added.
    """
    keep = []
    for info in infos:
        name = info.filename
        parts = name.split("/")
        if info.is_dir() or any(p in ("", ".", "..") for p in parts):
            continue
        if parts[0].lower() == "__macosx" or any(p.startswith(".") for p in parts):
            continue  # resource forks, __MACOSX, and dotfiles such as .DS_Store
        if not name.lower().endswith(tuple(IMAGE_SUFFIXES)):
            continue
        keep.append(info)
    if len(keep) > max_entries:
        raise IngestError(f"the zip holds {len(keep)} images; the limit is {max_entries}")
    total = sum(info.file_size for info in keep)
    if total > max_bytes:
        raise IngestError(f"the zip holds {total / 2**30:.1f} GiB of images; the limit is {max_bytes / 2**30:.1f} GiB")
    keep.sort(key=lambda info: natural_key(info.filename))
    return keep


class ZipFrameSource:
    type = "frame_dir"  # a zip is a directory of frames: same clip layout and semantics

    def __init__(
        self, path: Path, fps: float = 30.0, *, name: str | None = None,
        max_entries: int = MAX_ENTRIES, max_bytes: int = MAX_BYTES,
    ) -> None:
        self.path = Path(path)
        # `name` matters when the caller's file is a temp copy (`<uuid>_run.zip`); default to the stem.
        self.name = name or self.path.stem or self.path.name
        self.fps = float(fps)
        try:
            with zipfile.ZipFile(self.path) as archive:
                self.infos = _select(archive.infolist(), max_entries, max_bytes)
        except zipfile.BadZipFile as e:
            raise IngestError(f"cannot read {self.path.name} as a zip: {e}") from e
        if not self.infos:
            raise IngestError(f"no .jpg/.jpeg/.png files in {self.path.name}")

    def frames(self) -> Iterator[SourceFrame]:
        with zipfile.ZipFile(self.path) as archive:
            for info in self.infos:
                image = cv2.imdecode(np.frombuffer(archive.read(info), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise IngestError(f"unreadable image: {info.filename}")
                yield SourceFrame(image=image, original_name=info.filename, src_path=None)
