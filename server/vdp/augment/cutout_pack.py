"""Cutouts packed into one flat file that worker processes memory-map (one page-cache copy for all).

`cutouts.bin` holds the cutouts' RGB and mask bytes; `cutouts.json` lists, per cutout and in cutout order, its class
id, shape and the byte offsets of its image and mask in the `.bin`. `CutoutPackWriter` appends groups of cutouts as
they are made, in any order, so they never all need to be in memory at once.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np

from .composite import Cutout

BIN = "cutouts.bin"
INDEX = "cutouts.json"


class CutoutPackWriter:
    """Writes `cutouts.bin` as groups arrive (`add(key, cutouts)`, keys unique) and, on `close()`, `cutouts.json`
    listing every group's cutouts in ascending key order, which is the order `load_cutouts` returns them in."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._bin = open(self.directory / BIN, "wb")  # noqa: SIM115 -- open across add() calls; close() closes it
        self._offset = 0
        self._groups: dict[int, list[dict]] = {}

    def add(self, key: int, cutouts: Sequence[Cutout]) -> None:
        if key in self._groups:
            raise ValueError(f"cutout group {key} added twice")
        entries = []
        for c in cutouts:
            image = np.ascontiguousarray(c.image, dtype=np.uint8)
            mask = np.ascontiguousarray(c.mask, dtype=bool)
            entry = {"class_id": int(c.class_id), "shape": [int(image.shape[0]), int(image.shape[1])]}
            entry["image"] = self._write(image)
            entry["mask"] = self._write(mask)
            entries.append(entry)
        self._groups[key] = entries

    def _write(self, a: np.ndarray) -> int:
        offset = self._offset
        self._bin.write(a.tobytes())
        self._offset += a.nbytes
        return offset

    @property
    def count(self) -> int:
        return sum(len(g) for g in self._groups.values())

    def close(self) -> None:
        if self._bin.closed:
            return
        self._bin.close()
        entries = [e for key in sorted(self._groups) for e in self._groups[key]]
        (self.directory / INDEX).write_text(json.dumps(entries))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def pack_cutouts(cutouts: Sequence[Cutout], directory: Path) -> None:
    with CutoutPackWriter(directory) as writer:
        writer.add(0, cutouts)


def load_cutouts(directory: Path) -> list[Cutout]:
    """Cutouts as read-only views into a memory map of `cutouts.bin` (nothing is copied)."""
    directory = Path(directory)
    entries = json.loads((directory / INDEX).read_text())
    if not entries:
        return []
    data = np.memmap(directory / BIN, dtype=np.uint8, mode="r")
    cutouts = []
    for e in entries:
        h, w = e["shape"]
        image = data[e["image"]:e["image"] + h * w * 3].reshape(h, w, 3)
        mask = data[e["mask"]:e["mask"] + h * w].reshape(h, w).view(bool)
        cutouts.append(Cutout(e["class_id"], image, mask))
    return cutouts
