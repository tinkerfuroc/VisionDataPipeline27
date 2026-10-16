"""Cutouts packed into one flat file that worker processes memory-map (one page-cache copy for all).

`cutouts.bin` holds every cutout's RGB bytes, then every cutout's mask bytes, back to back; `cutouts.json`
lists, per cutout, its class id, shape and the byte offsets of its image and mask in the `.bin`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .composite import Cutout

BIN = "cutouts.bin"
INDEX = "cutouts.json"


def pack_cutouts(cutouts: Sequence[Cutout], directory: Path) -> None:
    directory = Path(directory)
    entries = []
    offset = 0
    for c in cutouts:
        entries.append({"class_id": int(c.class_id), "shape": [int(c.image.shape[0]), int(c.image.shape[1])],
                        "image": offset})
        offset += c.image.nbytes
    with open(directory / BIN, "wb") as f:
        for c in cutouts:
            f.write(np.ascontiguousarray(c.image, dtype=np.uint8).tobytes())
        for entry, c in zip(entries, cutouts, strict=True):
            entry["mask"] = offset
            mask = np.ascontiguousarray(c.mask, dtype=bool)
            f.write(mask.tobytes())
            offset += mask.nbytes
    (directory / INDEX).write_text(json.dumps(entries))


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
