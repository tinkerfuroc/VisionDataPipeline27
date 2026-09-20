"""Mask helpers. is_degenerate_mask is ported from tinkerfuroc/tk25_vision_train (MIT, (c) 2025
tinker@furoc), server/tk_vision/annotate/sam3.py @ 5874ba0."""

from __future__ import annotations

import numpy as np


def is_degenerate_mask(mask: np.ndarray, *, max_area_frac: float, min_area_px: int) -> bool:
    """True for empty/speck masks and for hallucinated frame-covering masks."""
    area = int(mask.sum())
    if area < min_area_px:
        return True
    return area > max_area_frac * mask.size


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
