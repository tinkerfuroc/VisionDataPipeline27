"""Frozen copy of the full-frame composite (the pre-ROI implementation), kept as the reference the
optimised one is compared against. Do not edit."""

from __future__ import annotations

import cv2
import numpy as np
from vdp.augment.composite import (
    ALPHA_SIGMA,
    MAX_OVERLAP,
    MIN_INSIDE,
    TRIES,
    Composite,
    Cutout,
    crop_background,
    transform_cutout,
)
from vdp.export.polygons import clean_mask


def _paste_mask(mask: np.ndarray, pos: tuple[int, int], canvas_hw: tuple[int, int]) -> np.ndarray:
    H, W = canvas_hw
    y, x = pos
    h, w = mask.shape
    full = np.zeros((H, W), bool)
    ty0, tx0, ty1, tx1 = max(0, y), max(0, x), min(H, y + h), min(W, x + w)
    if ty1 > ty0 and tx1 > tx0:
        full[ty0:ty1, tx0:tx1] = mask[ty0 - y:ty1 - y, tx0 - x:tx1 - x]
    return full


def place(rng: np.random.Generator, mask: np.ndarray, canvas_hw: tuple[int, int], placed: list[np.ndarray],
          check_overlap: bool) -> tuple[int, int] | None:
    H, W = canvas_hw
    h, w = mask.shape
    area = int(mask.sum())
    for _ in range(TRIES):
        y = int(rng.integers(-h // 2, max(-h // 2 + 1, H - h // 2)))
        x = int(rng.integers(-w // 2, max(-w // 2 + 1, W - w // 2)))
        full = _paste_mask(mask, (y, x), canvas_hw)
        visible = int(full.sum())
        if visible < MIN_INSIDE * area:
            continue
        if check_overlap and any(
            (full & other).sum() > MAX_OVERLAP * min(visible, int(other.sum())) for other in placed
        ):
            continue
        return y, x
    return None


def _colour_match(
    rng: np.random.Generator, obj: np.ndarray, mask: np.ndarray, under: np.ndarray,
) -> np.ndarray:
    out = obj.astype(np.float32)
    pull, contrast = rng.uniform(0.0, 0.5), rng.uniform(0.8, 1.2)
    for c in range(3):
        mo = out[..., c][mask].mean()
        mb = under[..., c][mask].mean()
        out[..., c] = (out[..., c] - mo) * contrast + mo + pull * (mb - mo)
    return np.clip(out, 0, 255)


def make_composite(rng: np.random.Generator, width: int, height: int, background: np.ndarray,
                   cutouts: list[Cutout], max_objects: int, min_area_px: int) -> Composite:
    canvas = crop_background(rng, background, width, height).astype(np.float32)
    k = int(rng.integers(1, max_objects + 1))
    picks = [cutouts[int(i)] for i in rng.integers(0, len(cutouts), k)]
    owner = np.full((height, width), -1, np.int16)
    placed: list[np.ndarray] = []
    classes: list[int] = []
    for j, cut in enumerate(picks):
        img, m = transform_cutout(rng, cut, min(width, height))
        pos = place(rng, m, (height, width), placed, check_overlap=j > 0)
        if pos is None:
            if j > 0:
                continue
            # the first object is always placed
            pos = ((height - m.shape[0]) // 2, (width - m.shape[1]) // 2)
        full = _paste_mask(m, pos, (height, width))
        if not full.any():
            continue
        y, x = pos
        src = np.zeros((height, width, 3), np.float32)
        ty0, tx0 = max(0, y), max(0, x)
        ty1, tx1 = min(height, y + m.shape[0]), min(width, x + m.shape[1])
        src[ty0:ty1, tx0:tx1] = img[ty0 - y:ty1 - y, tx0 - x:tx1 - x]
        matched = _colour_match(rng, src, full, canvas)
        # outside the hard mask the "object" is the background
        layer = np.where(full[..., None], matched, canvas)
        alpha = cv2.GaussianBlur(full.astype(np.float32), (0, 0), ALPHA_SIGMA)[..., None]
        # keep the object's own pixels solid
        alpha = np.where(full[..., None], np.maximum(alpha, 0.5), alpha)
        canvas = canvas * (1 - alpha) + layer * alpha
        owner[full] = len(placed)
        placed.append(full)
        classes.append(cut.class_id)
    if rng.random() < 0.3:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        canvas = canvas + rng.normal(0.0, rng.uniform(2.0, 8.0), canvas.shape)
    image = np.rint(np.clip(canvas, 0, 255)).astype(np.uint8)
    objects = []
    for idx, cls in enumerate(classes):
        m = clean_mask(owner == idx, min_area_px)
        if m.any():
            objects.append((cls, m))
    return Composite(image, objects)
