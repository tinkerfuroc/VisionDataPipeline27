"""One synthetic training image: labelled objects cut out by their masks and pasted onto a background
(spec B2). Pure and seeded: the same rng state gives the same composite."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..export.polygons import clean_mask, mask_bbox

MARGIN = 4
SCALE_RANGE = (0.15, 0.70)      # object's longer side as a fraction of the canvas' shorter side
MIN_INSIDE = 0.8                # at most 20 % of an object may fall outside the canvas
MAX_OVERLAP = 0.1               # of the smaller object
TRIES = 20
ALPHA_SIGMA = 1.5


@dataclass(frozen=True)
class Cutout:
    class_id: int
    image: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class Composite:
    image: np.ndarray
    objects: list[tuple[int, np.ndarray]]


def cutout_from(image: np.ndarray, mask: np.ndarray, class_id: int) -> Cutout | None:
    box = mask_bbox(mask)
    if box is None:
        return None
    x, y, w, h = box
    y0, x0 = max(0, y - MARGIN), max(0, x - MARGIN)
    y1, x1 = min(mask.shape[0], y + h + MARGIN), min(mask.shape[1], x + w + MARGIN)
    return Cutout(class_id, image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy())


def crop_background(rng: np.random.Generator, bg: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = bg.shape[:2]
    aspect = width / height
    side = min(h, w) * rng.uniform(0.5, 1.0)           # the crop's shorter side
    ch, cw = (side, side * aspect) if aspect >= 1 else (side / aspect, side)
    shrink = min(1.0, w / cw, h / ch)                  # keep the aspect, fit inside the image
    ch, cw = max(1, int(ch * shrink)), max(1, int(cw * shrink))
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    crop = bg[y0:y0 + ch, x0:x0 + cw]
    interp = cv2.INTER_AREA if cw > width else cv2.INTER_LINEAR
    return cv2.resize(crop, (width, height), interpolation=interp)


def transform_cutout(
    rng: np.random.Generator, cut: Cutout, canvas_short: int,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(cut.mask)
    longer = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    s = rng.uniform(*SCALE_RANGE) * canvas_short / longer
    angle = rng.uniform(-180.0, 180.0)
    img, m = cut.image, cut.mask.astype(np.uint8)
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    h, w = m.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    rot[0, 2] += nw / 2 - w / 2
    rot[1, 2] += nh / 2 - h / 2
    # BORDER_REPLICATE: when the cutout's mask touches the crop edge (no margin left to draw from), a constant
    # black border would bleed into the labelled edge pixels under linear interpolation. The mask itself still
    # gets a constant-0 border (BORDER_CONSTANT, the warpAffine default) so its outline stays exact.
    img2 = cv2.warpAffine(
        np.ascontiguousarray(img), rot, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    m2 = cv2.warpAffine(np.ascontiguousarray(m), rot, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    box = mask_bbox(m2)
    if box is None:
        # scaled below a pixel: keep a one-pixel object rather than failing. min_area_px > 1 (the normal case)
        # drops it via clean_mask once it reaches make_composite, so this is only ever a transient value.
        return img2[:1, :1], np.ones((1, 1), bool)
    x, y, bw, bh = box
    return img2[y:y + bh, x:x + bw], m2[y:y + bh, x:x + bw]


Placed = tuple[int, int, np.ndarray]    # (y0, x0, mask crop) in canvas coordinates, already clipped to the canvas

_PAD = math.ceil(4 * ALPHA_SIGMA)       # the alpha blur's reach: beyond this the blurred mask is exactly 0


def _visible(mask: np.ndarray, pos: tuple[int, int], canvas_hw: tuple[int, int]) -> Placed | None:
    """The part of `mask` (top-left at `pos`) that lies on the canvas, as a view; None if none does."""
    H, W = canvas_hw
    y, x = pos
    h, w = mask.shape
    ty0, tx0, ty1, tx1 = max(0, y), max(0, x), min(H, y + h), min(W, x + w)
    if ty1 <= ty0 or tx1 <= tx0:
        return None
    return ty0, tx0, mask[ty0 - y:ty1 - y, tx0 - x:tx1 - x]


def _overlap(a: Placed, b: Placed) -> int:
    ay, ax, am = a
    by, bx, bm = b
    y0, x0 = max(ay, by), max(ax, bx)
    y1, x1 = min(ay + am.shape[0], by + bm.shape[0]), min(ax + am.shape[1], bx + bm.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(np.count_nonzero(am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]))


def place(rng: np.random.Generator, mask: np.ndarray, canvas_hw: tuple[int, int], placed: list[Placed],
          check_overlap: bool) -> tuple[int, int] | None:
    H, W = canvas_hw
    h, w = mask.shape
    area = int(mask.sum())
    for _ in range(TRIES):
        y = int(rng.integers(-h // 2, max(-h // 2 + 1, H - h // 2)))
        x = int(rng.integers(-w // 2, max(-w // 2 + 1, W - w // 2)))
        vis = _visible(mask, (y, x), canvas_hw)
        visible = 0 if vis is None else int(np.count_nonzero(vis[2]))
        if visible < MIN_INSIDE * area:
            continue
        if check_overlap and any(
            _overlap(vis, other) > MAX_OVERLAP * min(visible, int(np.count_nonzero(other[2]))) for other in placed
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
    placed: list[Placed] = []
    classes: list[int] = []
    for j, cut in enumerate(picks):
        img, m = transform_cutout(rng, cut, min(width, height))
        pos = place(rng, m, (height, width), placed, check_overlap=j > 0)
        if pos is None:
            if j > 0:
                continue
            # the first object is always placed
            pos = ((height - m.shape[0]) // 2, (width - m.shape[1]) // 2)
        vis = _visible(m, pos, (height, width))
        if vis is None or not vis[2].any():
            continue
        y0, x0, crop = vis
        h, w = crop.shape
        y1, x1 = y0 + h, x0 + w
        src = img[y0 - pos[0]:y1 - pos[0], x0 - pos[1]:x1 - pos[1]]
        # everything below happens in the object's box plus the alpha blur's reach
        ry0, rx0, ry1, rx1 = max(0, y0 - _PAD), max(0, x0 - _PAD), min(height, y1 + _PAD), min(width, x1 + _PAD)
        roi = canvas[ry0:ry1, rx0:rx1]
        by, bx = y0 - ry0, x0 - rx0
        under = roi[by:by + h, bx:bx + w]
        matched = _colour_match(rng, src, crop, under)
        # outside the hard mask the "object" is the background
        layer = roi.copy()
        layer[by:by + h, bx:bx + w][crop] = matched[crop]
        full = np.zeros(roi.shape[:2], bool)
        full[by:by + h, bx:bx + w] = crop
        alpha = cv2.GaussianBlur(full.astype(np.float32), (0, 0), ALPHA_SIGMA)[..., None]
        # keep the object's own pixels solid
        alpha = np.where(full[..., None], np.maximum(alpha, 0.5), alpha)
        canvas[ry0:ry1, rx0:rx1] = roi * (1 - alpha) + layer * alpha
        owner[y0:y1, x0:x1][crop] = len(placed)
        placed.append((y0, x0, crop))
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
