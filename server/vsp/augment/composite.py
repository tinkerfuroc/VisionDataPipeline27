"""One synthetic training image: labelled objects cut out by their masks and pasted onto a background
(spec B2). Pure and seeded: the same rng state gives the same composite."""

from __future__ import annotations

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
    img2 = cv2.warpAffine(np.ascontiguousarray(img), rot, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    m2 = cv2.warpAffine(np.ascontiguousarray(m), rot, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    box = mask_bbox(m2)
    if box is None:  # scaled below a pixel: keep a one-pixel object rather than failing
        return img2[:1, :1], np.ones((1, 1), bool)
    x, y, bw, bh = box
    return img2[y:y + bh, x:x + bw], m2[y:y + bh, x:x + bw]


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
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    objects = []
    for idx, cls in enumerate(classes):
        m = clean_mask(owner == idx, min_area_px)
        if m.any():
            objects.append((cls, m))
    return Composite(image, objects)
