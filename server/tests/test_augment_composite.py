from __future__ import annotations

import cv2
import numpy as np
from vsp.augment.composite import (
    crop_background,
    cutout_from,
    make_composite,
    place,
    transform_cutout,
)


def _cut(class_id=0):
    img = np.zeros((40, 60, 3), np.uint8)
    img[:] = (200, 60, 30)
    mask = np.zeros((40, 60), bool)
    mask[5:35, 10:50] = True
    return cutout_from(img, mask, class_id)


def _bg():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)


def test_cutout_from_crops_with_margin():
    c = _cut()
    assert c.mask.shape == (38, 48) and c.image.shape == (38, 48, 3) and c.mask.sum() == 30 * 40
    assert cutout_from(np.zeros((5, 5, 3), np.uint8), np.zeros((5, 5), bool), 0) is None


def test_crop_background_shape():
    out = crop_background(np.random.default_rng(0), _bg(), 96, 160)
    assert out.shape == (160, 96, 3) and out.dtype == np.uint8


def test_transform_scales_to_range():
    for seed in range(20):
        img, m = transform_cutout(np.random.default_rng(seed), _cut(), 200)
        ys, xs = np.nonzero(m)
        longer = max(np.ptp(ys) + 1, np.ptp(xs) + 1)
        # rotation grows the box
        assert img.shape[:2] == m.shape and 0.15 * 200 * 0.9 <= longer <= 0.70 * 200 * 1.5


def test_place_keeps_eighty_percent_inside_and_avoids_overlap():
    rng = np.random.default_rng(3)
    m = np.ones((50, 50), bool)
    placed = []
    for i in range(3):
        pos = place(rng, m, (200, 200), placed, check_overlap=i > 0)
        assert pos is not None
        y, x = pos
        full = np.zeros((200, 200), bool)
        ys, xs = slice(max(0, y), min(200, y + 50)), slice(max(0, x), min(200, x + 50))
        full[ys, xs] = True
        assert full.sum() >= 0.8 * m.sum()
        for other in placed:
            assert (full & other).sum() <= 0.1 * min(full.sum(), other.sum())
        placed.append(full)


def test_make_composite_is_deterministic_and_labels_match():
    args = (160, 96, _bg(), [_cut(0), _cut(1)], 2, 64)
    a = make_composite(np.random.default_rng([7, 0]), *args)
    b = make_composite(np.random.default_rng([7, 0]), *args)
    assert a.image.shape == (96, 160, 3) and np.array_equal(a.image, b.image)
    assert 1 <= len(a.objects) <= 2
    for (ca, ma), (cb, mb) in zip(a.objects, b.objects):
        assert ca == cb and np.array_equal(ma, mb) and ma.shape == (96, 160) and ma.sum() >= 64
    if len(a.objects) == 2:
        assert not (a.objects[0][1] & a.objects[1][1]).any()  # occlusion: each pixel belongs to one object


def test_single_object_when_max_is_one():
    for seed in range(10):
        c = make_composite(np.random.default_rng([seed, 0]), 160, 96, _bg(), [_cut()], 1, 64)
        assert len(c.objects) == 1


def test_far_pixels_untouched_by_object_paste():
    """Pixels far from every object mask differ from the input only by the optional whole-image blur/noise. At
    least one seed in 0-20 fires neither finish step, so its far pixels equal the background crop exactly (the
    background crop is the rng's first draw).

    The alpha blur (sigma 1.5) is what makes an "outside the mask" pixel touch the blend at all, and OpenCV's
    float32 Gaussian kernel for that sigma has a support radius of about 6 px (ksize = round(sigma * 4 * 2 +
    1) | 1 = 13, radius 6): a margin of 4 px (the spec's cutout margin, reused loosely here) is too tight and
    catches genuine float32 rounding dust at the object's edge even when neither finish step fired, so the far
    region here uses an 8 px margin, safely past that kernel's support.
    """
    bg = _bg()
    cutouts = [_cut(0), _cut(1)]
    found = False
    for seed in range(21):
        rng = np.random.default_rng([seed, 0])
        composite = make_composite(rng, 160, 96, bg, cutouts, 2, 64)
        expected_bg = crop_background(np.random.default_rng([seed, 0]), bg, 160, 96)
        object_area = np.zeros((96, 160), np.uint8)
        for _, m in composite.objects:
            object_area |= m.astype(np.uint8)
        kernel = np.ones((17, 17), np.uint8)  # Chebyshev radius 8, past the alpha blur's float32 support
        dilated = cv2.dilate(object_area, kernel) > 0
        far = ~dilated
        if np.array_equal(composite.image[far], expected_bg[far]):
            found = True
            break
    assert found
