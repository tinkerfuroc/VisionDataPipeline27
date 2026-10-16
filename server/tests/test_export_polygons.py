from __future__ import annotations

import cv2
import numpy as np
import pytest
from polygons_reference import ref_clean_mask, ref_mask_bbox, ref_mask_to_polygons
from vdp.export.polygons import clean_mask, mask_bbox, mask_to_polygons, polygons_to_mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / float((a | b).sum())


def test_a_disc_survives_the_polygon_round_trip():
    mask = np.zeros((120, 160), np.uint8)
    cv2.circle(mask, (80, 60), 40, 1, -1)
    polys = mask_to_polygons(mask.astype(bool))
    assert len(polys) == 1
    assert _iou(polygons_to_mask(polys, 160, 120), mask.astype(bool)) > 0.95


def test_noise_is_removed_once_by_clean_mask_and_every_remaining_blob_gets_a_polygon():
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:40] = True
    mask[60:90, 55:95] = True
    mask[50, 50] = True  # a one-pixel speck
    assert len(mask_to_polygons(mask)) == 3  # the polygon step filters nothing by itself
    cleaned = clean_mask(mask, 64)
    assert not cleaned[50, 50] and cleaned.sum() == mask.sum() - 1
    polys = mask_to_polygons(cleaned)
    assert len(polys) == 2
    assert _iou(polygons_to_mask(polys, 100, 100), cleaned) > 0.95
    specks = np.zeros((100, 100), bool)
    specks[::5, ::5] = True
    assert specks.sum() == 400 and not clean_mask(specks, 64).any()  # plenty of pixels in total, no real blob


def test_bbox_is_inclusive_and_none_for_an_empty_mask():
    mask = np.zeros((50, 80), bool)
    mask[10:20, 30:70] = True
    assert mask_bbox(mask) == (30, 10, 40, 10)
    assert mask_bbox(np.zeros((5, 5), bool)) is None


# --- identity against the full-frame originals (the helpers crop to the mask's box internally) ---

def _edge_cases(h=40, w=56):
    out = [np.zeros((h, w), bool), np.ones((h, w), bool)]
    one = np.zeros((h, w), bool)
    one[17, 23] = True
    out.append(one)
    for ys, xs in [(slice(0, 5), slice(10, 20)), (slice(h - 5, h), slice(10, 20)), (slice(10, 20), slice(0, 5)),
                   (slice(10, 20), slice(w - 5, w)), (slice(0, 6), slice(0, 6)), (slice(0, 6), slice(w - 6, w)),
                   (slice(h - 6, h), slice(0, 6)), (slice(h - 6, h), slice(w - 6, w))]:
        m = np.zeros((h, w), bool)
        m[ys, xs] = True
        out.append(m)  # touching each edge and corner
    ring = np.zeros((h, w), bool)
    ring[5:30, 5:30] = True
    ring[10:25, 10:25] = False
    out.append(ring)  # a hole
    blobs = np.zeros((h, w), bool)
    blobs[2:9, 40:50] = True
    blobs[20:22, 3:5] = True
    blobs[30:38, 20:31] = True
    blobs[31, 45] = True
    out.append(blobs)  # several blobs, one tiny, out of raster order by size
    return out


def _random_masks(n=60, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h, w = int(rng.integers(8, 90)), int(rng.integers(8, 90))
        m = rng.random((h, w)) < rng.uniform(0.01, 0.6)
        if rng.random() < 0.5:  # clumpier blobs: dilate a sparse seed
            m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(rng.integers(1, 4))) > 0
        out.append(m)
    return out


CASES = _edge_cases() + _random_masks()


@pytest.mark.parametrize("min_area", [0, 1, 5, 64])
def test_clean_mask_matches_the_full_frame_reference(min_area):
    for m in CASES:
        got, want = clean_mask(m, min_area), ref_clean_mask(m, min_area)
        assert got.dtype == want.dtype and got.shape == want.shape and np.array_equal(got, want)


def test_mask_bbox_matches_the_full_frame_reference():
    for m in CASES:
        assert mask_bbox(m) == ref_mask_bbox(m)


@pytest.mark.parametrize("edge_scale", [1, 2, 3])
def test_mask_to_polygons_matches_the_full_frame_reference_vertex_for_vertex(edge_scale):
    for m in CASES:
        assert mask_to_polygons(m, edge_scale=edge_scale) == ref_mask_to_polygons(m, edge_scale=edge_scale)
        assert mask_to_polygons(m, 3.0, edge_scale) == ref_mask_to_polygons(m, 3.0, edge_scale)
