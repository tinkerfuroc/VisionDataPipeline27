from __future__ import annotations

import cv2
import numpy as np
from vsp.export.polygons import mask_bbox, mask_to_polygons, polygons_to_mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / float((a | b).sum())


def test_a_disc_survives_the_polygon_round_trip():
    mask = np.zeros((120, 160), np.uint8)
    cv2.circle(mask, (80, 60), 40, 1, -1)
    polys = mask_to_polygons(mask.astype(bool))
    assert len(polys) == 1
    assert _iou(polygons_to_mask(polys, 160, 120), mask.astype(bool)) > 0.95


def test_separate_blobs_become_separate_polygons_and_specks_are_dropped():
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:40] = True
    mask[60:90, 55:95] = True
    mask[50, 50] = True  # a one-pixel speck
    polys = mask_to_polygons(mask)
    assert len(polys) == 2
    assert _iou(polygons_to_mask(polys, 100, 100), mask) > 0.95


def test_bbox_is_inclusive_and_none_for_an_empty_mask():
    mask = np.zeros((50, 80), bool)
    mask[10:20, 30:70] = True
    assert mask_bbox(mask) == (30, 10, 40, 10)
    assert mask_bbox(np.zeros((5, 5), bool)) is None
