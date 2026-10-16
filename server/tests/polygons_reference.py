"""Frozen copy of the full-frame mask helpers (before they worked on the mask's bounding box), kept as the identity
reference the optimised `vdp.export.polygons` helpers are compared against. Do not edit."""

from __future__ import annotations

import cv2
import numpy as np
from vdp.export.polygons import EDGE_SCALE, Polygon


def ref_clean_mask(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """The mask without its connected blobs smaller than `min_area_px` (8-connectivity).

    This is THE noise rule of an export, applied once and used by every format: what survives is the object, for
    the polygon, the box, the area and the mask image alike. Filtering in only one of them makes the formats disagree
    (one stray tracker pixel used to stretch the YOLO-det and COCO boxes across the frame while the polygon ignored it).
    An object with nothing left is absent from the frame.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if min_area_px <= 1 or not binary.any():
        return binary.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px  # label 0 is the background
    return keep[labels]


def ref_mask_to_polygons(mask: np.ndarray, min_area_px: float = 0.0, edge_scale: int = EDGE_SCALE) -> list[Polygon]:
    """Outer contours of every connected blob (pass a `clean_mask`ed mask; nothing is filtered here by default, so
    every blob of the mask gets a polygon and the formats stay consistent). Holes are not represented: neither YOLO-seg nor the plain COCO
    polygon form can carry them, and SAM masks of solid parts rarely have any.

    Coordinates are continuous image coordinates (pixel i covers [i, i + 1)), the convention of `mask_bbox`, of
    normalised YOLO boxes and of pycocotools. OpenCV alone (`edge_scale=1`) puts vertices on the CENTRES of the
    boundary pixels, which shaves half a pixel off every side: a 5% loss on a 36 px part, and a half-pixel
    disagreement with the box formats. Tracing the mask enlarged `edge_scale` times leaves 0.5/edge_scale px.
    Loaders that truncate to integer pixels before filling (Ultralytics' polygon2mask) recover the mask exactly.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    k = max(1, int(edge_scale))
    if k > 1:
        binary = cv2.resize(binary, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) / (k * k) < min_area_px:
            continue
        points = contour.reshape(-1, 2)
        if k > 1:
            polygons.append([((float(x) + 0.5) / k, (float(y) + 0.5) / k) for x, y in points])
        else:
            polygons.append([(float(x), float(y)) for x, y in points])
    return polygons


def ref_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """`x, y, w, h` of the mask's bounding box in pixels (inclusive extent), or None for an empty mask."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1
