"""Mask <-> polygon conversion for the polygon-based formats (YOLO-seg, COCO)."""

from __future__ import annotations

import cv2
import numpy as np

Polygon = list[tuple[float, float]]


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 4.0) -> list[Polygon]:
    """Outer contours of every connected blob, in pixel coordinates. Holes are not represented:
    neither YOLO-seg nor the plain COCO polygon form can carry them, and SAM masks of solid parts rarely have any."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < min_area_px:
            continue
        polygons.append([(float(x), float(y)) for x, y in contour.reshape(-1, 2)])
    return polygons


def polygons_to_mask(polygons: list[Polygon], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [np.round(np.array(polygon)).astype(np.int32)], 1)
    return canvas.astype(bool)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """`x, y, w, h` of the mask's bounding box in pixels (inclusive extent), or None for an empty mask."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1
