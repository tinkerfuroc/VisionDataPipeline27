"""Mask <-> polygon conversion for the polygon-based formats (YOLO-seg, COCO)."""

from __future__ import annotations

import cv2
import numpy as np

Polygon = list[tuple[float, float]]


# Every exporter uses the same convention, so a YOLO-seg polygon, a YOLO-det box and a COCO bbox of one object agree.
EDGE_SCALE = 2  # measured: IoU 1.0 through pycocotools on a 36 px disc (0.93 at scale 1)


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 4.0, edge_scale: int = EDGE_SCALE) -> list[Polygon]:
    """Outer contours of every connected blob. Holes are not represented: neither YOLO-seg nor the plain COCO
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


def polygons_to_mask(polygons: list[Polygon], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        # floor, not round: continuous coordinate c lies in pixel floor(c) (Ultralytics truncates the same way)
        cv2.fillPoly(canvas, [np.floor(np.array(polygon)).astype(np.int32)], 1)
    return canvas.astype(bool)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """`x, y, w, h` of the mask's bounding box in pixels (inclusive extent), or None for an empty mask."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1
