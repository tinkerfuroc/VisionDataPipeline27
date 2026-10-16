"""Mask <-> polygon conversion for the polygon-based formats (YOLO-seg, COCO)."""

from __future__ import annotations

import cv2
import numpy as np

Polygon = list[tuple[float, float]]


# Every exporter uses the same convention, so a YOLO-seg polygon, a YOLO-det box and a COCO bbox of one object agree.
EDGE_SCALE = 2  # measured: IoU 1.0 through pycocotools on a 36 px disc (0.93 at scale 1)


def _box(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    """`y0, y1, x0, x1` (exclusive ends) of the non-zero pixels of a bool mask, or None when there are none. Two
    reductions and a scan of one row and one column: far cheaper than np.nonzero over a frame where the object is a
    few percent. The column reduction only runs over the occupied rows."""
    rows = np.flatnonzero(binary.any(axis=1))
    if rows.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    cols = np.flatnonzero(binary[y0:y1].any(axis=0))
    return y0, y1, int(cols[0]), int(cols[-1]) + 1


def clean_mask(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """The mask without its connected blobs smaller than `min_area_px` (8-connectivity).

    This is THE noise rule of an export, applied once and used by every format: what survives is the object, for
    the polygon, the box, the area and the mask image alike. Filtering in only one of them makes the formats disagree
    (one stray tracker pixel used to stretch the YOLO-det and COCO boxes across the frame while the polygon ignored it).
    An object with nothing left is absent from the frame.
    """
    binary = np.asarray(mask) > 0
    box = _box(binary) if min_area_px > 1 else None
    if box is None:
        return binary
    # A component never leaves its mask's box, so labelling the box alone gives the same blobs and the same areas.
    y0, y1, x0, x1 = box
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px  # label 0 is the background
    out = np.zeros(binary.shape, bool)
    out[y0:y1, x0:x1] = keep[labels]
    return out


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 0.0, edge_scale: int = EDGE_SCALE) -> list[Polygon]:
    """Outer contours of every connected blob (pass a `clean_mask`ed mask; nothing is filtered here by default, so
    every blob of the mask gets a polygon and the formats stay consistent). Holes are not represented: neither YOLO-seg nor the plain COCO
    polygon form can carry them, and SAM masks of solid parts rarely have any.

    Coordinates are continuous image coordinates (pixel i covers [i, i + 1)), the convention of `mask_bbox`, of
    normalised YOLO boxes and of pycocotools. OpenCV alone (`edge_scale=1`) puts vertices on the CENTRES of the
    boundary pixels, which shaves half a pixel off every side: a 5% loss on a 36 px part, and a half-pixel
    disagreement with the box formats. Tracing the mask enlarged `edge_scale` times leaves 0.5/edge_scale px.
    Loaders that truncate to integer pixels before filling (Ultralytics' polygon2mask) recover the mask exactly.
    """
    binary = np.asarray(mask) > 0
    box = _box(binary)
    if box is None:
        return []
    # Traced on the mask's box with a 1 px margin (clipped to the frame, so a blob on the frame edge meets the same
    # border as before); integer vertices are shifted back before the float maths, so the result is bit-identical.
    h, w = binary.shape
    y0, y1, x0, x1 = box
    py0, py1, px0, px1 = max(0, y0 - 1), min(h, y1 + 1), max(0, x0 - 1), min(w, x1 + 1)
    crop = binary[py0:py1, px0:px1].astype(np.uint8)
    k = max(1, int(edge_scale))
    if k > 1:
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) / (k * k) < min_area_px:
            continue
        points = contour.reshape(-1, 2) + (px0 * k, py0 * k)
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
    m = np.asarray(mask)
    box = _box(m if m.dtype == bool else m != 0)  # != 0: the truthiness np.nonzero used
    if box is None:
        return None
    y0, y1, x0, x1 = box
    return x0, y0, x1 - x0, y1 - y0
