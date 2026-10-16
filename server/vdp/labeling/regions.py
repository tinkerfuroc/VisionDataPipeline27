"""Clip regions: per frame range, a crop (keep only this rectangle) and blocked boxes (exclude these).

The excluded area is everything outside the crop plus the blocks. SAM3 sees it grey, masks written while a region
applies are cleared there, and exports crop to the crop and grey the blocks. Saved masks are never rewritten
because a region changed."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..store import Clip, RegionRange
from .errors import BadRequest

GREY = (114, 114, 114)  # YOLO's letterbox padding colour
MIN_CROP_PX = 32
MIN_KEPT_AREA_PX = 64  # Sam3Engine.MIN_MASK_AREA_PX: a candidate smaller than this once trimmed is dropped


def region_for(clip: Clip, idx: int) -> RegionRange | None:
    """The region that applies to frame `idx`, or None (no range, or a range with no crop and no blocks)."""
    for r in clip.regions:
        if r.start <= idx <= r.end:
            return r if (r.crop is not None or r.blocks) else None
    return None


def region_key(region: RegionRange | None) -> tuple | None:
    """A hashable value that changes whenever the pixels SAM3 sees change (part of the engine's frame cache key)."""
    if region is None:
        return None
    return (tuple(region.crop) if region.crop is not None else None, tuple(tuple(b) for b in region.blocks))


def kept_mask(region: RegionRange, width: int, height: int) -> np.ndarray:
    """HxW bool, True where the frame is kept."""
    if region.crop is None:
        keep = np.ones((height, width), dtype=bool)
    else:
        keep = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = region.crop
        keep[y0:y1, x0:x1] = True
    for x0, y0, x1, y1 in region.blocks:
        keep[y0:y1, x0:x1] = False
    return keep


def apply_to_image(image: np.ndarray, region: RegionRange | None) -> np.ndarray:
    if region is None:
        return image
    out = image.copy()
    out[~kept_mask(region, image.shape[1], image.shape[0])] = GREY
    return out


def apply_to_mask(mask: np.ndarray, region: RegionRange | None) -> np.ndarray:
    if region is None:
        return mask
    return mask & kept_mask(region, mask.shape[1], mask.shape[0])


def crop_to(arr: np.ndarray, region: RegionRange | None) -> np.ndarray:
    if region is None or region.crop is None:
        return arr
    x0, y0, x1, y1 = region.crop
    return arr[y0:y1, x0:x1]


def crop_size(region: RegionRange | None, width: int, height: int) -> tuple[int, int]:
    if region is None or region.crop is None:
        return width, height
    x0, y0, x1, y1 = region.crop
    return x1 - x0, y1 - y0


def is_excluded(region: RegionRange | None, x: int, y: int) -> bool:
    if region is None:
        return False
    if region.crop is not None:
        x0, y0, x1, y1 = region.crop
        if not (x0 <= x < x1 and y0 <= y < y1):
            return True
    return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in region.blocks)


def _intersects(a: Sequence[int], b: Sequence[int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def validate_regions(regions: Sequence[RegionRange], width: int, height: int, frame_count: int) -> None:
    ordered = sorted(regions, key=lambda r: r.start)
    for i, r in enumerate(ordered):
        name = f"region {r.start}–{r.end}"
        if r.start > r.end:
            raise BadRequest(f"{name}: start is after end")
        if r.start < 0 or r.end >= frame_count:
            raise BadRequest(f"{name}: frames must be within 0..{frame_count - 1}")
        if i > 0 and r.start <= ordered[i - 1].end:
            raise BadRequest(f"{name} overlaps region {ordered[i - 1].start}–{ordered[i - 1].end}")
        rects = ([("crop", r.crop)] if r.crop is not None else []) + [("block", b) for b in r.blocks]
        for kind, (x0, y0, x1, y1) in rects:
            if x0 >= x1 or y0 >= y1:
                raise BadRequest(f"{name}: the {kind} is empty")
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                raise BadRequest(f"{name}: the {kind} is outside the frame ({width}×{height})")
        if r.crop is not None:
            x0, y0, x1, y1 = r.crop
            if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
                raise BadRequest(f"{name}: the crop must be at least {MIN_CROP_PX}×{MIN_CROP_PX} px")
            for b in r.blocks:
                if not _intersects(b, r.crop):
                    raise BadRequest(f"{name}: a blocked box lies outside the crop")


def check_regions(clip: Clip) -> None:
    """validate_regions for a stored clip (a hand-edited clip.json is checked where its regions are used)."""
    try:
        validate_regions(clip.regions, clip.width, clip.height, clip.frame_count)
    except BadRequest as e:
        raise BadRequest(f"clip {clip.id}: {e}") from e
