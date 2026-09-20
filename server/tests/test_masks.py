import numpy as np
from vsp.engine.masks import bbox_of, is_degenerate_mask


def _m(h=100, w=100):
    return np.zeros((h, w), dtype=bool)


def test_empty_and_tiny_masks_are_degenerate():
    m = _m()
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)
    m[0:7, 0:9] = True  # 63 px
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)
    m[0:8, 0:8] = True  # now >= 64 px
    assert not is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)


def test_frame_covering_mask_is_degenerate():
    m = _m()
    m[:50, :] = True  # 50 %
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)


def test_bbox_of_is_xyxy_exclusive_max():
    m = _m()
    m[10:20, 30:45] = True
    assert bbox_of(m) == (30, 10, 45, 20)
