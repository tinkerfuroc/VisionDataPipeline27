import json

import numpy as np
import pytest
from vdp.labeling.errors import BadRequest
from vdp.labeling.regions import (
    GREY,
    apply_to_image,
    apply_to_mask,
    check_regions,
    crop_size,
    crop_to,
    is_excluded,
    kept_mask,
    region_for,
    region_key,
    validate_regions,
)
from vdp.store import RegionRange

W, H, N = 120, 60, 12


def R(start, end, crop=None, blocks=()):
    return RegionRange(start=start, end=end, crop=crop, blocks=list(blocks))


def test_old_clips_load_with_no_regions(store, make_clip):
    _, clip = make_clip()
    assert clip.regions == []
    path = store.clip_dir(clip.id) / "clip.json"
    raw = json.loads(path.read_text())
    raw.pop("regions", None)
    path.write_text(json.dumps(raw))
    assert store.read_clip(clip.id).regions == []


def test_region_for_picks_the_range_and_ignores_empty_ranges(store, make_clip):
    _, clip = make_clip()
    clip.regions = [R(0, 3, blocks=[(0, 0, 10, 10)]), R(4, 6), R(7, 11, crop=(10, 10, 60, 50))]
    assert region_for(clip, 2).blocks == [(0, 0, 10, 10)]
    assert region_for(clip, 5) is None  # a no-op range behaves as no region
    assert region_for(clip, 8).crop == (10, 10, 60, 50)


def test_kept_mask_image_and_mask_application():
    region = R(0, 0, crop=(10, 5, 100, 55), blocks=[(20, 10, 40, 30)])
    keep = kept_mask(region, W, H)
    assert keep.shape == (H, W) and keep[20, 50] and not keep[0, 0] and not keep[15, 25]
    img = np.full((H, W, 3), 7, np.uint8)
    out = apply_to_image(img, region)
    assert tuple(out[0, 0]) == GREY and tuple(out[15, 25]) == GREY and tuple(out[20, 50]) == (7, 7, 7)
    assert img[0, 0, 0] == 7  # a copy, not in place
    mask = np.ones((H, W), bool)
    trimmed = apply_to_mask(mask, region)
    assert trimmed.sum() == keep.sum() and mask.all()
    assert is_excluded(region, 25, 15) and is_excluded(region, 0, 0) and not is_excluded(region, 50, 20)
    assert not is_excluded(None, 0, 0)


def test_crop_to_and_crop_size():
    region = R(0, 0, crop=(10, 5, 100, 55))
    assert crop_to(np.zeros((H, W)), region).shape == (50, 90)
    assert crop_size(region, W, H) == (90, 50)
    assert crop_size(R(0, 0, blocks=[(0, 0, 5, 5)]), W, H) == (W, H)
    assert crop_to(np.zeros((H, W)), None).shape == (H, W)


def test_region_key_changes_with_the_region():
    a = R(0, 5, blocks=[(0, 0, 10, 10)])
    b = R(0, 5, blocks=[(0, 0, 11, 10)])
    assert region_key(None) is None
    assert region_key(a) != region_key(b) and region_key(a) == region_key(R(0, 9, blocks=[(0, 0, 10, 10)]))


@pytest.mark.parametrize("regions, message", [
    ([R(0, 5), R(5, 8)], "overlaps"),
    ([R(3, 2)], "start"),
    ([R(0, N)], "frame"),
    ([R(0, 1, blocks=[(0, 0, W + 1, 10)])], "outside the frame"),
    ([R(0, 1, blocks=[(5, 5, 5, 10)])], "empty"),
    ([R(0, 1, crop=(0, 0, 31, 40))], "32"),
    ([R(0, 1, crop=(0, 0, 40, 40), blocks=[(50, 0, 60, 10)])], "outside the crop"),
])
def test_validation_rejects_and_names_the_range(regions, message):
    with pytest.raises(BadRequest, match=message) as e:
        validate_regions(regions, W, H, N)
    assert "region" in str(e.value)


def test_validation_accepts_good_regions():
    validate_regions([R(0, 3, crop=(0, 0, 40, 40), blocks=[(30, 30, 50, 50)]), R(4, 11)], W, H, N)


def test_a_hand_edited_invalid_file_still_loads_and_check_names_the_clip(store, make_clip):
    project, clip = make_clip()
    path = store.clip_dir(clip.id) / "clip.json"
    raw = json.loads(path.read_text())
    raw["regions"] = [{"start": 0, "end": 5, "crop": None, "blocks": []}, {"start": 3, "end": 8, "crop": None, "blocks": []}]
    path.write_text(json.dumps(raw))
    assert [c.id for c in store.list_clips(project.id)] == [clip.id]
    with pytest.raises(BadRequest, match=f"clip {clip.id}: region 3–8 overlaps"):
        check_regions(store.read_clip(clip.id))
