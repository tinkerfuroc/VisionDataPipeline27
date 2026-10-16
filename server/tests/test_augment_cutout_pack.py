from __future__ import annotations

import numpy as np
from vdp.augment.composite import Cutout
from vdp.augment.cutout_pack import load_cutouts, pack_cutouts


def _cut(rng, class_id, h, w):
    return Cutout(class_id, rng.integers(0, 255, (h, w, 3), dtype=np.uint8), rng.random((h, w)) > 0.5)


def _maps_the_file(a) -> bool:
    """True when `a` is a view whose `.base` chain reaches a memory map of the file (nothing was copied)."""
    while a is not None:
        if isinstance(a, np.memmap) and getattr(a, "_mmap", None) is not None:
            return True
        a = a.base
    return False


def test_pack_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    cuts = [_cut(rng, 2, 5, 7), _cut(rng, 0, 1, 1), _cut(rng, 3, 9, 4)]
    pack_cutouts(cuts, tmp_path)
    assert (tmp_path / "cutouts.bin").is_file() and (tmp_path / "cutouts.json").is_file()
    loaded = load_cutouts(tmp_path)
    assert [c.class_id for c in loaded] == [2, 0, 3]
    for a, b in zip(cuts, loaded, strict=True):
        assert np.array_equal(a.image, b.image) and np.array_equal(a.mask, b.mask)
        assert b.image.dtype == np.uint8 and b.mask.dtype == bool
        assert _maps_the_file(b.image) and _maps_the_file(b.mask)
        assert not b.image.flags.writeable and not b.mask.flags.writeable


def test_pack_and_load_zero_cutouts(tmp_path):
    pack_cutouts([], tmp_path)
    assert load_cutouts(tmp_path) == []
