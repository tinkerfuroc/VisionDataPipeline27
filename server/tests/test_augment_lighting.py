from __future__ import annotations

import numpy as np
import pytest
from vdp.augment.lighting import relight


def _scene(seed=0):
    """A mid-grey scene with a bright patch and a dark patch, like a real frame's range."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(120, 25, (120, 160, 3)), 0, 255).astype(np.uint8)
    img[10:40, 10:60] = 235
    img[70:110, 90:150] = 25
    return img


def _luma(img):
    return img.astype(np.float32).mean(axis=2)


def test_relight_keeps_shape_and_dtype():
    img = _scene()
    for kind in ("low", "over"):
        out = relight(img, np.random.default_rng(1), kind)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_low_light_darkens_and_adds_grain():
    img = _scene()
    for seed in range(10):
        out = relight(img, np.random.default_rng(seed), "low")
        assert _luma(out).mean() < 0.85 * _luma(img).mean()
        # a flat patch of the source picks up sensor noise
        assert out[15:35, 15:55].std() > img[15:35, 15:55].std()


def test_over_exposure_brightens_and_clips_highlights():
    img = _scene()
    for seed in range(10):
        out = relight(img, np.random.default_rng(seed), "over")
        assert _luma(out).mean() > 1.1 * _luma(img).mean()
        assert (out[10:40, 10:60] == 255).all()  # the bright patch blows out
        assert (out < 255).any()  # mild: not a white frame


def test_relight_is_deterministic_per_seed():
    img = _scene()
    a = relight(img, np.random.default_rng(7), "low")
    b = relight(img, np.random.default_rng(7), "low")
    c = relight(img, np.random.default_rng(8), "low")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_warm_cast_follows_channel_order():
    grey = np.full((32, 32, 3), 128, np.uint8)
    warm_seed = next(s for s in range(100) if _is_warm(relight(grey, np.random.default_rng(s), "low")))
    rgb = relight(grey, np.random.default_rng(warm_seed), "low")
    bgr = relight(grey, np.random.default_rng(warm_seed), "low", bgr=True)
    # the cast lands on red in either order (the noise differs per channel position, so not pixel-exact)
    rgb_means = rgb.reshape(-1, 3).mean(axis=0)
    bgr_means = bgr.reshape(-1, 3).mean(axis=0)
    assert np.allclose(rgb_means, bgr_means[::-1], atol=1.0)


def _is_warm(rgb):
    means = rgb.reshape(-1, 3).mean(axis=0)
    return means[0] > means[2] + 1.0


def test_relight_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        relight(_scene(), np.random.default_rng(0), "dusk")
