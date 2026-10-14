"""Lighting simulation: low light and mild over-exposure, for training-time augmentation and augmented copies.

Exposure is changed in (approximately) linear light, as a camera would, not by scaling the encoded pixel values:
the image is decoded with a 2.2 gamma, scaled, and encoded again. Low light also gets sensor noise (shot noise
that grows with the signal, a read-noise floor and some chroma noise) and sometimes a warm tungsten cast, so a
dark frame looks grainy like real low-light footage rather than just dim. Over-exposure clips highlights and
lifts the shadows a little (haze). Only pixel values change, so every label of the frame stays valid.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

LightingKind = Literal["low", "over"]
GAMMA = 2.2
LOW_SHARE = 0.6  # of relit images, this share is darkened; the rest is over-exposed
LOW_EXPOSURE = (0.1, 0.5)  # about 1 to 3.3 stops under
OVER_EXPOSURE = (1.4, 2.2)  # about 0.5 to 1.1 stops over
WARM_CAST_P = 0.5
WARM_CAST = (0.0, 0.15)  # red up, blue down by this fraction
SHOT_NOISE = (0.0005, 0.005)  # variance per unit of linear signal
READ_NOISE = (0.0005, 0.002)  # sigma, linear full scale
CHROMA_NOISE = 0.5  # per-channel noise relative to the shared luma noise
HAZE = (0.0, 0.03)  # linear lift of the shadows when over-exposed
_DECODE = ((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA).astype(np.float32)


def relight(image: np.ndarray, rng: np.random.Generator, kind: LightingKind, *, bgr: bool = False) -> np.ndarray:
    """`image` (H×W×3 uint8, RGB, or BGR with `bgr=True`) as if shot in low light or mildly over-exposed."""
    if kind not in ("low", "over"):
        raise ValueError(f"unknown lighting kind {kind!r}; expected 'low' or 'over'")
    lin = _DECODE[image]
    if kind == "low":
        lin *= rng.uniform(*LOW_EXPOSURE)
        if rng.random() < WARM_CAST_P:
            warm = rng.uniform(*WARM_CAST)
            gains = np.array([1.0 + warm, 1.0, 1.0 - warm], np.float32)
            lin *= gains[::-1] if bgr else gains
        sigma = np.sqrt(rng.uniform(*SHOT_NOISE) * lin + rng.uniform(*READ_NOISE) ** 2)
        h, w = lin.shape[:2]
        noise = rng.standard_normal((h, w, 1), dtype=np.float32)
        noise = noise + CHROMA_NOISE * rng.standard_normal((h, w, 3), dtype=np.float32)
        lin += sigma * noise
    else:
        lin = lin * rng.uniform(*OVER_EXPOSURE) + rng.uniform(*HAZE)
    np.clip(lin, 0.0, 1.0, out=lin)
    return np.rint(lin ** (1.0 / GAMMA) * 255.0).astype(np.uint8)
