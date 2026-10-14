"""Albumentations wrapper around `vdp.augment.lighting.relight`, for the training worker.

Kept in its own importable module (not in `worker.py`, which runs as `__main__`): albumentations registers a
transform under its module path, and Ultralytics rebuilds the run's transforms from `A.to_dict` dicts with
`A.from_dict`, also on Resume, so the class must be found under a stable name.
"""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # never phone home for a version check

import albumentations as A
import numpy as np

from ..augment.lighting import LOW_SHARE, relight

LIGHTING_P = 0.3


class Lighting(A.ImageOnlyTransform):
    """Low light (share `low_share`) or mild over-exposure. Ultralytics hands albumentations BGR images."""

    def __init__(self, low_share: float = LOW_SHARE, bgr: bool = True, p: float = LIGHTING_P):
        super().__init__(p=p)
        self.low_share = low_share
        self.bgr = bgr

    def get_params(self) -> dict:
        kind = "low" if self.py_random.random() < self.low_share else "over"
        return {"kind": kind, "seed": int(self.random_generator.integers(2**31))}

    def apply(self, img: np.ndarray, kind: str = "low", seed: int = 0, **params) -> np.ndarray:
        return relight(img, np.random.default_rng(seed), kind, bgr=self.bgr)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("low_share", "bgr")
