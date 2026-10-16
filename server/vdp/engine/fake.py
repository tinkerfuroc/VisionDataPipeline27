"""Deterministic Engine for tests: masks are rectangles, tracking is 'shift right by dx per frame'."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np

from .masks import bbox_of, is_degenerate_mask
from .types import ChunkOOMError, Detection, Point, PreparedChunk


class FakeEngine:
    def __init__(
        self,
        dx: int = 2,
        n_text_candidates: int = 2,
        oom_if_more_than: int | None = None,
        is_empty: Callable[[np.ndarray], bool] | None = None,
        is_empty_obj: Callable[[np.ndarray, int], bool] | None = None,
        max_area_frac: float = 0.45,
        min_area_px: int = 64,
    ) -> None:
        self.dx = dx
        self.n_text_candidates = n_text_candidates
        self.oom_if_more_than = oom_if_more_than
        self.is_empty = is_empty
        self.is_empty_obj = is_empty_obj
        self.max_area_frac = max_area_frac
        self.min_area_px = min_area_px
        self.prepare_calls: list[int] = []
        self.propagate_calls: list[int] = []
        self.seed_calls: list[list[int]] = []
        self.images: list[np.ndarray] = []  # every image predict_text / predict_prompt was given
        self.frame_keys: list[tuple | None] = []

    def _is_degenerate(self, mask: np.ndarray) -> bool:
        return is_degenerate_mask(mask, max_area_frac=self.max_area_frac, min_area_px=self.min_area_px)

    def is_loaded(self) -> bool:
        return True

    def predict_text(self, image_bgr: np.ndarray, phrase: str, *, frame_key: tuple | None = None) -> list[Detection]:
        self.images.append(image_bgr)
        self.frame_keys.append(frame_key)
        h, w = image_bgr.shape[:2]
        dets = []
        for i in range(self.n_text_candidates):
            mask = np.zeros((h, w), dtype=bool)
            x0 = 5 + i * 30
            mask[5:25, x0:x0 + 20] = True
            if self._is_degenerate(mask):
                continue
            dets.append(Detection(mask=mask, bbox=bbox_of(mask), score=0.9 - 0.1 * i, phrase=phrase))
        return dets

    def predict_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        points: Sequence[Point] = (),
        box: tuple[int, int, int, int] | None = None,
        frame_key: tuple | None = None,
    ) -> Detection | None:
        self.images.append(image_bgr)
        self.frame_keys.append(frame_key)
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        positives = [(x, y) for x, y, lbl in points if lbl == 1]
        if box is not None:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
        elif positives:
            x, y = positives[0]
            mask[max(0, y - 10):y + 10, max(0, x - 10):x + 10] = True
        else:
            return None
        for x, y, lbl in points:
            if lbl == 0:
                mask[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = False
        if not mask.any() or self._is_degenerate(mask):
            return None
        return Detection(mask=mask, bbox=bbox_of(mask), score=0.95)

    def prepare(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        # The fake's "prepared chunk" is the frame list itself, so test doubles that override
        # `propagate` and index `frames` keep working.
        self.prepare_calls.append(len(frames))
        return list(frames)

    def propagate(
        self,
        frames: list[np.ndarray] | PreparedChunk,
        seeds: Mapping[int, np.ndarray],
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, dict[int, np.ndarray | None]]]:
        self.propagate_calls.append(len(frames))
        self.seed_calls.append(sorted(seeds))
        if self.oom_if_more_than is not None and len(frames) > self.oom_if_more_than:
            raise ChunkOOMError(f"fake OOM: {len(frames)} frames > {self.oom_if_more_than}")
        if not seeds:
            raise ValueError("at least one seed is required")
        h, w = frames[0].shape[:2]
        for oid, seed in seeds.items():
            if seed.shape != (h, w):
                raise ValueError(f"seed {oid} shape {seed.shape} does not match frames {(h, w)}")
            if not seed.any():
                raise ValueError(f"seed {oid} must be non-empty")
        for offset, frame in enumerate(frames):
            if cancel_cb is not None and cancel_cb():
                return
            out: dict[int, np.ndarray | None] = {}
            for oid, seed in seeds.items():
                if (self.is_empty is not None and self.is_empty(frame)) or (
                    self.is_empty_obj is not None and self.is_empty_obj(frame, oid)
                ):
                    out[oid] = None
                    continue
                s = offset * self.dx
                shifted = seed.copy() if s == 0 else np.zeros_like(seed)
                if 0 < s < seed.shape[1]:
                    shifted[:, s:] = seed[:, :-s]
                out[oid] = None if self._is_degenerate(shifted) else shifted
            yield offset, out
