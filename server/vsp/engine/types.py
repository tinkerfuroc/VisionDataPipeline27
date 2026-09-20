"""The contract between the labeling layer and any segmentation engine."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

Point = tuple[int, int, int]  # (x, y, label); label 1 = positive, 0 = negative


@dataclass
class Detection:
    mask: np.ndarray  # bool, HxW
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (max exclusive)
    score: float
    phrase: str | None = None


class ChunkOOMError(RuntimeError):
    """CUDA ran out of memory while propagating a chunk; retry with fewer frames."""


class EngineNotLoaded(RuntimeError):
    pass


class Engine(Protocol):
    def is_loaded(self) -> bool: ...

    def predict_text(
        self, image_bgr: np.ndarray, phrase: str, *, frame_key: tuple | None = None
    ) -> list[Detection]: ...

    def predict_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        points: Sequence[Point] = (),
        box: tuple[int, int, int, int] | None = None,
        frame_key: tuple | None = None,
    ) -> Detection | None: ...

    def propagate(
        self,
        frames: list[np.ndarray],
        seed_mask: np.ndarray,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        """Track `seed_mask` (given for `frames[0]`) forward through `frames`.

        Implementations are generators: argument validation and all work happen lazily, starting
        on the first `next()` call, not when `propagate()` is called. An implementation backed by
        a single GPU/model resource typically holds a non-reentrant lock from the first `next()`
        until the generator is exhausted or closed -- callers MUST exhaust it or call `.close()`
        (e.g. wrap consumption in `contextlib.closing`). While such a generator is open, calling
        another method on the same engine from the same thread can deadlock; from another thread
        it blocks until the generator finishes or is closed.
        """
        ...
