"""The contract between the labeling layer and any segmentation engine."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

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


# Engines dispatch with isinstance on their own concrete prepared-chunk class, so this alias documents
# intent rather than constraining the type.
PreparedChunk: TypeAlias = Any
"""Opaque, engine-specific output of `Engine.prepare()`; only the engine that produced it reads it."""


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

    def prepare(self, frames: list[np.ndarray]) -> PreparedChunk:
        """CPU-only preprocessing of a chunk for `propagate()`.

        Lock-free and thread-safe: it may run on another thread while a `propagate()` generator
        from this engine is open, so a caller can prepare the next chunk while the current one is
        on the GPU. Never touches the GPU. The result is opaque; pass it to `propagate()` in place
        of the raw frame list.
        """
        ...

    def propagate(
        self,
        frames: list[np.ndarray] | PreparedChunk,
        seeds: Mapping[int, np.ndarray],
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, dict[int, np.ndarray | None]]]:
        """Track every seed (object id -> bool mask, all given for `frames[0]`) forward through `frames`.

        Yields `(offset, {object_id: mask or None})` with exactly the seeded ids as keys; None means no usable
        mask for that object on that frame. At least one seed; each non-empty and of the frames' shape, else
        ValueError.

        Implementations are generators: argument validation and all work happen lazily, starting
        on the first `next()` call, not when `propagate()` is called. `frames` is either the raw
        BGR frames or the result of `prepare()` on them.

        An implementation backed by a single GPU/model resource typically holds a non-reentrant lock
        from the first `next()` until the generator is exhausted or closed -- callers MUST exhaust it
        or call `.close()` (e.g. wrap consumption in `contextlib.closing`). While such a generator is
        open, calling another method on the same engine from the same thread can deadlock; from
        another thread it blocks until the generator finishes or is closed. The one exception is
        `prepare()`, which never blocks on the lock.
        """
        ...
