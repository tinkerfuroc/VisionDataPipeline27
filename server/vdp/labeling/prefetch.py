"""Read and preprocess the next propagation chunk on a CPU thread while the current one is on the GPU."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import Self

import numpy as np

from ..engine.types import Engine, PreparedChunk
from ..store import Clip, ProjectStore
from .prompts import load_frame

Key = tuple[int, tuple[int, ...]]  # (anchor frame, chunk frames)


class ChunkPrefetcher:
    """One chunk ahead: `schedule()` the chunk the loop expects to need next, `take()` the one it actually needs.

    After `take()` the cache holds only the chunk just taken; after `schedule()` it holds that chunk
    plus the one now loading in the background. A request that does not match the schedule (an OOM
    retry on a prefix of the same frames, an anchor that is not the chunk's last frame) first waits
    for the in-flight schedule to finish -- there is one worker and the cache must be quiescent --
    so a mispredicted chunk costs one extra chunk read and prepare before its own; it is then built
    synchronously, reading only the frames it does not already hold, so only the preprocessing is
    redone. Frames on disk never change while a clip exists (deletion is soft), so cached pixels
    cannot go stale.

    Use as a context manager: `close()` joins the worker, so no thread outlives the propagation run.
    """

    def __init__(self, store: ProjectStore, clip: Clip, engine: Engine) -> None:
        self._store, self._clip, self._engine = store, clip, engine
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vdp-prefetch")
        self._lock = threading.Lock()  # guards _frames and reads; the worker writes both
        self._frames: dict[int, np.ndarray] = {}
        self._taken: Key | None = None
        self._scheduled: tuple[Key, Future[PreparedChunk]] | None = None
        self.reads = 0

    # ------------------------------------------------------------------ API
    def schedule(self, anchor_idx: int, indices: list[int]) -> None:
        """Start loading and preparing `[anchor] + indices` in the background. Replaces any earlier schedule."""
        self._settle()
        key: Key = (anchor_idx, tuple(indices))
        taken = {self._taken[0], *self._taken[1]} if self._taken is not None else set()
        self._retain(taken | {anchor_idx, *indices})
        self._scheduled = (key, self._pool.submit(self._build, key))

    def take(self, anchor_idx: int, indices: list[int]) -> PreparedChunk:
        """Return the prepared chunk `[anchor] + indices`, from the schedule when it matches, else built now."""
        key: Key = (anchor_idx, tuple(indices))
        scheduled = self._scheduled
        self._scheduled = None
        if scheduled is not None and scheduled[0] == key:
            future = scheduled[1]
        else:
            if scheduled is not None:
                with suppress(Exception):  # a failed load of a chunk nobody needs is not an error
                    scheduled[1].result()
            future = None
        self._taken = key
        result = future.result() if future is not None else self._build(key)
        self._retain({anchor_idx, *indices})  # keep only this chunk: take() owns the cache until scheduled
        return result

    def close(self) -> None:
        self._settle()
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._frames.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def cached(self) -> list[int]:
        with self._lock:
            return sorted(self._frames)

    # ------------------------------------------------------------ internals
    def _build(self, key: Key) -> PreparedChunk:
        anchor_idx, indices = key
        frames = [self._load(anchor_idx)] + [self._load(i) for i in indices]
        return self._engine.prepare(frames)

    def _load(self, idx: int) -> np.ndarray:
        with self._lock:
            frame = self._frames.get(idx)
        if frame is None:
            frame = load_frame(self._store, self._clip, idx)
            with self._lock:
                self._frames[idx] = frame
                self.reads += 1
        return frame

    def _retain(self, wanted: set[int]) -> None:
        with self._lock:
            for idx in [i for i in self._frames if i not in wanted]:
                del self._frames[idx]

    def _settle(self) -> None:
        """Wait for a still-running schedule so the cache is quiescent before it is edited or the pool closed."""
        if self._scheduled is not None:
            with suppress(Exception):
                self._scheduled[1].result()
            self._scheduled = None
