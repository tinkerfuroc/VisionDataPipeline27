"""Run independent work units across a spawn-based process pool, results in unit order."""
from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import TypeVar

from ..export import ExportCancelled

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 4)


def run_units(
    fn: Callable[[T], R], units: Sequence[T], *, workers: int | None, weight: Callable[[T], int] = lambda u: 1,
    on_done: Callable[[int], None] = lambda n: None, cancel: Callable[[], bool] = lambda: False,
    initializer: Callable[..., None] | None = None, initargs: tuple = (),
) -> list[R]:
    """Run `fn(unit)` for every unit; return results in unit order. `fn`/`initializer` must be module-level
    (picklable). One worker (or no units) runs inline. `on_done(weight(unit))` follows each unit; `cancel()` is
    checked between results, and a True drops queued units and raises ExportCancelled. A unit's exception
    propagates once the pool is shut down."""
    n = min(workers or default_workers(), len(units))
    if n <= 1:
        if initializer:
            initializer(*initargs)
        results = []
        for unit in units:
            if cancel():
                raise ExportCancelled()
            results.append(fn(unit))
            on_done(weight(unit))
        return results
    # spawn, not fork: the server runs jobs in threads and may hold a CUDA context
    pool = ProcessPoolExecutor(
        max_workers=n, mp_context=multiprocessing.get_context("spawn"), initializer=initializer, initargs=initargs,
    )
    try:
        futures = {pool.submit(fn, unit): i for i, unit in enumerate(units)}
        results: list = [None] * len(units)
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                results[futures[fut]] = fut.result()  # raises the unit's error
                on_done(weight(units[futures[fut]]))
            if pending and cancel():  # a cancel after the last unit finished is too late
                raise ExportCancelled()
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()
    return results
