"""Run independent work units across a spawn-based process pool, results in unit order."""
from __future__ import annotations

import contextlib
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import TypeVar

from ..export import ExportCancelled

T = TypeVar("T")
R = TypeVar("R")


WORKER_NICE = 10  # pool workers yield the CPU to a concurrent training run
PARENT_POLL_S = 1.0
CANCEL_POLL_S = 0.5


def default_workers() -> int:
    return max(1, len(os.sched_getaffinity(0)) - 4)


def _exit_when_orphaned(parent_pid: int) -> None:
    # a SIGKILLed parent never closes the call queue, so a spawn worker would otherwise wait on it forever
    while os.getppid() == parent_pid:
        time.sleep(PARENT_POLL_S)
    os._exit(1)


def _init_worker(parent_pid: int, initializer: Callable[..., None] | None, initargs: tuple) -> None:
    """Every pool worker: exit once the parent is gone, run at lower priority with one OpenCV thread (n workers
    times OpenCV's default of one thread per core would oversubscribe the CPU), then the caller's initializer."""
    threading.Thread(target=_exit_when_orphaned, args=(parent_pid,), name="parent-watch", daemon=True).start()
    with contextlib.suppress(Exception):
        import cv2

        cv2.setNumThreads(1)
    with contextlib.suppress(OSError):
        os.nice(max(0, WORKER_NICE - os.nice(0)))
    if initializer:
        initializer(*initargs)


def run_units(
    fn: Callable[[T], R], units: Sequence[T], *, workers: int | None, weight: Callable[[T], int] = lambda u: 1,
    on_done: Callable[[int], None] = lambda n: None, cancel: Callable[[], bool] = lambda: False,
    initializer: Callable[..., None] | None = None, initargs: tuple = (), deadline: float | None = None,
) -> list[R | None]:
    """Run `fn(unit)` for every unit; return results in unit order. `fn`/`initializer` must be module-level
    (picklable). One worker (or no units) runs inline. `on_done(weight(unit))` follows each unit; `cancel()` is
    checked between results and every CANCEL_POLL_S while units run; a True drops queued units, waits for the
    running ones and raises ExportCancelled. A unit's exception propagates once the pool is shut down. Pool
    workers exit if this process dies, and run niced with one OpenCV thread (see `_init_worker`).
    `deadline` (a time.monotonic() value) is a soft stop: once reached, queued units are dropped, running ones
    finish and are kept, and the result has None for every dropped unit; no exception. Cancel wins over it."""
    n = min(workers or default_workers(), len(units))
    if n <= 1:
        if initializer:
            initializer(*initargs)
        results = []
        for unit in units:
            if cancel():
                raise ExportCancelled()
            if deadline is not None and time.monotonic() >= deadline:
                results.extend([None] * (len(units) - len(results)))
                break
            results.append(fn(unit))
            on_done(weight(unit))
        return results
    # spawn, not fork: the server runs jobs in threads and may hold a CUDA context
    pool = ProcessPoolExecutor(
        max_workers=n, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker,
        initargs=(os.getpid(), initializer, initargs),
    )
    try:
        futures = {pool.submit(fn, unit): i for i, unit in enumerate(units)}
        results: list = [None] * len(units)
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=CANCEL_POLL_S, return_when=FIRST_COMPLETED)
            for fut in finished:
                results[futures[fut]] = fut.result()  # raises the unit's error
                on_done(weight(units[futures[fut]]))
            if pending and cancel():  # a cancel after the last unit finished is too late
                raise ExportCancelled()
            if pending and deadline is not None and time.monotonic() >= deadline:
                pool.shutdown(wait=True, cancel_futures=True)  # running units finish
                for fut in pending:
                    if not fut.cancelled():
                        results[futures[fut]] = fut.result()
                        on_done(weight(units[futures[fut]]))
                break
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()
    return results
