"""Loading the SAM3 engine in the background, and pausing it (freeing its VRAM) while training runs on its GPU.

The pause lives in memory only: after a restart the engine loads at startup as usual. Lock order is this
module's lock, then the scheduler's (a paused start holds ours across `TrainScheduler.submit`; the idle
listener holds ours while it asks `gpu_idle`), so a run queued by a paused start can never be missed by
an idle report racing it.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter

from ..engine.types import EngineNotLoaded
from .deps import GPU, State

log = logging.getLogger(__name__)
router = APIRouter(tags=["engine"])
T = TypeVar("T")


def sam3_gpu(engine) -> int | None:
    device = getattr(engine, "device", None)
    if isinstance(device, str) and device.startswith("cuda:"):
        with contextlib.suppress(ValueError):
            return int(device.removeprefix("cuda:"))
    return None


class EngineControl:
    def __init__(self, engine, jobs, train) -> None:
        self.engine = engine
        self.jobs = jobs
        self.train = train
        self._lock = threading.Lock()
        self._loading = False
        self.paused_gpu: int | None = None  # SAM3's GPU while it is paused for training there

    @property
    def paused(self) -> bool:
        return self.paused_gpu is not None

    def start_load(self) -> bool:
        """Loads the engine on a background thread and ends any pause. A no-op (False) when it is
        already loaded or loading, or cannot load at all (a test engine without `load`)."""
        with self._lock:
            return self._start_load_locked()

    def _start_load_locked(self) -> bool:
        self.paused_gpu = None
        if self._loading or self.engine.is_loaded() or not hasattr(self.engine, "load"):
            return False
        self._loading = True
        threading.Thread(target=self._load, name="engine-load", daemon=True).start()
        return True

    def _load(self) -> None:
        try:
            with contextlib.suppress(EngineNotLoaded):  # the reason is kept in engine.load_error for /api/health
                self.engine.load()
        finally:
            with self._lock:
                self._loading = False

    def start_training(self, pause: bool, gpus: set[int], submit: Callable[[Callable[[], None] | None], T]) -> T:
        """Runs `submit(before_queue)` -- a TrainScheduler request -- releasing the engine first when
        `pause` asks for it and any of `gpus` is SAM3's. The release happens in `before_queue`, once the
        request has been validated (a bad request never unloads SAM3), and fails with JobBusy (409),
        queueing nothing, while a propagation job holds the GPU: the lease also keeps one from starting
        mid-release. `pause` is ignored when no run goes to SAM3's GPU or the engine is not loaded."""
        if not pause:
            return submit(None)
        with self._lock:
            gpu = sam3_gpu(self.engine)
            if gpu is None or gpu not in gpus or not self.engine.is_loaded() or not hasattr(self.engine, "release"):
                return submit(None)
            released = False

            def release() -> None:
                nonlocal released
                with self.jobs.lease(GPU):  # a propagation holds the engine lock for its whole run
                    self.engine.release()
                released = True
                self.paused_gpu = gpu

            try:
                return submit(release)
            except BaseException:
                if released:  # released, then nothing was queued after all: bring SAM3 back now
                    self._start_load_locked()
                raise

    def gpu_idle(self, gpu: int) -> None:
        """`TrainScheduler.on_gpu_idle`: reloads SAM3 once nothing is queued or running on the GPU it was
        paused for (re-checked here, under our lock: the report may be stale)."""
        with self._lock:
            if self.paused_gpu == gpu and self.train.gpu_idle(gpu):
                log.info("no training left on cuda:%d; reloading SAM3", gpu)
                self._start_load_locked()


@router.post("/api/engine/load", status_code=202)
def load_engine(state: State) -> dict:
    """Loads SAM3 now, ending a training pause (the training on its GPU then shares its memory again)."""
    return {"started": state.engine_ctl.start_load()}
