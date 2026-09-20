"""Request-scoped access to the application state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Request

from ..config import Settings
from ..engine.types import EngineNotLoaded
from ..jobs import JobManager
from ..store import ProjectStore

GPU = "gpu"  # exclusive key: one GPU job at a time


@dataclass
class AppState:
    settings: Settings
    store: ProjectStore
    engine: Any  # an Engine; Sam3Engine additionally has load(), device, load_error
    jobs: JobManager


def get_state(request: Request) -> AppState:
    return request.app.state.vsp


State = Annotated[AppState, Depends(get_state)]


def require_engine(state: AppState):
    if not state.engine.is_loaded():
        reason = getattr(state.engine, "load_error", None) or "the SAM3 engine is still loading"
        raise EngineNotLoaded(reason)
    return state.engine


def gpu_lease(state: AppState):
    """An interactive hold on the GPU: excludes `POST /propagate` from starting for as long as the
    `with` body runs, atomically with the engine call inside it. Fails fast with 409 (`job_id`) if a
    propagation job already holds the GPU, instead of blocking a worker thread on the engine lock.
    """
    return state.jobs.lease(GPU)
