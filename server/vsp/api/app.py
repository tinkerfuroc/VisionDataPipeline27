"""FastAPI application factory. Tests inject FakeEngine; `vsp serve` builds the real Sam3Engine."""

from __future__ import annotations

import contextlib
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import REPO_ROOT, Settings
from ..engine.types import EngineNotLoaded
from ..ingest import IngestError, UploadTooLarge, clear_uploads
from ..jobs import JobBusy, JobManager, JobNotFound
from ..labeling.errors import BadRequest, Conflict, PropagationFailed
from ..store import NotFound, ProjectStore, StoreError
from ..train.scheduler import TrainScheduler, detect_gpus
from . import clips, export, jobs, labeling, projects
from . import train as train_api
from .deps import AppState, State

log = logging.getLogger(__name__)

_STATUS = [  # most specific first
    (NotFound, 404), (JobNotFound, 404), (Conflict, 409), (EngineNotLoaded, 503),
    (train_api.TrainUnavailable, 503), (PropagationFailed, 500),
    (UploadTooLarge, 413), (BadRequest, 400), (IngestError, 400), (StoreError, 400),
]


def _build_engine(settings: Settings):
    from ..engine.sam3 import Sam3Engine  # imports torch; keep it out of test collection paths

    cfg = settings.sam3
    return Sam3Engine(
        cfg.model_dir, device=cfg.device, dtype=cfg.dtype,
        text_threshold=cfg.text_threshold, score_mode=cfg.score_mode,
    )


def create_app(settings: Settings, *, engine=None, load_engine: bool = True, train=None) -> FastAPI:
    state = AppState(
        settings=settings,
        store=ProjectStore(settings.project.data_root),
        engine=engine if engine is not None else _build_engine(settings),
        jobs=JobManager(),
        train=train if train is not None else TrainScheduler(settings.project.data_root, detect_gpus()),
    )

    def _load() -> None:
        with contextlib.suppress(EngineNotLoaded):  # the reason is kept in engine.load_error for /api/health
            state.engine.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_engine and hasattr(state.engine, "load"):
            threading.Thread(target=_load, name="engine-load", daemon=True).start()
        state.train.start()
        yield
        state.train.shutdown()

    app = FastAPI(title="vision-sampler", version=__version__, lifespan=lifespan)
    app.state.vsp = state

    def _handler(status: int):
        async def handle(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc.args[0]) if exc.args else str(exc)})

        return handle

    for exc_type, status in _STATUS:
        app.add_exception_handler(exc_type, _handler(status))

    @app.exception_handler(JobBusy)
    async def _busy(request: Request, exc: JobBusy) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "job_id": exc.job_id})

    @app.get("/api/health")
    def health(st: State) -> dict:
        eng = st.engine
        return {
            "status": "ok",
            "version": __version__,
            "engine": {
                "loaded": bool(eng.is_loaded()),
                "device": getattr(eng, "device", None),
                "error": getattr(eng, "load_error", None),
            },
            "upload": {"max_bytes": st.settings.ingest.max_upload_bytes},
        }

    app.include_router(projects.router)
    app.include_router(clips.router)
    app.include_router(labeling.router)
    app.include_router(export.router)
    app.include_router(train_api.router)
    app.include_router(jobs.router)

    removed = clear_uploads(state.settings.project.data_root)  # litter from a previous process; see clear_uploads
    if removed:
        log.info("removed %d stale upload(s) from a previous run", removed)

    dist = REPO_ROOT / "web" / "dist"  # built by Plan 3; absent until then
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="spa")
    return app
