"""FastAPI application factory. Tests inject FakeEngine; `vdp serve` builds the real Sam3Engine."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..auth import AuthStore
from ..config import REPO_ROOT, Settings
from ..engine.types import EngineNotLoaded
from ..ingest import IngestError, UploadTooLarge, clear_uploads
from ..ingest.resumable import OffsetConflict, ResumableStore, UploadNotFound
from ..jobs import JobBusy, JobManager, JobNotFound
from ..labeling.errors import BadRequest, Conflict, PropagationFailed
from ..store import NotFound, ProjectStore, StoreError
from ..train.scheduler import TrainScheduler, detect_gpus
from . import augment as augment_api
from . import clips, export, jobs, labeling, projects
from . import engine as engine_api
from . import gate as gate_api
from . import train as train_api
from . import uploads as uploads_api
from .deps import AppState, State

log = logging.getLogger(__name__)

# Job kinds that shutdown waits for once cancelled: they stop within a work unit and then delete their
# half-written export, which a daemon job thread killed at interpreter exit would leave behind (and an augment
# job's pool would first drain every queued unit in concurrent.futures' exit hook).
SHUTDOWN_JOIN_KINDS = frozenset({augment_api.AUGMENT, "export"})
SHUTDOWN_JOIN_S = 30.0

_STATUS = [  # most specific first
    (UploadNotFound, 404), (NotFound, 404), (JobNotFound, 404), (Conflict, 409), (EngineNotLoaded, 503),
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


def create_app(
    settings: Settings, *, engine=None, load_engine: bool = True, train=None, auth_store: AuthStore | None = None,
) -> FastAPI:
    state = AppState(
        settings=settings,
        store=ProjectStore(settings.project.data_root),
        engine=engine if engine is not None else _build_engine(settings),
        jobs=JobManager(),
        train=train if train is not None else TrainScheduler(settings.project.data_root, detect_gpus()),
        uploads=ResumableStore(settings.project.data_root / "uploads" / "resumable", settings.ingest.max_upload_bytes),
    )
    state.engine_ctl = engine_api.EngineControl(state.engine, state.jobs, state.train)
    state.train.on_gpu_idle = state.engine_ctl.gpu_idle  # reloads SAM3 after a training pause
    gate = gate_api.GateState(auth_store if auth_store is not None else AuthStore(settings.project.data_root))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_engine:
            state.engine_ctl.start_load()
        state.train.start()
        yield
        deadline = time.monotonic() + SHUTDOWN_JOIN_S
        for job in state.jobs.cancel_all():
            if job.kind in SHUTDOWN_JOIN_KINDS:
                job.join(max(0.0, deadline - time.monotonic()))
        state.train.shutdown()

    app = FastAPI(title="VisionDataPipeline27", version=__version__, lifespan=lifespan)
    app.state.vdp = state
    app.state.gate = gate
    app.add_middleware(gate_api.AuthGate, state=gate)

    def _handler(status: int):
        async def handle(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc.args[0]) if exc.args else str(exc)})

        return handle

    for exc_type, status in _STATUS:
        app.add_exception_handler(exc_type, _handler(status))

    @app.exception_handler(JobBusy)
    async def _busy(request: Request, exc: JobBusy) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "job_id": exc.job_id})

    @app.exception_handler(OffsetConflict)
    async def _offset(request: Request, exc: OffsetConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "received": exc.received})

    @app.get("/api/health")
    def health(st: State, request: Request) -> dict:
        eng = st.engine
        return {
            "status": "ok",
            "version": __version__,
            "engine": {
                "loaded": bool(eng.is_loaded()),
                "device": getattr(eng, "device", None),
                "error": getattr(eng, "load_error", None),
                "paused": st.engine_ctl.paused,
            },
            "upload": {"max_bytes": st.settings.ingest.max_upload_bytes},
            "auth": {"enabled": request.app.state.gate.enabled()},
        }

    app.include_router(projects.router)
    app.include_router(clips.router)
    app.include_router(labeling.router)
    app.include_router(export.router)
    app.include_router(train_api.router)
    app.include_router(augment_api.router)
    app.include_router(jobs.router)
    app.include_router(engine_api.router)
    app.include_router(gate_api.router)
    app.include_router(uploads_api.router)

    removed = clear_uploads(state.settings.project.data_root)  # litter from a previous process; see clear_uploads
    if removed:
        log.info("removed %d stale upload(s) from a previous run", removed)
    state.uploads.drop_orphaned_completions()  # their import jobs died with the previous process
    state.uploads.sweep()

    dist = REPO_ROOT / "web" / "dist"  # built by Plan 3; absent until then
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="spa")
    return app
