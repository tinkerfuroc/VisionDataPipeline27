"""Background pools and augmented exports (spec B1, B2)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..augment.backgrounds import POOLS, FetchCancelled, fetch_pool, list_images, pool_dir
from ..jobs import Job
from .deps import State

router = APIRouter(tags=["augment"])
BACKGROUNDS = "backgrounds"  # job kind and exclusive key: one download at a time


class FetchBody(BaseModel):
    force: bool = False


@router.get("/api/backgrounds")
def backgrounds(state: State) -> list[dict]:
    root = state.settings.project.data_root
    fetching = state.jobs.active(BACKGROUNDS)
    return [{"name": n, "label": p["label"], "images": len(list_images(root, n)), "size_bytes": p["size"],
             "fetching": fetching} for n, p in POOLS.items()]


@router.post("/api/backgrounds/{name}/fetch", status_code=202)
def fetch(name: str, state: State, body: FetchBody | None = None) -> dict:
    root = state.settings.project.data_root
    pool_dir(root, name)  # 404 for an unknown pool, before a job exists
    force = body.force if body else False

    def emit_progress(phase: str, done: int, total: int, job: Job) -> None:
        job.emit({"type": "progress", "phase": phase, "done": done, "total": total})

    def work(job: Job) -> dict:
        try:
            return fetch_pool(root, name, force=force, cancel=job.cancel_requested,
                              progress=lambda ph, d, t: emit_progress(ph, d, t, job))
        except FetchCancelled:
            return {"cancelled": True}

    return {"job_id": state.jobs.start(BACKGROUNDS, work, exclusive=BACKGROUNDS).id}
