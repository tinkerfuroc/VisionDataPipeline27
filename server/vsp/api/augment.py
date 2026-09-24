"""Background pools and augmented exports (spec B1, B2)."""

from __future__ import annotations

import random
from pathlib import Path

from fastapi import APIRouter, Query
from PIL import Image
from pydantic import BaseModel, Field

from ..augment.backgrounds import POOLS, FetchCancelled, fetch_pool, list_images, pool_dir
from ..augment.derive import AugmentOptions, check_augment, run_augment
from ..export import ExportCancelled
from ..jobs import Job
from ..train.models import find_export
from ..train.results import read_ground_truth
from .deps import State

router = APIRouter(tags=["augment"])
BACKGROUNDS = "backgrounds"  # job kind and exclusive key: one download at a time
AUGMENT = "augment"  # job kind


class FetchBody(BaseModel):
    force: bool = False


class AugmentBody(BaseModel):
    per_image: int = Field(default=3, ge=1, le=10)
    max_objects: int = Field(default=2, ge=1, le=3)
    seed: int = Field(default=0, ge=0)
    pool: str = "dtd"


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


@router.post("/api/exports/{export_id}/augment", status_code=202)
def augment(export_id: str, state: State, body: AugmentBody | None = None) -> dict:
    b = body or AugmentBody()
    opts = AugmentOptions(per_image=b.per_image, max_objects=b.max_objects, seed=b.seed, pool=b.pool)
    data_root = state.settings.project.data_root
    check_augment(data_root, export_id, opts)  # 400 / 404 now, not as a job error

    def work(job: Job) -> dict:
        try:
            return run_augment(
                data_root, export_id, opts,
                progress=lambda done, total: job.emit({"type": "progress", "done": done, "total": total}),
                cancel=job.cancel_requested,
            )
        except ExportCancelled:
            return {"cancelled": True}

    return {"job_id": state.jobs.start(AUGMENT, work).id}


@router.get("/api/exports/{export_id}/samples")
def samples(
    export_id: str, state: State, limit: int = Query(12, ge=1, le=48), seed: int = Query(0),
) -> list[dict]:
    export = find_export(state.settings.project.data_root, export_id)
    images_dir = export.yolo_dir / "images" / "train"
    labels_dir = export.yolo_dir / "labels" / "train"
    names = sorted(p.name for p in images_dir.glob("syn_*.jpg"))
    if not names:
        names = sorted(p.name for p in images_dir.glob("*.jpg"))
    chosen = random.Random(seed).sample(names, min(limit, len(names)))
    out = []
    for name in chosen:
        with Image.open(images_dir / name) as img:
            width, height = img.size
        out.append({
            "name": name, "width": width, "height": height,
            "ground_truth": read_ground_truth(labels_dir / f"{Path(name).stem}.txt"),
        })
    return out
