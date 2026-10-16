"""POST /api/projects/{id}/export: start an export job (spec §4, §8)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..export import ExportCancelled, ExportOptions, check_request, clean_name, run_export
from ..jobs import Job, JobBusy
from .deps import GPU, State

router = APIRouter(tags=["export"])


class ExportOptionsBody(BaseModel):
    split_mode: Literal["per_clip", "contiguous"] | None = None
    val_clips: list[str] | None = None
    val_fraction: float | None = Field(default=None, ge=0, lt=1)
    stride: int = Field(default=1, ge=1)
    include_empty: bool = True
    min_area_px: int | None = Field(default=None, ge=0)
    classes: list[int] | None = None  # project class ids; null = every class


class ExportBody(BaseModel):
    clips: list[str]
    formats: list[str]
    options: ExportOptionsBody = Field(default_factory=ExportOptionsBody)
    name: str | None = None


@router.post("/api/projects/{project_id}/export", status_code=202)
def start_export(project_id: str, body: ExportBody, state: State) -> dict:
    cfg = state.settings.export
    o = body.options
    options = ExportOptions(
        split_mode=o.split_mode, val_clip_ids=None if o.val_clips is None else tuple(o.val_clips),
        val_fraction=cfg.val_fraction if o.val_fraction is None else o.val_fraction, stride=o.stride,
        include_empty=o.include_empty, min_area_px=cfg.min_area_px if o.min_area_px is None else o.min_area_px,
        class_ids=None if o.classes is None else tuple(o.classes),
    )
    check_request(state.store, project_id, body.clips, body.formats, options)  # 400 / 404 now, not as a job error
    name = clean_name(body.name)  # 400 now too, before any job starts
    # Export only reads, so it does not take the GPU. But a propagation is rewriting the very masks an export
    # would read, so refuse to start while one runs (the reverse race, a propagation started mid-export, only
    # affects masks the export may already have passed; mask files are written atomically, so none is torn).
    busy = state.jobs.running(GPU)
    if busy is not None:
        raise JobBusy(busy)
    exports_root = state.settings.project.data_root / "exports"

    def work(job: Job) -> dict:
        try:
            result = run_export(
                state.store, project_id, body.clips, body.formats, options, exports_root,
                progress=lambda done, total: job.emit({"type": "progress", "done": done, "total": total}),
                cancel=job.cancel_requested, name=name, workers=state.settings.export.workers,
            )
        except ExportCancelled:
            return {"cancelled": True}
        state.warmer.request(Path(result["path"]))  # so training on it starts sooner; never raises or blocks
        return result

    return {"job_id": state.jobs.start("export", work).id}
