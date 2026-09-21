from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..ingest import (
    FrameDirSource,
    ImportCancelled,
    IngestError,
    VideoFileSource,
    import_clip,
    resolve_allowed,
)
from ..jobs import Job, JobBusy
from ..labeling.prompts import OBJECT_ID
from ..store import NotFound
from .deps import AppState, State, gpu_lease
from .views import clip_detail

router = APIRouter(tags=["clips"])


class ImportFramesBody(BaseModel):
    path: str
    fps: float = Field(default=30.0, gt=0)
    stride: int = Field(default=1, ge=1)


def _start_import(state: AppState, project_id: str, source, stride: int, cleanup: Path | None = None) -> dict:
    cfg = state.settings.ingest

    def work(job: Job) -> dict:
        try:
            clip = import_clip(
                state.store, project_id, source, stride=stride, jpeg_quality=cfg.jpeg_quality,
                thumb_width=cfg.thumb_width, progress=lambda n: job.emit({"type": "progress", "done": n}),
                cancel=job.cancel_requested,
            )
            return {"clip_id": clip.id, "frame_count": clip.frame_count}
        except ImportCancelled:
            return {"cancelled": True}
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

    return {"job_id": state.jobs.start("import", work).id}


@router.post("/api/projects/{project_id}/clips/import-frames", status_code=202)
def import_frames(project_id: str, body: ImportFramesBody, state: State) -> dict:
    state.store.read_project(project_id)
    path = resolve_allowed(body.path, state.settings.ingest.allowed_roots)
    return _start_import(state, project_id, FrameDirSource(path, fps=body.fps), body.stride)


@router.post("/api/projects/{project_id}/clips/import-video", status_code=202)
def import_video(
    project_id: str,
    state: State,
    file: Annotated[UploadFile, File()],
    stride: Annotated[int, Form(ge=1)] = 1,
) -> dict:
    state.store.read_project(project_id)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename or "upload").name)[:80] or "upload"
    tmp_dir = state.settings.project.data_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}_{name}"
    try:
        with tmp.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    try:
        source = VideoFileSource(tmp, name=name)
    except IngestError as e:
        # Report an unreadable upload as a failed job so the UI has one code path, and remove the upload.
        tmp.unlink(missing_ok=True)
        error = IngestError(str(e))  # `e` is unbound once this block ends; the job runs later on another thread

        def fail(job: Job) -> dict:
            raise error

        return {"job_id": state.jobs.start("import", fail).id}
    return _start_import(state, project_id, source, stride, cleanup=tmp)


@router.get("/api/clips/{clip_id}")
def get_clip(clip_id: str, state: State) -> dict:
    return clip_detail(state, state.store.read_clip(clip_id))


@router.delete("/api/clips/{clip_id}", status_code=204)
def delete_clip(clip_id: str, state: State) -> Response:
    with gpu_lease(state):  # coarse across clips, deliberately: don't race a running GPU job's writes
        exporting = state.jobs.active("export")
        if exporting is not None:  # an export is reading clips; pulling one out from under it fails the whole export
            raise JobBusy(exporting)
        state.store.delete_clip(clip_id)
    return Response(status_code=204)


def _image(path: Path, media_type: str, *, headers: dict[str, str] | None = None) -> FileResponse:
    if not path.is_file():
        raise NotFound(f"{path.name} not found")
    return FileResponse(path, media_type=media_type, headers=headers)


@router.get("/api/clips/{clip_id}/frames/{idx}")
def get_frame(clip_id: str, idx: int, state: State) -> FileResponse:
    return _image(state.store.frame_path(clip_id, idx), "image/jpeg")


@router.get("/api/clips/{clip_id}/thumbs/{idx}")
def get_thumb(clip_id: str, idx: int, state: State) -> FileResponse:
    return _image(state.store.thumb_path(clip_id, idx), "image/jpeg")


@router.get("/api/clips/{clip_id}/masks/{idx}")
def get_mask(clip_id: str, idx: int, state: State) -> FileResponse:
    # masks are overwritten in place by re-propagation, unlike immutable frames/thumbs
    return _image(state.store.mask_path(clip_id, OBJECT_ID, idx), "image/png", headers={"Cache-Control": "no-cache"})
