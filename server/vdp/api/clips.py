from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from ..ingest import (
    FrameDirSource,
    FrameSource,
    ImportCancelled,
    IngestError,
    UploadTooLarge,
    VideoFileSource,
    ZipFrameSource,
    import_clip,
    resolve_allowed,
    save_stream,
)
from ..jobs import Job, JobBusy
from ..labeling.objects import find_object
from ..store import NotFound
from .deps import AppState, State, gpu_lease
from .views import clip_detail

router = APIRouter(tags=["clips"])

VIDEO_SUFFIXES = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".mts", ".m2ts",
})


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:80] or "upload"


def _upload_kind(name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix == ".zip":
        return "zip"
    return "video" if suffix in VIDEO_SUFFIXES else None


class ImportFramesBody(BaseModel):
    path: str
    fps: float = Field(default=30.0, gt=0)
    stride: int = Field(default=1, ge=1)


def _source_for(kind: str, path: Path, name: str, fps: float, cfg) -> FrameSource:
    """The frame source for an uploaded file: a video decodes itself; a zip of images needs `fps`."""
    if kind == "video":
        return VideoFileSource(path, name=name)
    return ZipFrameSource(
        path, fps=fps, name=Path(name).stem or name, max_entries=cfg.max_zip_entries, max_bytes=cfg.max_zip_bytes,
    )


def _start_import(
    state: AppState, project_id: str, build_source: Callable[[], FrameSource], stride: int,
    cleanup: Path | None = None, after: Callable[[], None] | None = None,
) -> dict:
    cfg = state.settings.ingest

    def work(job: Job) -> dict:
        try:
            source = build_source()  # a decoder/zip failure here is the job's error, reported like any other
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
            if after is not None:
                after()

    return {"job_id": state.jobs.start("import", work).id}


@router.post("/api/projects/{project_id}/clips/import-frames", status_code=202)
def import_frames(project_id: str, body: ImportFramesBody, state: State) -> dict:
    state.store.read_project(project_id)
    path = resolve_allowed(body.path, state.settings.ingest.allowed_roots)
    source = FrameDirSource(path, fps=body.fps)  # eager: a bad directory is a 400, not a job error
    return _start_import(state, project_id, lambda: source, body.stride)


@router.post("/api/projects/{project_id}/clips/upload", status_code=202)
async def upload_clip(
    project_id: str,
    request: Request,
    state: State,
    name: Annotated[str, Query(min_length=1, max_length=1000)],
    stride: Annotated[int, Query(ge=1)] = 1,
    fps: Annotated[float, Query(gt=0)] = 30.0,
) -> dict:
    """Start an import from a file uploaded as the raw request body (not multipart: see the plan's header).

    `fps` is used by a zip only; a video's frame rate comes from its own container.
    """
    state.store.read_project(project_id)
    clean = _safe_name(name)
    kind = _upload_kind(clean)
    if kind is None:
        raise IngestError(f"{clean} is not a video or a .zip of images")
    limit = state.settings.ingest.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        # Refuse before reading a byte. A browser may see this as a broken connection rather than the 413,
        # because it is still sending; the GUI checks the same limit from /api/health before it starts.
        raise UploadTooLarge(f"the upload is {int(declared)} bytes; the limit is {limit}")
    uploads = state.settings.project.data_root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    tmp = uploads / f"{uuid.uuid4().hex}_{clean}"
    try:
        await save_stream(request.stream(), tmp, limit)
    except ClientDisconnect as e:  # the tab closed, or the SSH tunnel dropped
        raise IngestError("the upload was interrupted") from e
    cfg = state.settings.ingest
    return _start_import(state, project_id, lambda: _source_for(kind, tmp, clean, fps, cfg), stride, cleanup=tmp)


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


@router.get("/api/clips/{clip_id}/objects/{object_id}/masks/{idx}")
def get_object_mask(clip_id: str, object_id: int, idx: int, state: State) -> FileResponse:
    # masks are overwritten in place by re-propagation, unlike immutable frames/thumbs
    clip = state.store.read_clip(clip_id)
    find_object(clip, object_id)  # 404 for an unknown object, same as a missing mask file
    return _image(state.store.mask_path(clip_id, object_id, idx), "image/png", headers={"Cache-Control": "no-cache"})
