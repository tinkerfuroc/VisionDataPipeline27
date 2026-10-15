"""Resumable uploads (spec: uploads from phones): create, send chunks, check progress, list, cancel. The last chunk
starts the same import job as a one-request upload. State and its rules live in `vdp.ingest.resumable`."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from ..ingest import IngestError, UploadTooLarge
from ..ingest.resumable import CHUNK_SIZE, UploadRecord
from ..jobs import TERMINAL, JobNotFound
from ..labeling.errors import Conflict
from .clips import _safe_name, _source_for, _start_import, _upload_kind
from .deps import AppState, State

router = APIRouter()


class CreateUploadBody(BaseModel):
    name: str = Field(min_length=1, max_length=1000)
    size: int = Field(ge=1)
    stride: int = Field(default=1, ge=1)
    fps: float = Field(default=30.0, gt=0)


@router.post("/api/projects/{project_id}/uploads", status_code=201)
def create_upload(project_id: str, body: CreateUploadBody, state: State) -> dict:
    state.store.read_project(project_id)
    clean = _safe_name(body.name)
    kind = _upload_kind(clean)
    if kind is None:
        raise IngestError(f"{clean} is not a video or a .zip of images")
    rec = state.uploads.create(project_id, clean, kind, body.size, body.stride, body.fps)
    return {"upload_id": rec.id, "chunk_size": CHUNK_SIZE, "received": rec.received}


async def _read_chunk(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > CHUNK_SIZE:
        raise UploadTooLarge(f"a chunk is at most {CHUNK_SIZE} bytes")
    data = bytearray()
    try:
        async for piece in request.stream():
            data += piece
            if len(data) > CHUNK_SIZE:
                raise UploadTooLarge(f"a chunk is at most {CHUNK_SIZE} bytes")
    except ClientDisconnect as e:  # nothing of this chunk is kept; the phone sends it again
        raise IngestError("the chunk was interrupted") from e
    return bytes(data)


def _start(state: AppState, rec: UploadRecord) -> str:
    cfg = state.settings.ingest
    path = state.uploads.part_path(rec.id)
    started = _start_import(
        state, rec.project_id, lambda: _source_for(rec.kind, path, rec.name, rec.fps, cfg), rec.stride,
        after=lambda: state.uploads.finish(rec.id),
    )
    return started["job_id"]


@router.put("/api/uploads/{upload_id}")
async def put_chunk(upload_id: str, request: Request, state: State, offset: Annotated[int, Query(ge=0)]) -> dict:
    data = await _read_chunk(request)
    rec = await run_in_threadpool(state.uploads.write_chunk, upload_id, offset, data)
    if rec.complete:
        rec = await run_in_threadpool(state.uploads.complete, upload_id, lambda r: _start(state, r))
    reply: dict = {"received": rec.received}
    if rec.job_id is not None:
        reply["job_id"] = rec.job_id
    return reply


@router.get("/api/uploads/{upload_id}")
def get_upload(upload_id: str, state: State) -> dict:
    return state.uploads.get(upload_id).public()


@router.get("/api/projects/{project_id}/uploads")
def list_uploads(project_id: str, state: State) -> list[dict]:
    state.store.read_project(project_id)
    return [rec.public() for rec in state.uploads.list(project_id)]


@router.delete("/api/uploads/{upload_id}", status_code=204)
def delete_upload(upload_id: str, state: State) -> Response:
    rec = state.uploads.get(upload_id)
    if rec.job_id is not None:
        try:
            running = state.jobs.get(rec.job_id).status not in TERMINAL
        except JobNotFound:
            running = False
        if running:
            raise Conflict(f"the import of {rec.name} is running; cancel its job instead")
    state.uploads.delete(upload_id)
    return Response(status_code=204)


@router.get("/upload", include_in_schema=False)
def phone_upload_page(request: Request) -> RedirectResponse:
    """A real path for the phone link: the GUI's pages live after `#`, which browsers never send to the server,
    so a `/#/upload` link would lose its page on the way through the login redirect."""
    query = request.url.query
    return RedirectResponse("/#/upload" + (f"?{query}" if query else ""), status_code=303)
