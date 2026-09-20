"""Prompt, accept, propagate and frame-cleanup routes (spec 4 API table, 5.2-5.4)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..jobs import Job
from ..labeling import frames as frame_ops
from ..labeling.errors import BadRequest
from ..labeling.prompts import PromptSpec, accept, preview
from ..labeling.propagate import plan_indices, run_propagation
from .deps import GPU, State, gpu_lease, require_engine
from .views import candidate_view, clip_detail

router = APIRouter(prefix="/api/clips/{clip_id}", tags=["labeling"])


class PromptBody(BaseModel):
    frame: int = Field(ge=0)
    prompt: PromptSpec


class AcceptBody(PromptBody):
    class_id: int | None = None
    candidate: int = Field(default=0, ge=0)


class PropagateBody(BaseModel):
    from_frame: int = Field(ge=0)
    direction: Literal["forward", "backward", "both"]


class PruneBody(BaseModel):
    from_: int = Field(alias="from", ge=0)


class RangeBody(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)


@router.post("/prompt")
def prompt_preview(clip_id: str, body: PromptBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    with gpu_lease(state):
        detections = preview(state.store, engine, clip, body.frame, body.prompt)
    return {"candidates": [candidate_view(d) for d in detections]}


@router.post("/accept")
def prompt_accept(clip_id: str, body: AcceptBody, state: State) -> dict:
    state.store.read_clip(clip_id)
    engine = require_engine(state)
    with gpu_lease(state):
        clip = accept(
            state.store, engine, clip_id, body.frame, body.prompt, class_id=body.class_id, candidate=body.candidate
        )
        return clip_detail(state, clip)


@router.post("/propagate", status_code=202)
def propagate(clip_id: str, body: PropagateBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    if not clip.objects:
        raise BadRequest("nothing to propagate: accept a prompt first")
    directions = ["forward", "backward"] if body.direction == "both" else [body.direction]
    for d in directions:  # validate now so a bad request is a 400, not a failed job
        plan_indices(clip, clip.objects[0], body.from_frame, d)
    cfg = state.settings.propagate

    def work(job: Job) -> dict:
        result = run_propagation(
            state.store, engine, clip_id, body.from_frame, body.direction,
            chunk_size=cfg.chunk_size, min_chunk_size=cfg.min_chunk_size, lost_after=cfg.lost_after,
            emit=job.emit, cancel=job.cancel_requested,
        )
        return result.as_dict()

    return {"job_id": state.jobs.start("propagate", work, exclusive=GPU).id}


@router.post("/frames/{idx}/delete")
def delete_frame(clip_id: str, idx: int, state: State) -> dict:
    return clip_detail(state, frame_ops.delete_frame(state.store, clip_id, idx))


@router.post("/frames/{idx}/restore")
def restore_frame(clip_id: str, idx: int, state: State) -> dict:
    return clip_detail(state, frame_ops.restore_frame(state.store, clip_id, idx))


@router.post("/prune")
def prune(clip_id: str, body: PruneBody, state: State) -> dict:
    return clip_detail(state, frame_ops.prune_from(state.store, clip_id, body.from_))


@router.post("/clear-range")
def clear_range(clip_id: str, body: RangeBody, state: State) -> dict:
    state.store.read_clip(clip_id)
    with gpu_lease(state):  # it deletes masks and trims ranges a running propagation is writing
        return clip_detail(state, frame_ops.clear_range(state.store, clip_id, body.start, body.end))
