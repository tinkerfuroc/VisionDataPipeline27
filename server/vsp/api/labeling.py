"""Prompt, accept, propagate, paint-mask and frame-cleanup routes (spec 4 API table, 5.2-5.4)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..jobs import Job
from ..labeling import frames as frame_ops
from ..labeling.objects import relabel_object
from ..labeling.paint import decode_mask_png, set_mask
from ..labeling.prompts import PromptSpec, accept, preview
from ..labeling.propagate import plan_indices, resolve_participants, run_propagation
from .deps import GPU, State, gpu_lease, require_engine
from .views import candidate_view, clip_detail

router = APIRouter(prefix="/api/clips/{clip_id}", tags=["labeling"])


class PromptBody(BaseModel):
    frame: int = Field(ge=0)
    prompt: PromptSpec


class AcceptBody(PromptBody):
    class_id: int | None = None
    candidate: int = Field(default=0, ge=0)


class MaskBody(BaseModel):
    frame: int = Field(ge=0)
    mask_png_b64: str
    class_id: int | None = None


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
        # legacy shim: removed in plan B -- click box derived from the clip's first object, as today
        object_id = clip.objects[0].id if clip.objects else None
        detections = preview(state.store, engine, clip, body.frame, body.prompt, object_id=object_id)
    return {"candidates": [candidate_view(d) for d in detections]}


@router.post("/accept")
def prompt_accept(clip_id: str, body: AcceptBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    with gpu_lease(state):
        # legacy shim: removed in plan B -- write to the clip's first object, or create one when there is none
        object_id = clip.objects[0].id if clip.objects else None
        clip, _ = accept(
            state.store, engine, clip_id, body.frame, body.prompt,
            object_id=object_id,
            class_id=body.class_id if object_id is None else None,
            candidate=body.candidate,
        )
        # legacy shim: removed in plan B -- a class_id for an existing object relabels it, only once the accept
        # above has succeeded, so a failed accept changes nothing
        if object_id is not None and body.class_id is not None:
            clip = relabel_object(state.store, clip_id, object_id, body.class_id)
        return clip_detail(state, clip)


@router.post("/mask")
def paint_mask(clip_id: str, body: MaskBody, state: State) -> dict:
    """A hand-painted mask for one frame (brush spec). The frame becomes a `paint` keyframe."""
    clip = state.store.read_clip(clip_id)
    mask = decode_mask_png(body.mask_png_b64)
    with gpu_lease(state):  # a propagation writes the very masks this replaces
        # legacy shim: removed in plan B -- write to the clip's first object, or create one when there is none
        object_id = clip.objects[0].id if clip.objects else None
        clip, _ = set_mask(
            state.store, clip_id, body.frame, mask,
            object_id=object_id,
            class_id=body.class_id if object_id is None else None,
        )
        # legacy shim: removed in plan B -- a class_id for an existing object relabels it, only once the paint
        # above has succeeded, so a failed save changes nothing
        if object_id is not None and body.class_id is not None:
            clip = relabel_object(state.store, clip_id, object_id, body.class_id)
        return clip_detail(state, clip)


@router.post("/propagate", status_code=202)
def propagate(clip_id: str, body: PropagateBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    participants, _ = resolve_participants(state.store, clip, body.from_frame, "all", strict=True)
    directions = ["forward", "backward"] if body.direction == "both" else [body.direction]
    for d in directions:  # validate now so a bad request is a 400, not a failed job
        plan_indices(clip, participants[0][0], body.from_frame, d)
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
