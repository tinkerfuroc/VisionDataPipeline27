"""Prompt, accept, propagate, paint-mask and frame-cleanup routes (spec 4 API table, 5.2-5.4)."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..jobs import Job, JobBusy
from ..labeling import frames as frame_ops
from ..labeling.objects import delete_object, find_object, relabel_object
from ..labeling.paint import decode_mask_png, set_mask
from ..labeling.prompts import PromptSpec, accept, preview
from ..labeling.propagate import plan_indices, resolve_participants, run_propagation
from .deps import GPU, State, gpu_lease, require_engine
from .views import candidate_view, clip_detail

router = APIRouter(prefix="/api/clips/{clip_id}", tags=["labeling"])

# "all" (default) or an explicit, non-empty list of object ids -- an empty list is a 422, not a
# misleading "nothing to do".
ObjectSelection = Literal["all"] | Annotated[list[int], Field(min_length=1)]


class PromptBody(BaseModel):
    frame: int = Field(ge=0)
    prompt: PromptSpec
    # required, null allowed: null = a new object (no click box derived from an existing mask); an id = that object
    object_id: int | None


class AcceptBody(PromptBody):
    class_id: int | None = None
    candidate: int = Field(default=0, ge=0)
    candidates: list[int] | None = None


class MaskBody(BaseModel):
    frame: int = Field(ge=0)
    mask_png_b64: str
    object_id: int | None  # required, null allowed: null = a new object, which needs class_id
    class_id: int | None = None


class PropagateBody(BaseModel):
    from_frame: int = Field(ge=0)
    direction: Literal["forward", "backward", "both"]
    objects: ObjectSelection = "all"


class PruneBody(BaseModel):
    from_: int = Field(alias="from", ge=0)


class RangeBody(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    objects: ObjectSelection = "all"


class RelabelBody(BaseModel):
    class_id: int


@router.post("/prompt")
def prompt_preview(clip_id: str, body: PromptBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    with gpu_lease(state):
        if body.object_id is not None:
            find_object(clip, body.object_id)  # 404 for an unknown object, before running any inference
        detections = preview(state.store, engine, clip, body.frame, body.prompt, object_id=body.object_id)
    return {"candidates": [candidate_view(d) for d in detections]}


@router.post("/accept")
def prompt_accept(clip_id: str, body: AcceptBody, state: State) -> dict:
    state.store.read_clip(clip_id)  # 404 before requiring the engine, same order as every other route here
    engine = require_engine(state)
    with gpu_lease(state):
        clip, ids = accept(
            state.store, engine, clip_id, body.frame, body.prompt,
            object_id=body.object_id, class_id=body.class_id, candidate=body.candidate, candidates=body.candidates,
        )
        return {**clip_detail(state, clip), "object_ids": ids}


@router.post("/mask")
def paint_mask(clip_id: str, body: MaskBody, state: State) -> dict:
    """A hand-painted mask for one frame of one object (brush spec). The frame becomes a `paint` keyframe."""
    state.store.read_clip(clip_id)
    mask = decode_mask_png(body.mask_png_b64)
    with gpu_lease(state):  # a propagation writes the very masks this replaces
        clip, object_id = set_mask(
            state.store, clip_id, body.frame, mask, object_id=body.object_id, class_id=body.class_id,
        )
        return {**clip_detail(state, clip), "object_id": object_id}


@router.post("/propagate", status_code=202)
def propagate(clip_id: str, body: PropagateBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    participants, _ = resolve_participants(state.store, clip, body.from_frame, body.objects, strict=True)
    directions = ["forward", "backward"] if body.direction == "both" else [body.direction]
    for obj, _ in participants:  # validate now so a bad request is a 400, not a failed job
        for d in directions:
            plan_indices(clip, obj, body.from_frame, d)
    cfg = state.settings.propagate
    objects = body.objects

    def work(job: Job) -> dict:
        result = run_propagation(
            state.store, engine, clip_id, body.from_frame, body.direction,
            objects=objects,
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
        clip = frame_ops.clear_range(state.store, clip_id, body.start, body.end, objects=body.objects)
        return clip_detail(state, clip)


@router.patch("/objects/{object_id}")
def patch_object(clip_id: str, object_id: int, body: RelabelBody, state: State) -> dict:
    return clip_detail(state, relabel_object(state.store, clip_id, object_id, body.class_id))


@router.delete("/objects/{object_id}")
def delete_object_route(clip_id: str, object_id: int, state: State) -> dict:
    with gpu_lease(state):  # a running propagation makes gpu_lease itself fail with 409
        exporting = state.jobs.active("export")
        if exporting is not None:  # an export is reading the object's masks; don't pull them out from under it
            raise JobBusy(exporting)
        clip = delete_object(state.store, clip_id, object_id)
    return clip_detail(state, clip)
