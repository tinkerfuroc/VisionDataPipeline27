"""Prompt preview and accept (spec 5.2). Stateless: accept recomputes the mask from the prompt."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..engine.masks import bbox_of
from ..engine.types import Detection, Engine
from ..store import Clip, Keyframe, LabelObject, NotFound, ProjectStore, Prompt, now_iso
from .errors import BadRequest, Conflict

OBJECT_ID = 1  # v1: one object per clip
BOX_FROM_MASK_PAD = 10


class PromptSpec(BaseModel):
    type: Literal["text", "clicks", "box"]
    phrase: str | None = None
    points: list[tuple[int, int, int]] = Field(default_factory=list)  # (x, y, label); 1 = positive, 0 = negative
    box: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2

    @model_validator(mode="after")
    def _check(self) -> PromptSpec:
        if self.type == "text" and not (self.phrase and self.phrase.strip()):
            raise ValueError("a text prompt needs a non-blank phrase")
        if self.type == "clicks" and not self.points:
            raise ValueError("a clicks prompt needs at least one point")
        if self.type == "box" and self.box is None:
            raise ValueError("a box prompt needs a box")
        if any(label not in (0, 1) for _, _, label in self.points):
            raise ValueError("point labels must be 0 (negative) or 1 (positive)")
        return self


def load_frame(store: ProjectStore, clip: Clip, idx: int) -> np.ndarray:
    if not 0 <= idx < clip.frame_count:
        raise BadRequest(f"frame {idx} is out of range 0..{clip.frame_count - 1}")
    img = cv2.imread(str(store.frame_path(clip.id, idx)), cv2.IMREAD_COLOR)
    if img is None:
        raise NotFound(f"frame {idx} of clip {clip.id} is missing on disk")
    return img


def padded_box(mask: np.ndarray, pad: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_of(mask)
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def create_object(clip: Clip, class_id: int) -> LabelObject:
    """The single place the v1 one-object-per-clip rule is enforced."""
    if clip.objects:
        raise Conflict("v1 supports one object per clip; import the footage again to label another component")
    obj = LabelObject(id=OBJECT_ID, class_id=class_id)
    clip.objects.append(obj)
    return obj


def _effective_box(
    store: ProjectStore, clip: Clip, frame: int, prompt: PromptSpec
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    """(box to send to the engine, box derived from the existing mask or None)."""
    if prompt.type == "clicks" and prompt.box is None and clip.objects:
        existing = store.read_mask(clip.id, clip.objects[0].id, frame)
        if existing is not None:
            derived = padded_box(existing, BOX_FROM_MASK_PAD, clip.width, clip.height)
            return derived, derived
    return prompt.box, None


def preview(store: ProjectStore, engine: Engine, clip: Clip, frame: int, prompt: PromptSpec) -> list[Detection]:
    image = load_frame(store, clip, frame)
    key = (clip.id, frame)
    if prompt.type == "text":
        return engine.predict_text(image, prompt.phrase.strip(), frame_key=key)
    box, _ = _effective_box(store, clip, frame, prompt)
    det = engine.predict_prompt(image, points=list(prompt.points), box=box, frame_key=key)
    return [det] if det is not None else []


def accept(
    store: ProjectStore,
    engine: Engine,
    clip_id: str,
    frame: int,
    prompt: PromptSpec,
    *,
    class_id: int | None = None,
    candidate: int = 0,
) -> Clip:
    clip = store.read_clip(clip_id)
    if frame in clip.deleted_frames:
        raise BadRequest(f"frame {frame} is deleted; restore it before labeling")
    _, derived = _effective_box(store, clip, frame, prompt)
    detections = preview(store, engine, clip, frame, prompt)
    if not detections:
        raise BadRequest("the prompt produced no usable mask (nothing found, or the mask is too small or too large)")
    if not 0 <= candidate < len(detections):
        raise BadRequest(f"candidate {candidate} does not exist; the prompt produced {len(detections)}")
    mask = detections[candidate].mask

    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        if clip.objects:
            obj = clip.objects[0]
        else:
            if class_id is None:
                raise BadRequest("class_id is required for the first accept on a clip")
            obj = None
        if class_id is not None:
            known = {c.id for c in store.read_project(clip.project_id).classes}
            if class_id not in known:
                raise BadRequest(f"unknown class id {class_id}")
        if obj is None:
            obj = create_object(clip, class_id)
        elif class_id is not None:
            obj.class_id = class_id
        store.write_mask(clip.id, obj.id, frame, mask)
        payload = prompt.model_dump(mode="json", exclude_none=True)
        payload["candidate"] = candidate
        if derived is not None:
            payload["box_from_mask"] = list(derived)
        keyframe = Keyframe(frame_idx=frame, prompt=Prompt(type=prompt.type, payload=payload), created_at=now_iso())
        obj.keyframes = sorted(
            [k for k in obj.keyframes if k.frame_idx != frame] + [keyframe], key=lambda k: k.frame_idx
        )
        store.write_clip(clip)
        return clip
