"""Prompt preview and accept (spec 5.2). Stateless: accept recomputes the mask from the prompt."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..engine.masks import bbox_of
from ..engine.types import Detection, Engine
from ..store import Clip, Keyframe, LabelObject, NotFound, ProjectStore, Prompt, now_iso
from .errors import BadRequest
from .objects import resolve_target

BOX_FROM_MASK_PAD = 10


class PromptSpec(BaseModel):
    type: Literal["text", "clicks", "box"]
    phrase: str | None = None
    points: list[tuple[int, int, int]] = Field(default_factory=list)  # (x, y, label); 1 = positive, 0 = negative
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2

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
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise BadRequest("box coordinates must be finite numbers")
            # a drag from bottom-right to top-left is an ordinary gesture, not a malformed box
            self.box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
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
    """Append a new object to `clip` (in memory; the caller writes the clip). Ids are never reused."""
    new_id = max(clip.next_object_id, max((o.id for o in clip.objects), default=0) + 1)
    obj = LabelObject(id=new_id, class_id=class_id)
    clip.objects.append(obj)
    clip.next_object_id = new_id + 1
    return obj


def _validate_points(points: list[tuple[int, int, int]], width: int, height: int) -> None:
    for x, y, _ in points:
        if not (0 <= x < width and 0 <= y < height):
            raise BadRequest(f"point ({x}, {y}) is outside the frame")


def _clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(width), x2), min(float(height), y2)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        raise BadRequest("the box is empty or outside the frame")
    return int(cx1), int(cy1), int(cx2), int(cy2)


def _effective_box(
    store: ProjectStore, clip: Clip, frame: int, prompt: PromptSpec, object_id: int | None
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    """(box to send to the engine, box derived from the existing mask or None)."""
    _validate_points(prompt.points, clip.width, clip.height)
    if prompt.type == "clicks" and prompt.box is None and object_id is not None:
        existing = store.read_mask(clip.id, object_id, frame)
        if existing is not None:
            derived = padded_box(existing, BOX_FROM_MASK_PAD, clip.width, clip.height)
            return derived, derived
    if prompt.box is not None:
        return _clamp_box(prompt.box, clip.width, clip.height), None
    return None, None


def preview(
    store: ProjectStore, engine: Engine, clip: Clip, frame: int, prompt: PromptSpec, *, object_id: int | None = None
) -> list[Detection]:
    image = load_frame(store, clip, frame)
    key = (clip.id, frame)
    if prompt.type == "text":
        return engine.predict_text(image, prompt.phrase.strip(), frame_key=key)
    box, _ = _effective_box(store, clip, frame, prompt, object_id)
    det = engine.predict_prompt(image, points=list(prompt.points), box=box, frame_key=key)
    return [det] if det is not None else []


def accept(
    store: ProjectStore,
    engine: Engine,
    clip_id: str,
    frame: int,
    prompt: PromptSpec,
    *,
    object_id: int | None,
    class_id: int | None = None,
    candidate: int = 0,
    candidates: list[int] | None = None,
) -> tuple[Clip, list[int]]:
    if candidates is not None and object_id is not None:
        raise BadRequest("candidates needs object_id null")
    clip = store.read_clip(clip_id)
    if frame in clip.deleted_frames:
        raise BadRequest(f"frame {frame} is deleted; restore it before labeling")
    box, derived = _effective_box(store, clip, frame, prompt, object_id)
    detections = preview(store, engine, clip, frame, prompt, object_id=object_id)
    if not detections:
        raise BadRequest("the prompt produced no usable mask (nothing found, or the mask is too small or too large)")

    indices = candidates if candidates is not None else [candidate]
    seen: set[int] = set()
    for c in indices:
        if not 0 <= c < len(detections):
            raise BadRequest(f"candidate {c} does not exist; the prompt produced {len(detections)}")
        if c in seen:
            raise BadRequest(f"duplicate candidate {c}")
        seen.add(c)

    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        if frame in clip.deleted_frames:  # re-check: a delete may have raced the unlocked read above
            raise BadRequest(f"frame {frame} is deleted; restore it before labeling")
        ids = []
        for c in indices:
            obj = resolve_target(store, clip, object_id, class_id)
            store.write_mask(clip.id, obj.id, frame, detections[c].mask)
            payload = prompt.model_dump(mode="json", exclude_none=True)
            payload["candidate"] = c
            if derived is not None:
                payload["box_from_mask"] = list(derived)
            elif box is not None:
                payload["box"] = list(box)  # the normalised/clamped box, not the raw client input
            keyframe = Keyframe(
                frame_idx=frame, prompt=Prompt(type=prompt.type, payload=payload), created_at=now_iso()
            )
            obj.keyframes = sorted(
                [k for k in obj.keyframes if k.frame_idx != frame] + [keyframe], key=lambda k: k.frame_idx
            )
            ids.append(obj.id)
        store.write_clip(clip)
        return clip, ids
