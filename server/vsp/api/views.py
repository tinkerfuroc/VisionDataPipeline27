"""JSON shapes shared by several routers."""

from __future__ import annotations

import base64

from ..engine.types import Detection
from ..store import Clip, encode_mask_png
from .deps import AppState


def clip_summary(state: AppState, clip: Clip) -> dict:
    states = state.store.frame_states(clip)
    body = clip.model_dump(mode="json", exclude={"original_names"})
    body["labeled_frames"] = sum(1 for s in states if s in ("masked", "keyframe"))
    return body


def clip_detail(state: AppState, clip: Clip) -> dict:
    project = state.store.read_project(clip.project_id)
    return {
        "clip": clip.model_dump(mode="json"),
        "states": state.store.frame_states(clip),
        "classes": [c.model_dump(mode="json") for c in project.classes],
        "object_states": {str(o.id): state.store.object_states(clip, o) for o in clip.objects},
    }


def candidate_view(det: Detection) -> dict:
    return {
        "score": round(float(det.score), 4),
        "bbox": list(det.bbox),
        "area": int(det.mask.sum()),
        "phrase": det.phrase,
        "mask_png_b64": base64.b64encode(encode_mask_png(det.mask)).decode("ascii"),
    }
