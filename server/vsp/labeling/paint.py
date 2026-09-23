"""Hand-painted masks (brush spec): the whole mask arrives as a PNG and the frame becomes a `paint` keyframe."""

from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..store import Clip, Keyframe, ProjectStore, Prompt, now_iso
from .errors import BadRequest
from .prompts import create_object
from .ranges import subtract_range


def decode_mask_png(b64: str) -> np.ndarray:
    """A base64 PNG -> bool HxW; any pixel brighter than mid-grey is in the mask."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
        img.load()
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise BadRequest(f"mask_png_b64 is not a PNG image: {e}") from e
    if "A" in img.getbands():  # flatten onto black, so a mask drawn only in the alpha channel still counts
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
    return np.array(img.convert("L")) > 127


def set_mask(
    store: ProjectStore, clip_id: str, frame: int, mask: np.ndarray, *, class_id: int | None = None
) -> Clip:
    """Store `mask` as the object's mask on `frame` and make the frame a `paint` keyframe.

    An empty mask removes the frame's mask and keyframe instead. Either way the frame leaves
    `propagated_ranges`. The object (and its class) always stays.
    """
    store.clip_dir(clip_id)  # raises NotFound before the lock table grows an entry for a bogus id
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        if not 0 <= frame < clip.frame_count:
            raise BadRequest(f"frame {frame} is out of range 0..{clip.frame_count - 1}")
        if frame in clip.deleted_frames:
            raise BadRequest(f"frame {frame} is deleted; restore it before labeling")
        if mask.shape != (clip.height, clip.width):
            raise BadRequest(f"mask is {mask.shape[1]}×{mask.shape[0]}, the clip is {clip.width}×{clip.height}")
        if not clip.objects and not mask.any():
            raise BadRequest("the mask is empty and the clip has no object yet; paint something first")
        if clip.objects:
            obj = clip.objects[0]
        elif class_id is None:
            raise BadRequest("class_id is required for the first accept on a clip")
        else:
            obj = None
        if class_id is not None:
            known = {c.id for c in store.read_project(clip.project_id).classes}
            if class_id not in known:
                raise BadRequest(f"unknown class id {class_id}")
        if obj is None:
            obj = create_object(clip, class_id)
        elif class_id is not None:
            obj.class_id = class_id
        others = [k for k in obj.keyframes if k.frame_idx != frame]
        if mask.any():
            store.write_mask(clip.id, obj.id, frame, mask)
            painted = Keyframe(frame_idx=frame, prompt=Prompt(type="paint", payload={}), created_at=now_iso())
            obj.keyframes = sorted([*others, painted], key=lambda k: k.frame_idx)
        else:
            store.delete_mask(clip.id, obj.id, frame)
            obj.keyframes = others
        obj.propagated_ranges = subtract_range(obj.propagated_ranges, frame, frame)
        store.write_clip(clip)
        return clip
