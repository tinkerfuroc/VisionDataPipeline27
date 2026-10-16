"""On-disk schema. A clip holds any number of objects; ids are unique within the clip
and never reused."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassDef(BaseModel):
    id: int
    name: str
    color: str


class Project(BaseModel):
    id: str
    name: str
    created_at: str
    classes: list[ClassDef] = Field(default_factory=list)


class Prompt(BaseModel):
    type: Literal["text", "clicks", "box", "paint"]
    payload: dict


class Keyframe(BaseModel):
    frame_idx: int
    prompt: Prompt
    created_at: str


class LabelObject(BaseModel):
    id: int
    class_id: int
    keyframes: list[Keyframe] = Field(default_factory=list)
    propagated_ranges: list[tuple[int, int]] = Field(default_factory=list)


class ClipSource(BaseModel):
    type: Literal["video", "frame_dir"]
    name: str


# [x0, y0, x1, y1], integer pixels of the original frame, half-open; checked by labeling.regions.validate_regions
Rect = tuple[int, int, int, int]


class RegionRange(BaseModel):
    """Frames start..end (inclusive) share one crop (keep only this rectangle) and blocked boxes (exclude these).
    Only the shape is checked here, so a clip always loads; the geometry is checked where regions are used."""

    start: int
    end: int
    crop: Rect | None = None
    blocks: list[Rect] = Field(default_factory=list)


class Clip(BaseModel):
    id: str
    project_id: str
    source: ClipSource
    fps: float
    stride: int
    width: int
    height: int
    frame_count: int
    created_at: str
    original_names: dict[int, str] = Field(default_factory=dict)
    deleted_frames: list[int] = Field(default_factory=list)
    objects: list[LabelObject] = Field(default_factory=list)
    # ids are never reused: a deleted object's stale mask URLs and job events can never reach a new object
    next_object_id: int = 0
    regions: list[RegionRange] = Field(default_factory=list)
