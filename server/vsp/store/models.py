"""On-disk schema. `objects` is a list for forward compatibility; v1 uses at most one."""

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
