"""The export plan: which frames, which objects, which split. Computed once; every exporter serialises the same
plan, so the formats are mutually consistent (spec §8)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ..labeling.errors import BadRequest
from ..store import ProjectStore
from ..store.models import ClassDef, Clip

FORMATS = ("yolo_seg", "yolo_det", "coco", "masks")
Split = Literal["train", "val"]


@dataclass(frozen=True)
class ExportOptions:
    split_mode: Literal["per_clip", "contiguous"] | None = None  # None: per_clip with two or more clips, else contiguous
    val_clip_ids: tuple[str, ...] | None = None  # per_clip only; None: the most recently imported ceil(val_fraction * n)
    val_fraction: float = 0.2
    stride: int = 1
    include_empty: bool = True
    min_area_px: int = 64


@dataclass(frozen=True)
class PlanObject:
    object_id: int
    class_id: int
    mask_path: Path


@dataclass(frozen=True)
class PlanFrame:
    clip_id: str
    frame_idx: int
    stem: str
    split: Split
    image_path: Path
    width: int
    height: int
    objects: tuple[PlanObject, ...]


@dataclass(frozen=True)
class ExportPlan:
    project_id: str
    project_name: str
    classes: tuple[ClassDef, ...]
    options: ExportOptions  # resolved: split_mode is never None here
    clip_ids: tuple[str, ...]
    clip_names: dict[str, str]
    val_clip_ids: tuple[str, ...]
    frames: tuple[PlanFrame, ...]

    def of_split(self, split: Split) -> list[PlanFrame]:
        return [f for f in self.frames if f.split == split]


def _read_clips(store: ProjectStore, project_id: str, clip_ids: list[str]) -> list[Clip]:
    if not clip_ids:
        raise BadRequest("choose at least one clip to export")
    if len(set(clip_ids)) != len(clip_ids):
        raise BadRequest("a clip is listed twice")
    clips = [store.read_clip(cid) for cid in clip_ids]
    for clip in clips:
        if clip.project_id != project_id:
            raise BadRequest(f"clip {clip.id} does not belong to project {project_id}")
    return clips


def check_request(
    store: ProjectStore, project_id: str, clip_ids: list[str], formats: list[str], options: ExportOptions
) -> ExportOptions:
    """Cheap validation (no mask is read), shared by the API route and the job. Returns the resolved options."""
    store.read_project(project_id)
    clips = _read_clips(store, project_id, clip_ids)
    if not formats:
        raise BadRequest("choose at least one format")
    for name in formats:
        if name not in FORMATS:
            raise BadRequest(f"unknown format {name!r}; choose from {', '.join(FORMATS)}")
    if not 0 <= options.val_fraction < 1:
        raise BadRequest("val_fraction must be at least 0 and below 1")
    if options.stride < 1:
        raise BadRequest("stride must be 1 or more")
    mode = options.split_mode or ("per_clip" if len(clips) >= 2 else "contiguous")
    val_ids: tuple[str, ...] = ()
    if mode == "per_clip":
        if len(clips) < 2:
            raise BadRequest("a per-clip split needs at least two clips; use the contiguous split for one clip")
        if options.val_clip_ids is None:
            newest_first = sorted(clips, key=lambda c: (c.created_at, c.id), reverse=True)
            n_val = min(len(clips) - 1, math.ceil(options.val_fraction * len(clips)))
            val_ids = tuple(c.id for c in newest_first[:n_val])
        else:
            unknown = [cid for cid in options.val_clip_ids if cid not in clip_ids]
            if unknown:
                raise BadRequest(f"validation clip {unknown[0]} is not among the selected clips")
            if len(set(options.val_clip_ids)) >= len(clips):
                raise BadRequest("every selected clip is marked for validation; leave at least one for training")
            val_ids = tuple(dict.fromkeys(options.val_clip_ids))
    return replace(options, split_mode=mode, val_clip_ids=val_ids)


def _stems(clip: Clip, frames: list[int]) -> dict[int, str]:
    """`<clip_id>_<original stem | 6-digit idx>`; a colliding original stem falls back to the index form."""
    out: dict[int, str] = {}
    used: set[str] = set()
    for idx in frames:
        original = clip.original_names.get(idx)
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original).stem) if original else ""
        stem = f"{clip.id}_{base}" if base else f"{clip.id}_{idx:06d}"
        if stem in used:
            stem = f"{clip.id}_{idx:06d}"
        used.add(stem)
        out[idx] = stem
    return out


def build_plan(
    store: ProjectStore, project_id: str, clip_ids: list[str], options: ExportOptions,
    tick: Callable[[], None] | None = None,
) -> ExportPlan:
    """`tick` is called once per frame considered (it may raise to cancel)."""
    resolved = check_request(store, project_id, clip_ids, list(FORMATS[:1]), options)
    project = store.read_project(project_id)
    clips = _read_clips(store, project_id, clip_ids)
    frames: list[PlanFrame] = []
    for clip in clips:
        deleted = set(clip.deleted_frames)
        kept = [i for i in range(clip.frame_count) if i not in deleted][:: resolved.stride]
        if resolved.split_mode == "contiguous":
            n_val = math.ceil(resolved.val_fraction * len(kept))
            n_val = min(n_val, max(0, len(kept) - 1))  # never the whole clip
            val_frames = set(kept[len(kept) - n_val :]) if n_val else set()
        else:
            val_frames = set(kept) if clip.id in (resolved.val_clip_ids or ()) else set()
        stems = _stems(clip, kept)
        for idx in kept:
            if tick is not None:
                tick()
            objects = []
            for obj in clip.objects:
                mask = store.read_mask(clip.id, obj.id, idx)
                if mask is None or int(mask.sum()) < resolved.min_area_px:
                    continue  # absent or degenerate mask = the object is not in this frame
                objects.append(PlanObject(obj.id, obj.class_id, store.mask_path(clip.id, obj.id, idx)))
            if not objects and not resolved.include_empty:
                continue
            frames.append(PlanFrame(
                clip_id=clip.id, frame_idx=idx, stem=stems[idx], split="val" if idx in val_frames else "train",
                image_path=store.frame_path(clip.id, idx), width=clip.width, height=clip.height, objects=tuple(objects),
            ))
    if not frames:
        raise BadRequest("nothing to export: no frame of the selected clips has a usable mask")
    return ExportPlan(
        project_id=project.id, project_name=project.name, classes=tuple(project.classes), options=resolved,
        clip_ids=tuple(clip_ids), clip_names={c.id: c.source.name for c in clips},
        val_clip_ids=tuple(resolved.val_clip_ids or ()), frames=tuple(frames),
    )
