"""The export plan: which frames, which objects, which split. Objects of unselected classes are dropped here, and
the selected classes renumbered, so every exporter serialises the same export ids."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

from ..labeling.errors import BadRequest
from ..labeling.regions import apply_to_mask, check_regions, crop_size, region_for
from ..store import ProjectStore
from ..store.models import ClassDef, Clip, RegionRange
from .polygons import clean_mask

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
    # project class ids to export (None = every class); always PROJECT ids, also in derived copies
    class_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PlanObject:
    object_id: int
    class_id: int
    mask_path: Path
    min_area_px: int  # exporters clean the mask with the same rule the plan used to decide the object is present
    region: RegionRange | None = None  # cut (excluded area cleared, then cropped) before the mask is used


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
    region: RegionRange | None = None  # the image is greyed and cropped; width/height are the crop's


@dataclass(frozen=True)
class ExportPlan:
    project_id: str
    project_name: str
    classes: tuple[ClassDef, ...]  # ids are EXPORT ids 0..k-1 (see select_classes)
    class_map: dict[int, int]  # export id -> project class id
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
    project = store.read_project(project_id)
    clips = _read_clips(store, project_id, clip_ids)
    if not formats:
        raise BadRequest("choose at least one format")
    if len(set(formats)) != len(formats):
        raise BadRequest("a format is listed twice")
    for name in formats:
        if name not in FORMATS:
            raise BadRequest(f"unknown format {name!r}; choose from {', '.join(FORMATS)}")
    if not 0 <= options.val_fraction < 1:
        raise BadRequest("val_fraction must be at least 0 and below 1")
    if options.stride < 1:
        raise BadRequest("stride must be 1 or more")
    class_ids: tuple[int, ...] | None = None
    if options.class_ids is not None:
        if not options.class_ids:
            raise BadRequest("choose at least one class to export")
        if len(set(options.class_ids)) != len(options.class_ids):
            raise BadRequest("a class is listed twice")
        known = {c.id for c in project.classes}
        unknown = [i for i in options.class_ids if i not in known]
        if unknown:
            raise BadRequest(f"unknown class {unknown[0]} in project {project_id}")
        class_ids = tuple(sorted(options.class_ids))
    mode = options.split_mode or ("per_clip" if len(clips) >= 2 else "contiguous")
    val_ids: tuple[str, ...] = ()
    if mode == "contiguous" and options.val_clip_ids:
        raise BadRequest("validation clips apply to the per-clip split only; the contiguous split takes the tail of every clip")
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
    return replace(options, split_mode=mode, val_clip_ids=val_ids, class_ids=class_ids)


def select_classes(
    classes: Sequence[ClassDef], class_ids: tuple[int, ...] | None,
) -> tuple[tuple[ClassDef, ...], dict[int, int]]:
    """The exported classes and the map export id -> project class id (spec §10). The selected classes, ordered
    by project class id, get export ids 0..k-1; with every class selected the map is the identity (project class
    ids are contiguous from 0)."""
    chosen = sorted((c for c in classes if class_ids is None or c.id in class_ids), key=lambda c: c.id)
    renumbered = tuple(c.model_copy(update={"id": i}) for i, c in enumerate(chosen))
    return renumbered, {i: c.id for i, c in enumerate(chosen)}


def options_from_manifest(options: Mapping[str, Any]) -> ExportOptions:
    """An export's resolved options as `export.json` stores them (lists for tuples; keys an older manifest lacks
    take their defaults, keys this version does not know are ignored)."""
    known = {f.name for f in fields(ExportOptions)}
    values = {k: v for k, v in options.items() if k in known}
    values["val_clip_ids"] = tuple(values.get("val_clip_ids") or ())
    if values.get("class_ids") is not None:
        values["class_ids"] = tuple(int(i) for i in values["class_ids"])
    return ExportOptions(**values)


def classes_from_manifest(manifest: Mapping[str, Any]) -> tuple[tuple[ClassDef, ...], dict[int, int]]:
    """A derived copy's classes (export ids, as the source wrote them) and its class map. A manifest written
    before class selection has no `project_class_id`: its export ids are the project ids."""
    entries = sorted(manifest["classes"], key=lambda c: c["id"])
    classes = tuple(ClassDef(id=int(c["id"]), name=c["name"], color=c["color"]) for c in entries)
    return classes, {int(c["id"]): int(c.get("project_class_id", c["id"])) for c in entries}


def _stems(clip: Clip, frames: list[int]) -> dict[int, str]:
    """`<clip_id>_<original stem | 6-digit idx>`; a colliding original stem falls back to the index form."""
    fallback = {idx: f"{clip.id}_{idx:06d}" for idx in frames}
    reserved = set(fallback.values())  # an original stem may not take another frame's index form either
    out: dict[int, str] = {}
    used: set[str] = set()
    for idx in frames:
        original = clip.original_names.get(idx)
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original).stem) if original else ""
        stem = f"{clip.id}_{base}" if base else fallback[idx]
        if stem in used or (stem in reserved and stem != fallback[idx]):
            stem = fallback[idx]
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
    classes, class_map = select_classes(project.classes, resolved.class_ids)
    export_id_of = {project_class: export_class for export_class, project_class in class_map.items()}
    frames: list[PlanFrame] = []
    for clip in clips:
        check_regions(clip)
        deleted = set(clip.deleted_frames)
        kept = [i for i in range(clip.frame_count) if i not in deleted][:: resolved.stride]
        stems = _stems(clip, kept)
        candidates: list[tuple[int, tuple[PlanObject, ...], RegionRange | None]] = []
        for idx in kept:
            if tick is not None:
                tick()
            region = region_for(clip, idx)
            objects = []
            for obj in clip.objects:
                export_class = export_id_of.get(obj.class_id)
                if export_class is None:  # an unselected class: dropped BEFORE the empty-frame filter below
                    continue
                mask = store.read_mask(clip.id, obj.id, idx)
                if mask is not None:
                    mask = apply_to_mask(mask, region)
                # absent, or nothing but noise once cut and cleaned = the object is not in this frame
                if mask is None or not clean_mask(mask, resolved.min_area_px).any():
                    continue
                objects.append(PlanObject(
                    obj.id, export_class, store.mask_path(clip.id, obj.id, idx), resolved.min_area_px, region,
                ))
            if objects or resolved.include_empty:
                candidates.append((idx, tuple(objects), region))
        # The split comes AFTER the empty-frame filter: the tail of what is actually exported, not of the frame range
        # (otherwise a clip labelled only at its start would get no validation frames at all).
        if resolved.split_mode == "contiguous":
            n_val = math.ceil(resolved.val_fraction * len(candidates))
            n_val = min(n_val, max(0, len(candidates) - 1))  # never the whole clip
            val_frames = {idx for idx, _, _ in candidates[len(candidates) - n_val :]} if n_val else set()
        else:
            val_frames = {idx for idx, _, _ in candidates} if clip.id in (resolved.val_clip_ids or ()) else set()
        for idx, objects, region in candidates:
            w, h = crop_size(region, clip.width, clip.height)
            frames.append(PlanFrame(
                clip_id=clip.id, frame_idx=idx, stem=stems[idx], split="val" if idx in val_frames else "train",
                image_path=store.frame_path(clip.id, idx), width=w, height=h, objects=objects, region=region,
            ))
    if not frames:
        raise BadRequest("nothing to export: no frame of the selected clips has a usable mask")
    return ExportPlan(
        project_id=project.id, project_name=project.name, classes=classes, class_map=class_map, options=resolved,
        clip_ids=tuple(clip_ids), clip_names={c.id: c.source.name for c in clips},
        val_clip_ids=tuple(resolved.val_clip_ids or ()), frames=tuple(frames),
    )
