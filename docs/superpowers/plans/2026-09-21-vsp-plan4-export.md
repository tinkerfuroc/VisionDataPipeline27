# Vision Sampler Plan 4 — Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export labelled clips as YOLO-seg, YOLO-det, COCO JSON and indexed mask PNGs with a leak-free train/val split, started from a new Export page in the web GUI and run as a background job.

**Architecture:** A new `vsp.export` package reads from the store only. `build_plan()` computes one `ExportPlan` (frames, objects, split); four small `Exporter`s serialise that same plan, so the formats agree with each other. `run_export()` copies each image once into `exports/<id>/images/`, hardlinks it into every format directory (copy fallback), writes `export.json`, and runs inside the existing `JobManager`. The web GUI gets an Export page (clip selection on the left, options in a right-hand sidebar) that follows the job with the existing `useJob` hook.

**Tech Stack:** Python 3.11, numpy, opencv-python-headless, pillow, pydantic, FastAPI (all already dependencies); `pycocotools` as a dev-only dependency for one validation test. Web: React 18 + TypeScript + Tailwind + vitest, no new packages.

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md` §8 (Export), §4 API table, §9, §10. GUI rules: `docs/superpowers/specs/2026-09-21-gui-design-pass.md` and the "GUI design pass" section of `docs/superpowers/plans/2026-09-20-vsp-roadmap.md`.

## Global Constraints

- Python is managed with uv only: `uv run pytest`, `uv add --dev <pkg>`. Never pip, never conda.
- Run Python tests as `HF_HUB_OFFLINE=1 uv run pytest -q` from the repository root; lint with `uv run ruff check .` (line length and rules come from `pyproject.toml`; keep it clean).
- Web commands run as `bash -c 'source ~/.nvm/nvm.sh && cd web && <command>'`. Web checks: `npx tsc --noEmit`, `npx vitest run`. Do not add npm packages.
- Never stage or commit anything under `vision_train/`, `data/`, `web/node_modules/`, `web/dist/`, or any `*.pt` / `*.safetensors`. Stage by explicit path only (`git add <path> ...`), never `git add -A` / `git add .`.
- Do not push, do not create a PR, do not touch the global git config.
- `vsp.export` reads from the store only: it never writes inside `data/projects/` and never links store files outside the store (images are COPIED from the store into the export, and only those copies are hardlinked).
- Exact values from the spec: `val_fraction` default 0.2, `stride` default 1, `include_empty` default true, `min_area_px` default 64; format names `yolo_seg`, `yolo_det`, `coco`, `masks`; mask PNG value = `class_id + 1`, 0 = background; file stem = `<clip_id>_<original stem | 6-digit idx>`; split modes `per_clip` and `contiguous`; random per-frame splitting is not offered.
- Store facts: frames are `frames/<idx:06d>.jpg`; `store.read_mask(clip_id, object_id, idx)` returns a bool array or `None` (absent mask = object not present); `clip.deleted_frames` lists excluded frame indices; `clip.original_names` maps frame index → original file name (frame-directory clips only); class ids come from `project.classes[*].id`.
- Errors: raise `vsp.labeling.errors.BadRequest` for invalid requests (the app maps it to HTTP 400 with `{"detail": str}`); `vsp.jobs.JobBusy(job_id)` maps to 409.
- GUI rules: every page renders its own `AppHeader`; controls live in a RIGHT-hand sidebar; chrome has no hue — use the `btn`, `btn-on`, `btn-danger`, `field`, `heading`, `note`, `alert` classes and the `surround/well/panel/line/ink/dim/danger` tokens; sentence-case copy, no all-caps labels, no middle-dot strings, no "→"; a button names exactly what it will do.
- Web test conventions: component tests start with `// @vitest-environment jsdom`, call `cleanup()` in `afterEach`, mock the API with `vi.mock("../api/rest", async (orig) => ({ ...(await orig<typeof import("../api/rest")>()), api: { ... } }))`, and use `vi.resetAllMocks()` (not per-mock `.mockReset()`) in `beforeEach`.
- Commit after every task with a conventional message ending in the two attribution lines the dispatcher gives you.

---

### Task 1: `export/polygons.py` and `export/plan.py`

**Files:**
- Create: `server/vsp/export/__init__.py`, `server/vsp/export/polygons.py`, `server/vsp/export/plan.py`
- Test: `server/tests/test_export_polygons.py`, `server/tests/test_export_plan.py`

**Interfaces:**
- Consumes: `vsp.store.ProjectStore` (`read_project`, `read_clip`, `read_mask`, `frame_path`, `mask_path`), `vsp.store.models.ClassDef`, `vsp.labeling.errors.BadRequest`.
- Produces (later tasks rely on these exact names):
  - `polygons.mask_to_polygons(mask, min_area_px=4.0) -> list[list[tuple[float, float]]]`
  - `polygons.polygons_to_mask(polygons, width, height) -> np.ndarray` (bool)
  - `polygons.mask_bbox(mask) -> tuple[int, int, int, int] | None` (`x, y, w, h`)
  - `plan.ExportOptions`, `plan.PlanObject`, `plan.PlanFrame`, `plan.ExportPlan`, `plan.FORMATS`, `plan.check_request(store, project_id, clip_ids, formats, options) -> ExportOptions`, `plan.build_plan(store, project_id, clip_ids, options, tick=None) -> ExportPlan`

- [ ] **Step 1: Write the failing polygon tests** — `server/tests/test_export_polygons.py`:

```python
from __future__ import annotations

import cv2
import numpy as np
from vsp.export.polygons import mask_bbox, mask_to_polygons, polygons_to_mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / float((a | b).sum())


def test_a_disc_survives_the_polygon_round_trip():
    mask = np.zeros((120, 160), np.uint8)
    cv2.circle(mask, (80, 60), 40, 1, -1)
    polys = mask_to_polygons(mask.astype(bool))
    assert len(polys) == 1
    assert _iou(polygons_to_mask(polys, 160, 120), mask.astype(bool)) > 0.95


def test_separate_blobs_become_separate_polygons_and_specks_are_dropped():
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:40] = True
    mask[60:90, 55:95] = True
    mask[50, 50] = True  # a one-pixel speck
    polys = mask_to_polygons(mask)
    assert len(polys) == 2
    assert _iou(polygons_to_mask(polys, 100, 100), mask) > 0.95


def test_bbox_is_inclusive_and_none_for_an_empty_mask():
    mask = np.zeros((50, 80), bool)
    mask[10:20, 30:70] = True
    assert mask_bbox(mask) == (30, 10, 40, 10)
    assert mask_bbox(np.zeros((5, 5), bool)) is None
```

- [ ] **Step 2: Run them** — `HF_HUB_OFFLINE=1 uv run pytest server/tests/test_export_polygons.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'vsp.export'`.

- [ ] **Step 3: Implement** — `server/vsp/export/__init__.py`:

```python
"""Dataset export. Reads from the store only; one ExportPlan, many Exporters (spec §8)."""
```

`server/vsp/export/polygons.py`:

```python
"""Mask <-> polygon conversion for the polygon-based formats (YOLO-seg, COCO)."""

from __future__ import annotations

import cv2
import numpy as np

Polygon = list[tuple[float, float]]


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 4.0) -> list[Polygon]:
    """Outer contours of every connected blob, in pixel coordinates. Holes are not represented:
    neither YOLO-seg nor the plain COCO polygon form can carry them, and SAM masks of solid parts rarely have any."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < min_area_px:
            continue
        polygons.append([(float(x), float(y)) for x, y in contour.reshape(-1, 2)])
    return polygons


def polygons_to_mask(polygons: list[Polygon], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [np.round(np.array(polygon)).astype(np.int32)], 1)
    return canvas.astype(bool)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """`x, y, w, h` of the mask's bounding box in pixels (inclusive extent), or None for an empty mask."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1
```

- [ ] **Step 4: Run the polygon tests** — expected: 3 passed.

- [ ] **Step 5: Write the failing plan tests** — `server/tests/test_export_plan.py`. The `store` and `make_clip` fixtures come from `server/tests/conftest.py` (`make_clip(n_frames=12, white=(), classes=("part",))` returns `(project, clip)`; frames are 120×60). Note `make_clip` creates a NEW project per call, so the multi-clip helper below imports extra clips into one project itself.

```python
from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest
from vsp.export.plan import ExportOptions, build_plan, check_request
from vsp.ingest import FrameDirSource, import_clip
from vsp.labeling.errors import BadRequest
from vsp.store.models import LabelObject


def _mask(x0=10, size=20):
    m = np.zeros((60, 120), bool)
    m[10 : 10 + size, x0 : x0 + size] = True
    return m


def _label(store, clip, frames, class_id=0, size=20):
    clip.objects = [LabelObject(id=1, class_id=class_id)]
    store.write_clip(clip)
    for i in frames:
        store.write_mask(clip.id, 1, i, _mask(size=size))
    return store.read_clip(clip.id)


def _extra_clip(store, project_id, tmp_path, n=6):
    src = tmp_path / f"more_{uuid.uuid4().hex[:6]}"
    src.mkdir()
    for i in range(n):
        assert cv2.imwrite(str(src / f"{i:04d}.png"), np.full((60, 120, 3), 90, np.uint8))
    return import_clip(store, project_id, FrameDirSource(src))


def test_contiguous_split_puts_the_tail_of_the_clip_in_val(store, make_clip):
    project, clip = make_clip(n_frames=10)
    _label(store, clip, range(10))
    plan = build_plan(store, project.id, [clip.id], ExportOptions())
    assert plan.options.split_mode == "contiguous"
    assert [f.frame_idx for f in plan.frames if f.split == "val"] == [8, 9]
    assert [f.frame_idx for f in plan.frames if f.split == "train"] == list(range(8))
    assert plan.frames[0].stem == f"{clip.id}_0000"  # the original file stem of a frame-directory clip
    assert plan.frames[0].objects[0].class_id == 0


def test_deleted_frames_are_left_out_and_stride_thins_what_remains(store, make_clip):
    project, clip = make_clip(n_frames=10)
    clip = _label(store, clip, range(10))
    clip.deleted_frames = [1, 2]
    store.write_clip(clip)
    plan = build_plan(store, project.id, [clip.id], ExportOptions(stride=3, val_fraction=0.0))
    assert [f.frame_idx for f in plan.frames] == [0, 5, 8]  # kept = 0,3,4,5,6,7,8,9 -> every third
    assert {f.split for f in plan.frames} == {"train"}


def test_include_empty_keeps_or_drops_frames_without_a_usable_mask(store, make_clip):
    project, clip = make_clip(n_frames=6)
    clip = _label(store, clip, [0, 1, 2])
    store.write_mask(clip.id, 1, 3, _mask(size=5))  # 25 px: below min_area_px, so "absent"
    keep = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0))
    assert [len(f.objects) for f in keep.frames] == [1, 1, 1, 0, 0, 0]
    drop = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, include_empty=False))
    assert [f.frame_idx for f in drop.frames] == [0, 1, 2]


def test_per_clip_split_never_shares_a_clip_between_train_and_val(store, make_clip, tmp_path):
    project, first = make_clip(n_frames=6)
    _label(store, first, range(6))
    others = [_extra_clip(store, project.id, tmp_path) for _ in range(2)]
    for c in others:
        _label(store, c, range(6))
    ids = [first.id, *(c.id for c in others)]
    plan = build_plan(store, project.id, ids, ExportOptions())
    assert plan.options.split_mode == "per_clip"
    assert len(plan.val_clip_ids) == 1  # ceil(0.2 * 3)
    by_clip = {cid: {f.split for f in plan.frames if f.clip_id == cid} for cid in ids}
    assert all(len(splits) == 1 for splits in by_clip.values())
    assert by_clip[plan.val_clip_ids[0]] == {"val"}
    chosen = build_plan(store, project.id, ids, ExportOptions(val_clip_ids=(first.id,)))
    assert {f.split for f in chosen.frames if f.clip_id == first.id} == {"val"}
    stems = [f.stem for f in chosen.frames]
    assert len(stems) == len(set(stems))


def test_invalid_requests_are_refused(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=4)
    _label(store, clip, range(4))
    other_project, foreign = make_clip(n_frames=4)
    ok = ExportOptions()
    with pytest.raises(BadRequest, match="at least one clip"):
        check_request(store, project.id, [], ["coco"], ok)
    with pytest.raises(BadRequest, match="at least one format"):
        check_request(store, project.id, [clip.id], [], ok)
    with pytest.raises(BadRequest, match="unknown format"):
        check_request(store, project.id, [clip.id], ["voc"], ok)
    with pytest.raises(BadRequest, match="does not belong"):
        check_request(store, project.id, [foreign.id], ["coco"], ok)
    with pytest.raises(BadRequest, match="at least two clips"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(split_mode="per_clip"))
    with pytest.raises(BadRequest, match="every selected clip"):
        second = _extra_clip(store, project.id, tmp_path)
        check_request(store, project.id, [clip.id, second.id], ["coco"], ExportOptions(val_clip_ids=(clip.id, second.id)))
    with pytest.raises(BadRequest, match="nothing to export"):
        build_plan(store, project.id, [clip.id], ExportOptions(include_empty=False, min_area_px=10_000))
```

- [ ] **Step 6: Run them** — expected: FAIL with `ModuleNotFoundError: No module named 'vsp.export.plan'`.

- [ ] **Step 7: Implement** — `server/vsp/export/plan.py`:

```python
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
```

- [ ] **Step 8: Run all export tests, the whole suite and ruff** — `HF_HUB_OFFLINE=1 uv run pytest -q` (expected: previous 167 + 8 new, all pass) and `uv run ruff check .` (clean). If ruff reformats nothing but complains about line length in the code above, wrap the line; do not change behaviour.

- [ ] **Step 9: Commit**

```bash
git add server/vsp/export/__init__.py server/vsp/export/polygons.py server/vsp/export/plan.py server/tests/test_export_polygons.py server/tests/test_export_plan.py
git commit -m "feat(export): polygons and the export plan (split, stride, include-empty, min area)"
```

---

### Task 2: `Exporter` protocol and the four exporters

**Files:**
- Create: `server/vsp/export/base.py`, `server/vsp/export/yolo.py`, `server/vsp/export/coco.py`, `server/vsp/export/mask_png.py`
- Modify: `server/vsp/export/__init__.py`, `pyproject.toml` (dev dependency, through `uv add --dev pycocotools`)
- Test: `server/tests/test_export_formats.py`

**Interfaces:**
- Consumes: everything Task 1 produces.
- Produces: `base.ExportSummary(name, images, annotations, linked)`, `base.Exporter` (Protocol: `name: str`, `write(plan, out_dir, tick) -> ExportSummary`), `base.shared_image(out_dir, frame) -> Path` (= `out_dir / "images" / f"{frame.stem}.jpg"`), `base.place_image(src, dst) -> bool` (True when hardlinked), `base.read_mask(obj) -> np.ndarray`, and `vsp.export.EXPORTERS: dict[str, Exporter]` keyed by `yolo_seg`, `yolo_det`, `coco`, `masks`.
- Contract every exporter follows: it writes only under `out_dir / <its name>/`; it takes each image from `shared_image(out_dir, frame)` (Task 3's runner puts them there first) via `place_image`; it calls `tick()` once per frame written.

- [ ] **Step 1: Add the dev dependency** — `uv add --dev pycocotools`. If the install fails (no wheel for this platform), do NOT work around it with pip: skip the dependency, keep the `pytest.importorskip("pycocotools")` line in the test below (the test then skips), and say so in your report.

- [ ] **Step 2: Write the failing tests** — `server/tests/test_export_formats.py`:

```python
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image
from vsp.export import EXPORTERS
from vsp.export import base as export_base
from vsp.export.plan import ExportOptions, build_plan
from vsp.export.polygons import polygons_to_mask
from vsp.store.models import LabelObject

W, H = 120, 60


def _disc(cx: int) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - 30) ** 2 <= 18**2


@pytest.fixture
def planned(store, make_clip, tmp_path):
    """10 frames; a disc moves right on frames 0-7, frames 8-9 are empty; contiguous split -> val = 8, 9."""
    project, clip = make_clip(n_frames=10, classes=("bolt", "nut"))
    clip.objects = [LabelObject(id=1, class_id=1)]
    store.write_clip(clip)
    for i in range(8):
        store.write_mask(clip.id, 1, i, _disc(30 + 6 * i))
    plan = build_plan(store, project.id, [clip.id], ExportOptions())
    out = tmp_path / "export"
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    return plan, out, clip


def _write(name, plan, out):
    ticks = []
    summary = EXPORTERS[name].write(plan, out, lambda: ticks.append(1))
    assert len(ticks) == len(plan.frames)
    return summary


def test_yolo_seg_writes_normalised_polygons_that_rasterise_back(planned):
    plan, out, clip = planned
    summary = _write("yolo_seg", plan, out)
    assert summary.images == {"train": 8, "val": 2} and summary.annotations == {"train": 8, "val": 0}
    root = out / "yolo_seg"
    assert sorted(p.name for p in (root / "images" / "train").iterdir())[0] == f"{clip.id}_0000.jpg"
    line = (root / "labels" / "train" / f"{clip.id}_0000.txt").read_text().strip().splitlines()[0].split()
    assert line[0] == "1"
    coords = np.array(line[1:], float).reshape(-1, 2)
    assert coords.min() >= 0 and coords.max() <= 1
    poly = [(x * W, y * H) for x, y in coords]
    truth = _disc(30)
    back = polygons_to_mask([poly], W, H)
    assert (back & truth).sum() / (back | truth).sum() > 0.95
    assert (root / "labels" / "val" / f"{clip.id}_0008.txt").read_text() == ""  # an empty frame is a background negative
    yaml = (root / "data.yaml").read_text()
    assert "train: images/train" in yaml and "val: images/val" in yaml and "0: bolt" in yaml and "1: nut" in yaml


def test_yolo_det_writes_the_normalised_box_of_the_mask(planned):
    plan, out, clip = planned
    _write("yolo_det", plan, out)
    cls, cx, cy, w, h = (out / "yolo_det" / "labels" / "train" / f"{clip.id}_0000.txt").read_text().split()
    assert cls == "1"
    assert float(cx) == pytest.approx(30.5 / W, abs=1e-4) and float(cy) == pytest.approx(30.5 / H, abs=1e-4)
    assert float(w) == pytest.approx(37 / W, abs=1e-4) and float(h) == pytest.approx(37 / H, abs=1e-4)


def test_coco_is_loadable_and_consistent(planned):
    plan, out, clip = planned
    summary = _write("coco", plan, out)
    assert summary.annotations == {"train": 8, "val": 0}
    path = out / "coco" / "annotations" / "instances_train.json"
    data = json.loads(path.read_text())
    assert [c["name"] for c in data["categories"]] == ["bolt", "nut"] and [c["id"] for c in data["categories"]] == [1, 2]
    ann = data["annotations"][0]
    assert ann["category_id"] == 2 and ann["iscrowd"] == 0 and ann["bbox"] == [12, 12, 37, 37]
    assert ann["area"] == float(_disc(30).sum())
    assert (out / "coco" / "images" / "train" / data["images"][0]["file_name"]).is_file()
    val = json.loads((out / "coco" / "annotations" / "instances_val.json").read_text())
    assert len(val["images"]) == 2 and val["annotations"] == []
    coco_api = pytest.importorskip("pycocotools.coco")
    loaded = coco_api.COCO(str(path))
    assert len(loaded.getImgIds()) == 8 and len(loaded.getAnnIds()) == 8
    assert loaded.annToMask(loaded.loadAnns(loaded.getAnnIds()[:1])[0]).sum() > 0.95 * _disc(30).sum()


def test_mask_pngs_are_indexed_with_class_id_plus_one(planned):
    plan, out, clip = planned
    _write("masks", plan, out)
    img = Image.open(out / "masks" / "masks" / "train" / f"{clip.id}_0000.png")
    assert img.mode == "P"
    values = np.array(img)
    assert set(np.unique(values)) == {0, 2} and (values == 2).sum() == _disc(30).sum()
    empty = np.array(Image.open(out / "masks" / "masks" / "val" / f"{clip.id}_0009.png"))
    assert set(np.unique(empty)) == {0}
    assert (out / "masks" / "images" / "val" / f"{clip.id}_0009.jpg").is_file()


def test_images_are_hardlinked_with_a_copy_fallback(planned, monkeypatch):
    plan, out, clip = planned
    assert _write("yolo_det", plan, out).linked is True
    placed = out / "yolo_det" / "images" / "train" / f"{clip.id}_0000.jpg"
    assert os.stat(placed).st_nlink >= 2

    def refuse(src, dst):
        raise OSError("cross-device link")

    monkeypatch.setattr(export_base.os, "link", refuse)
    assert _write("yolo_seg", plan, out).linked is False
    assert (out / "yolo_seg" / "images" / "train" / f"{clip.id}_0000.jpg").is_file()
```

- [ ] **Step 3: Run them** — expected: FAIL (`ImportError: cannot import name 'EXPORTERS'`).

- [ ] **Step 4: Implement** — `server/vsp/export/base.py`:

```python
"""What every exporter shares: the protocol, image placement, mask loading."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .plan import ExportPlan, PlanFrame, PlanObject

Tick = Callable[[], None]


@dataclass
class ExportSummary:
    name: str
    images: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0})
    annotations: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0})
    linked: bool = True  # False as soon as one image had to be copied instead of hardlinked


class Exporter(Protocol):
    name: str

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary: ...


def shared_image(out_dir: Path, frame: PlanFrame) -> Path:
    """The export's own copy of a frame; format directories link to it, never to the store."""
    return out_dir / "images" / f"{frame.stem}.jpg"


def place_image(src: Path, dst: Path) -> bool:
    """Hardlink `src` at `dst`, copying when the filesystem refuses. Returns True when linked."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


def read_mask(obj: PlanObject) -> np.ndarray:
    return np.array(Image.open(obj.mask_path).convert("L")) > 127
```

`server/vsp/export/yolo.py`:

```python
"""YOLO-seg (normalised polygons, one line per contour) and YOLO-det (normalised cx cy w h of the mask's box)."""

from __future__ import annotations

from pathlib import Path

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan, PlanFrame
from .polygons import mask_bbox, mask_to_polygons


def _data_yaml(plan: ExportPlan) -> str:
    names = "\n".join(f"  {c.id}: {c.name}" for c in sorted(plan.classes, key=lambda c: c.id))
    return f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n"


class _Yolo:
    name = ""

    def lines(self, frame: PlanFrame) -> list[str]:
        raise NotImplementedError

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary:
        root = out_dir / self.name
        summary = ExportSummary(self.name)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for frame in plan.frames:
            linked = place_image(shared_image(out_dir, frame), root / "images" / frame.split / f"{frame.stem}.jpg")
            summary.linked = summary.linked and linked
            lines = self.lines(frame)
            # an empty label file is YOLO's way of saying "background image"
            (root / "labels" / frame.split / f"{frame.stem}.txt").write_text("".join(f"{ln}\n" for ln in lines))
            summary.images[frame.split] += 1
            summary.annotations[frame.split] += len(lines)
            tick()
        (root / "data.yaml").write_text(_data_yaml(plan))
        return summary


class YoloSeg(_Yolo):
    name = "yolo_seg"

    def lines(self, frame: PlanFrame) -> list[str]:
        out = []
        for obj in frame.objects:
            for polygon in mask_to_polygons(read_mask(obj)):
                coords = " ".join(f"{x / frame.width:.6f} {y / frame.height:.6f}" for x, y in polygon)
                out.append(f"{obj.class_id} {coords}")
        return out


class YoloDet(_Yolo):
    name = "yolo_det"

    def lines(self, frame: PlanFrame) -> list[str]:
        out = []
        for obj in frame.objects:
            box = mask_bbox(read_mask(obj))
            if box is None:
                continue
            x, y, w, h = box
            out.append(
                f"{obj.class_id} {(x + w / 2) / frame.width:.6f} {(y + h / 2) / frame.height:.6f} "
                f"{w / frame.width:.6f} {h / frame.height:.6f}"
            )
        return out
```

`server/vsp/export/coco.py`:

```python
"""COCO instances JSON: polygon segmentation, bbox, area, iscrowd 0, one category per project class.
Category ids are class_id + 1 because many COCO consumers reserve 0 for background."""

from __future__ import annotations

import json
from pathlib import Path

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan
from .polygons import mask_bbox, mask_to_polygons


class Coco:
    name = "coco"

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary:
        root = out_dir / self.name
        summary = ExportSummary(self.name)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        categories = [{"id": c.id + 1, "name": c.name, "supercategory": "object"} for c in sorted(plan.classes, key=lambda c: c.id)]
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            images, annotations = [], []
            for frame in plan.of_split(split):
                linked = place_image(shared_image(out_dir, frame), root / "images" / split / f"{frame.stem}.jpg")
                summary.linked = summary.linked and linked
                image_id = len(images) + 1
                images.append({"id": image_id, "file_name": f"{frame.stem}.jpg", "width": frame.width, "height": frame.height})
                for obj in frame.objects:
                    mask = read_mask(obj)
                    polygons = mask_to_polygons(mask)
                    box = mask_bbox(mask)
                    if not polygons or box is None:
                        continue
                    annotations.append({
                        "id": len(annotations) + 1, "image_id": image_id, "category_id": obj.class_id + 1,
                        "segmentation": [[round(v, 2) for point in polygon for v in point] for polygon in polygons],
                        "bbox": list(box), "area": float(mask.sum()), "iscrowd": 0,
                    })
                tick()
            summary.images[split] = len(images)
            summary.annotations[split] = len(annotations)
            body = {"info": {"description": plan.project_name}, "licenses": [], "images": images,
                    "annotations": annotations, "categories": categories}
            (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(body))
        return summary
```

`server/vsp/export/mask_png.py`:

```python
"""Semantic masks: 8-bit indexed PNG, 0 = background, class_id + 1 = the class, palette from the class colours."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import ExportSummary, Tick, place_image, read_mask, shared_image
from .plan import ExportPlan


def _palette(plan: ExportPlan) -> list[int]:
    palette = [0, 0, 0] * 256
    for c in plan.classes:
        if 0 <= c.id + 1 <= 255:
            rgb = c.color.lstrip("#")
            palette[(c.id + 1) * 3 : (c.id + 1) * 3 + 3] = [int(rgb[i : i + 2], 16) for i in (0, 2, 4)]
    return palette


class MaskPng:
    name = "masks"

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> ExportSummary:
        root = out_dir / self.name
        summary = ExportSummary(self.name)
        palette = _palette(plan)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for frame in plan.frames:
            linked = place_image(shared_image(out_dir, frame), root / "images" / frame.split / f"{frame.stem}.jpg")
            summary.linked = summary.linked and linked
            canvas = np.zeros((frame.height, frame.width), np.uint8)
            for obj in frame.objects:
                canvas[read_mask(obj)] = obj.class_id + 1
            img = Image.fromarray(canvas)  # mode "L"; putpalette turns it into an indexed ("P") image in place
            img.putpalette(palette)
            img.save(root / "masks" / frame.split / f"{frame.stem}.png")
            summary.images[frame.split] += 1
            summary.annotations[frame.split] += len(frame.objects)
            tick()
        return summary
```

Replace `server/vsp/export/__init__.py` with:

```python
"""Dataset export. Reads from the store only; one ExportPlan, many Exporters (spec §8).
Adding a format = one module with a class that has `name` and `write`, registered here."""

from .base import Exporter, ExportSummary
from .coco import Coco
from .mask_png import MaskPng
from .plan import FORMATS, ExportOptions, ExportPlan, build_plan, check_request
from .yolo import YoloDet, YoloSeg

EXPORTERS: dict[str, Exporter] = {e.name: e for e in (YoloSeg(), YoloDet(), Coco(), MaskPng())}
assert tuple(EXPORTERS) == FORMATS

__all__ = ["EXPORTERS", "FORMATS", "ExportOptions", "ExportPlan", "ExportSummary", "Exporter", "build_plan", "check_request"]
```

Note for the implementer: the spec's module list names `yolo_seg.py` and `yolo_det.py`; this plan merges them into `yolo.py` because they share everything except the label line (DRY). That is a deliberate ruling by the plan author, not an omission.

- [ ] **Step 5: Run the format tests, then the whole suite and ruff.** Expected: 5 new tests pass (the COCO test's last three lines are skipped only if pycocotools could not be installed). The numbers in the tests come from the disc `(x-30)² + (y-30)² ≤ 18²`: x and y span 12..48, so the box is `[12, 12, 37, 37]` and its centre is 30.5. If a bbox assertion fails, print the actual box before changing anything: the expected values are correct for an inclusive pixel extent.

- [ ] **Step 6: Commit**

```bash
git add server/vsp/export/__init__.py server/vsp/export/base.py server/vsp/export/yolo.py server/vsp/export/coco.py server/vsp/export/mask_png.py server/tests/test_export_formats.py pyproject.toml uv.lock
git commit -m "feat(export): YOLO-seg, YOLO-det, COCO and indexed mask PNG exporters behind one protocol"
```

---

### Task 3: the export runner, the API route, and the job

**Files:**
- Create: `server/vsp/export/runner.py`, `server/vsp/api/export.py`
- Modify: `server/vsp/export/__init__.py` (export `run_export`, `ExportCancelled`), `server/vsp/api/app.py` (register the router), `README.md` (API table row + "Exporting" paragraph)
- Test: `server/tests/test_export_runner.py`, `server/tests/test_api_export.py`

**Interfaces:**
- Consumes: Task 1 and 2; `vsp.api.deps` (`State`, `AppState`, `GPU`), `vsp.jobs` (`Job`, `JobBusy`), `state.jobs.start(kind, fn)`, `state.jobs.running(GPU) -> str | None`, `job.emit({...})`, `job.cancel_requested()`. A job function that returns `{"cancelled": True}` ends with status `cancelled`.
- Produces:
  - `runner.run_export(store, project_id, clip_ids, formats, options, exports_root, progress=None, cancel=None) -> dict` returning `{"export_id", "path", "formats", "counts": {"train": {"images", "annotated", "empty"}, "val": {...}}, "linked"}`; raises `ExportCancelled` after removing the partial directory. `progress(done, total)` with `total = frames × (2 + len(formats))` (plan, copy, one pass per format).
  - `POST /api/projects/{project_id}/export` body `{"clips": [str], "formats": [str], "options": {"split_mode"?, "val_clips"?, "val_fraction"?, "stride"?, "include_empty"?, "min_area_px"?}}` → `202 {"job_id"}`; 400 for an invalid request; 404 for an unknown project or clip; 409 `{"detail", "job_id"}` while a GPU job (a propagation) is running. Job kind `"export"`, events `{"type": "progress", "done", "total"}`, result as above.

- [ ] **Step 1: Write the failing runner tests** — `server/tests/test_export_runner.py`:

```python
from __future__ import annotations

import json

import numpy as np
import pytest
from vsp.export import ExportCancelled, ExportOptions, run_export
from vsp.store.models import LabelObject


def _labelled(store, make_clip, n=6):
    project, clip = make_clip(n_frames=n)
    clip.objects = [LabelObject(id=1, class_id=0)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(n):
        store.write_mask(clip.id, 1, i, mask)
    return project, clip


def test_run_export_writes_every_format_a_manifest_and_reports_progress(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    seen = []
    result = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), tmp_path / "exports",
                        progress=lambda done, total: seen.append((done, total)))
    out = tmp_path / "exports" / result["export_id"]
    assert result["path"] == str(out) and result["formats"] == ["yolo_seg", "masks"]
    assert result["counts"] == {"train": {"images": 4, "annotated": 4, "empty": 0}, "val": {"images": 2, "annotated": 2, "empty": 0}}
    assert len(list((out / "images").glob("*.jpg"))) == 6
    assert (out / "yolo_seg" / "data.yaml").is_file() and (out / "masks" / "masks" / "val").is_dir()
    assert not (out / "coco").exists()
    manifest = json.loads((out / "export.json").read_text())
    assert manifest["options"]["split_mode"] == "contiguous" and manifest["options"]["min_area_px"] == 64
    assert manifest["clips"] == [{"id": clip.id, "name": clip.source.name, "train": 4, "val": 2}]
    assert manifest["classes"] == [{"id": 0, "name": "part", "color": manifest["classes"][0]["color"], "yolo_id": 0, "coco_id": 1, "mask_value": 1}]
    assert seen[-1] == (6 * 4, 6 * 4) and [d for d, _ in seen] == sorted(d for d, _ in seen)
    # the store was only read: no stray file, and the export holds copies, not links to store frames
    import os
    assert os.stat(store.frame_path(clip.id, 0)).st_nlink == 1


def test_a_cancelled_export_leaves_nothing_behind(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 8

    with pytest.raises(ExportCancelled):
        run_export(store, project.id, [clip.id], ["coco"], ExportOptions(), tmp_path / "exports", cancel=cancel)
    assert list((tmp_path / "exports").iterdir()) == []
```

- [ ] **Step 2: Run them** — expected: FAIL (`ImportError: cannot import name 'ExportCancelled'`).

- [ ] **Step 3: Implement** — `server/vsp/export/runner.py`:

```python
"""Run one export: plan, copy each image once, let every chosen exporter serialise the plan, write the manifest."""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from ..store import ProjectStore, now_iso
from .base import shared_image
from .plan import ExportOptions, ExportPlan, build_plan, check_request


class ExportCancelled(Exception):
    pass


def _counts(plan: ExportPlan) -> dict:
    out = {}
    for split in ("train", "val"):
        frames = plan.of_split(split)
        annotated = sum(1 for f in frames if f.objects)
        out[split] = {"images": len(frames), "annotated": annotated, "empty": len(frames) - annotated}
    return out


def run_export(
    store: ProjectStore, project_id: str, clip_ids: list[str], formats: list[str], options: ExportOptions,
    exports_root: Path, progress: Callable[[int, int], None] | None = None, cancel: Callable[[], bool] | None = None,
) -> dict:
    from . import EXPORTERS  # here, not at module level: the package __init__ imports this module

    check_request(store, project_id, clip_ids, formats, options)
    n_candidates = sum(
        len([i for i in range(c.frame_count) if i not in set(c.deleted_frames)][:: options.stride])
        for c in (store.read_clip(cid) for cid in clip_ids)
    )
    state = {"done": 0, "total": max(1, n_candidates * (2 + len(formats)))}

    def tick() -> None:
        if cancel is not None and cancel():
            raise ExportCancelled
        state["done"] += 1
        if progress is not None:
            progress(min(state["done"], state["total"]), state["total"])

    exports_root = Path(exports_root)
    exports_root.mkdir(parents=True, exist_ok=True)
    export_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = exports_root / export_id
    try:
        plan = build_plan(store, project_id, clip_ids, options, tick=tick)
        # include_empty=False may have dropped frames: shrink the total so the bar still ends at 100%
        state["total"] = state["done"] + len(plan.frames) * (1 + len(formats))
        (out / "images").mkdir(parents=True)
        for frame in plan.frames:
            shutil.copy2(frame.image_path, shared_image(out, frame))  # a copy: store frames are never linked outside
            tick()
        summaries = [EXPORTERS[name].write(plan, out, tick) for name in formats]
        counts = _counts(plan)
        linked = all(s.linked for s in summaries)
        manifest = {
            "export_id": export_id, "created_at": now_iso(),
            "project": {"id": plan.project_id, "name": plan.project_name},
            "options": {**asdict(plan.options), "val_clip_ids": list(plan.val_clip_ids)},
            "clips": [
                {"id": cid, "name": plan.clip_names[cid],
                 "train": sum(1 for f in plan.frames if f.clip_id == cid and f.split == "train"),
                 "val": sum(1 for f in plan.frames if f.clip_id == cid and f.split == "val")}
                for cid in plan.clip_ids
            ],
            "classes": [
                {"id": c.id, "name": c.name, "color": c.color, "yolo_id": c.id, "coco_id": c.id + 1, "mask_value": c.id + 1}
                for c in sorted(plan.classes, key=lambda c: c.id)
            ],
            "formats": [asdict(s) for s in summaries],
            "counts": counts,
            "images_hardlinked": linked,  # false: the filesystem refused hardlinks, so images were copied per format
        }
        import json

        (out / "export.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # never leave a half-written dataset that looks complete
        raise
    return {"export_id": export_id, "path": str(out), "formats": list(formats), "counts": counts, "linked": linked}
```

Move the two function-level imports (`json`, and only `json`) to the top of the file; `from . import EXPORTERS` stays inside the function for the reason its comment gives. In `server/vsp/export/__init__.py` add `from .runner import ExportCancelled, run_export` AFTER the `EXPORTERS = ...` lines and add both names to `__all__`.

- [ ] **Step 4: Run the runner tests** — expected: 2 passed. `now_iso` is exported by `vsp.store` (`server/vsp/store/__init__.py`); if it is not, import it from `vsp.store.project`.

- [ ] **Step 5: Write the failing API tests** — `server/tests/test_api_export.py`. Reuse the helpers of `server/tests/test_api_core.py` by importing them (`from test_api_core import _frames_dir, _project, _settings, client, wait_job  # noqa: F401`); if that import style fails under this repo's pytest rootdir, copy the five helpers into the new file instead.

```python
from __future__ import annotations

import json
from pathlib import Path

from test_api_core import _frames_dir, _project, client, wait_job  # noqa: F401


def _clip_with_masks(client, tmp_path, project_id):
    started = client.post(f"/api/projects/{project_id}/clips/import-frames", json={"path": str(_frames_dir(tmp_path, n=6)), "fps": 10})
    clip_id = wait_job(client, started.json()["job_id"])["result"]["clip_id"]
    accepted = client.post(f"/api/clips/{clip_id}/accept", json={"frame": 0, "prompt": {"type": "clicks", "points": [[30, 30, 1]]}, "class_id": 0})
    assert accepted.status_code == 200, accepted.text
    job = client.post(f"/api/clips/{clip_id}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]
    assert wait_job(client, job)["status"] == "done"
    return clip_id


def test_export_runs_as_a_job_and_reports_where_it_wrote(client, tmp_path):
    project = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project["id"])
    started = client.post(f"/api/projects/{project['id']}/export", json={"clips": [clip_id], "formats": ["yolo_det", "coco"]})
    assert started.status_code == 202, started.text
    snap = wait_job(client, started.json()["job_id"])
    assert snap["status"] == "done" and snap["kind"] == "export"
    out = Path(snap["result"]["path"])
    assert out.parent.name == "exports" and (out / "coco" / "annotations" / "instances_train.json").is_file()
    assert json.loads((out / "export.json").read_text())["options"]["val_fraction"] == 0.2


def test_export_request_errors(client, tmp_path):
    project = _project(client)
    clip_id = _clip_with_masks(client, tmp_path, project["id"])
    url = f"/api/projects/{project['id']}/export"
    assert client.post(url, json={"clips": [clip_id], "formats": ["voc"]}).status_code == 400
    assert client.post(url, json={"clips": [], "formats": ["coco"]}).status_code == 400
    assert client.post(url, json={"clips": ["nope"], "formats": ["coco"]}).status_code == 404
    assert client.post("/api/projects/nope/export", json={"clips": [clip_id], "formats": ["coco"]}).status_code == 404
    assert client.post(url, json={"clips": [clip_id], "formats": ["coco"], "options": {"val_fraction": 1.5}}).status_code == 422
    body = client.post(url, json={"clips": [clip_id], "formats": ["coco"], "options": {"split_mode": "per_clip"}})
    assert body.status_code == 400 and "at least two clips" in body.json()["detail"]
```

Progress events are covered by the runner test, so the API test does not repeat them.

- [ ] **Step 6: Implement the route** — `server/vsp/api/export.py`:

```python
"""POST /api/projects/{id}/export: start an export job (spec §4, §8)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..export import ExportCancelled, ExportOptions, check_request, run_export
from ..jobs import Job, JobBusy
from .deps import GPU, State

router = APIRouter(tags=["export"])


class ExportOptionsBody(BaseModel):
    split_mode: Literal["per_clip", "contiguous"] | None = None
    val_clips: list[str] | None = None
    val_fraction: float | None = Field(default=None, ge=0, lt=1)
    stride: int = Field(default=1, ge=1)
    include_empty: bool = True
    min_area_px: int | None = Field(default=None, ge=0)


class ExportBody(BaseModel):
    clips: list[str]
    formats: list[str]
    options: ExportOptionsBody = Field(default_factory=ExportOptionsBody)


@router.post("/api/projects/{project_id}/export", status_code=202)
def start_export(project_id: str, body: ExportBody, state: State) -> dict:
    cfg = state.settings.export
    o = body.options
    options = ExportOptions(
        split_mode=o.split_mode, val_clip_ids=None if o.val_clips is None else tuple(o.val_clips),
        val_fraction=cfg.val_fraction if o.val_fraction is None else o.val_fraction, stride=o.stride,
        include_empty=o.include_empty, min_area_px=cfg.min_area_px if o.min_area_px is None else o.min_area_px,
    )
    check_request(state.store, project_id, body.clips, body.formats, options)  # 400 / 404 now, not as a job error
    # Export only reads, so it does not take the GPU. But a propagation is rewriting the very masks an export
    # would read, so refuse to start while one runs (the reverse race, a propagation started mid-export, only
    # affects masks the export may already have passed; mask files are written atomically, so none is torn).
    busy = state.jobs.running(GPU)
    if busy is not None:
        raise JobBusy(busy)
    exports_root = state.settings.project.data_root / "exports"

    def work(job: Job) -> dict:
        try:
            return run_export(
                state.store, project_id, body.clips, body.formats, options, exports_root,
                progress=lambda done, total: job.emit({"type": "progress", "done": done, "total": total}),
                cancel=job.cancel_requested,
            )
        except ExportCancelled:
            return {"cancelled": True}

    return {"job_id": state.jobs.start("export", work).id}
```

In `server/vsp/api/app.py`: import the module next to the other routers (`from . import clips, export, jobs, labeling, projects` — match the existing import line's style) and add `app.include_router(export.router)` after `app.include_router(labeling.router)`. Check how `NotFound` from the store is mapped there (the 404 assertions rely on the existing handler).

- [ ] **Step 7: Run the API tests, the whole suite, ruff.** All green. A progress event is emitted per frame per pass; for a 1-minute clip at 30 fps and four formats that is about 11 000 small dicts in the job's event list, which is acceptable for v1 — do not add throttling.

- [ ] **Step 8: README** — in the API table of `README.md` add the row `POST /api/projects/{p}/export` with body `{clips, formats, options}` → `202 {job_id}`, "400 invalid request, 409 while a propagation runs". After the "Web GUI" section's demo-data paragraph add a short "Exporting" paragraph: where exports land (`data/exports/<export_id>/`), the four directory names, that `export.json` records options, clips, the class map (`yolo_id`, `coco_id`, `mask_value`) and counts, that the split is per clip (whole clips) or contiguous (the tail of each clip) and never random per frame because neighbouring video frames would leak between train and val, and that a cancelled or failed export removes its directory.

- [ ] **Step 9: Commit**

```bash
git add server/vsp/export/__init__.py server/vsp/export/runner.py server/vsp/api/export.py server/vsp/api/app.py server/tests/test_export_runner.py server/tests/test_api_export.py README.md
git commit -m "feat(export): export runner with manifest and cleanup, and the export job route"
```

---

### Task 4: the Export page in the web GUI

**Files:**
- Create: `web/src/lib/exportSelection.ts`, `web/src/lib/exportSelection.test.ts`, `web/src/pages/Export.tsx`, `web/src/pages/Export.test.tsx`
- Modify: `web/src/api/types.ts`, `web/src/api/rest.ts`, `web/src/App.tsx` (route), `web/src/pages/Clips.tsx` (link to the page)

**Interfaces:**
- Consumes: `POST /api/projects/{id}/export` (Task 3); existing `api.listProjects()`, `api.listClips(projectId)`, `api.cancelJob(jobId)`, `thumbUrl(clipId, idx)`, `ApiError` (`status`, `jobId`), `useJob(jobId)` → `{status, done, total, result, error, ...}`, `isTerminal(status)`, `AppHeader`, `ClipSummary` (`id, source.name, frame_count, width, height, fps, created_at`). Read `web/src/pages/Clips.tsx`, `web/src/components/ImportJobRow.tsx` and `web/src/hooks/useJob.ts` before writing anything: the page must look and behave like its siblings.
- Produces: route `/projects/:projectId/export`; `api.startExport(projectId, body: ExportRequest): Promise<{ job_id: string }>`.

**Ruling recorded by the plan author:** the spec (§7) calls this an "Export dialog". The user has since ruled that every page keeps its controls in a right-hand sidebar, so this is a PAGE: clip selection on the left, options and the Start button in a right sidebar. Same content as the spec's dialog.

- [ ] **Step 1: Types and client.** In `web/src/api/types.ts` add:

```ts
export type ExportFormat = "yolo_seg" | "yolo_det" | "coco" | "masks";
export type SplitMode = "per_clip" | "contiguous";
export type ExportRequest = {
  clips: string[];
  formats: ExportFormat[];
  options: { split_mode: SplitMode; val_clips?: string[]; val_fraction: number; stride: number; include_empty: boolean };
};
export type SplitCounts = { images: number; annotated: number; empty: number };
export type ExportResult = { export_id: string; path: string; formats: ExportFormat[]; counts: { train: SplitCounts; val: SplitCounts }; linked: boolean };
```

In `web/src/api/rest.ts`, inside the `api` object and in the style of its neighbours (they use the file's `request<T>(path, init)` helper with a JSON body — copy how `propagate` does it), add `startExport: (projectId: string, body: ExportRequest) => request<{ job_id: string }>(...)` posting to `/api/projects/${projectId}/export`. Add a test for it in `web/src/api/rest.test.ts` following the existing pattern for a POST with a JSON body (assert URL, method and parsed body).

- [ ] **Step 2: Write the failing selection-logic tests** — `web/src/lib/exportSelection.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { ClipSummary } from "../api/types";
import { defaultValClips, exportProblem, splitModeFor } from "./exportSelection";

const clip = (id: string, created_at: string): ClipSummary =>
  ({ id, created_at, project_id: "p", source: { type: "video", name: `${id}.mp4` }, fps: 10, stride: 1, width: 640, height: 480,
     frame_count: 40, deleted_frames: [], original_names: null, objects: [] }) as unknown as ClipSummary;
const clips = [clip("a", "2026-01-01"), clip("b", "2026-01-03"), clip("c", "2026-01-02")];

describe("export selection", () => {
  it("uses the per-clip split for two or more clips and the contiguous split for one", () => {
    expect(splitModeFor(1)).toBe("contiguous");
    expect(splitModeFor(2)).toBe("per_clip");
  });
  it("pre-selects the most recently imported clips for validation, never all of them", () => {
    expect(defaultValClips(clips, ["a", "b", "c"], 0.2)).toEqual(["b"]);
    expect(defaultValClips(clips, ["a", "b", "c"], 0.5)).toEqual(["b", "c"]);
    expect(defaultValClips(clips, ["a", "c"], 0.9)).toEqual(["c"]);
    expect(defaultValClips(clips, ["a", "b"], 0)).toEqual([]);
    expect(defaultValClips(clips, ["a"], 0.2)).toEqual([]);
  });
  it("says what stops an export, or null", () => {
    const ok = { selected: ["a", "b"], formats: ["coco"], valClips: ["b"], valFraction: 0.2, stride: 1 };
    expect(exportProblem(ok)).toBeNull();
    expect(exportProblem({ ...ok, selected: [] })).toBe("Choose at least one clip.");
    expect(exportProblem({ ...ok, formats: [] })).toBe("Choose at least one format.");
    expect(exportProblem({ ...ok, valClips: ["a", "b"] })).toBe("Leave at least one clip for training.");
    expect(exportProblem({ ...ok, valFraction: 1 })).toBe("The validation share must be below 100%.");
    expect(exportProblem({ ...ok, stride: 0 })).toBe("Keep every Nth frame needs a whole number of 1 or more.");
  });
});
```

- [ ] **Step 3: Implement** — `web/src/lib/exportSelection.ts`:

```ts
import type { ClipSummary, SplitMode } from "../api/types";

/** Whole clips go to train or val when there are several; one clip is split at its tail. Never per frame. */
export const splitModeFor = (selectedCount: number): SplitMode => (selectedCount >= 2 ? "per_clip" : "contiguous");

/** The most recently imported ceil(fraction × n) of the selected clips, but never all of them (spec §8). */
export function defaultValClips(clips: ClipSummary[], selected: string[], fraction: number): string[] {
  if (selected.length < 2 || fraction <= 0) return [];
  const chosen = clips.filter((c) => selected.includes(c.id));
  const newestFirst = [...chosen].sort((a, b) => (a.created_at === b.created_at ? b.id.localeCompare(a.id) : b.created_at.localeCompare(a.created_at)));
  const n = Math.min(chosen.length - 1, Math.ceil(fraction * chosen.length));
  return newestFirst.slice(0, n).map((c) => c.id);
}

type Draft = { selected: string[]; formats: string[]; valClips: string[]; valFraction: number; stride: number };

export function exportProblem(d: Draft): string | null {
  if (d.selected.length === 0) return "Choose at least one clip.";
  if (d.formats.length === 0) return "Choose at least one format.";
  if (!(d.valFraction >= 0 && d.valFraction < 1)) return "The validation share must be below 100%.";
  if (!Number.isInteger(d.stride) || d.stride < 1) return "Keep every Nth frame needs a whole number of 1 or more.";
  if (splitModeFor(d.selected.length) === "per_clip" && d.valClips.filter((id) => d.selected.includes(id)).length >= d.selected.length)
    return "Leave at least one clip for training.";
  return null;
}
```

Check the real `ClipSummary` type in `web/src/api/types.ts`; if a field in the test's `clip()` helper does not exist there, drop it from the helper (the cast makes extra fields harmless, missing ones matter only if the code reads them).

- [ ] **Step 4: Write the failing page tests** — `web/src/pages/Export.test.tsx`, following `web/src/pages/Clips.test.tsx` for the harness (jsdom docblock, `vi.mock("../api/rest", ...)` spreading the real module, a `FakeSocket` stubbed as the global `WebSocket` in `beforeEach` — copy the `FakeSocket` class from `Clips.test.tsx`; re-stub it in `beforeEach` because `vi.resetAllMocks()` does not undo `vi.stubGlobal`). Render with:

```tsx
render(
  <MemoryRouter initialEntries={["/projects/p1/export"]}>
    <Routes><Route path="/projects/:projectId/export" element={<Export />} /></Routes>
  </MemoryRouter>,
);
```

Mock `api.listProjects` → one project `{id: "p1", name: "bench", classes: [...]}` and `api.listClips` → three clips `a` (oldest), `b` (newest), `c`. Required tests (write them all, with real assertions):

1. *"selects every clip, pre-selects the newest for validation, and starts the export with exactly what is shown"* — after load, three checkboxes named `Include a.mp4` … are checked; the card of `b.mp4` shows the text `Validation`, the others `Training`; all four format checkboxes (`YOLO segmentation`, `YOLO detection`, `COCO`, `Mask images`) are checked. Click the button named `Export 3 clips`. Expect `api.startExport` called with `"p1"` and `{clips: ["a","b","c"], formats: ["yolo_seg","yolo_det","coco","masks"], options: {split_mode: "per_clip", val_clips: ["b"], val_fraction: 0.2, stride: 1, include_empty: true}}`.
2. *"lets the user move a clip between training and validation"* — click the button named `Use a.mp4 for validation`; then start; `val_clips` is `["a","b"]` in selection order of the clip list (`["a","b"]`). Click `Use a.mp4 for training` puts it back.
3. *"with one clip selected it splits the tail of the clip and sends no validation clips"* — untick `Include b.mp4` and `Include c.mp4`; the button reads `Export 1 clip`; the note text matches `/last 20% of its frames/`; the request has `split_mode: "contiguous"` and NO `val_clips` key.
4. *"says why it cannot start"* — untick all formats: the Start button is disabled and the text `Choose at least one format.` is shown.
5. *"shows progress, then where the dataset was written"* — `api.startExport` resolves `{job_id: "j1"}`; drive the FakeSocket: `{type:"progress", done: 10, total: 40, seq: 0}` → text `10 of 40` is shown and a `Cancel` button exists (clicking it calls `api.cancelJob("j1")`); then `{type:"done", result: {export_id:"e1", path:"/data/exports/e1", formats:["coco"], counts:{train:{images:32,annotated:30,empty:2}, val:{images:8,annotated:8,empty:0}}, linked:true}, seq: 1}` → the path `/data/exports/e1` is shown, plus `32 training images` and `8 validation images`.
6. *"reports a refused start"* — `api.startExport` rejects with `new ApiError(409, "job abc is still running", "abc")` (check `ApiError`'s real constructor argument order in `rest.ts`) → an alert contains `A propagation is running. Wait for it to finish, then export.`; any other error shows its message.

- [ ] **Step 5: Implement `web/src/pages/Export.tsx`.** Structure (match `Clips.tsx` for layout classes and request guards):

```
<div className="flex h-full flex-col">
  <AppHeader>  ‹ Clips link (aria-label "Back to clips", to "/")   <h1 className="truncate font-medium">Export {project.name}</h1>  </AppHeader>
  <div className="flex min-h-0 flex-1 flex-col overflow-y-auto md:flex-row md:overflow-hidden">
    <div className="flex min-w-0 flex-1 flex-col gap-4 p-4 md:overflow-y-auto">      ← LEFT: clip cards
    <aside className="... md:w-80 md:border-l ... bg-panel p-3">                     ← RIGHT: options, Start, progress, result
```

Left — one card per clip (same card look as `Clips.tsx`): thumbnail, a checkbox with accessible name `Include <name>`, the meta line `<frame_count> frames, <percent>% labeled` (reuse the percent logic from `Clips.tsx`; if it is a local function there, move it to `web/src/lib/timeline.ts` as an exported `percentLabeled(clip)` and import it in both pages rather than duplicating it). When the split is per clip and the clip is included, show its role as plain text `Training` or `Validation` plus a small `btn` named `Use <name> for validation` / `Use <name> for training`. Excluded clips show `Not exported` in `text-dim`. Validation cards get the inverted `btn-on` look on the role chip only (no hue).

Right sidebar sections, each a `<section className="flex flex-col gap-2">` with an `<h2 className="heading">`:
- **Formats** — four checkboxes (labels exactly as in test 1), each followed by a `note` naming the directory it writes (`yolo_seg/`, `yolo_det/`, `coco/`, `masks/`).
- **Split** — a `note` that explains the active mode: per clip → `Whole clips go to training or validation, so neighbouring frames never end up on both sides. Choose the validation clips on the left.`; contiguous → `With one clip, the last 20% of its frames become the validation set.` (the percentage follows the field). A number field `Validation share (%)` (default 20, min 0, max 99). Changing it re-runs `defaultValClips` ONLY while the user has not changed a role by hand (keep a `touched` flag; selecting/deselecting clips also re-runs it while untouched).
- **Frames** — number field `Keep every Nth frame` (default 1, min 1); checkbox `Include frames without a mask` (default checked) with the note `They teach the model what the background looks like.`
- **Start** — the problem text from `exportProblem` (in `note text-ink`) when there is one; the button `Export <n> clip(s)` (`btn btn-on`, disabled when there is a problem, a job is running, or a start request is in flight). While the job runs: `<progress>`, the text `<done> of <total>` and a `Cancel` button (`btn`) calling `api.cancelJob`. On `done`: the heading `Dataset written`, the path in a `field`-styled read-only `<input aria-label="Dataset folder">` (so it can be selected and copied), and the counts as `32 training images` / `8 validation images` with `, <n> without a mask` appended when `empty > 0`; when `linked` is false add the note `This disk does not support hardlinks, so each format holds its own copy of the images.` On `cancelled`: `Export cancelled. Nothing was written.` On `error`: `role="alert"` with the job error.

Behaviour rules: guard every async continuation with a `closed` flag or a request id (copy the `clipsSeq` pattern from `Clips.tsx`); use `useJob(jobId)` for the job and read `job.result` as `ExportResult` only when `job.status === "done"`; keep the in-flight guard as a ref plus state (as `Clips.tsx` does with `pending`); an unknown project id (not in `listProjects`) shows an `alert` `This project no longer exists.` with the back link. Do not persist anything in `localStorage`.

In `web/src/App.tsx` add `<Route path="/projects/:projectId/export" element={<Export />} />`. In `web/src/pages/Clips.tsx`, at the top of the right sidebar (above Classes) when the project has at least one clip, add `<Link to={`/projects/${project.id}/export`} className="btn btn-on">Export dataset</Link>`, and add one assertion for it to `Clips.test.tsx` (`getByRole("link", { name: "Export dataset" })` has `href` `/projects/<id>/export`).

- [ ] **Step 6: Run everything** — `npx tsc --noEmit` clean; `npx vitest run` all green (the previous 143 plus the new ones); `npm run build` succeeds. `HF_HUB_OFFLINE=1 uv run pytest -q` still green.

- [ ] **Step 7: Commit**

```bash
git add web/src/api/types.ts web/src/api/rest.ts web/src/api/rest.test.ts web/src/lib/exportSelection.ts web/src/lib/exportSelection.test.ts web/src/lib/timeline.ts web/src/pages/Export.tsx web/src/pages/Export.test.tsx web/src/pages/Clips.tsx web/src/pages/Clips.test.tsx web/src/App.tsx
git commit -m "feat(web): Export page - clip and role selection, formats, split, progress, dataset path"
```

(Leave `web/src/lib/timeline.ts` out of `git add` if you did not need to touch it.)

---

### Task 5 (controller, not a subagent): real-server check and documentation

- [ ] Start `HF_HUB_OFFLINE=1 uv run vsp serve`, seed with `uv run python scripts/seed_demo.py` twice into ONE project is not possible with the script (it creates a project per run), so: seed once, then import the same `data/demo_frames` directory a second time into that project through the API and label + propagate it, so a two-clip per-clip export can be exercised.
- [ ] Export all four formats through the API; verify on disk: image counts per split, no stem in both splits, `export.json` content, hardlink count of a format image ≥ 2, a YOLO-seg polygon rasterised back against its mask (IoU > 0.95), `pycocotools` loads both COCO files, mask PNG values.
- [ ] Open the Export page in headless Firefox (recipe in the design-pass brief, including the `getBoundingClientRect()` flush) and check the screenshot: right-hand sidebar, roles on the cards, no hue in the chrome.
- [ ] README manual QA checklist: add an item for exporting (per-clip with a hand-picked validation clip; one clip contiguous; cancel midway and confirm the folder is gone; load the YOLO `data.yaml` / COCO JSON in the training tool of choice).
- [ ] Roadmap: mark Plan 4 done, append "As executed" notes to this plan.

---

## As executed (2026-09-21)

Tasks 1-2 by Haiku, 3-4 by Sonnet, each reviewed by Sonnet; fixes by the controller. What changed against the text above:

- **Task 1 (review):** `_stems` let an original file stem take ANOTHER frame's 6-digit index form (frames 3 "000007.png"
  and 7 shared a stem); all index forms are now reserved first. `val_clip_ids` sent with the contiguous split was
  silently ignored; it is a BadRequest now.
- **Task 2:** the implementer lowered the pycocotools assertion to 0.9494. Root cause: OpenCV contour vertices sit on
  pixel CENTRES while pycocotools (and normalised YOLO boxes, and `mask_bbox`) use continuous coordinates, so polygons
  lost half a pixel per side (IoU 0.93 on a 36 px disc) and YOLO-seg disagreed with YOLO-det/COCO by half a pixel.
  `mask_to_polygons` now traces the mask enlarged `EDGE_SCALE = 2` times and returns continuous coordinates for EVERY
  format; `polygons_to_mask` floors (as Ultralytics' loader truncates). Round trips are exact. Also from the review:
  class names in `data.yaml` are JSON-quoted (":" and "#" broke the file), and more than 255 classes is a BadRequest
  for the mask PNG format.
- **Task 3:** the plan's API test treated `_project()` as returning a dict (it returns the id); helpers were copied
  into `test_api_export.py` because importing the `client` fixture trips ruff F811; the README had no API table, so the
  route is described in prose. Review fixes: the export directory is created up front with `exist_ok=False` outside the
  cleanup `try` (an id collision could otherwise delete another export); `DELETE /api/clips/{id}` answers 409 while an
  export runs (`JobManager.active(kind)`); API tests for 409-during-propagation, failed-export cleanup, and
  delete-during-export.
- **Task 4:** the plan's sample `done` event nested the result under `result`; the real event is flat
  (`JobManager._run` spreads the result onto the event) and the page follows the real shape.
- **Final whole-branch review (Opus):** two silent data-correctness defects, both fixed test-first. (1) The noise
  filter lived only in `mask_to_polygons`, so one stray tracker pixel stretched the YOLO-det and COCO boxes across the
  frame while the polygon ignored it, and a mask of scattered specks was "annotated" in two formats and absent in the
  other two. `clean_mask(mask, min_area_px)` now removes blobs below `min_area_px` ONCE; the plan decides presence with
  it and every exporter reads the cleaned mask (`PlanObject.min_area_px`), so `min_area_px` is a per-blob threshold
  that reaches every format. (2) With `include_empty=False` the contiguous tail was taken before the empty frames were
  dropped (a clip labelled only at its start got no validation frames); the split now follows the filter. Also:
  duplicate formats are refused; the Clips page explains a refused delete instead of showing a job id; tests for
  image/label/annotation/mask pairing per split, cross-format consistency on one noisy frame, and a clip with no masks.
  Left as notes: a killed server leaves a half-written export without `export.json`; job events are never evicted.
