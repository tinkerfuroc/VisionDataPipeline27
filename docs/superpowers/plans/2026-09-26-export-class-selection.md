# Plan C: Export Class Selection and Per-Object Cutouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** an export can be limited to some of the project's classes (renumbered 0..k-1, with a class map back to
the project in the manifest); the semantic mask PNG keeps a small object visible on top of a larger one and
reports the overlap; augmented copies cut composites per object (from the source's COCO annotations) instead of
per class region.

**Architecture:** the class filter lives in the plan: `build_plan` drops objects of unselected classes before the
empty-frame filter and renumbers the selected classes, so every exporter keeps reading `plan.classes` /
`PlanObject.class_id` exactly as today and needs no change for the renumbering. `ExportPlan.class_map` (export id →
project class id) is written into the manifest as `project_class_id` and rebuilt from it by derived copies through
two small helpers in `plan.py` (`options_from_manifest`, `classes_from_manifest`). The mask PNG paints through a
pure `paint()` function (largest area first, overlap counted) and reports `overlap_px` in its own summary
subclass. `_cutouts_and_sizes` rasterises the source's COCO train polygons per object, and falls back to the
per-class regions of the mask PNGs (flagged in the derived manifest and noted in the augment panel).

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, numpy, OpenCV, Pillow, pytest (uv-managed env: run everything
with `uv run`); React + TypeScript + Vitest for the web page (run through `node node_modules/.bin/...`, no npm).

**Spec:** `docs/superpowers/specs/2026-09-26-multi-object-labeling-and-export-selection-design.md` (§10, §12, §13,
§14 plan C row).

## Global Constraints

- Python tooling only through uv: `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server`. Never
  pip/conda. Web: `cd web && node node_modules/.bin/vitest run`, `node node_modules/.bin/tsc --noEmit`,
  `node node_modules/.bin/vite build` (no npm).
- **Class selection:** `ExportOptions.class_ids: tuple[int, ...] | None = None` (None = every class). It always
  holds **project** class ids (also in derived copies). `check_request` resolves it to a sorted tuple (or keeps
  None). An empty tuple → 400 `"choose at least one class to export"`; an id that is not a project class → 400
  `"unknown class <id> in project <project_id>"`; a repeated id → 400 `"a class is listed twice"`.
- **API:** `ExportBody.options.classes: list[int] | null` (default null) maps to `class_ids` (list → tuple).
- **Order in `build_plan`:** objects of unselected classes are dropped **before** the empty-frame filter (a frame
  left without objects is kept or skipped by `include_empty`), and the split is computed after that filter
  (unchanged order). Dropped objects' masks are not even read.
- **Renumbering:** the selected classes, ordered by project class id, get export ids `0..k-1`.
  `plan.classes` holds `ClassDef`s whose `id` is the export id; `PlanObject.class_id` is the export id;
  `ExportPlan.class_map: dict[int, int]` (export id → project class id) is a **required** field. With every class
  selected the map is the identity. YOLO class = export id; COCO `category_id` = mask PNG value = export id + 1.
- **Manifest:** every `classes` entry is `{"id", "project_class_id", "name", "color", "yolo_id", "coco_id",
  "mask_value"}` where `id`/`yolo_id` are export ids, `coco_id`/`mask_value` export id + 1. `options.class_ids` is
  written as a JSON list (or null) and read back as a tuple (`options_from_manifest`).
- **Derived copies** build `classes` and `class_map` with `classes_from_manifest(manifest)`, whose map is
  `{c["id"]: c.get("project_class_id", c["id"])}`, so an older manifest maps to itself and the mapping survives any
  chain of derivations. A derived plan never re-runs the class filter.
- **Semantic mask PNG:** objects are painted largest area first (area of the cleaned mask; ties keep plan order),
  so a smaller object on top of a larger one stays visible. `overlap_px` = pixels claimed by more than one object,
  summed over every frame of both splits, reported **only** in the `masks` format summary (`MaskSummary`), computed
  in `MaskPng.write`, and summed by `_merged_format_summaries` (derived copies; a side without the key counts 0).
- **Cutouts:** with `coco` in the source's formats, `_cutouts_and_sizes` cuts one cutout per COCO train
  annotation (`coco/annotations/instances_train.json`, `file_name` stem → `images/<stem>.jpg`, polygons
  rasterised by `polygons_to_mask`, class = `category_id - 1`); otherwise per class region of the mask PNGs. The
  derived manifest records `augment.cutouts: "per_object" | "per_class_region"` (and the job result `cutouts`). The
  `masks` format stays required for augmenting.
- Every commit leaves `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server` and the three web
  commands green.
- Commit messages end with the two attribution lines: a `Co-Authored-By:` line naming the model that wrote the
  commit (for example `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`) and
  `Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6`.
- Stage only the files your task touches (`git add <paths>`), never `git add -A`. Other agents may be working in
  the same worktree.
- All controls stay in the page's right-hand sidebar.

## File map

| File | Change |
|---|---|
| `server/vsp/export/plan.py` | `ExportOptions.class_ids`, validation in `check_request`, `select_classes`, filter + renumbering in `build_plan`, `ExportPlan.class_map`, `options_from_manifest`, `classes_from_manifest` |
| `server/vsp/export/__init__.py` | export the new helpers |
| `server/vsp/api/export.py` | `ExportOptionsBody.classes` |
| `server/vsp/export/runner.py` | `manifest_for`: `project_class_id`, `class_ids` as a list |
| `server/vsp/export/mask_png.py` | `MaskSummary`, `paint()` (largest first, overlap count) |
| `server/vsp/export/yolo.py`, `coco.py`, `base.py`, `polygons.py` | no change (they already read export ids from the plan; tests pin it) |
| `server/vsp/augment/derive.py` | plan from manifest helpers, `overlap_px` merge, per-object cutouts + `cutouts` flag |
| `server/vsp/augment/composite.py` | no change (`cutout_from` is reused as is) |
| `server/vsp/train/models.py` | no change: `find_export` sorts class names by `yolo_id`, which is the export id |
| `web/src/api/types.ts` | `ExportRequest.options.classes` |
| `web/src/lib/exportSelection.ts` (+ test) | `classObjectCounts`, "no class" problem |
| `web/src/pages/Export.tsx` (+ test) | Classes section in the right sidebar |
| `web/src/components/AugmentPanel.tsx` (+ test) | note on the per-class-region fallback |
| `README.md` | export classes, manifest map, overlap, cutouts, QA step |
| tests | `test_export_plan.py`, `test_api_export.py`, `test_export_formats.py`, `test_export_runner.py`, `test_augment_derive.py` |

Existing tests that compare exactly and must be updated: `test_export_runner.py::
test_run_export_writes_every_format_a_manifest_and_reports_progress` (the `manifest["classes"] == [...]` line gains
`"project_class_id": 0`; Task 2) and `Export.test.tsx` "selects every clip, pre-selects the newest …" (the exact
`options` object gains `classes: null`; Task 5), and `exportSelection.test.ts` "says what stops an export" (the
`ok` draft gains `classes`/`classCount`; Task 5), and `web/src/api/rest.test.ts` "posts an export request with the
server's field names" (its `options` gains `classes: [0, 2]`, otherwise `tsc --noEmit` fails; Task 5). No other test compares manifests, options or summaries exactly
(checked: `test_api_augment.py` compares only the background fetch result; `train_fixtures.make_export` writes its
own manifest and is read by `find_export`, which ignores the new keys).

---

### Task 1: Plan — class filter, renumbering, class map, options; API body; derived plan from the manifest

**Files:**
- Modify: `server/vsp/export/plan.py`, `server/vsp/export/__init__.py`, `server/vsp/api/export.py`,
  `server/vsp/augment/derive.py` (only the `ExportPlan(...)` construction in `run_augment` and its imports)
- Test: `server/tests/test_export_plan.py`, `server/tests/test_api_export.py`

**Interfaces:**
- Produces:
  - `ExportOptions.class_ids: tuple[int, ...] | None = None`
  - `select_classes(classes: Sequence[ClassDef], class_ids: tuple[int, ...] | None) -> tuple[tuple[ClassDef, ...], dict[int, int]]`
  - `ExportPlan.class_map: dict[int, int]` (required, export id → project class id)
  - `options_from_manifest(options: Mapping[str, Any]) -> ExportOptions`
  - `classes_from_manifest(manifest: Mapping[str, Any]) -> tuple[tuple[ClassDef, ...], dict[int, int]]`
  - `ExportOptionsBody.classes: list[int] | None = None`
- Consumes: `ProjectStore.read_project(project_id).classes` (`list[ClassDef]`, ids contiguous from 0).

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_export_plan.py` (extend the import line to
the wrapped import below — ruff's line length is 110):

```python
from vsp.export.plan import (
    ExportOptions,
    build_plan,
    check_request,
    classes_from_manifest,
    options_from_manifest,
)
```

```python
def _cup_and_plate(store, make_clip, n=4):
    """Classes cup (0), bowl (1), plate (2); a cup on every frame, a plate on the first half; no bowl anywhere."""
    project, clip = make_clip(n_frames=n, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=2)]
    store.write_clip(clip)
    cup = np.zeros((60, 120), bool)
    cup[10:30, 5:30] = True
    plate = np.zeros((60, 120), bool)
    plate[10:30, 70:100] = True
    for i in range(n):
        store.write_mask(clip.id, 1, i, cup)
    for i in range(n // 2):
        store.write_mask(clip.id, 2, i, plate)
    return project, store.read_clip(clip.id)


def test_selected_classes_are_renumbered_in_project_class_order(store, make_clip):
    project, clip = _cup_and_plate(store, make_clip)
    only_plate = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, class_ids=(2,)))
    assert [(c.id, c.name) for c in only_plate.classes] == [(0, "plate")]
    assert only_plate.class_map == {0: 2}
    assert only_plate.options.class_ids == (2,)
    assert [[(o.object_id, o.class_id) for o in f.objects] for f in only_plate.frames] == [[(2, 0)], [(2, 0)], [], []]

    both = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, class_ids=(2, 0)))
    assert [(c.id, c.name) for c in both.classes] == [(0, "cup"), (1, "plate")]
    assert both.class_map == {0: 0, 1: 2}
    assert both.options.class_ids == (0, 2)  # resolved: sorted
    assert [(o.object_id, o.class_id) for o in both.frames[0].objects] == [(1, 0), (2, 1)]


def test_every_class_selected_is_the_identity_map(store, make_clip):
    project, clip = _cup_and_plate(store, make_clip)
    plan = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0))
    assert plan.options.class_ids is None
    assert [c.id for c in plan.classes] == [0, 1, 2] and plan.class_map == {0: 0, 1: 1, 2: 2}
    assert [(o.object_id, o.class_id) for o in plan.frames[0].objects] == [(1, 0), (2, 2)]


def test_dropped_objects_leave_their_frames_to_include_empty_and_the_split_follows(store, make_clip):
    project, clip = _cup_and_plate(store, make_clip, n=10)  # plate on frames 0-4
    keep = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0, class_ids=(2,)))
    assert [len(f.objects) for f in keep.frames] == [1] * 5 + [0] * 5
    drop = build_plan(store, project.id, [clip.id], ExportOptions(class_ids=(2,), include_empty=False))
    # the contiguous tail is taken from the 5 frames left after the filter: ceil(0.2 * 5) = 1
    assert [(f.frame_idx, f.split) for f in drop.frames] == [
        (0, "train"), (1, "train"), (2, "train"), (3, "train"), (4, "val"),
    ]
    with pytest.raises(BadRequest, match="nothing to export"):  # bowl: selected, but on no frame
        build_plan(store, project.id, [clip.id], ExportOptions(class_ids=(1,), include_empty=False))
    empty = build_plan(store, project.id, [clip.id], ExportOptions(class_ids=(1,)))
    assert len(empty.frames) == 10 and not any(f.objects for f in empty.frames)


def test_class_selection_is_validated(store, make_clip):
    project, clip = make_clip(n_frames=2, classes=("cup", "bowl"))
    with pytest.raises(BadRequest, match="at least one class"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=()))
    with pytest.raises(BadRequest, match="unknown class 5"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=(0, 5)))
    with pytest.raises(BadRequest, match="class is listed twice"):
        check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=(1, 1)))
    assert check_request(store, project.id, [clip.id], ["coco"], ExportOptions(class_ids=(1, 0))).class_ids == (0, 1)
    assert check_request(store, project.id, [clip.id], ["coco"], ExportOptions()).class_ids is None


def test_manifest_helpers_restore_the_options_and_the_class_map():
    stored = {"split_mode": "contiguous", "val_clip_ids": [], "val_fraction": 0.2, "stride": 1,
              "include_empty": True, "min_area_px": 64, "class_ids": [2]}
    opts = options_from_manifest(stored)
    assert opts.class_ids == (2,) and opts.val_clip_ids == () and opts.split_mode == "contiguous"
    old = {k: v for k, v in stored.items() if k != "class_ids"}
    assert options_from_manifest({**old, "val_clip_ids": None}).class_ids is None
    assert options_from_manifest({**old, "val_clip_ids": None}).val_clip_ids == ()

    classes, class_map = classes_from_manifest({"classes": [
        {"id": 1, "project_class_id": 4, "name": "plate", "color": "#00ff00", "yolo_id": 1, "coco_id": 2,
         "mask_value": 2},
        {"id": 0, "project_class_id": 2, "name": "cup", "color": "#ff0000", "yolo_id": 0, "coco_id": 1,
         "mask_value": 1},
    ]})
    assert [(c.id, c.name, c.color) for c in classes] == [(0, "cup", "#ff0000"), (1, "plate", "#00ff00")]
    assert class_map == {0: 2, 1: 4}
    _, old_map = classes_from_manifest({"classes": [
        {"id": 0, "name": "a", "color": "#000000", "yolo_id": 0, "coco_id": 1, "mask_value": 1},
        {"id": 1, "name": "b", "color": "#111111", "yolo_id": 1, "coco_id": 2, "mask_value": 2},
    ]})
    assert old_map == {0: 0, 1: 1}  # an export written before class selection maps to itself
```

Append to `server/tests/test_api_export.py`:

```python
def test_export_class_selection_is_validated_before_the_job_and_applied(client, tmp_path):
    project_id = _project(client, classes=("part", "cup"))
    clip_id = _clip_with_masks(client, tmp_path, project_id)  # one object of class 0 ("part")
    url = f"/api/projects/{project_id}/export"
    empty = client.post(url, json={"clips": [clip_id], "formats": ["coco"], "options": {"classes": []}})
    assert empty.status_code == 400 and "at least one class" in empty.json()["detail"]
    unknown = client.post(url, json={"clips": [clip_id], "formats": ["coco"], "options": {"classes": [7]}})
    assert unknown.status_code == 400 and "unknown class 7" in unknown.json()["detail"]
    started = client.post(url, json={"clips": [clip_id], "formats": ["yolo_seg"], "options": {"classes": [0]}})
    assert started.status_code == 202, started.text
    snap = wait_job(client, started.json()["job_id"])
    assert snap["status"] == "done", snap
    manifest = json.loads((Path(snap["result"]["path"]) / "export.json").read_text())
    assert manifest["options"]["class_ids"] == [0]
    assert [c["name"] for c in manifest["classes"]] == ["part"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest server/tests/test_export_plan.py server/tests/test_api_export.py -q`
Expected: FAIL — `ImportError` for `classes_from_manifest` / `options_from_manifest` at collection of
`test_export_plan.py`, and the API test gets 202 for `classes: []` (the body field is ignored). If ruff reports I001
on any new import you add (e.g. in `export/__init__.py`), run `uv run ruff check server --fix`.

- [ ] **Step 3: Implement**

`server/vsp/export/plan.py` — imports: add `from collections.abc import Callable, Mapping, Sequence`,
`from dataclasses import dataclass, fields, replace`, `from typing import Any, Literal`.

`ExportOptions` gains a last field:

```python
    # project class ids to export (None = every class); always PROJECT ids, also in derived copies
    class_ids: tuple[int, ...] | None = None
```

`ExportPlan` gains, directly after `classes`:

```python
    classes: tuple[ClassDef, ...]  # ids are EXPORT ids 0..k-1 (see select_classes)
    class_map: dict[int, int]  # export id -> project class id
```

`check_request`: keep the project (`project = store.read_project(project_id)` instead of the bare call) and, before
`mode = ...`, validate the classes:

```python
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
```

and the final line becomes `return replace(options, split_mode=mode, val_clip_ids=val_ids, class_ids=class_ids)`.

New functions (after `check_request`):

```python
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
```

`build_plan`: after `project = store.read_project(project_id)`:

```python
    classes, class_map = select_classes(project.classes, resolved.class_ids)
    export_id_of = {project_class: export_class for export_class, project_class in class_map.items()}
```

The object loop becomes:

```python
            objects = []
            for obj in clip.objects:
                export_class = export_id_of.get(obj.class_id)
                if export_class is None:  # an unselected class: dropped BEFORE the empty-frame filter below
                    continue
                mask = store.read_mask(clip.id, obj.id, idx)
                # absent, or nothing but noise once cleaned = the object is not in this frame
                if mask is None or not clean_mask(mask, resolved.min_area_px).any():
                    continue
                objects.append(
                    PlanObject(obj.id, export_class, store.mask_path(clip.id, obj.id, idx), resolved.min_area_px)
                )
```

and the returned plan passes `classes=classes, class_map=class_map` (instead of `classes=tuple(project.classes)`).
Update the module docstring's second sentence to: `Objects of unselected classes are dropped here, and the
selected classes renumbered, so every exporter serialises the same export ids.`

`server/vsp/export/__init__.py`: import and list `classes_from_manifest`, `options_from_manifest`,
`select_classes` from `.plan` (add them to `__all__`, keeping it sorted).

`server/vsp/api/export.py`: `ExportOptionsBody` gains `classes: list[int] | None = None` (comment: `# project class
ids; null = every class`), and the `ExportOptions(...)` call gains
`class_ids=None if o.classes is None else tuple(o.classes),`.

`server/vsp/augment/derive.py`: replace the three lines building `classes`, `val_clip_ids` and `options`, and the
plan construction, with:

```python
        classes, class_map = classes_from_manifest(manifest)
        options = options_from_manifest(manifest["options"])
        plan = ExportPlan(
            project_id=manifest["project"]["id"], project_name=manifest["project"]["name"], classes=classes,
            class_map=class_map, options=options, clip_ids=tuple(c["id"] for c in manifest["clips"]),
            clip_names={c["id"]: c["name"] for c in manifest["clips"]}, val_clip_ids=options.val_clip_ids or (),
            frames=tuple(synthetic),  # the exporters run on the synthetic frames only; real ones are copied
        )
```

Imports in `derive.py`: `from ..export.plan import ExportPlan, PlanFrame, PlanObject, classes_from_manifest,
options_from_manifest`; drop `ExportOptions` and `from ..store.models import ClassDef` (now unused; ruff F401).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest server/tests -q -m "not gpu"` — Expected: all pass (the augment tests exercise the new derived
plan construction). `uv run ruff check server` — clean.

- [ ] **Step 5: Commit**

```bash
git add server/vsp/export/plan.py server/vsp/export/__init__.py server/vsp/api/export.py server/vsp/augment/derive.py server/tests/test_export_plan.py server/tests/test_api_export.py
git commit -m "Export: choose which classes to export (filter before the empty-frame rule, renumber 0..k-1)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 2: Formats and manifest — export ids everywhere, `project_class_id`, derived copies keep the map

**Files:**
- Modify: `server/vsp/export/runner.py`
- Test: `server/tests/test_export_formats.py`, `server/tests/test_export_runner.py`,
  `server/tests/test_augment_derive.py`

**Interfaces:**
- Produces: manifest `classes[i]` = `{"id", "project_class_id", "name", "color", "yolo_id", "coco_id",
  "mask_value"}`; manifest `options.class_ids`: `list[int] | None`.
- Consumes: `ExportPlan.class_map`, `classes_from_manifest` (Task 1).

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_export_formats.py`:

```python
def _cup_and_plate(store, make_clip, n=2):
    project, clip = make_clip(n_frames=n, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=2)]
    store.write_clip(clip)
    cup = np.zeros((H, W), bool)
    cup[10:30, 5:30] = True  # 500 px
    plate = np.zeros((H, W), bool)
    plate[10:30, 70:100] = True  # 600 px
    for i in range(n):
        store.write_mask(clip.id, 1, i, cup)
        store.write_mask(clip.id, 2, i, plate)
    return project, clip


def _export_to(store, project, clip, out, options):
    plan = build_plan(store, project.id, [clip.id], options)
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    for name in EXPORTERS:
        EXPORTERS[name].write(plan, out, lambda: None)
    return plan


def _rgb(color: str) -> list[int]:
    return [int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]


def test_one_selected_class_is_export_id_0_in_every_format(store, make_clip, tmp_path):
    project, clip = _cup_and_plate(store, make_clip)
    out = tmp_path / "plate"
    plan = _export_to(store, project, clip, out, ExportOptions(val_fraction=0.0, class_ids=(2,)))
    stem = plan.frames[0].stem
    seg = (out / "yolo_seg" / "labels" / "train" / f"{stem}.txt").read_text().splitlines()
    assert [ln.split()[0] for ln in seg] == ["0"]
    assert (out / "yolo_det" / "labels" / "train" / f"{stem}.txt").read_text().split()[0] == "0"
    assert yaml_lib.safe_load((out / "yolo_seg" / "data.yaml").read_text())["names"] == {0: "plate"}
    coco = json.loads((out / "coco" / "annotations" / "instances_train.json").read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(1, "plate")]
    assert [a["category_id"] for a in coco["annotations"]] == [1, 1]  # one plate per frame, no cup
    png = Image.open(out / "masks" / "masks" / "train" / f"{stem}.png")
    values = np.array(png)
    assert set(np.unique(values)) == {0, 1} and (values == 1).sum() == 600
    assert png.getpalette()[3:6] == _rgb(project.classes[2].color)  # value 1 is drawn in the plate's colour


def test_two_selected_classes_keep_project_order_with_a_gap_closed(store, make_clip, tmp_path):
    project, clip = _cup_and_plate(store, make_clip)
    out = tmp_path / "cup_plate"
    plan = _export_to(store, project, clip, out, ExportOptions(val_fraction=0.0, class_ids=(0, 2)))
    stem = plan.frames[0].stem
    seg = (out / "yolo_seg" / "labels" / "train" / f"{stem}.txt").read_text().splitlines()
    assert sorted(ln.split()[0] for ln in seg) == ["0", "1"]
    assert yaml_lib.safe_load((out / "yolo_det" / "data.yaml").read_text())["names"] == {0: "cup", 1: "plate"}
    coco = json.loads((out / "coco" / "annotations" / "instances_train.json").read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(1, "cup"), (2, "plate")]
    assert sorted({a["category_id"] for a in coco["annotations"]}) == [1, 2]
    png = Image.open(out / "masks" / "masks" / "train" / f"{stem}.png")
    values = np.array(png)
    assert set(np.unique(values)) == {0, 1, 2} and (values == 1).sum() == 500 and (values == 2).sum() == 600
    assert png.getpalette()[6:9] == _rgb(project.classes[2].color)
```

In `server/tests/test_export_runner.py`, **update** the exact `manifest["classes"]` assertion in
`test_run_export_writes_every_format_a_manifest_and_reports_progress` to:

```python
    assert manifest["classes"] == [{"id": 0, "project_class_id": 0, "name": "part", "color": manifest["classes"][0]["color"], "yolo_id": 0, "coco_id": 1, "mask_value": 1}]
    assert manifest["options"]["class_ids"] is None
```

and append:

```python
def test_the_manifest_maps_export_ids_back_to_project_classes(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=4, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=2)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(4):
        store.write_mask(clip.id, 1, i, mask)
    result = run_export(store, project.id, [clip.id], ["yolo_seg"], ExportOptions(class_ids=(2,)),
                        tmp_path / "exports")
    manifest = json.loads((tmp_path / "exports" / result["export_id"] / "export.json").read_text())
    assert manifest["options"]["class_ids"] == [2]
    assert manifest["classes"] == [{
        "id": 0, "project_class_id": 2, "name": "plate", "color": project.classes[2].color,
        "yolo_id": 0, "coco_id": 1, "mask_value": 1,
    }]
```

Append to `server/tests/test_augment_derive.py`:

```python
def test_a_derived_copy_keeps_the_class_map_and_the_export_ids(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=6, classes=("cup", "bowl", "plate"))
    clip.objects = [LabelObject(id=1, class_id=2)]
    store.write_clip(clip)
    mask = np.zeros((60, 120), bool)
    mask[10:40, 20:60] = True
    for i in range(6):
        store.write_mask(clip.id, 1, i, mask)
    data_root = tmp_path / "data"
    src = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(class_ids=(2,)),
                     data_root / "exports")
    _make_pool(data_root)
    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    manifest = json.loads((Path(result["path"]) / "export.json").read_text())
    assert [(c["id"], c["project_class_id"], c["name"]) for c in manifest["classes"]] == [(0, 2, "plate")]
    assert manifest["options"]["class_ids"] == [2]
    syn = list((Path(result["path"]) / "yolo_seg" / "labels" / "train").glob("syn_*.txt"))
    assert syn
    for p in syn:
        assert {line.split()[0] for line in p.read_text().splitlines()} == {"0"}


def test_a_source_manifest_without_project_class_id_maps_to_itself(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    manifest_path = Path(src["path"]) / "export.json"
    old = json.loads(manifest_path.read_text())
    for c in old["classes"]:
        del c["project_class_id"]  # as an export written before class selection
    del old["options"]["class_ids"]
    manifest_path.write_text(json.dumps(old))

    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    manifest = json.loads((Path(result["path"]) / "export.json").read_text())
    assert [(c["id"], c["project_class_id"]) for c in manifest["classes"]] == [(0, 0)]
    assert manifest["options"]["class_ids"] is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest server/tests/test_export_formats.py server/tests/test_export_runner.py server/tests/test_augment_derive.py -q`
Expected: the runner and derive tests FAIL (`KeyError: 'project_class_id'` / the classes entry lacks the key). The
two new format tests already PASS: the exporters read export ids from the plan since Task 1; they pin that
behaviour.

- [ ] **Step 3: Implement**

`server/vsp/export/runner.py` — in `manifest_for`, the `options` and `classes` entries become:

```python
        "options": {
            **asdict(plan.options), "val_clip_ids": list(plan.val_clip_ids),
            # project class ids (null = every class), a list in JSON; read back with options_from_manifest
            "class_ids": None if plan.options.class_ids is None else list(plan.options.class_ids),
        },
```

```python
        # id / yolo_id are EXPORT ids (0..k-1 over the selected classes); project_class_id maps them back
        "classes": [
            {"id": c.id, "project_class_id": plan.class_map[c.id], "name": c.name, "color": c.color,
             "yolo_id": c.id, "coco_id": c.id + 1, "mask_value": c.id + 1}
            for c in sorted(plan.classes, key=lambda c: c.id)
        ],
```

Update `manifest_for`'s docstring: `... (and a derived export starts its own manifest from). Derived plans carry the
source's class map (classes_from_manifest), so project_class_id survives any chain of derivations.`

No change in `derive.py`: it already builds `class_map` from the source (Task 1) and passes the plan to
`manifest_for`, and its `options` come from `options_from_manifest`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest server/tests -q -m "not gpu"` — all pass. `uv run ruff check server` — clean.

- [ ] **Step 5: Commit**

```bash
git add server/vsp/export/runner.py server/tests/test_export_formats.py server/tests/test_export_runner.py server/tests/test_augment_derive.py
git commit -m "Export manifest: project_class_id per class, class_ids option; derived copies keep the map" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 3: Semantic mask PNG — largest object first, `overlap_px` (and merged in derived copies)

**Files:**
- Modify: `server/vsp/export/mask_png.py`, `server/vsp/augment/derive.py` (`_merged_format_summaries` only)
- Test: `server/tests/test_export_formats.py`, `server/tests/test_export_runner.py`,
  `server/tests/test_augment_derive.py`

**Interfaces:**
- Produces:
  - `@dataclass class MaskSummary(ExportSummary): overlap_px: int = 0` (in `mask_png.py`)
  - `paint(objects: Sequence[tuple[int, np.ndarray]], height: int, width: int) -> tuple[np.ndarray, int]`
    (`(value, bool mask)` pairs → uint8 canvas, pixels claimed by more than one object)
  - `MaskPng.write(...) -> MaskSummary`
  - `_merged_format_summaries(...)` entries carry `overlap_px` when either side has it.

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_export_formats.py` (add `from dataclasses import asdict` and
`from vsp.export.mask_png import paint` to the imports):

```python
def test_paint_draws_the_smaller_object_on_top_whatever_the_order_and_counts_the_overlap():
    big = np.zeros((H, W), bool)
    big[10:50, 10:60] = True  # 2000 px
    small = np.zeros((H, W), bool)
    small[20:30, 20:30] = True  # 100 px, inside big
    for objects in ([(2, small), (1, big)], [(1, big), (2, small)]):
        canvas, overlap = paint(objects, H, W)
        assert canvas.dtype == np.uint8
        assert (canvas == 2).sum() == 100 and (canvas == 1).sum() == 1900 and overlap == 100
    apart = np.zeros((H, W), bool)
    apart[0:5, 100:110] = True
    assert paint([(1, apart), (2, small)], H, W)[1] == 0
    assert paint([], H, W)[1] == 0


def test_the_mask_image_keeps_a_small_object_visible_on_a_large_one(store, make_clip, tmp_path):
    project, clip = make_clip(n_frames=2, classes=("screw", "board"))
    clip.objects = [LabelObject(id=1, class_id=0), LabelObject(id=2, class_id=1)]  # the small one comes first
    store.write_clip(clip)
    screw = np.zeros((H, W), bool)
    screw[20:40, 40:60] = True  # 400 px, lying on the board
    board = np.zeros((H, W), bool)
    board[5:55, 10:110] = True  # 5000 px
    for i in range(2):
        store.write_mask(clip.id, 1, i, screw)
        store.write_mask(clip.id, 2, i, board)
    plan = build_plan(store, project.id, [clip.id], ExportOptions(val_fraction=0.0))
    out = tmp_path / "overlap"
    (out / "images").mkdir(parents=True)
    for frame in plan.frames:
        shutil.copy2(frame.image_path, export_base.shared_image(out, frame))
    summary = EXPORTERS["masks"].write(plan, out, lambda: None)
    values = np.array(Image.open(out / "masks" / "masks" / "train" / f"{plan.frames[0].stem}.png"))
    assert (values == 1).sum() == 400 and (values == 2).sum() == 4600
    assert summary.overlap_px == 2 * 400
    assert asdict(summary)["overlap_px"] == 800
    assert "overlap_px" not in asdict(EXPORTERS["yolo_seg"].write(plan, out, lambda: None))
```

Append to `server/tests/test_export_runner.py`:

```python
def test_overlap_px_is_reported_for_the_mask_format_only(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    result = run_export(store, project.id, [clip.id], ["yolo_seg", "masks"], ExportOptions(), tmp_path / "exports")
    manifest = json.loads((tmp_path / "exports" / result["export_id"] / "export.json").read_text())
    formats = {f["name"]: f for f in manifest["formats"]}
    assert formats["masks"]["overlap_px"] == 0  # one object per frame
    assert "overlap_px" not in formats["yolo_seg"]
```

Append to `server/tests/test_augment_derive.py` (add `from vsp.augment.derive import _merged_format_summaries`,
`from vsp.export.base import ExportSummary` and `from vsp.export.mask_png import MaskSummary` to the imports):

```python
def test_merged_summaries_sum_overlap_px_and_leave_other_formats_alone():
    counts = {"images": {"train": 4, "val": 1}, "annotations": {"train": 4, "val": 1}, "linked": True}
    src = [{"name": "masks", **counts, "overlap_px": 30}, {"name": "yolo_seg", **counts}]
    synth = [MaskSummary("masks", overlap_px=5), ExportSummary("yolo_seg")]
    merged = {m["name"]: m for m in _merged_format_summaries(src, synth, True)}
    assert merged["masks"]["overlap_px"] == 35
    assert "overlap_px" not in merged["yolo_seg"]
    older = _merged_format_summaries([{"name": "masks", **counts}], [MaskSummary("masks", overlap_px=0)], True)
    assert older[0]["overlap_px"] == 0  # a source written before overlap_px counts as 0


def test_the_derived_mask_summary_carries_the_source_overlap(store, make_clip, tmp_path):
    project, clip = _multi(store, make_clip)
    data_root = tmp_path / "data"
    src = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    result = run_augment(data_root, src["export_id"], AugmentOptions(per_image=1, seed=0))
    src_masks = next(f for f in json.loads((Path(src["path"]) / "export.json").read_text())["formats"]
                     if f["name"] == "masks")
    new_masks = next(f for f in json.loads((Path(result["path"]) / "export.json").read_text())["formats"]
                     if f["name"] == "masks")
    # _multi: m1&m2 share column 29 on rows 10-19 (10 px), and rows 20-29, columns 20-44 hold 2+ objects (250 px)
    assert src_masks["overlap_px"] == 6 * 260
    assert new_masks["overlap_px"] == src_masks["overlap_px"]  # composites never overlap: one owner per pixel
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest server/tests/test_export_formats.py server/tests/test_export_runner.py server/tests/test_augment_derive.py -q`
Expected: FAIL — `ImportError: cannot import name 'paint'` (test_export_formats collection) and
`cannot import name 'MaskSummary'` (test_augment_derive collection); the runner test `KeyError: 'overlap_px'`.

- [ ] **Step 3: Implement**

`server/vsp/export/mask_png.py` (insert `MaskSummary` and `paint` after the imports, keep `_palette` as it is, and
replace `class MaskPng` with the block below; add `from collections.abc import Sequence` and
`from dataclasses import dataclass`; module docstring: `"""Semantic masks: 8-bit indexed PNG, 0 = background,
export class id + 1 = the class, palette from the class colours. One value per pixel: objects are painted largest
first, so a smaller object lying on a larger one stays visible; the pixels two objects share are counted."""`):

```python
@dataclass
class MaskSummary(ExportSummary):
    overlap_px: int = 0  # pixels claimed by more than one object, over every frame of both splits


def paint(objects: Sequence[tuple[int, np.ndarray]], height: int, width: int) -> tuple[np.ndarray, int]:
    """`(value, mask)` pairs -> the semantic canvas and the number of pixels claimed by more than one object.
    Largest area first (ties keep the given order), so the smaller object wins where two overlap."""
    canvas = np.zeros((height, width), np.uint8)
    claims = np.zeros((height, width), np.uint16)
    for value, mask in sorted(objects, key=lambda vm: -int(vm[1].sum())):  # sorted() is stable
        canvas[mask] = value
        claims += mask
    return canvas, int((claims > 1).sum())


class MaskPng:
    name = "masks"

    def write(self, plan: ExportPlan, out_dir: Path, tick: Tick) -> MaskSummary:
        if any(c.id + 1 > 255 for c in plan.classes):
            raise BadRequest("the mask image format holds at most 255 classes (8-bit pixels, 0 is the background)")
        root = out_dir / self.name
        summary = MaskSummary(self.name)
        palette = _palette(plan)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for frame in plan.frames:
            linked = place_image(shared_image(out_dir, frame), root / "images" / frame.split / f"{frame.stem}.jpg")
            summary.linked = summary.linked and linked
            canvas, overlap = paint(
                [(obj.class_id + 1, read_mask(obj)) for obj in frame.objects], frame.height, frame.width,
            )
            summary.overlap_px += overlap
            img = Image.fromarray(canvas)  # mode "L"; putpalette turns it into an indexed ("P") image in place
            img.putpalette(palette)
            img.save(root / "masks" / frame.split / f"{frame.stem}.png")
            summary.images[frame.split] += 1
            summary.annotations[frame.split] += len(frame.objects)
            tick()
        return summary
```

(`_palette` is unchanged.) `write_dataset` and `manifest_for` need no change: `asdict(MaskSummary)` includes
`overlap_px`, and only the masks summary is a `MaskSummary`.

`server/vsp/augment/derive.py` — `_merged_format_summaries` builds each entry into a local and adds the sum:

```python
    for src in src_formats:
        synth = synth_by_name.get(src["name"], empty)
        entry = {
            "name": src["name"],
            "images": {sp: src["images"][sp] + synth["images"][sp] for sp in ("train", "val")},
            "annotations": {sp: src["annotations"][sp] + synth["annotations"][sp] for sp in ("train", "val")},
            "linked": bool(src["linked"]) and bool(synth["linked"]) and real_linked,
        }
        if "overlap_px" in src or "overlap_px" in synth:  # the masks format; an older source counts as 0
            entry["overlap_px"] = int(src.get("overlap_px", 0)) + int(synth.get("overlap_px", 0))
        merged.append(entry)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest server/tests -q -m "not gpu"` — all pass. `uv run ruff check server` — clean.

- [ ] **Step 5: Commit**

```bash
git add server/vsp/export/mask_png.py server/vsp/augment/derive.py server/tests/test_export_formats.py server/tests/test_export_runner.py server/tests/test_augment_derive.py
git commit -m "Mask images: paint the largest object first, report overlap_px (summed in derived copies)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 4: Augmented copies — per-object cutouts from COCO, flagged fallback, augment panel note

**Files:**
- Modify: `server/vsp/augment/derive.py`, `web/src/components/AugmentPanel.tsx`
- Test: `server/tests/test_augment_derive.py`, `web/src/components/AugmentPanel.test.tsx`

**Interfaces:**
- Produces:
  - `CutoutMode = Literal["per_object", "per_class_region"]`
  - `_coco_train_objects(source_dir: Path) -> dict[str, list[tuple[int, list[Polygon]]]]`
  - `_cutouts_and_sizes(source_dir: Path, formats: list[str], min_area_px: int, tick: Callable[[], None]) -> tuple[list[Cutout], list[tuple[int, int]], CutoutMode]`
  - derived manifest `augment.cutouts`; `run_augment(...)` result gains `"cutouts"`.
- Consumes: `polygons_to_mask`, `clean_mask` (`export/polygons.py`), `cutout_from` (`augment/composite.py`),
  the COCO exporter's layout (`coco/annotations/instances_train.json`, `category_id` = export id + 1).

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_augment_derive.py` (extend the derive import to
the wrapped import below — ruff's line length is 110):

```python
from vsp.augment.derive import (
    AugmentOptions,
    _cutouts_and_sizes,
    _merged_format_summaries,
    check_augment,
    run_augment,
)
```

```python
def _train_images(src) -> int:
    return json.loads((Path(src["path"]) / "export.json").read_text())["counts"]["train"]["images"]


def test_cutouts_are_per_object_when_the_source_has_coco(store, make_clip, tmp_path):
    project, clip = _multi(store, make_clip)  # two touching "a" objects and an overlapping "b" on every frame
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "coco", "masks"]
    src = _export(store, project, clip, formats, data_root)
    ticks = []
    cutouts, sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: ticks.append(1))
    n = _train_images(src)
    assert mode == "per_object"
    assert sizes == [(120, 60)] * n and len(ticks) == n
    assert len(cutouts) == 3 * n
    # each object minus the pixels another object also claims: m1 (rows 10-29, cols 5-29) loses col 29 to m2 and
    # rows 20-29 x cols 20-29 to m3 -> 500 - 110 = 390; m2 (cols 29-49) loses col 29 and rows 20-29 x cols 29-44 ->
    # 420 - 170 = 250; m3 (rows 20-44, cols 20-44) loses rows 20-29 x cols 20-44 -> 625 - 250 = 375
    assert sorted((c.class_id, int(c.mask.sum())) for c in cutouts[:3]) == [(0, 250), (0, 390), (1, 375)]


def test_cutouts_fall_back_to_class_regions_without_coco(store, make_clip, tmp_path):
    project, clip = _multi(store, make_clip)
    data_root = tmp_path / "data"
    formats = ["yolo_seg", "masks"]
    src = _export(store, project, clip, formats, data_root)
    cutouts, _sizes, mode = _cutouts_and_sizes(Path(src["path"]), formats, 64, lambda: None)
    assert mode == "per_class_region"
    assert len(cutouts) == 2 * _train_images(src)
    # the two touching "a" objects come out as one 20x45 region; "b" keeps only what the mask PNG shows of it
    assert sorted((c.class_id, int(c.mask.sum())) for c in cutouts[:2]) == [(0, 900), (1, 375)]


def test_the_derived_manifest_records_how_cutouts_were_made(store, make_clip, tmp_path):
    project, clip = _labelled(store, make_clip)
    data_root = tmp_path / "data"
    with_coco = _export(store, project, clip, ["yolo_seg", "coco", "masks"], data_root)
    without = _export(store, project, clip, ["yolo_seg", "masks"], data_root)
    _make_pool(data_root)
    per_object = run_augment(data_root, with_coco["export_id"], AugmentOptions(per_image=1, seed=0))
    fallback = run_augment(data_root, without["export_id"], AugmentOptions(per_image=1, seed=0))
    assert per_object["cutouts"] == "per_object" and fallback["cutouts"] == "per_class_region"
    assert json.loads((Path(per_object["path"]) / "export.json").read_text())["augment"]["cutouts"] == "per_object"
    assert json.loads((Path(fallback["path"]) / "export.json").read_text())["augment"]["cutouts"] == \
        "per_class_region"
```

(The fallback numbers rely on Task 3's painting order: m3 (625 px) is painted first, then m1 and m2 over it, so
class "a" = rows 10–29 × columns 5–49 = 900 px and class "b" keeps rows 30–44 × columns 20–44 = 375 px.)

Append to `web/src/components/AugmentPanel.test.tsx` inside the `describe`:

```tsx
  it("notes that an export without COCO cuts objects per class region", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    expect(await screen.findByText(/has no COCO format, so composites cut objects per class region/)).toBeTruthy();
    cleanup();
    render(<AugmentPanel exportInfo={exp({ formats: ["yolo_seg", "coco", "masks"] })} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    expect(screen.queryByText(/has no COCO format/)).toBeNull();
  });
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest server/tests/test_augment_derive.py -q` — Expected: FAIL (`_cutouts_and_sizes()` takes 3
positional arguments / returns two values; `KeyError: 'cutouts'`).
Run: `cd web && node node_modules/.bin/vitest run src/components/AugmentPanel.test.tsx` — Expected: FAIL (the note is
not rendered).

- [ ] **Step 3: Implement**

`server/vsp/augment/derive.py` — imports: add `from typing import Literal`, and
`from ..export.polygons import Polygon, clean_mask, polygons_to_mask` (replacing the `clean_mask`-only import).
Replace the module docstring's last paragraph with:

```
Cutouts for composites are cut per object from the source's COCO train annotations when the source has the
`coco` format (two touching objects of one class stay two cutouts), and per class region of the semantic `masks`
PNGs otherwise; the derived manifest records which (`augment.cutouts`). Either way they feed composites only,
never a real frame's own labels.
```

Replace `_cutouts_and_sizes` with:

```python
CutoutMode = Literal["per_object", "per_class_region"]


def _coco_train_objects(source_dir: Path) -> dict[str, list[tuple[int, list[Polygon]]]]:
    """Each train image's objects, by image stem: `(export class id, polygons)`, in annotation order. The COCO
    exporter writes one annotation per object, so objects that touch or overlap stay apart."""
    data = json.loads((source_dir / "coco" / "annotations" / "instances_train.json").read_text())
    stems = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
    out: dict[str, list[tuple[int, list[Polygon]]]] = {}
    for ann in sorted(data["annotations"], key=lambda a: a["id"]):
        polygons = [[(float(p[i]), float(p[i + 1])) for i in range(0, len(p) - 1, 2)] for p in ann["segmentation"]]
        out.setdefault(stems[ann["image_id"]], []).append((int(ann["category_id"]) - 1, polygons))
    return out


def _cutouts_and_sizes(
    source_dir: Path, formats: list[str], min_area_px: int, tick: Callable[[], None],
) -> tuple[list[Cutout], list[tuple[int, int]], CutoutMode]:
    """Cutouts and `(width, height)` of every real train frame (the frames are listed from the `masks` format).
    Per object from the COCO train annotations when the source has `coco` (each object minus the pixels another
    object also claims), else per class region of the indexed mask PNGs (touching objects of one class then come out
    as one). Feeds composites only."""
    per_object = "coco" in formats
    by_stem = _coco_train_objects(source_dir) if per_object else {}
    cutouts: list[Cutout] = []
    sizes: list[tuple[int, int]] = []
    for mask_path in sorted((source_dir / "masks" / "masks" / "train").glob("*.png")):
        tick()
        with Image.open(mask_path) as png:
            width, height = png.size
            indexed = None if per_object else np.array(png)
        sizes.append((width, height))
        if indexed is None:
            found = [(cls, polygons_to_mask(polygons, width, height)) for cls, polygons in by_stem.get(mask_path.stem, [])]
            # Where labelled masks overlap, whose pixels those are is unknown: an object's cutout keeps only the pixels
            # no other object claims, so it never carries a piece of its neighbour into a composite.
            owners = sum((mask.astype(np.uint8) for _, mask in found), np.zeros((height, width), np.uint8))
            found = [(cls, mask & (owners == 1)) for cls, mask in found]
        else:
            found = [(int(v) - 1, indexed == v) for v in sorted(int(v) for v in np.unique(indexed) if v > 0)]
        found = [(cls, clean_mask(mask, min_area_px)) for cls, mask in found]
        found = [(cls, mask) for cls, mask in found if mask.any()]
        if not found:
            continue
        image = np.array(Image.open(source_dir / "images" / f"{mask_path.stem}.jpg").convert("RGB"))
        for cls, mask in found:
            cut = cutout_from(image, mask, cls)
            if cut is not None:
                cutouts.append(cut)
    return cutouts, sizes, "per_object" if per_object else "per_class_region"
```

In `run_augment`: `cutouts, sizes, cutout_mode = _cutouts_and_sizes(source_dir, formats, min_area_px, tick)`; the
`augment` block gains `"cutouts": cutout_mode`; the returned dict gains `"cutouts": cutout_mode`. The progress total
is unchanged (still one tick per real train frame here).

`web/src/components/AugmentPanel.tsx` — after the `{fetchJob.status === "cancelled" && ...}` line, add:

```tsx
      {!exportInfo.formats.includes("coco") && (
        <p className="note">
          This export has no COCO format, so composites cut objects per class region: touching objects of one class
          are cut as one. Export with COCO ticked for per-object cutouts.
        </p>
      )}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server`, and
`cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit && node node_modules/.bin/vite build`
— all green. (`test_same_seed_same_composites` still passes: the same source and seed give the same cutouts; a
source with COCO now gives a different cutout set than before this change, which spec §12 accepts.)

- [ ] **Step 5: Commit**

```bash
git add server/vsp/augment/derive.py server/tests/test_augment_derive.py web/src/components/AugmentPanel.tsx web/src/components/AugmentPanel.test.tsx
git commit -m "Augmented copies: cut composites per object from COCO, flag the per-class-region fallback" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 5: Export page — Classes section in the right sidebar

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/lib/exportSelection.ts`, `web/src/pages/Export.tsx`
- Test: `web/src/lib/exportSelection.test.ts`, `web/src/pages/Export.test.tsx`, `web/src/api/rest.test.ts` (in
  "posts an export request with the server's field names" the `options` becomes
  `options: { split_mode: "per_clip", val_clips: ["b"], val_fraction: 0.2, stride: 1, include_empty: true, classes: [0, 2] },`
  — the new field is required, so without this `tsc --noEmit` fails)

**Interfaces:**
- Produces:
  - `ExportRequest["options"]` gains `classes: number[] | null` (null = every class).
  - `classObjectCounts(clips: ClipSummary[], included: string[]): Map<number, number>` (class id → objects in the
    included clips).
  - `exportProblem(d: Draft)` with `Draft` gaining `classes: number[]` (ticked ids) and `classCount: number`
    (project classes); new message `"Choose at least one class."` (only when the project has classes).
- Consumes: `Project.classes`, `ClipSummary.objects` (`LabelObject.class_id`), `POST /export` `options.classes`
  (Task 1).

- [ ] **Step 1: Write the failing tests**

`web/src/lib/exportSelection.test.ts` — import `classObjectCounts` too; in "says what stops an export, or null",
change `ok` to include the classes and add the new case:

```ts
    const ok = { selected: ["a", "b"], formats: ["coco"], classes: [0], classCount: 1, valClips: ["b"], valFraction: 0.2, stride: 1 };
    // ...existing expectations unchanged...
    expect(exportProblem({ ...ok, classes: [] })).toBe("Choose at least one class.");
    expect(exportProblem({ ...ok, classes: [], classCount: 0 })).toBeNull(); // a project without classes can still export backgrounds
```

and add:

```ts
  it("counts each class's objects over the included clips only", () => {
    const withObjects = (c: ClipSummary, classIds: number[]) =>
      ({ ...c, objects: classIds.map((class_id, i) => ({ id: i + 1, class_id, keyframes: [], propagated_ranges: [] })) }) as ClipSummary;
    const list = [withObjects(clips[0], [1, 1, 0]), withObjects(clips[1], [2]), withObjects(clips[2], [2])];
    expect(Object.fromEntries(classObjectCounts(list, ["a", "b"]))).toEqual({ 0: 1, 1: 2, 2: 1 });
    expect(classObjectCounts(list, []).size).toBe(0);
  });
```

`web/src/pages/Export.test.tsx` — **update** the exact options in the first test to
`options: { split_mode: "per_clip", val_clips: ["b"], val_fraction: 0.2, stride: 1, include_empty: true, classes: null },`
and add inside the `describe`:

```tsx
  const threeClasses: Project = {
    ...project,
    classes: [
      { id: 0, name: "connector", color: "#e6194b" },
      { id: 1, name: "cup", color: "#3cb44b" },
      { id: 2, name: "bowl", color: "#ffe119" },
    ],
  };
  const obj = (id: number, class_id: number) => ({ id, class_id, keyframes: [], propagated_ranges: [] });
  const classClips = [
    { ...clipA, objects: [obj(1, 1), obj(2, 1), obj(3, 0)] },
    { ...clipB, objects: [obj(1, 2)] },
    clipC,
  ];
  const classRow = (name: string) => screen.getByRole("checkbox", { name: `Class ${name}` }).closest("label")!;

  it("lists every class ticked, with its object count over the included clips, and sends null for all", async () => {
    m.listProjects.mockResolvedValue([threeClasses]);
    m.listClips.mockResolvedValue(classClips);
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");
    for (const name of ["connector", "cup", "bowl"]) {
      expect((screen.getByRole("checkbox", { name: `Class ${name}` }) as HTMLInputElement).checked).toBe(true);
    }
    expect(classRow("connector").textContent).toContain("1 object");
    expect(classRow("cup").textContent).toContain("2 objects");
    expect(classRow("bowl").textContent).toContain("1 object");

    fireEvent.click(screen.getByRole("checkbox", { name: "Include b.mp4" }));
    expect(classRow("bowl").textContent).toContain("0 objects");

    fireEvent.click(screen.getByRole("button", { name: "Export 2 clips" }));
    await waitFor(() => expect(m.startExport).toHaveBeenCalled());
    expect(m.startExport.mock.calls[0][1].options.classes).toBeNull();
  });

  it("sends only the ticked classes, in class order", async () => {
    m.listProjects.mockResolvedValue([threeClasses]);
    m.listClips.mockResolvedValue(classClips);
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");
    fireEvent.click(screen.getByRole("checkbox", { name: "Class connector" }));
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() => expect(m.startExport).toHaveBeenCalled());
    expect(m.startExport.mock.calls[0][1].options.classes).toEqual([1, 2]);
  });

  it("says why it cannot start when no class is ticked", async () => {
    m.listProjects.mockResolvedValue([threeClasses]);
    m.listClips.mockResolvedValue(classClips);
    renderPage();
    await screen.findByText("a.mp4");
    for (const name of ["connector", "cup", "bowl"]) fireEvent.click(screen.getByRole("checkbox", { name: `Class ${name}` }));
    expect(screen.getByText("Choose at least one class.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Export 3 clips" }) as HTMLButtonElement).disabled).toBe(true);
  });
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && node node_modules/.bin/vitest run src/lib/exportSelection.test.ts src/pages/Export.test.tsx`
Expected: FAIL (`classObjectCounts` is not exported; no "Class …" checkboxes; `classes` missing from the request).

- [ ] **Step 3: Implement**

`web/src/api/types.ts` — `ExportRequest.options` becomes:

```ts
  options: {
    split_mode: SplitMode; val_clips?: string[]; val_fraction: number; stride: number; include_empty: boolean;
    /** Project class ids to export; null = every class. */
    classes: number[] | null;
  };
```

`web/src/lib/exportSelection.ts`:

```ts
/** Objects per class over the included clips (from the clip summaries' `objects`), for the Classes section. */
export function classObjectCounts(clips: ClipSummary[], included: string[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const c of clips) {
    if (!included.includes(c.id)) continue;
    for (const o of c.objects) counts.set(o.class_id, (counts.get(o.class_id) ?? 0) + 1);
  }
  return counts;
}

type Draft = {
  selected: string[]; formats: string[]; classes: number[]; classCount: number; valClips: string[];
  valFraction: number; stride: number;
};
```

and in `exportProblem`, directly after the formats check:
`if (d.classCount > 0 && d.classes.length === 0) return "Choose at least one class.";`

`web/src/pages/Export.tsx`:
- import `classObjectCounts` from `../lib/exportSelection`.
- state next to `includeEmpty`: `const [excludedClasses, setExcludedClasses] = useState<Set<number>>(new Set());`
  (excluded rather than ticked, so every class — including one added later — starts ticked).
- in `loadClips`, next to `setTouched(false)`: `setExcludedClasses(new Set());`
- after `orderedFormats`:

```tsx
  const classes = useMemo(() => [...(project?.classes ?? [])].sort((a, b) => a.id - b.id), [project]);
  const objectCounts = useMemo(() => classObjectCounts(clips, orderedIncluded), [clips, orderedIncluded]);
  const chosenClasses = classes.filter((c) => !excludedClasses.has(c.id)).map((c) => c.id);
```

- `exportProblem({...})` gains `classes: chosenClasses, classCount: classes.length,`.
- a toggle next to `toggleFormat`:

```tsx
  const toggleClass = (id: number) => {
    setExcludedClasses((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
```

- in `startExport`, `options` gains `classes: chosenClasses.length === classes.length ? null : chosenClasses,`.
- in the `<aside>`, a new section directly after the Formats section:

```tsx
          <section className="flex flex-col gap-2">
            <h2 className="heading">Classes</h2>
            {classes.length === 0 && <p className="note">This project has no classes yet.</p>}
            {classes.map((c) => {
              const n = objectCounts.get(c.id) ?? 0;
              return (
                <label key={c.id} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    aria-label={`Class ${c.name}`}
                    checked={!excludedClasses.has(c.id)}
                    onChange={() => toggleClass(c.id)}
                  />
                  <span aria-hidden className="inline-block h-3 w-3 shrink-0 rounded-sm" style={{ backgroundColor: c.color }} />
                  <span className="min-w-0 flex-1 truncate">{c.name}</span>
                  <span className="font-cond text-dim">
                    {n} object{n === 1 ? "" : "s"}
                  </span>
                </label>
              );
            })}
            <p className="note">
              Objects of unticked classes are left out; a frame left without objects follows "Include frames without a
              mask". The exported classes are numbered 0, 1, … in this order.
            </p>
          </section>
```

- [ ] **Step 4: Run the tests**

Run: `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit && node node_modules/.bin/vite build`
— green. `uv run pytest server/tests -q -m "not gpu"` — still green (no server change).

- [ ] **Step 5: Commit**

```bash
git add web/src/api/types.ts web/src/api/rest.test.ts web/src/lib/exportSelection.ts web/src/lib/exportSelection.test.ts web/src/pages/Export.tsx web/src/pages/Export.test.tsx
git commit -m "Export page: Classes section with object counts; send the chosen classes" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 6: README

**Files:** Modify `README.md` only. Another agent may be committing README changes: run `git diff README.md`
first; if it shows changes you did not make, wait until they are committed (`git log -1 -- README.md`), then read
the current file before editing. Edit only the paragraphs named here.

- [ ] **Step 1: Edit**
  1. API paragraph (`POST /api/projects/{id}/export` ...): the body is `{clips, formats, options, name}`;
     `options.classes` is a list of project class ids to export (null or absent = every class); an empty list, a
     repeated id or an unknown class id is a 400. Also note under the augmented-copy text: with COCO, each object is
     cut out minus the pixels another object also claims.
  2. "Exporting": replace "the class map (`yolo_id`, `coco_id`, `mask_value` per class)" with "the class map (per
     exported class: `id`/`yolo_id` = export id, `coco_id`/`mask_value` = export id + 1, and `project_class_id`, the
     project's own class id)". Add a paragraph: "The Export page's right sidebar has a Classes section: one checkbox
     per class with its swatch and its object count over the included clips, all ticked by default. Objects of
     unticked classes are left out before anything else, so a frame left without objects is kept or dropped by
     "Include frames without a mask", and the split is taken after that. The exported classes are renumbered
     0..k-1 in project class order (YOLO ids; COCO category ids and mask values are one higher);
     `options.class_ids` in `export.json` records the chosen project class ids (null = every class)." Add: "Where
     objects overlap, the mask image holds one value per pixel: objects are painted largest first, so a smaller
     object lying on a larger one stays visible. The `masks` entry of `export.json`'s `formats` reports
     `overlap_px`, the pixels claimed by more than one object (the other formats keep every object whole)."
  3. "Augmented copies": replace "(per-class, so two objects of the same class in one frame are cut together)"
     with "(one cutout per object, rasterised from the source's COCO train annotations when the source has the
     COCO format; without it, one cutout per class region of the mask images, so touching objects of one class are
     cut together — the panel says so)", and add: "`export.json`'s `augment.cutouts` records which
     (`per_object` / `per_class_region`); the same seed on an export made before this change may give different
     composites."
  4. Manual QA list: append the next number: "Class selection: on a project with three classes, untick one on the
     Export page (the counts follow the included clips) and export all four formats; `yolo_seg/data.yaml` lists
     the remaining classes as 0 and 1, COCO categories are 1 and 2, the mask PNG values are 1 and 2, and
     `export.json` maps them back with `project_class_id`. Untick every class: Start says "Choose at least one
     class." Put a small object on a large one: the mask PNG shows the small one, and `overlap_px` is non-zero.
     Make an augmented copy of an export with COCO: `augment.cutouts` is `per_object`; of one without COCO: the
     panel notes the fallback and it is `per_class_region`."
- [ ] **Step 2:** Run `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server`, and
  `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit && node node_modules/.bin/vite build`.
- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Docs: export class selection, class map, mask overlap, per-object cutouts" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

## Note for the negatives branch

`negative-crops` (not merged here; checked against `git show negative-crops:server/vsp/augment/negatives.py` and
`git show negative-crops:server/vsp/augment/derive.py`) adds `server/vsp/augment/negatives.py`, which rebuilds a
plan from a source manifest with `ExportOptions(**manifest["options"])` and passes it to `manifest_for`. Its own
`derive.py` — forked before this plan's Task 4 — still defines `merge_real_frames`, `merge_coco` and
`merged_format_summaries` as public names (no leading underscore), and `negatives.py` imports them by those names
(`from .derive import merge_coco, merge_real_frames, merged_format_summaries`); this plan renames all three to
`_merge_real_frames`, `_merge_coco`, `_merged_format_summaries`. So merging `negative-crops` conflicts twice:
in `derive.py`'s own definitions (public vs. private) and in `negatives.py`'s import line, which needs the
leading underscores added; the `ExportPlan(...)` block (point 2 below) is a separate, third conflict. When it is
merged on top of this plan:

1. Build options with `options_from_manifest(manifest["options"])` instead of `ExportOptions(**...)`: it turns
   `class_ids` (and `val_clip_ids`) back into tuples, fills keys an older manifest lacks, and ignores unknown keys.
2. Build the plan's classes with `classes, class_map = classes_from_manifest(manifest)` and pass
   `class_map=class_map` to `ExportPlan(...)` — the field is required, so the branch fails with a `TypeError` until
   it does. Never call `select_classes` / `build_plan`'s filter on a derived copy: the source's ids are already
   export ids, and `options.class_ids` stays the source's project class ids (informational).
3. Negative frames carry no objects, so no class id is involved; any object it does copy keeps the source's export
   id.
4. Merge format summaries through the renamed `_merged_format_summaries`, not the branch's own copy: its own
   `merged_format_summaries` has no `overlap_px` handling at all today, so the rule (the masks entry sums
   `overlap_px`, a side without the key counting 0; negative frames add 0) must be carried into the renamed
   function, not dropped when the two are reconciled.
5. If it cuts anything from the source's objects, use `_cutouts_and_sizes(source_dir, formats, min_area_px, tick)`
   (three return values now) rather than the mask PNG regions.
6. The branch writes no `name` to its result manifest today (`result_manifest` has no `"name"` key). It must gain
   one, following the dataset-name rules (spec §11: the source's display name — name, else id — plus a suffix,
   truncated to `MAX_NAME` with the suffix kept whole, as `_derived_name` already does for augmented copies) —
   here the suffix is ` +<N>% negatives`, not ` ×<N> augmented`.
7. After switching to `options_from_manifest` / `classes_from_manifest` (points 1-2), its
   `from ..export.plan import ExportOptions` and `from ..store.models import ClassDef` imports become unused
   (ruff F401) and should be dropped.
8. Its exclusion mask is built from the `masks` PNG (`mask = indexed > 0`): on a filtered export (`class_ids` set),
   objects of unticked classes are already absent from that PNG, so a negative crop's free-rectangle search treats
   their pixels as background too and may place a crop over one. This is consistent with how the rest of a
   filtered export treats unticked classes (as if they were never labelled), but it should be stated in the
   branch's own docs/comments so it isn't mistaken for a bug later.

Tests to add there: a negatives copy of an export made with `class_ids=(2,)` keeps
`[(0, 2, "plate")]` as `(id, project_class_id, name)`, and one of a manifest without `project_class_id` maps to
itself (mirror of Task 2's derive tests).

## Self-review against the spec

- §10 `class_ids` field / list↔tuple / API `classes` / 400s: Task 1 (plan tests, API test), Task 2 (manifest list).
- §10 drop before the empty-frame filter, split after: Task 1 `test_dropped_objects_...`.
- §10 renumbering, required `class_map`, identity with all classes, YOLO = export id, COCO / mask value = +1: Task 1
  (plan), Task 2 (every format, palette colour follows the renumbered class).
- §10 `project_class_id` in the manifest; derived copies map with the `c.get("project_class_id", c["id"])`
  fallback; older manifests: Task 1 (`classes_from_manifest`), Task 2 (derive tests).
- §10 overlap: largest first, `overlap_px` in the masks summary, computed in `MaskPng.write`, summed by the merge:
  Task 3.
- §10 Export page Classes section (swatch, counts over included clips, all ticked, Start problem text): Task 5.
- §12 per-object cutouts from COCO (`file_name` → `images/<stem>.jpg`, `polygons_to_mask`, `category_id - 1`),
  fallback per class region, `masks` still required, `cutouts` in the derived manifest, panel note, seed change
  accepted: Task 4, README Task 6.
- §13 every selected class absent: "nothing to export" with `include_empty` off, all-empty dataset with it on, 0
  objects shown in the Classes section: Task 1 test, Task 5 test (bowl shows "0 objects").
- §14 plan C rows: all covered above; README: Task 6.
