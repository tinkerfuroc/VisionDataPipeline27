# Several objects per clip, class selection on export, named datasets

Date: 2026-09-26. Status: design approved in conversation (approach A, "object list with an active object"); audited
against the code once (20 findings, all folded in below).

## 1. Goal

1. A clip can hold any number of labelled objects, each with its own class (several of the same class too, e.g. two
   cups and a bowl). Each object is prompted, painted, propagated, cleared and exported on its own.
2. An export can be limited to some of the project's classes.
3. An export can be given a name: a free-text display name, and a folder id derived from it.

Out of scope: one object changing class over time; merging or splitting objects; object ids shared across clips
(an object belongs to its clip); per-object frame deletion (deleting a frame stays clip-wide).

## 2. Decisions taken in conversation

| Topic | Decision |
|---|---|
| Multi-class meaning | several objects per clip |
| Propagate | both: all objects together (default), or the selected object only |
| Timeline | one lane per object, stacked; the active lane is taller; clicking a lane selects its object |
| Excluded classes on export | their objects are dropped; a frame left with none follows "Include frames without a mask" |
| Dataset name | both: free-text display name (editable later) and a folder id slugged from it |
| Overlay outline (previous change) | the active object gets the configurable outline; other objects are fill only |

## 3. Delivery: four plans

| Plan | Scope | Depends on |
|---|---|---|
| A. Server multi-object | §4 store, §5 engine, §6 propagation, §7 labelling API, their tests | — |
| B. Label page | §8, §9 | A |
| C. Export class selection | §10, §12 (cutouts use A's multi-object fixtures) | A (fixtures only) |
| D. Export names | §11 | — |

D can run before or alongside A. Each plan ends with the test suites green and README updated for its part.

## 4. Data model and storage

No migration. `Clip.objects: list[LabelObject]` and `masks/<object_id>/<frame:06d>.png` already exist; v1 only ever
created object id 1 (`OBJECT_ID`), which existing clips keep.

- `create_object(clip, class_id)` no longer refuses a second object. Ids are never reused within a clip, so a
  deleted object's stale mask URLs and job events can never be mistaken for a new object's: `Clip` gains
  `next_object_id: int = 0` (0 in old clip.json files), the new id is
  `max(clip.next_object_id, max(existing ids, default=0) + 1)`, and `next_object_id` becomes that id + 1.
- `OBJECT_ID` is removed; every server path takes an explicit object id.
- Deleting an object removes its `LabelObject` entry and its `masks/<id>/` directory (§7 for when it is allowed).
- `ProjectStore.frame_states(clip)` keeps its meaning (the clip-level aggregate: deleted > keyframe of any object >
  masked by any object > empty) and gains `object_states(clip, obj) -> list[FrameState]` with the same priority rule
  for one object (a deleted frame is `deleted` in every lane).

## 5. Engine

`Engine.propagate(frames, seeds: Mapping[int, np.ndarray], cancel_cb=None) -> Iterator[tuple[int, dict[int, np.ndarray | None]]]`

- `seeds` maps object id → bool mask for `frames[0]`; at least one, each non-empty and of the frames' shape, else
  `ValueError` (raised lazily, as today). All seeds belong to the same frame: the caller groups objects by anchor
  (§6).
- Each yielded dict has exactly the seeded ids as keys; `None` = no usable mask for that object on that frame (the
  existing degenerate-mask rule, applied per object).
- `Sam3Engine._propagate_locked`: for each seed, `obj_idx = session.obj_id_to_idx(obj_id)` and
  `session.add_mask_inputs(obj_idx, 0, seed)`; `session.obj_with_new_inputs` = every seeded id. Each output's
  `pred_masks` has one row per tracked object, in the order of `out.object_ids`; row i is binarised on its own and
  belongs to `out.object_ids[i]` (`_binarize` gains a row argument instead of taking `[0, 0]`). The tracker head
  runs once per object per frame, so tracker cost grows linearly with the object count; frame encoding is shared.
  The OOM back-off is unchanged.
- `predict_text` / `predict_prompt` are unchanged (single object, preview only).
- `FakeEngine` and the engine contract tests move to the new signature. One object is simply one seed.
- A `gpu`-marked test propagates two objects on a test clip and checks each output stays near its own seed.

## 6. Propagation

`POST /api/clips/{id}/propagate` body: `{from_frame, direction, objects: "all" | [object_id, ...]}` (`objects`
defaults to `"all"`; "all" includes objects hidden in the GUI).

- **Participants**: the requested objects that have a mask on `from_frame`. For `"all"`, objects without one are
  skipped and reported (`skipped: [ids]`); for an explicit list, a listed object without a seed is a 400 naming it;
  an unknown id is a 404; no participant at all is a 400 (today's message). The route validates for early errors,
  but the **job resolves participants again** from a fresh clip read: an object deleted in between is skipped, not
  an error, and `_record_range` ignores an object that no longer exists (no bare `next(...)`).
- **Per-object bound**: each participant's frames are `plan_indices(clip, obj, from_frame, direction)` (up to its own
  next keyframe). The direction covers the union of the participants' lists; `total` = the sum over directions of
  the union's size.
- **Per-object anchors** (the rule today, kept per object): an object's anchor is its own last frame with a mask in
  this run (`from_frame` at the start); the anchor image is prepended to the chunk and the object is seeded with its
  mask there. Frames where it had no mask are skipped over, exactly as the single object's are today. For each
  chunk, the still-active objects are **grouped by anchor frame** and each group runs as one engine call on
  `[its anchor] + chunk`. Usually there is one group; a second group costs one extra pass over the chunk. The
  prefetcher is keyed per group (`(anchor, chunk)`, as today), and `schedule()` predicts the group of the object
  that wrote the chunk's last frame.
- **Inside a chunk**: an object that reaches its bound or gets lost mid-chunk stays in that engine call, but its
  outputs past that point are discarded. Progress (`done`, `processed`) counts frames covered by at least one
  object; when groups diverge the lagging group re-emits frame events for earlier frames, so events are not in frame
  order. Masks are written per object (deleted frames are not written; `None` deletes that object's stored
  mask on that frame). `_record_range(obj, frames)` is called per object with only the frames that object covered,
  on every exit path, as today. "Stopped short" compares the frames yielded with the chunk's length, per group.
- **Lost**: counted per object (`lost_after` consecutive empties). A lost object stops being seeded. A direction ends
  when every participant is lost or bounded. Result:
  `lost_by_direction: {direction: {object_id: frame}}`, `lost: {object_id: frame}` (the first direction in run order
  that lost it), and `lost_at` = in the first direction (run order) where anything was lost, the lost frame nearest
  `from_frame`. The page seeks to `lost_at`, as today.
- **Frame event**: `{type: "frame", frame, objects: [{id, has_mask}], done, total}`, listing only the objects still
  active on that frame. There is no aggregate `has_mask`: the client derives the aggregate state from its per-object
  lanes (a non-participating object's mask on that frame must keep the frame "masked").
- OOM back-off, cancel latching and the "engine stopped short" error are unchanged, applied per engine call.

## 7. Labelling API

- `GET /api/clips/{id}` (`clip_detail`) adds `object_states: {object_id: FrameState[]}` (JSON keys are strings);
  `states` stays the aggregate (Clips page, ClipStats). `clip_summary` is unchanged (its `clip` dump already carries
  `objects` and will carry `next_object_id`).
- Masks: `GET /api/clips/{id}/objects/{object_id}/masks/{idx}` replaces `/masks/{idx}` (only the web app used it;
  existing tests move to the new route).
- `POST /prompt` takes `object_id: int | null`: a clicks prompt without a box derives its box from **that** object's
  mask on the frame (`_effective_box`); null = no derived box. Preview and accept then compute the same mask.
- `POST /accept` and `POST /mask` take `object_id: int | null`. `null` creates a new object and requires `class_id`;
  an id writes to that object (404 if unknown). `class_id` together with an `object_id` is a 400: relabelling is its
  own call.
- `POST /accept` with `object_id: null` also takes `candidates: [int, ...]` (default `[candidate]`): each listed
  text-prompt candidate becomes its own new object of `class_id`, in one call ("Accept all as new objects").
- `PATCH /api/clips/{id}/objects/{object_id}` `{class_id}`: relabel at once. Takes the clip lock only, not the GPU
  lease, so it works during a propagation (it touches no mask).
- `DELETE /api/clips/{id}/objects/{object_id}`: clip lock + GPU lease; 409 while a propagation or an **export** job is
  running (as `DELETE` clip does today), since both read that object's masks.
- `POST /clear-range` takes `objects: "all" | [id, ...]` (default `"all"`, today's behaviour).
- Frame delete / restore / prune stay clip-wide.
- Locking, stated: accept, `/mask`, clear-range and object delete keep taking the GPU lease, so while any
  propagation runs — including a "selected object only" run — every object on every clip is read-only for masks.
  The GUI shows the same "A propagation is running" reason as today.

## 8. Label page

- **Objects panel** (replaces "Object", right sidebar): one row per object — class swatch, `#id class`, masked-frame
  count, an eye toggle (hide from the canvas; display only, per page session). Clicking a row makes it active. Below
  the list: the active object's class select (applies at once via PATCH), "+ New object" with a class select (the
  new object only exists once its first mask is accepted or painted; until then the active row reads "New object
  (class)"), Delete object (confirm dialog naming its frame count). The Add-class form stays.
- **Keys**: `O` / `Shift+O` next / previous object, `N` new object (unbound today; `useKeyboard` binds only Enter,
  Esc, arrows, Space, Del, `[`, `]`, `-`, `=`, `+`, `z`, `k`, `g`, `m`). `K`, `G` and the "has a mask here" checks
  (`hasMask`, prefetch, propagate-blocked reason) use the **active object's lane**, not the aggregate `states`.
- **Canvas**: every visible object is drawn in its class colour at `MASK_ALPHA`; the active one also gets the outline
  from the Mask outline settings, others none. `FrameStore` composites a frame's visible objects into **one** cached
  layer keyed by the frame's mask version, the ordered visible ids, the active id, class colours and outline style,
  so the 48-layer LRU still holds 48 frames regardless of object count. Prefetch fetches the visible objects' masks.
- **Brush**: `useBrush` loads its base from the **active object's own** mask URL; while brushing, the composite
  leaves the active object out (the brush layer is drawn in its place). Save posts `/mask` with the active
  `object_id`, or `object_id: null` + `class_id` on a "New object" row.
- **Prompting**: `usePrompting` sends `object_id` with `/prompt` and `/accept`; its current "send the class when it
  changed" logic is removed (relabel is PATCH). Text prompts with several candidates offer "Accept selected" and
  "Accept all as new objects" (the latter only when the active row is "New object").
- **Propagate panel**: a two-way choice "All objects" / "Selected: #2 cup" above the direction buttons; the note
  names skipped objects and each lost object. "Selected" is disabled with a reason on a "New object" row.
- **Cleanup panel**: "Clear masks" gains "Active object" / "All objects" (default active object). Delete / prune
  unchanged.
- **Mask versions and lanes**: `useClip.patchFrames` takes the new event shape, bumps the frame's version, updates
  each listed object's lane with today's `applyFrameEvent` rule, then recomputes the aggregate state for that frame
  from all lanes.
- Deleting the active object makes the next object (or "New object" when none is left) active. An object with no
  masks left stays listed until deleted.

## 9. Timeline (one lane per object)

- One lane per object in object-id order; inactive lanes are 10 px, the active lane 30 px (today's band), 2 px gaps.
  Each lane uses today's encoding for its object (filled in the class colour = masked, hollow = no mask, amber
  diamond = keyframe); deleted frames are hatched across every lane. Hidden objects' lanes are drawn at 40 % opacity.
- A fixed 72 px left gutter names each lane `#id class`; clicking a lane or its label selects that object, and
  clicking inside a lane also seeks (as today).
- The playhead spans all lanes. Height grows with the object count up to 6 lanes, then the lane area scrolls
  vertically and keeps the active lane in view.
- The hover tooltip names the frame and each object's state on it.
- Shift-drag marks a range as today; what "Clear masks" then clears follows the Cleanup panel's choice.
- A clip with no objects shows a single empty lane labelled "No objects yet".

## 10. Export: class selection

- `ExportOptions.class_ids: tuple[int, ...] | None = None` (None = every class) — an `ExportOptions` field, because
  derived copies rebuild options with `ExportOptions(**manifest["options"])`; stored in the manifest as a list and
  converted back to a tuple when read. API: `ExportBody.options.classes: [int] | null`; empty list or an unknown id
  is a 400.
- `build_plan` drops objects of unselected classes before the empty-frame filter, so a frame left without objects is
  kept or skipped by `include_empty`, and the split is computed after that (unchanged order).
- **Renumbering**: the selected classes, ordered by project class id, get export ids 0..k-1. `PlanObject.class_id` and
  `plan.classes` carry export ids; `ExportPlan.class_map: dict[int, int]` (export id → project class id) is a
  required field. With every class selected the mapping is the identity (project class ids are contiguous from 0:
  `add_class` uses max+1 and classes cannot be deleted). YOLO names/labels use the export id; COCO `category_id` and
  the mask PNG value use export id + 1 (unchanged convention).
- `manifest_for` writes each `classes` entry with `project_class_id` (from `class_map`); `id`, `yolo_id`, `coco_id`,
  `mask_value` are export ids. **Derived copies** (augmented, and negatives if that feature lands) build their plan's
  `class_map` from the source manifest with `c.get("project_class_id", c["id"])`, so older exports map to themselves
  and the mapping survives any chain of derivations.
- **Overlap in the semantic mask PNG**: objects are painted largest area first, so a smaller object on top of a
  larger one stays visible. `overlap_px` (pixels claimed by more than one object) is reported in the `masks` format's
  summary, computed in `MaskPng.write`, and summed by the derived copies' summary merge.
- Export page: a "Classes" section in the right sidebar, one checkbox per class with its swatch and its object count
  across the included clips (from the clip summaries' `objects`); all ticked by default; the Start problem text covers
  "no class selected".

## 11. Export: names

- `ExportBody.name: str | null` (top level; stripped, at most 80 characters, any text).
- Folder id: `slug(name)` = lowercase; runs of characters outside `[a-z0-9._-]` become `-`; trimmed of leading
  non-alphanumerics and trailing `-._`; at most 48 characters. An empty slug (or no name) falls back to today's
  `YYYYmmdd-HHMMSS-xxxxxx` id. The id is reserved with `mkdir(exist_ok=False)`; on `FileExistsError` it tries
  `-2`, `-3`, ..., so two concurrent exports never share a folder.
- `export.json` gains `name` (free text or null; a missing key in older manifests reads as null). The job result
  returns `export_id` and `name`.
- `ExportInfo` gains `name`; `to_json` and the web export type include it. Everywhere an export is shown: `name` when
  set, with the id beneath; otherwise the id. Runs store only `export_id` (run.json), so the Runs table and the
  prediction gallery look the name up in the `/api/exports` list; without a name the Runs table keeps today's
  project name over the id.
- Rename: `PATCH /api/exports/{id}` `{name}` rewrites `export.json`'s `name` atomically under one global lock, not a
  per-export one (renames are rare, so a single lock is the simpler choice); the folder id never changes, so runs
  that reference it stay valid. Applies to exports listed on the Train page: those with a yolo_seg dataset. Train
  page: a rename field for the selected export in the sidebar.
- `list_exports` sorts by the manifest's `created_at`, newest first, ties broken by id (its docstring no longer says
  ids start with a timestamp).
- Derived copies keep their id rules (`<source id>-aug<N>x-<4hex>`, and `<source id>-neg<N>-<4hex>` if the
  negatives feature lands) and take the source's display name (name, else id) plus a suffix: ` ×<N> augmented`,
  ` +<N>% negatives`.
- `vsp train --list` prints a name column; `--export` takes the folder id (unchanged).

## 12. Augmented copies: per-object cutouts

With several objects per frame, cutting composites per class region of the semantic PNG would cut two touching
objects of one class as one. `_cutouts_and_sizes` therefore cuts per object from the source's COCO train annotations
(`coco/annotations/instances_train.json`; each annotation's `file_name` maps to `images/<stem>.jpg`; rasterised with
the export's own `polygons_to_mask`; class = `category_id - 1`, an export id; each object minus the pixels another
object in the frame also claims, so a cutout never carries a neighbour's pixels) when the source has the `coco` format,
and falls back to per-class regions of the mask PNGs otherwise. The `masks` format stays required (real frames are
listed from it). The derived manifest records `cutouts: "per_object" | "per_class_region"`, and the Train page's
augment panel notes the fallback. The same seed on an older export may now produce different composites (different
cutout set); this is expected.

## 13. Errors and edge cases

- Relabel during a propagation is allowed (class only); delete object is 409 during a propagation or an export.
- Propagate "Selected" on a "New object" row is disabled with the reason.
- Export where every selected class is absent from the chosen clips: the existing "nothing to export" 400 (with
  `include_empty` off) or an all-empty dataset (with it on); the Classes section shows 0 objects for such a class.

## 14. Testing

- **Plan A**: store (`next_object_id`, delete, `object_states`); engine contract with 1 and 3 seeds (fake engine);
  propagation: per-object keyframe bounds, one object lost while another continues, per-object anchors diverging
  into two groups, skipped objects, explicit list with a missing seed, object deleted between route and job,
  cancel and OOM with two objects, `lost_at` rules, `total`; prefetch with per-group keys; API: `/prompt` with
  `object_id`, accept new vs existing object, accept-all candidates, `class_id` + `object_id` 400, PATCH during a
  propagation, DELETE during a propagation and during an export (409), per-object mask route, clear-range by object,
  propagate body validation; existing `/masks/{idx}` tests moved. GPU: two-object SAM3 smoke test.
- **Plan B**: Timeline lanes (heights, selection click, seek, hover text, hatching across lanes, >6 lanes scroll),
  Objects panel, `FrameStore` composite keying, `useClip.patchFrames` aggregate recompute, `usePrompting` and
  `useBrush` with object ids, Propagate/Cleanup choices, keys.
- **Plan C**: class filter + renumbering in YOLO/COCO/masks, manifest mapping, empty-frame handling after the filter,
  overlap order and `overlap_px`, derived manifests keep `project_class_id`, old manifests without it, per-object
  cutouts from COCO and the flagged fallback; Export page Classes section.
- **Plan D**: slug rules, collision suffixes, concurrent reservation, rename (and concurrent renames), old manifests
  without `name`, `list_exports` order, `--list` column; Export page Name field, Train rename, Runs table name lookup.
- README: Label page (objects, keys), Export (classes, names), manual QA steps for multi-object propagation.

## 15. Note on work in progress

At the time of writing the working tree also holds an uncommitted "negatives" feature (`server/vsp/augment/negatives.py`,
`vsp negatives`, edits to `derive.py` and `cli.py`) that is not part of this spec. §10 and §11 say how negatives copies
should behave if it lands; plan C picks that up only if it has been committed by then.
