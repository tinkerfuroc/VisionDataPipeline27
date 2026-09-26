# Several objects per clip, class selection on export, named datasets

Date: 2026-09-26. Status: design approved in conversation (approach A, "object list with an active object").

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
| Propagate | both: all objects together in one SAM3 session (default), or the selected object only |
| Timeline | one lane per object, stacked; the active lane is taller; clicking a lane selects its object |
| Excluded classes on export | their objects are dropped; a frame left with none follows "Include frames without a mask" |
| Dataset name | both: free-text display name (editable later) and a folder id slugged from it |
| Overlay outline (previous change) | the active object gets the configurable outline; other objects are fill only |

## 3. Data model and storage

No migration. `Clip.objects: list[LabelObject]` and `masks/<object_id>/<frame:06d>.png` already exist; v1 only ever
created object id 1 (`OBJECT_ID`), which existing clips keep.

- `create_object(clip, class_id)` no longer refuses a second object. Ids are never reused within a clip, so a
  deleted object's stale mask URLs and job events can never be mistaken for a new object's: `Clip` gains
  `next_object_id: int = 0` (0 in old clip.json files), the new id is
  `max(clip.next_object_id, max(existing ids, default=0) + 1)`, and `next_object_id` becomes that id + 1.
- `OBJECT_ID` is removed; every server path takes an explicit object id.
- Deleting an object removes its `LabelObject` entry and its `masks/<id>/` directory, under the clip lock and the GPU
  lease (a propagation may be writing that directory).
- `ProjectStore.frame_states(clip)` keeps its meaning (the clip-level aggregate: deleted > keyframe of any object >
  masked by any object > empty) and gains `object_states(clip, obj) -> list[FrameState]` with the same priority rule
  for one object (a deleted frame is `deleted` in every lane).

## 4. Engine

`Engine.propagate(frames, seeds: Mapping[int, np.ndarray], cancel_cb=None) -> Iterator[tuple[int, dict[int, np.ndarray | None]]]`

- `seeds` maps object id → bool mask for `frames[0]`; at least one, each non-empty and of the frames' shape, else
  `ValueError` (raised lazily, as today).
- Each yielded dict has exactly the seeded ids; `None` = the object has no usable mask on that frame.
- `Sam3Engine`: every seed is added to the same inference session under its own `obj_id`
  (`session.obj_id_to_idx(obj_id)`, `add_mask_inputs`); outputs are split per object with the same post-processing
  and "no usable mask" rule as today, applied per object.
- `FakeEngine` and the engine contract tests move to the new signature. There is no single-object variant left; one
  object is simply one seed.
- A `gpu`-marked test propagates two objects on real footage fixtures and checks each output stays near its own seed.

## 5. Propagation

`POST /api/clips/{id}/propagate` body: `{from_frame, direction, objects: "all" | [object_id, ...]}` (`objects`
defaults to `"all"`).

- **Participants**: the requested objects that have a mask on `from_frame`. For `"all"`, objects without one are
  skipped and reported in the result (`skipped: [ids]`); for an explicit list, a listed object without a seed is a 400
  naming it. No participant at all is a 400 (today's message). An unknown object id is a 404.
- **Per-object bound**: each object's run stops before its own next keyframe in that direction (`plan_indices` per
  object). The run covers the union of the participants' ranges; an object whose bound is reached is no longer seeded
  and nothing more is written for it.
- **Chunks**: unchanged anchor scheme (the previous chunk's last frame is prepended). Each still-active object is
  seeded with its most recent mask: its mask on the anchor frame, or, if it had none there, its last mask within the
  current run of empties (at most `lost_after - 1` frames old — beyond that it is lost). Masks are written per object
  exactly as today (deleted frames are not written; a `None` deletes that object's stored mask on that frame).
- **Lost**: counted per object (`lost_after` consecutive empties). A lost object stops being seeded. A direction ends
  when every participant is lost or bounded. Result: `lost_at` becomes `lost: {object_id: frame}` plus the old
  `lost_at` = the earliest of them (the page seeks there, as today); `lost_by_direction` likewise per object.
- **Progress** counts frames, not frame×object. The `frame` event becomes
  `{type: "frame", frame, objects: {object_id: has_mask}, has_mask: any(...), done, total}`.
- `propagated_ranges` are recorded per object for the frames that object actually covered. OOM back-off, prefetch,
  cancel latching and the "engine stopped short" checks are unchanged.

## 6. Labelling API

- `GET /api/clips/{id}` (`clip_detail`) adds `object_states: {object_id: FrameState[]}`; `states` stays the
  aggregate (Clips page, ClipStats).
- Masks: `GET /api/clips/{id}/objects/{object_id}/masks/{idx}` replaces `/masks/{idx}` (only the web app used it).
- `POST /accept` and `POST /mask` take `object_id: int | null`. `null` creates a new object and requires `class_id`;
  an id writes to that object (404 if unknown). `class_id` is no longer accepted together with an `object_id`
  (400): relabelling is its own call. The clicks-without-box prompt derives its box from **that** object's mask on the
  frame (`_effective_box`), not object 1's.
- `POST /accept` with `object_id: null` also takes `candidates: [int, ...]` (default `[candidate]`): each listed
  text-prompt candidate becomes its own new object of `class_id`, in one call ("accept all").
- `PATCH /api/clips/{id}/objects/{object_id}` `{class_id}`: relabel immediately.
- `DELETE /api/clips/{id}/objects/{object_id}`: GPU lease + clip lock; 409 while a propagation runs (as `clear-range`).
- `POST /clear-range` takes `objects: "all" | [id, ...]` (default `"all"`, today's behaviour).
- Frame delete / restore / prune stay clip-wide.
- `clip_summary` adds `objects: [{id, class_id}]` for the Clips and Export pages (already in `clip` dump; keep it).

## 7. Label page

- **Objects panel** (replaces "Object", right sidebar): one row per object — class swatch, `#id class`, masked-frame
  count, an eye toggle (hide from the canvas; display only, per page session). Clicking a row makes it active. Below
  the list: the active object's class select (changes apply at once via PATCH), "+ New object" with a class select
  (the new object only exists once its first mask is accepted or painted; until then the active row reads "New
  object (class)"), Delete object (confirm dialog naming the frame count). The Add-class form stays.
- **Keys**: `O` / `Shift+O` next / previous object, `N` new object. (Free in `useKeyboard`: only Enter, Esc, arrows,
  Space, Del, `[`, `]`, `-`, `=`, `+`, `z`, `k`, `g`, `m` are bound.) `K`, `G`, `[`/`]` act on the active object.
- **Canvas**: every visible object is drawn in its class colour at `MASK_ALPHA`; the active one also gets the outline
  from the Mask outline settings, others none. `FrameStore` composites a frame's visible objects into **one** cached
  layer keyed by frame URL versions, the ordered visible ids, the active id, class colours and outline style, so the
  48-layer LRU still holds 48 frames regardless of object count. Prefetch fetches the visible objects' masks.
- **Prompting / brush** write to the active object (or create it when the active row is "New object"). Text prompts
  with several candidates offer "Accept selected" and "Accept all as new objects".
- **Propagate panel**: a two-way choice "All objects" / "Selected: #2 cup" above the direction buttons; the note
  names skipped objects and each lost object.
- **Cleanup panel**: "Clear masks" gains "Active object" / "All objects" (default active object). Delete / prune
  unchanged.
- **Mask versions**: `useClip.patchFrames` bumps the frame's version for any object event; `applyFrameEvent` updates
  both the aggregate and the per-object lanes.

## 8. Timeline (one lane per object)

- One lane per object in object-id order; inactive lanes are 10 px, the active lane is 30 px (today's band), with 2 px
  gaps. Each lane uses today's encoding for its object (filled in the class colour = masked, hollow = no mask, amber
  diamond = keyframe) and deleted frames are hatched across every lane. Hidden objects' lanes are drawn at 40 %
  opacity.
- A left gutter (fixed 72 px) names each lane `#id class`; clicking a lane or its label selects that object; clicking
  inside a lane also seeks (as today).
- The playhead spans all lanes. Height grows with the object count up to 6 lanes, then the lane area scrolls
  vertically and keeps the active lane in view.
- The hover tooltip names the frame, and each object's state on it.
- Shift-drag marks a range as today; what "Clear masks" then clears follows the Cleanup panel's choice.
- A clip with no objects shows a single empty lane labelled "No objects yet".

## 9. Export: class selection

- `ExportOptions.class_ids: tuple[int, ...] | None` (None = every class). The API body gains `classes: [int] | null`.
  Empty list or an unknown id is a 400.
- `build_plan` drops objects of unselected classes before the empty-frame filter, so a frame left without objects is
  kept or skipped by `include_empty`, and the split is computed after that (unchanged order).
- **Renumbering**: the selected classes, ordered by project class id, get export ids 0..k-1. `PlanObject.class_id`
  and `plan.classes` carry export ids; `plan.class_map` maps export id → project class id. YOLO names/labels use the
  export id, COCO `category_id` and the mask PNG value use export id + 1 (unchanged convention).
- `export.json` `classes` entries gain `project_class_id`; `id`, `yolo_id`, `coco_id`, `mask_value` are export ids.
  `options` records `class_ids`.
- **Overlap in the semantic mask PNG**: objects are painted largest area first, so a smaller object on top of a larger
  one stays visible; `counts` gains `overlap_px` (pixels claimed by more than one object) per split.
- Export page: a "Classes" section in the right sidebar, one checkbox per class with its swatch and its object count
  across the included clips; all ticked by default; the Start problem text covers "no class selected".

## 10. Export: names

- Body gains `name: str | null` (stripped, ≤ 80 chars, any text).
- Folder id: `slug(name)` = lowercase; runs of characters outside `[a-z0-9._-]` become `-`; trimmed of leading
  non-alphanumerics and trailing `-._`; at most 48 chars. An empty slug (or no name) falls back to today's
  `YYYYmmdd-HHMMSS-xxxxxx` id. The id is reserved with `mkdir(exist_ok=False)`; on `FileExistsError` it retries
  `-2`, `-3`, ... , so two concurrent exports can never share a folder.
- `export.json` gains `name` (the free text, or null) and the job result returns `export_id`, `name`.
- Everywhere an export is shown: `name` when set, with the id beneath; otherwise the id.
- Rename: `PATCH /api/train/exports/{id}` `{name}` rewrites `export.json`'s `name` atomically (the folder id never
  changes, so runs that reference it stay valid). Train page: a rename field for the selected export in the sidebar.
- `list_exports` sorts by the manifest's `created_at` (newest first) instead of by folder name.
- Augmented copies keep the id rule `<source id>-aug<N>x-<4hex>` and are named `<source display name> ×<N>
  augmented`.
- `vsp train --list` prints the name column; `--export` takes the folder id (unchanged).

## 11. Augmented copies: per-object cutouts

With several objects per frame, cutting composites per class region of the semantic PNG would cut two touching
objects of one class as one. `_cutouts_and_sizes` therefore cuts per object from the source's COCO train annotations
(`coco/annotations/instances_train.json`, rasterised with the export's own `polygons_to_mask`) when the source has
the `coco` format, and falls back to per-class regions of the mask PNGs otherwise; the derived manifest records
`cutouts: "per_object" | "per_class_region"` and the Train page's augment panel notes the fallback. Class ids in both
paths are export ids, so they match the manifest.

## 12. Errors and edge cases

- Deleting the active object makes the next object (or "New object" when none is left) active.
- Relabelling or deleting an object while a propagation runs: relabel is allowed (class only); delete is 409.
- An object with no masks left (all cleared) stays in the list until deleted, as the single object does today.
- Propagate "Selected" on a "New object" row is disabled with the reason.
- Export with every selected class absent from the chosen clips: the existing "nothing to export" 400 (with
  `include_empty` off) or an all-empty dataset (with it on) — the Classes section shows 0 objects for such a class.

## 13. Testing

- Server: store (`next_object_id`, delete, `object_states`), engine contract with 1 and 3 seeds (fake engine),
  propagation (per-object keyframe bounds, one object lost while another continues, anchor carry-over within
  `lost_after`, skipped objects, explicit list with a missing seed, cancel/OOM with two objects), API (accept new vs
  existing object, accept-all candidates, PATCH/DELETE object, per-object mask route, clear-range by object,
  propagate body validation), export (class filter + renumbering in YOLO/COCO/masks, manifest mapping, empty-frame
  handling after the filter, overlap order and `overlap_px`), names (slug rules, collision suffixes, concurrent
  reservation, rename, `list_exports` order), derive (per-object cutouts from COCO, fallback flagged).
- Web: Timeline lanes (heights, selection click, seek, hover text, hatching across lanes, >6 lanes scroll), Objects
  panel, `FrameStore` composite keying, Propagate/Cleanup choices, Export Classes and Name, Train rename.
- GPU: two-object SAM3 propagation smoke test.
- README: Label page (objects, keys), Export (classes, names), manual QA steps for multi-object propagation.
