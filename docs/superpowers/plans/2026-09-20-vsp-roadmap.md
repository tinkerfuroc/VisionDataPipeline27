# Vision Sampler — Implementation Roadmap

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`

The spec is implemented as four plans. Each ends in working, tested software and is written in full
detail only when the plan before it has landed, because each depends on interfaces the previous one
proves (most importantly: Plan 1 Task 6 is the first time SAM3 inference runs on this machine).

| # | Plan | Delivers | Needs GPU | Status |
|---|---|---|---|---|
| 1 | `2026-09-20-vsp-plan1-backend-core.md` | uv project, settings, `vsp doctor`, store, ingest, weights tooling, `Engine` protocol, `FakeEngine`, `Sam3Engine` verified on GPU | Task 6 only | **Done** (branch `plan1-backend-core`, 71 tests, 10 on GPU) |
| 2 | `2026-09-20-vsp-plan2-labeling-api.md` — labeling, jobs, HTTP API | Everything the GUI calls, fully testable with `FakeEngine` and `TestClient` | One added GPU test | **Written, ready to execute** |
| 3 | Web GUI | React SPA: Clips page, Label page, served by `vsp serve` | For manual QA | Outline below |
| 4 | Export | `ExportPlan` + YOLO-seg, YOLO-det, COCO, mask PNG exporters, Export dialog | No | Outline below |

Plans 3 and 4 are independent of each other once Plan 2 is done and can run in either order.

## Carried forward from Plan 1 (executed 2026-09-20, branch `plan1-backend-core`)

Measured on the RTX 3080 Ti (driver 535, torch 2.14.0+cu126, transformers 5.17.0): model load 3 s;
click → mask 65 ms with a warm vision cache (≈ 850 ms for the first prompt on a frame); click IoU 0.99,
box IoU 0.98, box + positive + negative click IoU 0.92; forward and backward tracking 19/19 frames at
IoU ≥ 0.98; **peak VRAM 2.47 GiB for a 24-frame chunk at 640×480** — `propagate.chunk_size` has a lot of
headroom; re-measure at the real footage resolution before raising it.

Facts Plan 2 must design around:

- `Sam3Engine.propagate()` is a generator that holds the (non-reentrant) GPU lock from the first `next()`
  until it is exhausted or closed. Always wrap it in `contextlib.closing`; cancel via `cancel_cb`; never
  call `release()` / `predict_*` from the thread that is consuming it. Validation is lazy (first `next()`).
- `frame_key` for the vision cache must be clip-scoped: `(clip_id, frame_idx)`.
- All three engine paths return `None` / drop masks under 64 px or over 45 % of the frame
  (`Sam3Engine.MIN_MASK_AREA_PX`, `MAX_MASK_AREA_FRAC`); `FakeEngine` mirrors this, so test masks must be
  ≥ 64 px. The UI needs a "no usable mask" message for prompts that land outside those bounds.
- `tracker_model.config.multimask_max_pt_num` is raised from the checkpoint's 1 to 32 at load, because box
  corners count as points and the default disabled best-of-3 mask selection for any box prompt. Verified
  for a box + 2 clicks only — **multi-click refinement (5+ clicks) is untested**; add a GPU test for it
  when `labeling/prompts.py` is written.
- Box-only prompts are slightly looser than click prompts; the UI should encourage a refining click.
- `Sam3Engine.load()` raises `EngineNotLoaded` for every failure and keeps the reason in `load_error` —
  this is what `/api/health` should report.
- Relative config paths resolve against the repository root (`vsp.config.REPO_ROOT`), so `vsp serve` may be
  started from any directory. `resolve_allowed()` exists in `vsp.ingest` but nothing calls it yet — the
  frame-directory import route must.
- The installed ruff's default rules include import sorting, pyupgrade and simplify; write plan code with
  sorted imports and `collections.abc`.

Deferred cosmetic minors (none blocking): `device.py` error message names the `sam3.device` key and has a
redundant `tuple()`; `cli.py` has an unreachable `return 2`; `convert_checkpoint` has no unit test (thin glue
over an external script).

## Plan 2 outline — labeling, jobs, HTTP API (spec §4 API table, §5.2–§5.4, §9)

Adds dependencies: `fastapi`, `uvicorn[standard]`, `python-multipart`; dev: `httpx`, `pytest-asyncio`.

1. **`labeling/prompts.py`** — `preview(clip, frame, prompt) -> list[Detection]` (nothing persisted);
   `accept(clip, frame, prompt, class_id, candidate) -> Clip` creates the object or adds a keyframe,
   writes the mask, records the prompt. Single-object rule enforced here and only here (409 later).
   Correction on an already-masked frame = padded box derived from the existing mask + the user's
   clicks (masks cannot be combined with points in transformers 5.17.0).
2. **`labeling/propagate.py`** — pure planning first: `plan_indices(clip, obj, from_frame, direction)`
   returns ordered frame indices capped at the neighboring keyframe (exclusive) or the clip end;
   `split_chunks(indices, size)`. Then the runner: anchor-prepended chunks (`[anchor] + chunk`, anchor =
   most recent frame in this run with a real mask, anchor output discarded), OOM back-off (halve to
   `min_chunk_size`), `lost_after` counter across chunks → early `done` with `lost_at`, keyframes never
   overwritten, deleted frames fed to the tracker but not written, atomic mask writes, `clip.json`
   updated once per chunk, cancel keeps landed masks. **Always exhaust or `close()` the engine
   generator** — `Sam3Engine.propagate` holds the GPU lock across yields.
3. **`labeling/frames.py`** — delete/restore, prune-from, clear-range (non-keyframes only; trims
   `propagated_ranges`).
4. **`jobs/manager.py` + `jobs/ws.py`** — one `JobManager` for import, propagate, export:
   start/poll/cancel, event stream `progress|frame|log|done|error|cancelled`, one GPU job at a time.
5. **`api/` + `app.py` + `vsp serve`** — the routers in the spec's API table; engine loads in the
   background at startup; `/api/health` reports state, device and `load_error`; frame-dir import goes
   through `resolve_allowed`; static SPA mount (empty until Plan 3).
6. **API tests** — `TestClient` with `FakeEngine`: full flow import → prompt → accept → propagate →
   correct → prune; WebSocket event order; every 409/400 path in spec §9.

## Plan 3 outline — web GUI (spec §7)

Prerequisite: Node 20+ installed user-level with nvm (not yet on this machine). Stack: React + Vite +
Tailwind, ported `rest.ts` and canvas-overlay code from the template (its `web/` directory is complete,
unlike its `server/`).

1. Vite scaffold, `rest.ts` typed client, `useJob` (WebSocket with polling fallback) + reducer tests.
2. Clips page — projects/classes, import video, import frame directory, progress, clip cards.
3. `FrameCanvas` — frame + mask overlay, zoom/pan, click/box capture, candidate numbering.
4. `PromptPanel` + `useClip` — Click/Box/Text modes, preview on every click, Accept/Clear, class picker.
5. `Timeline` + transport — state strip, seek, keyframe/empty-frame jumps, playback with prefetch;
   vitest for state derivation.
6. `PropagatePanel` — direction buttons, live streamed masks, Stop, seek to `lost_at`.
7. `CleanupPanel` + keyboard shortcuts; engine-not-loaded banner.
8. Manual QA checklist on real footage (this is where click-mask quality on real components is judged).

## Plan 4 outline — export (spec §8)

1. `export/plan.py` — `ExportPlan`: non-deleted frames, export stride, include-empty, split
   (`per_clip` with user-chosen val clips; `contiguous` tail split). No-leak tests.
2. `export/polygons.py` (ported) + `yolo_seg.py`, `yolo_det.py` — golden files; polygon→raster IoU > 0.95.
3. `coco.py` — validated by loading with `pycocotools`.
4. `mask_png.py` — 8-bit indexed, `0` = background, `class_id + 1`.
5. Runner — copy images once into `exports/<id>/images/`, hardlink into format dirs (copy fallback),
   `export.json` manifest, runs as a job; API route.
6. `ExportDialog` in the SPA (if Plan 3 has landed).
