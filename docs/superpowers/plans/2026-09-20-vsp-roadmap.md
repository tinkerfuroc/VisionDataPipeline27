# Vision Sampler — Implementation Roadmap

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`

The spec is implemented as four plans. Each ends in working, tested software and is written in full
detail only when the plan before it has landed, because each depends on interfaces the previous one
proves (most importantly: Plan 1 Task 6 is the first time SAM3 inference runs on this machine).

| # | Plan | Delivers | Needs GPU | Status |
|---|---|---|---|---|
| 1 | `2026-09-20-vsp-plan1-backend-core.md` | uv project, settings, `vsp doctor`, store, ingest, weights tooling, `Engine` protocol, `FakeEngine`, `Sam3Engine` verified on GPU | Task 6 only | **Written, ready to execute** |
| 2 | Labeling, jobs, HTTP API | Everything the GUI calls, fully testable with `FakeEngine` and `TestClient` | No | Outline below |
| 3 | Web GUI | React SPA: Clips page, Label page, served by `vsp serve` | For manual QA | Outline below |
| 4 | Export | `ExportPlan` + YOLO-seg, YOLO-det, COCO, mask PNG exporters, Export dialog | No | Outline below |

Plans 3 and 4 are independent of each other once Plan 2 is done and can run in either order.

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
