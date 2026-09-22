# Vision Sampler — Implementation Roadmap

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`

The spec is implemented as four plans. Each ends in working, tested software and is written in full
detail only when the plan before it has landed, because each depends on interfaces the previous one
proves (most importantly: Plan 1 Task 6 is the first time SAM3 inference runs on this machine).

| # | Plan | Delivers | Needs GPU | Status |
|---|---|---|---|---|
| 1 | `2026-09-20-vsp-plan1-backend-core.md` | uv project, settings, `vsp doctor`, store, ingest, weights tooling, `Engine` protocol, `FakeEngine`, `Sam3Engine` verified on GPU | Task 6 only | **Done** (branch `plan1-backend-core`, 71 tests, 10 on GPU) |
| 2 | `2026-09-20-vsp-plan2-labeling-api.md` — labeling, jobs, HTTP API | Everything the GUI calls, fully testable with `FakeEngine` and `TestClient` | Two added GPU tests | **Done** (branch `plan2-labeling-api`, 167 tests, 12 on GPU) |
| 3 | `2026-09-20-vsp-plan3-web-gui.md` — web GUI | React SPA: Clips page, Label page, served by `vsp serve`; no Python changes | For the end-to-end smoke and manual QA | **Done** (branch `plan3-web-gui`, 134 web tests; Python suite unchanged at 167) |
| 4 | Export | `ExportPlan` + YOLO-seg, YOLO-det, COCO, mask PNG exporters, Export dialog | No | **Done** (branch `plan4-exporters`; see the Plan 4 section below) |
| 5 | `2026-09-21-vsp-plan5-uploads.md` — web uploads | Streaming video and image-zip uploads from the Clips page, with progress and cancel, over the SSH-forwarded GUI | No | **Done** (branch `plan5-uploads`, 204 Python tests, 167 web tests) |

Plans 3, 4 and 5 are independent of each other once Plan 2 is done and can run in any order.

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

## Carried forward from Plan 2 (executed 2026-09-20, branch `plan2-labeling-api`)

Verified end to end with the real engine over HTTP: click prompt → mask of 6476 px for a true disc of ≈ 6362 px,
forward propagation 19/19 frames. Multi-click refinement measured on the 3080 Ti: IoU 0.95 with a box + 6 clicks,
0.99 with 6 clicks and no box (`multimask_max_pt_num = 32` holds up).

API behaviours the GUI (Plan 3) must design around:

- **GPU exclusivity.** `POST /prompt`, `/accept`, `/clear-range` and `DELETE /api/clips/{c}` hold an *interactive
  lease*: they return 409 `{"detail", "job_id": "<running job>"}` while a propagate job runs, and queue behind each
  other on the engine lock (overlapping previews never 409). `POST /propagate` returns 409 `{"job_id": "interactive"}`
  for the few milliseconds an interactive call is in flight — **`rest.ts` must retry this**, not show it.
  `delete` / `restore` / `prune` and all reads stay allowed during a job; prune does **not** stop a running propagation.
- **Error bodies.** Mapped errors are `{"detail": "<string>"}` (404, 409, 400, 503 engine not loaded, 500 propagation
  failed). FastAPI validation errors are 422 with `detail` as a *list* — handle both shapes. At 503 import, browsing
  and frame cleanup keep working.
- **Prompts.** The server normalises inverted boxes, clamps boxes to the frame and rejects boxes under 2 px and points
  outside the frame with 400 — the canvas should still clamp clicks to `[0, width) × [0, height)`.
  "No usable mask" (under 64 px or over 45 % of the frame) is a 400 with an explanatory `detail`.
  A `clicks` prompt on an already-masked frame becomes a correction automatically (`payload.box_from_mask` is recorded).
- **Health.** `loaded: true` | `loaded: false, error: null` (still loading in the background — normal for the first
  seconds) | `loaded: false, error: "<reason>"` (show the banner).
- **Jobs.** 202 `{"job_id"}` from propagate and both imports. `WS /ws/jobs/{id}` replays from `seq` 0, streams, closes;
  unknown id → close code 4404. Events: `progress {done, total}` (**`total` is absent for imports**), `frame {frame,
  has_mask, done, total}`, `log {message}`, then exactly one of `done` / `cancelled` (result spread in) / `error
  {message}`. The terminal event is appended before `status` flips. A job is `cancelled` only if its work actually
  stopped early; a cancelled import removes its partial clip. There is **no HTTP events endpoint** — a polling fallback
  sees status/result only (`Job.events_since` exists; adding `GET /api/jobs/{j}/events?since=N` is trivial if wanted).
  Stop takes effect at the next frame boundary inside a chunk; a WebSocket client that disconnects while no events are
  pending is only noticed at the next event.
- **Propagation result.** `{processed, written, cleared, lost_at, cancelled, oom_retries, lost_by_direction}`;
  `lost_at` is the forward loss when both directions lose. Seek the timeline there.
- **Clip detail.** `{clip, states, classes}`; `states[i]` ∈ `empty | masked | keyframe | deleted`. **Drive the timeline
  from `states`, never from `propagated_ranges`** — a range means "visited by propagation", including frames where the
  object was absent. Frame-op responses return the same shape; while a job is live prefer the WS `frame` events.
- **Classes** are created with `POST /api/projects/{p}/classes {name}` → 201 (not `PATCH /projects/{p}` as spec §4
  sketched); "type a new class name" in the UI is two calls. `class_id` is required on the first accept for a clip.
- **Files.** Frames and thumbs are immutable JPEGs (cacheable); masks are PNG with `Cache-Control: no-cache` and 404
  when the object is absent in that frame.
- **SPA.** `create_app` mounts `web/dist` at `/` when the directory exists; no server change needed once Plan 3 builds.

Deferred minors (none blocking): `_record_range` uses `next()` without a default; `store.clip_dir` globs per call
(µs against a ms-scale GPU step); `_effective_box` is computed twice per accept; `JobManager` never evicts finished jobs
or their events; `Job.status` is written without the manager lock (GIL-dependent); lease starvation is possible under an
unbroken stream of interactive calls; no upload size limit (closed by Plan 5); the upload-write cleanup and the WebSocket close
suppression have no test; `PromptSpec.box` accepts floats (truncated to int when clamped); a text prompt carrying a stray
`box` is validated in `accept` but not in `preview`; `delete_clip` takes the GPU lease but not the clip lock, so it can
still interleave with delete/restore/prune on the same clip; `ProjectStore.lock`'s parameter is named `clip_id` although
`add_class` passes a project id; propagate's synchronous validation is best-effort (a keyframe
removed between the route and the job thread surfaces as a job error).

## Carried forward from Plan 3 (executed 2026-09-20/21, branch `plan3-web-gui`)

Verified against the real server: `scripts/seed_demo.py` imports 40 synthetic frames, accepts one click and propagates
39 masks; headless Firefox renders both pages (frame, tinted overlay, timeline, all panels).

For Plan 4 (export):
- Reuse `useJob` (it resets during render on a new job id) and `ImportJobRow` as the template for an export progress row.
- Add `clip_id` (or a scope) to `Job.snapshot()` on the server: today a 409 names the job holding the global GPU lock with
  no way to tell whose it is, so the Label page tracks ownership client-side (only jobs it started, or re-attached from
  its own `sessionStorage` key, may patch its states; a foreign job only shows progress and Stop).
- The Export dialog is the one spec §7 item not built yet.

Deferred minors (none blocking): `test/setup-dom.ts` replaces `getContext` without restoring it; `localStorage` /
`sessionStorage` access is unguarded; the clip-card thumbnail is always frame 0; the propagate `starting` guard is
state-only; the burst test cannot observe intermediate frames in jsdom (coalescing is proven by `useClip.patchFrames`'s
test); `FrameCanvas`'s failed-frame fallback and the `key={clipId}` route wrapper have no test; the poll fallback retries
every second without backoff; the 320 px sidebar wraps some button labels (superseded by the planned design pass).

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

Prerequisite: Node 22 is installed user-level with nvm (source `~/.nvm/nvm.sh` explicitly in non-interactive shells). Stack: React + Vite +
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

## GUI design pass (done, 2026-09-21)

Brief and the revisions made after user review: `docs/superpowers/specs/2026-09-21-gui-design-pass.md`. What Plan 4 must
respect when it adds the Export dialog:
- Every page keeps its controls in a right-hand sidebar; pages render their own `AppHeader` (it carries the engine banner).
- Chrome has no hue: use the `btn` / `btn-on` / `btn-danger` / `field` / `heading` / `note` / `alert` classes from
  `web/src/styles.css` and the Tailwind tokens (`surround`, `well`, `panel`, `line`, `ink`, `dim`, `key`, `danger`).
- The interface scale is the single root `font-size` percentage in `web/src/styles.css`; sizes are rem.
- Prefer direct manipulation with a button that names exactly what it will do (the mark-then-clear range is the model).
- The mask overlay is outlined (`tintMask` with a size); exported mask PNGs are unaffected, it is display only.
- `scripts/seed_demo.py` builds a colourful tracking scene for judging overlays.

## Plan 4 — export (done, 2026-09-21, branch `plan4-exporters`)

Plan and "As executed" notes: `docs/superpowers/plans/2026-09-21-vsp-plan4-export.md`. All four plans of the original
design are now built. Verified against the real server: a two-clip, four-format export of the demo project (per-clip
split, stride 2, one deleted frame) has no file stem in both splits, only the validation clip in `val`, hardlinked
format images (the store's frames stay unlinked), a parseable `data.yaml`, indexed mask PNGs, and YOLO-seg / COCO
masks that rasterise back to the mask PNG at IoU 1.0 (COCO through pycocotools).

Still deferred: `clip_id`/scope on `Job.snapshot()`; listing past exports in the GUI (the job result and `export.json`
carry the path); a propagation started while an export runs is tolerated (mask writes are atomic) rather than refused;
progress events are one per frame per pass, unthrottled, and finished jobs are never evicted from memory; a server killed
mid-export leaves a half-written folder (recognisable by its missing `export.json`); holes in masks are not represented in polygon formats.
Known cost, measured in the scoped re-review of the final fix: each mask is decoded and cleaned five times per export
(once by the plan, once by each of the four writers, because writers run format by format): about 5-7 ms per call at
1920x1080, so roughly a minute of redundant work for an 1800-frame, four-format export. The fix when it matters is a
frame-major loop (read and clean each mask once, hand it to all writers); a cache cannot help in format-major order.
`clean_mask` uses 8-connectivity, so a diagonal staircase of single pixels counts as one blob.

## Plan 5 — web uploads (done, 2026-09-21, branch `plan5-uploads`)

Plan and "As executed" notes: `docs/superpowers/plans/2026-09-21-vsp-plan5-uploads.md`. One streaming upload route
replaces the multipart `import-video`: `POST /api/projects/{p}/clips/upload?name=&stride=&fps=` takes the file as the raw
request body, streams it once into `data/uploads/` under `ingest.max_upload_bytes` (413 over the cap), and hands the temp
file to `VideoFileSource` or the new `ZipFrameSource` inside the ordinary import job, which deletes it. `ZipFrameSource`
reads `.jpg/.jpeg/.png` entries straight out of the archive (nothing extracted), in natural order across folders, with
entry and uncompressed-size caps and `__MACOSX`/dotfile/traversal names refused. The Clips page uploads with
`XMLHttpRequest` (the only browser API with request progress and a real abort) and shows percentage progress plus Cancel;
`/api/health` advertises the cap so an over-size file is refused before the transfer starts. The old multipart route,
`api.importVideo` and their tests are gone, so there is exactly one import path per source kind.

This plan supersedes the spec's §4 `import-video` row and the §6 "multipart upload to a temp file" clause, and it closes
the Plan 2 deferred minor "no upload size limit".

Still deferred (designed, not built): resumable chunked uploads (tus-style sessions with an offset, so a dropped SSH
link resumes instead of restarting); the transfer needs the browser tab to stay open; a `.zip` is the only archive
format; one file per request, no client-side queue. A real `ssh -L` tunnel could not be exercised on this box (no sshd);
a byte-copying TCP relay stood in for the forwarded stream, which is all the server ever sees.
