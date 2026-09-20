# Vision Sampler — SAM3 Video Labeling Pipeline: Design

Date: 2026-09-20
Status: Draft for review

## 1. Purpose

A single-user, local web application that turns a short video (or a directory
of numbered frames) into a labeled dataset for one user-chosen component. The
user selects the component on any frame by text prompt, clicks, or a box; SAM3
tracks it through the clip; the user corrects drift by re-prompting and
re-propagating; the result is exported as YOLO-seg, YOLO-det, COCO JSON, and
mask PNGs.

### In scope

- Import: video file upload, or a server-side directory of numbered frames.
- Interactive selection: text prompt, positive/negative clicks, drag box.
- SAM3 tracking forward and backward from any prompted frame.
- Correction: re-prompt on a drifted frame, re-propagate from there.
- Cleanup: delete/restore frames, prune from a frame, clear masks in a range.
- Export to four formats with leak-free train/val splitting.

### Out of scope (v1)

- More than one object per clip (data model allows it; logic and UI do not).
- Live camera / RealSense capture, augmentation, model training, inference
  testing (all present in the template; intentionally dropped).
- Vertex-level polygon or brush editing.
- Multi-user access, authentication, resumable jobs.
- Clips longer than about one minute (no lazy decode or range selection).

## 2. Relationship to the template

Template: <https://github.com/tinkerfuroc/tk25_vision_train> (MIT, © 2025
tinker@furoc). This is a **new codebase** that ports proven modules rather
than forking the repository. Ported files keep a header comment crediting the
source, and the template's MIT notice is included in `THIRD_PARTY_NOTICES.md`.

| Template source | Destination | Treatment |
|---|---|---|
| `server/tk_vision/annotate/sam3.py` | `vsp/engine/sam3.py` | Port `Sam3Engine`, tracker_neck patch, vision-feature cache, `predict_text`, `predict_box`, `propagate_video`, `is_degenerate_mask`, `ChunkOOMError`. Add true point-prompt path (§5.2). |
| `server/tk_vision/annotate/chunker.py` | — | Not ported. Propagation runs over an ordered *list of frame indices* (ascending for forward, descending for backward), so chunking is a list split in `vsp/labeling/propagate.py`; the template's range/overlap chunker does not fit. |
| `server/tk_vision/capture/folder.py`, `source.py` | `vsp/ingest/` | Port `FrameSource`/`FolderSource` ideas into `FrameDirSource`; add natural sort and validation. |
| `server/tk_vision/services/polygons.py` | `vsp/export/polygons.py` | Port as-is. |
| `server/tk_vision/services/export_service.py` | `vsp/export/plan.py`, `yolo_seg.py` | Reuse split and YOLO-seg writing logic; restructure behind `Exporter` protocol. |
| `server/tk_vision/services/label_service.py` | `vsp/labeling/` | Rewrite; 521-line service split into focused units. Its chunk-seeding logic is *not* reused (see §5.3). |
| `server/tk_vision/data/persistence.py` | `vsp/store/` | **Missing from the published template**: its `.gitignore` has an unanchored `data/` rule that excluded the `tk_vision/data/` package (`ProjectStore`, `Clip`, `Track`, `read_mask`, …). `vsp/store` is therefore written from scratch, and any ported test that imports `ProjectStore` is adapted to the new store. Our `.gitignore` anchors the rule as `/data/`. |
| `server/tk_vision/api/propagate.py`, `ws/` | `vsp/jobs/` | Generalize the start/poll/cancel/WebSocket pattern into one `JobManager`. |
| `server/tk_vision/cli.py` (`serve`), `config.py`, `app.py` | `vsp/cli.py`, `config.py`, `app.py` | Port, trimmed; `fetch-weights` is replaced by `convert-weights` / `verify-weights` (§3). |
| `web/src/api/rest.ts`, canvas overlay code in `Label.tsx` | `web/src/` | Port; `Label.tsx` decomposed into components. |
| Tests for ported modules | `server/tests/` | Port alongside the module. |

Not ported: `capture/realsense_*`, `live_camera.py`, `augment_*`, `train_*`,
`infer_*`, `_train_runner.py`, `yolo_tuning/`, `Datasets.tsx`, `Test.tsx`,
`resource/ontology.json` (classes become per-project data edited in the GUI).

## 3. Environment

- Python 3.11 managed by **uv**: `uv venv`, `uv sync`, `uv run vsp serve`,
  `uv run pytest`, all from the repository root. Dependencies live in the
  root `pyproject.toml` (package source `server/vsp`, tests `server/tests`);
  `uv.lock` is committed. No conda files, no `requirements.txt`.
- PyTorch CUDA wheels are selected through a `[tool.uv.sources]` /
  `[[tool.uv.index]]` entry so `uv sync` resolves the correct build.
- `transformers` is pinned to a version that provides `Sam3VideoModel` and
  `Sam3TrackerVideoInferenceSession`.
- GPU: SAM3 runs in bfloat16 on the RTX 3080 Ti (12 GB), chosen automatically
  by `sam3.device: auto` (§11). The RTX 2080 Ti is not used (no bfloat16
  support).
- Node 20+ for building the SPA. `vsp serve` serves the built SPA statically.
- Video decoding uses OpenCV (`opencv-python-headless` bundles FFmpeg
  libraries), so **no system ffmpeg is required**.
- SAM3 weights — **verified 2026-09-20 with the user-supplied checkpoint:**
  - Supplied at `vision_train/sam3_checkpoint/`: Meta's raw `sam3.pt`
    (3.45 GB, 1465 fp32 tensors, prefixes `detector.` / `tracker.`) plus the
    official `config.json`, processor and tokenizer files. No
    `model.safetensors`, so conversion is required.
  - Converted to `vision_train/sam3_checkpoint_hf/` (the default
    `sam3.model_dir`) with Hugging Face's official
    `convert_sam3_video_to_hf.py` (transformers `main`, sha256 `7dcf1fd3…`; it
    is not shipped in the wheel). That script expects Meta-internal prefixes
    (`sam3_model.`, `sam2_predictor.model.`), so the public prefixes are
    renamed first. Result: **0 missing keys**; 38 unexpected keys, all benign
    (32 precomputed RoPE tables, 6 detector point-projection weights the HF
    model does not use); 22 `tracker_neck.*` tensors at top level, so the
    template's `tracker_model.tracker_neck.*` alias hack is not needed — the
    engine instead verifies after load that the live `tracker_neck`
    parameters equal the safetensors values.
  - Pitfall: run offline, the converter writes a 1.9 KB stub `tokenizer.json`.
    The official tokenizer/processor files from the upload are copied over it,
    and `recondition_on_trk_masks` is set to the official `false`.
  - `torchvision` is a hard requirement of the SAM3 image processor.
  - The template's `fetch-weights` is replaced by `vsp convert-weights`
    (wraps the steps above) and `vsp verify-weights` (file presence, real
    tokenizer, 22 tracker_neck tensors). No download command in v1.
  - **Not yet run:** inference on GPU with this checkpoint (text, click, box,
    20-frame propagation within 12 GB). This is the gate in the engine task.
- Dependencies dropped relative to the template: `ultralytics`,
  `albumentations`, `pyrealsense2`, `thefuzz`, `supervision`, `matplotlib`,
  `pandas`, `joblib`. `pycocotools` is a dev/test dependency only.

### Machine state found during design (2026-09-20)

Ubuntu 20.04, system Python 3.8.10, NVIDIA driver 535.183.01. **Not
installed:** `uv`, `node`, `ffmpeg`. Consequences:

- `uv` and Node (via nvm) need user-level installs before milestone 1; neither
  needs sudo. uv supplies Python 3.11 itself.
- `uv` 0.12.17 was installed to `~/.local/bin` during the verification spike
  (shell profiles untouched; that directory is already on PATH).
- **PyTorch build — verified in a throwaway venv (Python 3.11.16):**
  - PyPI's default `torch` (2.14.0) bundles CUDA 13 and needs driver ≥ 580; it
    cannot run on driver 535. The project must not resolve torch from PyPI.
  - `torch` and `torchvision` resolve from the official index
    `https://download.pytorch.org/whl/cu126`, declared in `pyproject.toml` as
    an explicit `[[tool.uv.index]]` and bound to those two packages through
    `[tool.uv.sources]`; everything else resolves from PyPI. A uv dry-run
    against this index resolved `torch==2.14.0+cu126` and
    `torchvision==0.29.0+cu126`.
  - Connectivity through this machine's local proxy is intermittent: the
    PyTorch index failed TLS 3/3 times early in the spike and succeeded 3/3
    later the same session. If it fails during setup, retry, or use the
    Aliyun mirror as a flat source
    (`uv sync --find-links https://mirrors.aliyun.com/pytorch-wheels/cu126/`),
    which is where the verified wheel was actually downloaded from (~20 MB/s).
    `huggingface.co` showed the same TLS failure at the time of writing, so
    which is one reason v1 has no weight-download command; the user-supplied
    checkpoint is converted locally.
  - **Pinned: `torch==2.14.0+cu126` with `transformers==5.17.0`**
    (`transformers` requires only `torch>=2.5`). On driver 535.183.01 this
    passed: CUDA available, bfloat16 matmul, bfloat16 autocast convolution at
    1×3×1008×1008 (cuDNN), bfloat16 scaled-dot-product attention, and imports
    of `Sam3VideoModel`, `Sam3VideoProcessor`, and
    `Sam3TrackerVideoInferenceSession`. No driver upgrade is needed.
  - PyTorch enumerates the RTX 3080 Ti as `cuda:0` while `nvidia-smi` lists it
    as GPU 1, confirming that `sam3.device: auto` (§11) is the right default
    and that a hard-coded index would be fragile.
  - The scaffold milestone keeps this check as `vsp doctor` so a future driver
    or wheel change is caught immediately.

## 4. Architecture

```
vision-sampler-pipeline/
├── pyproject.toml, uv.lock
├── server/
│   ├── vsp/
│   │   ├── engine/      types.py, device.py, masks.py, fake.py, sam3.py
│   │   ├── ingest/      source.py, video_file.py, frame_dir.py
│   │   ├── store/       project.py, models.py
│   │   ├── labeling/    prompts.py, propagate.py, frames.py
│   │   ├── export/      plan.py, base.py, polygons.py,
│   │   │                yolo_seg.py, yolo_det.py, coco.py, mask_png.py
│   │   ├── jobs/        manager.py, ws.py
│   │   ├── api/         health.py, projects.py, clips.py, prompts.py,
│   │   │                propagate.py, frames.py, export.py
│   │   ├── config.py, cli.py, app.py
│   └── tests/
├── web/src/
│   ├── api/rest.ts
│   ├── hooks/           useClip.ts, useJob.ts
│   ├── components/      FrameCanvas, PromptPanel, PropagatePanel,
│   │                    CleanupPanel, Timeline, ExportDialog
│   └── pages/           Clips.tsx, Label.tsx
├── configs/default.yaml
├── data/                (gitignored) projects/, exports/
└── docs/superpowers/specs/
```

### Module boundaries

- **engine** — arrays in, masks out. No knowledge of projects, disk layout, or
  HTTP. One `threading.Lock` serializes GPU access.
- **ingest** — turns a source into `frames/NNNNNN.jpg` + `thumbs/` + metadata.
  Nothing downstream knows which source type was used.
- **store** — the only module that reads or writes `data/projects/`. Pydantic
  models, atomic writes (temp file + rename), one lock per clip.
- **labeling** — orchestrates engine + store. Depends on an `Engine` protocol
  so tests use a `FakeEngine`.
- **export** — reads from store only. One `ExportPlan`, many `Exporter`s.
- **jobs** — `JobManager`: start, poll, cancel, WebSocket event stream
  (`progress | frame | log | done | error | cancelled`). Used by import,
  propagate, and export. At most one GPU job at a time; a second request
  returns HTTP 409.
- **api** — thin FastAPI routers; validation and wiring only.

### HTTP API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Engine state, selected device, reason if not loaded |
| `GET/POST /api/projects`, `GET/PATCH /api/projects/{p}` | Projects and their class lists |
| `GET /api/projects/{p}/clips` | Clip summaries |
| `POST /api/projects/{p}/clips/import-video` (multipart) | Start import job |
| `POST /api/projects/{p}/clips/import-frames` `{path, fps, stride}` | Start import job |
| `GET/DELETE /api/clips/{c}` | Clip detail (object, keyframes, per-frame state array) / delete |
| `GET /api/clips/{c}/frames/{i}`, `.../thumbs/{i}` | JPEG |
| `GET /api/clips/{c}/masks/{i}` | Mask PNG (404 when absent) |
| `POST /api/clips/{c}/prompt` `{frame, type, payload}` | Preview mask(s); nothing persisted |
| `POST /api/clips/{c}/accept` `{frame, prompt, class_id, candidate?}` | Create object or add keyframe |
| `POST /api/clips/{c}/propagate` `{from_frame, direction}` | Start job (409 if a GPU job is running) |
| `POST /api/clips/{c}/frames/{i}/delete`, `/restore`; `POST /api/clips/{c}/prune` `{from}`; `POST /api/clips/{c}/clear-range` `{start, end}` | Cleanup |
| `POST /api/projects/{p}/export` `{clips, formats, options}` | Start export job |
| `GET/DELETE /api/jobs/{j}`, `WS /ws/jobs/{j}` | Poll / cancel / event stream for every job type |

## 5. Data model and labeling flow

### 5.1 On-disk layout

```
data/projects/<project_id>/
├── project.json          name, created_at, classes: [{id, name, color}]
└── clips/<clip_id>/
    ├── clip.json         source {type: video|frame_dir, name}, fps, stride,
    │                     width, height, frame_count, original_names: {idx: name},
    │                     deleted_frames: [idx], objects: [Object]
    ├── frames/000000.jpg immutable after import
    ├── thumbs/000000.jpg
    └── masks/<object_id>/000000.png   1-bit PNG; absent = object not present
```

`Object`: `{id, class_id, keyframes: [Keyframe], propagated_ranges: [[start, end]]}`
`Keyframe`: `{frame_idx, prompt: {type: text|clicks|box, payload}, created_at}`

Rules:

- **Single object per clip (v1).** `objects` is a list for forward
  compatibility, but `labeling/prompts.py` refuses to create a second object.
  This is the only place the limit is enforced. To label another component in
  the same footage, import the footage again as a new clip.
- **Absent mask = not present.** Empty or degenerate tracker output (ported
  `is_degenerate_mask` area filter) writes no PNG. Exporters treat a missing
  PNG as "object absent in this frame".
- **Keyframes are never overwritten by propagation.** A keyframe's mask
  changes only when the user re-prompts on that frame.
- **Deleting a frame** adds its index to `deleted_frames`; no files are
  removed. Restore removes the index.

### 5.2 Prompting

All three prompt types produce a *preview* mask that is not persisted until
the user presses Accept. Accept creates the object (first time) or adds a
keyframe (subsequently), writes the mask, and records the prompt.

- **Text** — `predict_text(phrase)` returns up to 8 candidates; the UI numbers
  them and the user picks exactly one, which may then be refined with clicks.
- **Box** — `predict_box(xyxy)`; refinable with clicks.
- **Clicks** — positive/negative points. The template does not do real point
  prompting: it collapses positive clicks into a 12 px-padded box, runs
  `predict_box`, and erases a fixed 20–25 px square around each negative
  click. That is weak for small components next to similar parts. v1 aims to
  routes clicks through the SAM3 tracker (SAM2-style) prompt encoder: a
  one-frame `Sam3TrackerVideoInferenceSession` fed the cached vision features
  (same trick the template uses for propagation, so no extra weights are
  loaded), with points/box added via
  `Sam3TrackerVideoProcessor.add_inputs_to_inference_session`.
  **Verified by source inspection of `transformers` 5.17.0:** the session has
  `add_point_inputs`; the processor accepts `input_points` + `input_labels`
  and `input_boxes`, together or separately; `clear_old_inputs=False`
  accumulates clicks. **Verified constraint:** masks cannot be combined with
  points or boxes on the same frame (the processor and the model both raise),
  and this version does not feed a frame's previous mask logits back in.
  **Still to confirm on GPU with a checkpoint:** mask quality of this path on
  real footage. Fallback: port the template's `predict_clicks` unchanged.
- Each preview request sends the *full* click list for the frame (the UI owns
  it), so refinement is stateless on the server.
- Refining a frame that already has a mask (the correction case): because a
  mask prompt cannot be combined with clicks, the server derives a padded box
  from the existing mask and submits **box + the user's clicks**. The box
  localizes the object; positive/negative clicks fix the error. The prompt is
  recorded on the keyframe as `{type: clicks, payload: {box_from_mask, points}}`.
- Accept requires a class. The class field lets the user pick an existing
  project class or type a new name, which creates it.
- The vision embedding for the current frame is cached by `(clip_id, idx)` so
  each additional click is a decoder-only call (sub-second).

### 5.3 Propagation

Request: `{from_frame, direction: forward | backward | both}`. `from_frame`
must be a keyframe.

- Range is capped at the nearest neighboring keyframe (exclusive) or the clip
  boundary, so work beyond a later correction is preserved.
- Forward: frames `[k, end]`. Backward: the same engine call on the reversed
  slice, with indices mapped back. `propagate_video` seeds at index 0 of
  whatever list it receives, so no engine change is needed.
- Chunked (default 24 frames) because the tracker's memory does not survive
  across engine calls; only a seed mask carries identity between chunks.
- **Anchor-prepended chunks.** Each chunk's frame list is
  `[anchor frame] + next N frames`, seeded with the anchor's mask at index 0;
  the anchor's own output is discarded. The anchor is the most recent frame in
  this run with a non-empty mask (initially the keyframe). This guarantees the
  seed mask always matches the image it is applied to. The template instead
  applies the most recent prior mask to the chunk's first frame, which is
  wrong whenever the object was absent or moved at the boundary, and skips the
  chunk outright when no seed exists. The template's multi-frame overlap and
  `propagated_overlap` bookkeeping are not needed and are not ported.
- The `lost_after` counter runs across chunk boundaries.
- Deleted frames are still fed to the tracker (temporal continuity) but their
  masks are not shown or exported.
- Each mask is written atomically as it lands and a `frame` event is emitted;
  `clip.json` (`propagated_ranges`) is updated once per chunk.
- **Lost object:** after `propagate.lost_after` consecutive empty masks
  (default 15) the job ends as `done` with `lost_at: <idx>`; the UI seeks
  there.
- **OOM:** `ChunkOOMError` halves the chunk size and retries that chunk, down
  to 4 frames, then fails with an actionable message. Landed masks are kept.
- **Cancel:** stops after the current frame; landed masks are kept.

### 5.4 Cleanup

Delete/restore frame; prune from frame N to the end (bulk delete); clear the
object's masks in a frame range (for tracker hallucinations after the object
leaves view — removes PNGs for non-keyframes in the range and trims
`propagated_ranges`).

## 6. Ingest

Common options: `stride` (keep every Nth frame, default 1). Both sources run
as an import job and yield the same clip layout.

- **VideoFileSource** — multipart upload to a temp file; OpenCV
  `VideoCapture` reads fps/size and decodes sequentially (rotation metadata
  honored via `CAP_PROP_ORIENTATION_AUTO`); frames written as JPEG at
  `ingest.jpeg_quality`. `frame_count` is the number of frames actually
  decoded, not the container's reported count.
- **FrameDirSource** — a server-side directory path, which must resolve inside
  one of `ingest.allowed_roots`. Accepts `.jpg/.jpeg/.png`, natural sort
  (`frame_2` before `frame_10`), numbering gaps allowed. Rejects empty
  directories, unreadable files, and mixed resolutions with a specific error.
  JPEGs are copied (never hardlinked, so later edits to the user's source
  files cannot alter the store's immutable frames); PNGs are re-encoded to
  JPEG at `ingest.jpeg_quality`.
  `fps` is user-supplied (default 30). Original filenames are recorded in
  `original_names`.
- Failure at any point removes the partial clip directory.

## 7. GUI

Stack: React + Vite + Tailwind (same as template). Two pages.

**Clips** — project/class management, *Import video*, *Import frame
directory*, stride option, import progress, clip cards (thumbnail, frame
count, percent labeled, delete).

**Label** — three regions:

```
┌────────────────────────────────────────────────┬──────────────────────────┐
│              frame canvas                      │ OBJECT  class, stats     │
│   mask overlay · click dots · drag box         │ PROMPT  Click/Box/Text   │
│   zoom (wheel) · pan (middle-drag)             │         Accept / Clear   │
│                                                │ PROPAGATE ◀ Both ▶ Stop  │
│                                                │ CLEANUP delete/prune/    │
│                                                │         clear range      │
├────────────────────────────────────────────────┴──────────────────────────┤
│ transport · frame N / total · timeline: ◆ keyframe ▬ masked ░ empty ✕ del │
└───────────────────────────────────────────────────────────────────────────┘
```

- Click mode: left = positive, right = negative, `Z` undoes last click; each
  click refreshes the preview.
- Box mode: drag → preview → optional click refinement.
- Text mode: phrase → numbered candidates → click one.
- `Enter` accept, `Esc` clear, `←/→` step, `Shift+←/→` ±10, `Space` play with
  overlay at clip fps, `K`/`Shift+K` next/previous keyframe, `G` next empty
  frame, `Del` delete frame, `M` toggle overlay.
- The timeline is the primary QA tool: per-frame state strip, click to seek.
- Playback prefetches the next ~30 frames and masks so review at clip fps does
  not stall on per-frame requests.
- During a propagate job the canvas follows streamed masks and prompting is
  disabled; `Stop` cancels.
- A banner appears when `/api/health` reports the engine is not loaded, with
  the reason and the `vsp verify-weights` / `vsp convert-weights` commands.
- **Export dialog** — clip multi-select, format checkboxes, split mode,
  val fraction, export stride, include-empty toggle, progress, output path.

Components: `FrameCanvas`, `PromptPanel`, `PropagatePanel`, `CleanupPanel`,
`Timeline`, `ExportDialog`; hooks `useClip` (clip state + mutations) and
`useJob` (WebSocket job events with polling fallback).

## 8. Export

`ExportPlan` is computed once: for the selected clips, all non-deleted frames
(thinned by export stride), each frame's object masks, and a split assignment.
Every exporter serializes the same plan, so formats are mutually consistent.

- **Split modes:** `per_clip` (whole clips to train or val; default with ≥2
  clips — the Export dialog pre-selects the most recently imported
  ⌈`val_fraction` × n⌉ clips as val and the user can change the selection)
  and `contiguous` (last `val_fraction` of each clip's frames to val;
  default with 1 clip). Random per-frame splitting is not offered because
  adjacent video frames leak between train and val.
- **Options:** `val_fraction` (0.2), `stride` (1), `include_empty` (true —
  frames without a mask become background negatives), `min_area_px` (64).
- **Class ids** come from `project.json`, stable across clips.
- **File stems:** `<clip_id>_<original stem | 6-digit idx>`.

```
data/exports/<export_id>/
├── export.json                         options, clips, class map, counts
├── yolo_seg/{images,labels}/{train,val}/ + data.yaml
├── yolo_det/{images,labels}/{train,val}/ + data.yaml
├── coco/images/{train,val}/ + annotations/instances_{train,val}.json
└── masks/{images,masks}/{train,val}/   8-bit indexed PNG: 0 = bg, class_id+1
```

Images are copied once from the store into `data/exports/<export_id>/images/`
and hardlinked from there into each format directory (copy fallback). Store
frames are never linked outside the store, so modifying an exported image
cannot corrupt a project.
YOLO-seg: normalized polygons, multi-contour preserved (ported). YOLO-det:
normalized `cx cy w h` of the mask's bounding box. COCO: polygon
segmentation, `bbox`, `area`, `iscrowd: 0`, one category per project class.

`Exporter` protocol: `name: str`; `write(plan: ExportPlan, out_dir: Path,
progress: Callable[[int, int], None]) -> ExportSummary`. Adding a format means
adding one module and registering it.

## 9. Error handling summary

| Situation | Behavior |
|---|---|
| Engine cannot load | Server starts; health reports reason; UI banner; import/export still work |
| CUDA OOM in propagate | Halve chunk and retry to a floor of 4; then job error; landed masks kept |
| Tracker loses object | Early `done` with `lost_at`; UI seeks there |
| Second GPU job requested | HTTP 409 with the running job id |
| Server restart mid-job | No resume; atomic writes guarantee no corrupt masks; timeline shows progress |
| Bad video / frame dir | Import job error with cause; partial clip removed |
| Frame dir outside allowed roots | HTTP 400; never touches the path |
| Second object on a clip | HTTP 409 explaining the v1 single-object limit |
| Hardlink unsupported | Silent copy fallback, noted in `export.json` |

Server binds `127.0.0.1` by default; binding elsewhere logs a warning (ported).

## 10. Testing

Run with `uv run pytest` from `server/`.

- **engine** — device auto-selection, degenerate-mask filter, `FakeEngine`
  behavior; weights verification on a synthetic checkpoint directory.
- **labeling** (with `FakeEngine`: deterministic moving rectangle, scriptable
  empties and OOM): range capping at neighboring keyframes; backward index
  mapping; keyframes never overwritten; single-object enforcement; lost-object
  early stop; OOM back-off sequence; cancel keeps landed masks.
- **store** — model round-trip; atomic write leaves no partial file on
  simulated failure; delete/restore; clear-range trims `propagated_ranges`.
- **ingest** — synthesized 30-frame video; stride; natural sort; numbering
  gaps; mixed-size rejection; PNG re-encode; `original_names`; allowed-roots
  enforcement (ported path-safety tests); partial-clip cleanup on failure.
- **export** — golden output per format on synthetic masks; YOLO polygon →
  raster round-trip IoU > 0.95; COCO structure validated (pycocotools load);
  indexed PNG values; no frame appears in both splits; contiguous split
  boundaries; include-empty on/off; hardlink fallback.
- **api** — FastAPI `TestClient` with `FakeEngine`; job lifecycle and
  WebSocket event order; 409 paths.
- **GPU smoke** (skipped without CUDA + checkpoint; ported): tracker_neck
  patch; text, box, and click prediction; 20-frame forward and backward
  propagation.
- **web** — vitest for timeline state derivation, keyframe range logic, and
  the `useJob` reducer. No browser e2e in v1.
- **Manual QA checklist** (in README): import both source types → prompt by
  each method → propagate both directions → correct mid-clip → prune → export
  all formats → load `data.yaml` with Ultralytics and the COCO file with
  pycocotools.

## 11. Configuration (`configs/default.yaml`)

```yaml
server:    {host: 127.0.0.1, port: 28000}
project:   {data_root: ./data}
ingest:    {allowed_roots: ["~"], default_stride: 1, jpeg_quality: 95, thumb_width: 160}
sam3:      {model_dir: ./sam3_checkpoint_hf, device: auto, dtype: bfloat16,
            text_threshold: 0.45, box_threshold: 0.4, score_mode: native}
propagate: {chunk_size: 24, min_chunk_size: 4, lost_after: 15}
export:    {val_fraction: 0.2, min_area_px: 64}
```

`sam3.device: auto` selects the first CUDA device with compute capability
≥ 8.0 (native bfloat16), which on this machine is the RTX 3080 Ti regardless
of whether PyTorch enumerates it as index 0 or 1. The chosen device name is
logged at startup and reported by `/api/health`. An explicit `cuda:N`
overrides auto-selection. If no capable device exists the engine reports
"not loaded" with that reason (see §9).

## 12. Milestones (input to the implementation plan)

0. **Prerequisites** — uv is installed; install Node (user-level, via nvm);
   obtain a SAM3 checkpoint (user is supplying one; otherwise gated
   converted copy already exists at `vision_train/sam3_checkpoint_hf/`).
1. **Scaffold** — git init, uv project with the pinned torch build from the
   official cu126 index, `vsp doctor` GPU check (the verified spike script), config, CLI,
   FastAPI app + health, SPA shell served statically, local test command.
2. **Engine** — port `Sam3Engine` + tests; `convert-weights` / `verify-weights`; GPU smoke
   tests; point-prompt spike and decision.
3. **Ingest + store + Clips page** — both sources, import jobs, project/class
   management.
4. **Prompting** — prompts API, `FrameCanvas`, `PromptPanel`, preview/accept.
5. **Propagation + timeline** — `JobManager`, forward/backward, keyframe
   capping, lost-object stop, OOM back-off, `Timeline`, correction loop.
6. **Cleanup** — delete/restore/prune/clear-range, keyboard shortcuts.
7. **Export** — `ExportPlan`, four exporters, `ExportDialog`.
8. **Docs + QA** — README, manual QA checklist, third-party notices.
