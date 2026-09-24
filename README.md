# vision-sampler-pipeline

SAM3-driven video labeling for one user-chosen component per clip.
Design: `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`.

## Setup

    uv sync
    uv run vsp doctor          # torch build, GPU selection, bfloat16
    uv run vsp verify-weights  # converted SAM3 checkpoint is complete
    uv run pytest              # GPU tests skip themselves without CUDA + checkpoint

Python, torch (`2.14.0+cu126`) and transformers (`5.17.0`) are pinned in `pyproject.toml`.
Do not install torch from PyPI: its default build needs an NVIDIA driver >= 580.

Relative paths in the config (`project.data_root`, `sam3.model_dir`) are resolved against the
repository root, not the current working directory, so `vsp` can be run from any directory.

## Verified on this machine

- Date: 2026-09-20
- GPU: NVIDIA GeForce RTX 3080 Ti, driver 535.183.01, torch 2.14.0+cu126, transformers 5.17.0
- GPU smoke tests: 12 passed
- Peak VRAM, 24-frame propagate at 640x480: 2.47 GiB
- Text prompt "red circle" on the synthetic scene: 1 detection(s)

## Run the server

    HF_HUB_OFFLINE=1 uv run vsp serve        # http://127.0.0.1:28000 ; the model loads in the background
    curl -s --noproxy '*' http://127.0.0.1:28000/api/health

`/api/health` shows `engine.loaded` and, if loading failed, the reason. It also reports `upload.max_bytes`, the
size cap the GUI checks before it starts an upload. Interactive API docs: `/docs`.
Importing, browsing and frame cleanup work without a GPU; prompting and propagation return 503 until the
engine is loaded, and 409 (with the running `job_id`) while a propagation job holds the GPU. A propagate
request also returns 409 with `job_id: "interactive"` if a prompt/accept/clear-range call is in flight;
retry. Deleting a clip also returns 409 (with the running `job_id`) while a GPU job runs, so a delete can
never race a propagation's writes. Cancelling an import job removes the partial clip it had started.

`POST /api/projects/{id}/export` starts an export job; body `{clips, formats, options}` → `202 {job_id}`.
It returns 400 for an invalid request (unknown format, no clips, a bad option) and 409 (with the running
`job_id`) while a propagation job runs, since propagation rewrites the very masks an export reads.

### Reaching the GUI from another machine

`vsp serve` binds `127.0.0.1` by default. Forward the port instead of binding the server to the network:

    ssh -N -L 28000:127.0.0.1:28000 user@gpu-host

then open `http://127.0.0.1:28000/` locally; uploads travel through the tunnel. Binding a public interface
exposes an unauthenticated labeling server, which is why the CLI warns about it. `-o Compression=no` is
usually faster for already-compressed footage, and `-o ServerAliveInterval=30` keeps a NAT'd tunnel alive.

### Uploading footage

The Clips page uploads a video file or a `.zip` of images and imports it in one step. The browser sends the
file as the raw request body (not multipart) to
`POST /api/projects/{p}/clips/upload?name=<file>&stride=<n>&fps=<n>` → `202 {job_id}`. `stride` applies to
both kinds; `fps` only to a zip, because a video's frame rate comes from its container. A name is accepted if
it ends in one of the video extensions in `vsp.api.clips.VIDEO_SUFFIXES` or in `.zip`; a zip may hold
`.jpg/.jpeg/.png` entries in nested folders (natural order; `__MACOSX` and dotfiles are skipped). The upload
is streamed once into `data/uploads/` and deleted when the import job ends, and a server killed mid-upload
sweeps that directory at its next start. A body over `ingest.max_upload_bytes` (8 GiB by default) is refused
with 413; the GUI refuses the file before the transfer begins, using the cap from `/api/health`.

The transfer is a single request, so it needs the page to stay open: reloading or closing the tab cancels it
and leaves nothing behind. The GUI shows percentage progress and a Cancel button. Resumable uploads are not
implemented yet. Peak disk use during an import is the upload plus the JPEG frames and thumbnails it writes.

## Web GUI

Node 22 is installed with nvm (`source ~/.nvm/nvm.sh` in a fresh shell).

    cd web && npm ci            # once
    npm run build               # writes web/dist, which `vsp serve` serves at http://127.0.0.1:28000/
    npm run dev                 # or: hot-reloading dev server on http://127.0.0.1:5173 (proxies /api and /ws to :28000)
    npm test                    # vitest

### Demo data

With the server running, `uv run python scripts/seed_demo.py` creates a project "demo" with a synthetic 40-frame clip
(a disc moving across the frame), labels frame 0 with one click and propagates forward, then prints the URL to open.
`--no-label` imports only.

### Exporting

An export lands under `data/exports/<export_id>/`: an `images/` directory holding one copy of every
exported frame, plus one directory per requested format — `yolo_seg/`, `yolo_det/`, `coco/` and/or
`masks/` — and an `export.json` manifest recording the resolved options, the clips exported (with their
per-split frame counts), the class map (`yolo_id`, `coco_id`, `mask_value` per class) and the train/val
image counts. The train/val split is always either per clip (whole clips held out for validation) or
contiguous (the tail of every clip held out), never random per frame: neighbouring video frames are
nearly identical, so a random per-frame split would leak validation frames into training. A cancelled or
failed export removes its directory rather than leaving a partial dataset behind.

Noise is removed once, the same way for every format: connected blobs of a mask smaller than `min_area_px`
(64 by default) are dropped, what is left is the object for the polygon, the box, the area and the mask image alike,
and an object with nothing left counts as absent. With "include frames without a mask" off, the validation tail of
the contiguous split is taken from the frames that are actually exported.

Polygon and box coordinates share one convention in every format: continuous image coordinates, where pixel
`i` covers `[i, i + 1)`. An export refuses to start while a propagation is running (409), a clip cannot be
deleted while an export is reading it (409), and a request that turns out to have no usable mask at all is
reported as a failed job ("nothing to export"), because finding that out means reading the masks.

### Training

`uv sync` installs Ultralytics (YOLO26 and YOLO11 segmentation). Train from the Train page (header → Train) or from
the terminal:

    uv run vsp train --list                                  # exports that have a yolo_seg dataset
    uv run vsp train --export <id> --gpu 1                   # yolo26s/m-seg and yolo11s/m-seg, one after another
    uv run vsp train --export <id> --model yolo26s-seg --epochs 150 --imgsz 960 --batch 8

Each run lives in `data/runs/<run_id>/`: `run.json` (request and status), `worker.log`, Ultralytics' own output under
`train/` (`weights/best.pt`, `last.pt`, `results.csv`, plots) and `predictions.json` (best.pt on every val image, conf
≥ 0.25). Pretrained checkpoints are cached in `data/train_weights/`. A run is a separate process
(`python -m vsp.train.worker`) with `CUDA_VISIBLE_DEVICES` set to its GPU, so a crash or out-of-memory error ends
that run only.

The server keeps one queue per GPU: one run at a time per GPU, GPUs in parallel. The default GPU is the one SAM3 is not
on; choosing SAM3's GPU is allowed but shares its VRAM. Cancel stops a run after SIGTERM (SIGKILL after 30 s) and keeps
`last.pt`, so Resume continues from the last finished epoch. A run that was training when the server stopped is shown
as "interrupted" and can be resumed; queued runs are queued again. `vsp train` runs in the foreground and owns its
runs; Ctrl-C, SIGTERM and SIGHUP (a closed terminal, `kill`) all cancel them the same way. Do not start it on a GPU
the server is also training on. Do not set `CUDA_VISIBLE_DEVICES` for the server process itself — GPU indexes (from
`vsp doctor`, the Train page and `vsp train --gpu`) are passed to each worker as-is, so setting it on the server
would remap or hide devices under it. A run started by `vsp train` shows on the Train page's GPU queues as "running
in another process (vsp train)"; it can only be cancelled from the terminal it was started in.

Defaults: 100 epochs, early stop after 30 epochs without improvement, 960 px, batch 8, COCO-pretrained weights,
Ultralytics' default augmentation. Metrics come from `results.csv`; the comparison table ranks runs by the best mask
mAP50-95 epoch. ONNX export (`best.onnx`, static, square at the run's image size, 960 × 960 by default) runs on the run's GPU queue.

### Augmentation

**Presets:** `uv run vsp train --export <id> --augment strong` (or `default`, on the Train page select Default / Strong / Custom). The **Strong** preset raises geometric, photometric and copy-paste augmentation above Ultralytics' defaults: degrees 15, shear 5, perspective 0.0005, flip up 0.5, flip horizontal 0.5, scale 0.6, translate 0.2, HSV hue 0.02, saturation 0.7, value 0.5, mosaic 1.0, mixup 0.1, copy_paste 0.3, plus RandomBrightnessContrast, MotionBlur, GaussianBlur, GaussNoise, ImageCompression and CLAHE. The Custom preset on the GUI allows per-field adjustment.

Installing `albumentations` (a dependency since this branch, `uv sync` pulls it in) also lets Ultralytics enable its own built-in low-probability Blur, MedianBlur, ToGray and CLAHE transforms, independent of the preset above — so a **Default** run on this branch is not pixel-identical to a default run from before it.

**Background pools:** `uv run vsp backgrounds fetch dtd` (CLI, ~625 MB) or `POST /api/backgrounds/dtd/fetch` (web job) downloads the Describable Textures Dataset (5,640 images, one category per folder) to `data/backgrounds/dtd/`. The `--force` flag redownloads even if images are present.

**Augmented copies:** From the Train page, when an export has the `masks` format, an "Augmented copy" panel creates a new export with real train frames plus `per_image × (frame count)` synthetic composites named `syn_*`, seeded by a "Seed" field (0 by default, so two copies of the same export can differ). Each composite pastes `1–3` (user configurable) cutouts cut from real frames' mask regions (per-class, so two objects of the same class in one frame are cut together) onto random background textures, scaled, rotated, positioned and colour-matched. Val frames are unchanged (images hard-linked, labels copied verbatim per format). The source export is never modified and cannot itself be augmented if it is already derived. The copy carries over whichever of `yolo_seg`, `yolo_det`, `coco` and `masks` the source export has — not always all four. The copy's `export.json` names its source in a `derived_from` field (`null` for an ordinary export).

**Prediction strip:** On the Train page, when a run finishes, a "Validation predictions" gallery shows four images at a time, scrollable. Filter chips select All / Missed (ground truth present but no prediction) / Worst IoU. Click a card to enlarge it. Keyboard `←/→` pages the strip by four images at a time; once a card is enlarged, the same keys instead step one image at a time through the current filter, and `Esc` (or "Back to strip") returns to the strip positioned on that card. The overlay shows ground truth as a dashed outline and predictions filled blue with confidence scores. IoU values per image are computed with the export's polygon convention and cached in `<run_dir>/predictions_iou.json`, rebuilt when `predictions.json` changes.

### Keyboard

`Enter` accept · `Esc` clear · `Z` undo click · `←/→` step · `Shift+←/→` ±10 · `Space` play/pause · `K` / `Shift+K`
next / previous keyframe · `G` next empty frame · `Del` delete or restore frame · `[` / `]` mark the first / last frame to clear · `-` / `=` shrink / grow the brush (Brush tool) · `M` mask overlay · mouse wheel zoom ·
middle-drag pan · right click = negative point. `Enter`, `Esc` and `Z` act on the brush while it is selected (Save, Discard, Undo stroke) instead of accept / clear / undo click.

### Manual QA checklist (run on real footage before trusting a dataset)

1. Import a video and a frame directory (try a stride > 1); cancel one import midway and confirm no clip appears.
2. Label by each method: clicks (add a negative click), a box dragged bottom-right → top-left plus a refining click,
   and a text phrase with several candidates. Check the "no usable mask" message on an empty region.
3. Propagate forward, backward and both from a mid-clip keyframe; press Stop during one and confirm the masks so far stay.
4. Let the object leave the frame: propagation should stop early and seek to the "lost at" frame.
5. Correct a bad frame mid-clip (it becomes a keyframe) and propagate again: the run must stop at the next keyframe.
   Then Stop a run part-way and propagate again from the last masked frame: it continues from there, and the
   buttons stay disabled on a frame that has no mask.
6. Scrub the timeline and play at clip fps with the overlay on; delete a frame, restore it, prune the tail. Clear masks:
   Shift-drag a stretch of the timeline (or press `[` and `]`), check the button names the frames and the note counts
   the masks and keyframes, clear them, and confirm the keyframes inside the stretch are gone from the timeline too,
   the mark disappears, and a new prompt on the clip still lands on the same object and class.
7. Reload the page during a propagation: the panel should re-attach to the running job.
8. Stop the server: the banner says it is unreachable; start it without weights: the banner shows the reason and
   importing / cleanup still work.
9. Propagate twice in one page session (let the first run finish, then start a second from a keyframe) and reload
   the page partway through the second run: it must re-attach to the second job, not the first.
10. Open the same clip in two tabs; start a propagation in tab A, then click to prompt in tab B on a different clip:
    B must not move or show A's progress as its own.
11. Delete a keyframe: its marker disappears from the timeline and propagation is blocked from it until it is restored.
12. Read the timeline without relying on colour: masked runs are filled (in the class colour), frames without a mask
    are hollow, deleted frames are hatched and keyframes carry an amber diamond. Hover it for a thumbnail with the
    frame number and state. Zoom into the frame, then press Fit in the tool rail. Tab through the page: every control
    shows a focus ring.
13. Export (the "Export dataset" button on the Clips page): with two or more clips, move a clip between Training and
    Validation by hand and export all four formats; with one clip selected, check the note says the tail of the clip
    becomes the validation set. Cancel an export midway and confirm its folder under `data/exports/` is gone. Open
    `export.json`, then load the result in the tool you train with: `yolo_seg/data.yaml` or `yolo_det/data.yaml` in
    Ultralytics, `coco/annotations/instances_train.json` with pycocotools, and look at a few `masks/masks/` PNGs
    (pixel value = `mask_value` from the manifest's class map).
14. Upload footage from the Clips page over the SSH forward, once as a video and once as a zip of frames: the
    percentage must move, Cancel mid-transfer must stop it without an error message, and the finished import must
    appear as a clip whose name, frame count and fps are right. Reload the page during an upload and confirm the
    transfer stops, no partial clip appears, and `data/uploads/` is empty.
15. Upload a file larger than `ingest.max_upload_bytes`: the GUI must refuse it before the transfer starts, and
    `curl -X POST --data-binary @big.mp4 .../clips/upload?name=big.mp4` must answer 413.
16. Brush: pick the Brush tool on a masked frame, paint a missing corner (left-drag) and erase a spill (right-drag);
    the transport, the other tools, Propagate and Cleanup are disabled until you Save (Enter) or Discard (Esc). After
    Save the frame shows a keyframe diamond; propagate from it and check the run stops at the next keyframe. Erase the
    whole mask on another frame and Save: the frame becomes empty. The brush works with the engine unloaded.
