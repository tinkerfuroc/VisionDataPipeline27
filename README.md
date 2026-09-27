# vision-sampler-pipeline

SAM3-driven video labeling for any number of user-chosen objects per clip.
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

`POST /api/projects/{id}/export` starts an export job; body `{clips, formats, options, name}` → `202
{job_id}`. `options.classes` is a list of project class ids to export (null or absent = every class); an
empty list, a repeated id or an unknown class id is a 400. It returns 400 for an invalid request (unknown
format, no clips, a bad option) and 409 (with the running `job_id`) while a propagation job runs, since
propagation rewrites the very masks an export reads.

### Reaching the GUI from another machine

`vsp serve` binds `127.0.0.1` by default. Forward the port instead of binding the server to the network:

    ssh -N -L 28000:127.0.0.1:28000 user@gpu-host

then open `http://127.0.0.1:28000/` locally; uploads travel through the tunnel. Binding a public interface
exposes an unauthenticated labeling server, which is why the CLI warns about it. `-o Compression=no` is
usually faster for already-compressed footage, and `-o ServerAliveInterval=30` keeps a NAT'd tunnel alive.

### Classes

The Clips page's Classes section lists the project's classes, each with a swatch and an "Add class" form. The
swatch is a colour picker: changing it saves the class's colour to the project (all of its clips) a moment after
you stop dragging, and new exports' mask images pick it up too — an export already on disk keeps the colours it
was made with. The same picker is on the Label page's right sidebar (see "Mask colours" below).
`PATCH /api/projects/{id}/classes/{class_id}` (body `{"color": "#rrggbb"}`) does the save; an unknown class or
project is a 404, a malformed colour a 422.

### Uploading footage

The Clips page uploads a video file or a `.zip` of images and imports it in one step. The browser sends each
file as the raw request body (not multipart), one request per file, to
`POST /api/projects/{p}/clips/upload?name=<file>&stride=<n>&fps=<n>` → `202 {job_id}`. `stride` applies to
both kinds; `fps` only to a zip, because a video's frame rate comes from its container. A name is accepted if
it ends in one of the video extensions in `vsp.api.clips.VIDEO_SUFFIXES` or in `.zip`; a zip may hold
`.jpg/.jpeg/.png` entries in nested folders (natural order; `__MACOSX` and dotfiles are skipped). The upload
is streamed once into `data/uploads/` and deleted when the import job ends, and a server killed mid-upload
sweeps that directory at its next start. A body over `ingest.max_upload_bytes` (8 GiB by default) is refused
with 413; the GUI refuses a file before its transfer begins, using the cap from `/api/health`.

The file picker and the drop zone both take several videos and/or zips at once, and "Keep every Nth frame" and
"Frames per second (zip only)" sit above the picker because uploading starts as soon as files are chosen or
dropped — there is no separate upload button. Each file is validated and snapshots those two settings at the
moment it is chosen, so changing them afterwards only affects files chosen later; an invalid file (wrong type,
a bad setting, or over the size cap) is skipped with a note naming it and why, while the rest of the same
selection still queue. Queued files upload one at a time, in the order chosen, through a small queue shown in
the sidebar: the running file with its progress, the waiting files by name (each removable with its own ×
button), and any failed file with its error and a Dismiss button. A failure never stops the queue — it moves on
to the next file. "Cancel upload" cancels the running file and moves on; "Cancel all" (once more than one file
is queued or running) aborts the running file and clears the rest. Each transfer is a single request, so it
needs the page to stay open: reloading or closing the tab cancels it and leaves nothing behind. Resumable
uploads are not implemented yet. Peak disk use during an import is the upload plus the JPEG frames and
thumbnails it writes.

### Several objects per clip (server API)

A clip can hold more than one labelled object (`clip.objects`, each with its own `id`, `class_id`, keyframes and
`propagated_ranges`; ids are never reused). The web Label page edits them one active object at a time (see "Label
page: several objects" below).

`/prompt`, `/accept` and `/mask` require an `object_id` key (a body without it is a 422): `null` means a new object
(`/accept` and `/mask` then need `class_id`; `/prompt` derives no click box from an existing mask), an id targets
that object.

- `POST /api/clips/{id}/prompt` and `POST /api/clips/{id}/accept` take the required `object_id`. `/accept` also
  takes `candidates` (a list of candidate indices from a text prompt that found several detections) as an
  alternative to the single `candidate` index; `candidates` requires `object_id` to be `null`, since each accepted
  candidate becomes its own new object (with `class_id`), and their ids come back in `object_ids`. An existing
  `object_id` cannot be combined with `class_id` (relabel with the `PATCH` route below instead).
- `POST /api/clips/{id}/mask` (the brush) takes the same required `object_id` and optional `class_id` and returns
  the object id it wrote to as `object_id`.
- `POST /api/clips/{id}/propagate` and `POST /api/clips/{id}/clear-range` take an `objects` field: `"all"`
  (default) or a non-empty list of object ids. Propagate resolves and validates the objects up front, so an
  unknown id is a 404 and a listed object with no mask on `from_frame` is a 400, both before the job starts, not
  a failed job.
  The job result keeps `processed`/`written`/`cleared`/`lost_at`/`cancelled`/`oom_retries` and adds
  `lost_by_direction` (per direction, object id → the frame it was first empty on), `lost` (merged over both
  directions, object id → frame), `skipped` (object ids the job itself left out when it re-read the clip —
  deleted, or with no seed mask on `from_frame`) and `objects` (the ids that actually ran). Each `"frame"`
  progress event's `objects` list carries `{"id", "has_mask"}` per object that covered that frame. There is no
  clip-level `has_mask`: a frame's overall state follows from the objects' lanes.
- `PATCH /api/clips/{id}/objects/{object_id}` (body `{"class_id": ...}`) relabels an object;
  `DELETE /api/clips/{id}/objects/{object_id}` removes it and its mask files (409 while any export or GPU job
  is running, server-wide, as for deleting a clip).
- `GET /api/clips/{id}/objects/{object_id}/masks/{idx}` serves one object's mask PNG (404 for an unknown object).
- Clip detail responses (`GET /api/clips/{id}` and every route that returns one) add `object_states`: a map from
  each object's id (as a string key) to its own per-frame state list (`empty`/`masked`/`keyframe`/`deleted`),
  alongside the `states` field, which keeps reporting the whole clip (any object).

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

### Label page: several objects

A clip can hold any number of objects, each with its own class (two cups and a bowl are three objects). The
**Objects** panel in the right sidebar lists them as `#id class` with the number of frames each has a mask on; the
highlighted row is the **active object**, which every prompt, accept, brush save, "Selected" propagation and
"Active object" clear acts on. Click a row (or press `O` / `Shift+O`) to change it. The eye hides an object from the
frame (display only, for this page session). Picking a class from the select **always starts a new object** of it
("New object (class)"), whether or not one is already active — it never relabels, even if that class already has an
object. To relabel the active object instead, use the separate "Change class of #id" row: choose a class there and
press **Change** (disabled until you pick a class other than its own); this applies at once, also during a
propagation. **+ New object** (`N`) starts the same "New object (class)" row: the object is created by its first
accepted or painted mask. A text prompt that finds several candidates on a New object row offers **Accept all as new
objects**, one object per candidate. **Delete object** asks first, naming how many frames lose their mask; it is
refused while a propagation or an export runs.

The frame shows every visible object in its class colour; only the active object gets the outline. The timeline has
one lane per project class (the active object's class tall), including a class with no object in this clip (an
empty lane, its name with no count suffix); a class with more than one object gets one lane summarising all of them,
named "Class ×n". A name gutter sits on the left, and the playhead spans all lanes; past six lanes the lanes scroll
with the mouse wheel. Clicking a lane (or its name) selects that class: its lowest-id object if it has one, else it
starts a new object of it, same as the class select — clicking the class already active does nothing. Hovering a
frame lists every class's state on it. The Propagate panel runs **All objects** (every object with a mask on the
current frame; the others are named as skipped) or **Selected** only; lost objects are named one by one. Cleanup's
"Clear masks" acts on the **Active object** (default) or **All objects**. `K`, `G` and "has a mask here" follow the
active object's own lane, not its class's.

### Exporting

An export lands under `data/exports/<export_id>/`: an `images/` directory holding one copy of every
exported frame, plus one directory per requested format — `yolo_seg/`, `yolo_det/`, `coco/` and/or
`masks/` — and an `export.json` manifest recording the resolved options, the clips exported (with their
per-split frame counts), the class map (per exported class: `id`/`yolo_id` = export id, `coco_id`/`mask_value`
= export id + 1, and `project_class_id`, the project's own class id) and the train/val image counts. The
train/val split is always either per clip (whole clips held out for validation) or contiguous (the tail of
every clip held out), never random per frame: neighbouring video frames are nearly identical, so a random
per-frame split would leak validation frames into training. A cancelled or failed export removes its
directory rather than leaving a partial dataset behind.

The Export page's right sidebar has a Classes section: one checkbox per class with its swatch and its object
count over the included clips, all ticked by default. Deleted frames and the "every Nth frame" stride apply
first; objects of unticked classes are left out before the empty-frame rule, so a frame left without objects
is kept or dropped by "Include frames without a mask", and the split is taken after that. The exported classes are renumbered 0..k-1 in project class order (YOLO ids;
COCO category ids and mask values are one higher); `options.class_ids` in `export.json` records the chosen
project class ids (null = every class). Where objects overlap, the mask image holds one value per pixel:
objects are painted largest first, so a smaller object lying on a larger one stays visible. The `masks`
entry of `export.json`'s `formats` reports `overlap_px`, the pixels claimed by more than one object (the
other formats keep every object whole).

When you create an export, the Export page's right sidebar has a "Dataset name" field (optional, up to 80
characters). If you provide a name, the export folder is named using that name, lowercased, with any
characters outside `a-z 0-9 . _ -` converted to `-`, and truncated to 48 characters at most. If that folder
name already exists, the system appends `-2`, `-3`, etc. to avoid collision. If you leave the field empty or
the name contains no usable characters, the folder is named with a timestamp ID instead. The name is stored in
`export.json` under the `name` field and is displayed on the Train page (in the export list and Runs table)
alongside the folder ID. You can rename an export on the Train page using the "Rename export" field below
the export list, or via the API with `PATCH /api/exports/{id}` passing `{"name": ...}`. The folder ID never
changes when you rename, so existing training runs continue to work. Augmented copies are named
"`<source name or id> ×N augmented`".

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

    uv run vsp train --list                                  # exports that have a yolo_seg dataset (shows each export's folder id, name, project, image counts and classes)
    uv run vsp train --export <folder_id> --gpu 1            # yolo26s/m-seg and yolo11s/m-seg, one after another
    uv run vsp train --export <folder_id> --model yolo26s-seg --epochs 150 --imgsz 960 --batch 8

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

**Augmented copies:** From the Train page, when an export has the `masks` format, an "Augmented copy" panel creates a new export with real train frames plus `per_image × (frame count)` synthetic composites named `syn_*`, seeded by a "Seed" field (0 by default, so two copies of the same export can differ). Each composite pastes `1–3` (user configurable) cutouts cut from real frames (one cutout per object, rasterised from the source's COCO train annotations when the source has the COCO format; without it, one cutout per class region of the mask images, so touching objects of one class are cut together — the panel says so) onto random background textures, scaled, rotated, positioned and colour-matched. With COCO, shared pixels are resolved per class (the smaller object wins within a class; across classes the mask image decides) and holes come from the mask image, except a holed object lying on a larger object of the same class (the mask image can't tell those two apart, so the hole still shows the host's pixels — a known remaining limit). `export.json`'s `augment.cutouts` records which (`per_object` / `per_class_region`); the same seed on an export made before this change may give different composites. Val frames are unchanged (images hard-linked, labels copied verbatim per format). The source export is never modified and cannot itself be augmented if it is already derived. The copy carries over whichever of `yolo_seg`, `yolo_det`, `coco` and `masks` the source export has — not always all four. The copy's `export.json` names its source in a `derived_from` field (`null` for an ordinary export).

**Prediction strip:** On the Train page, when a run finishes, a "Validation predictions" gallery shows four images at a time, scrollable. Filter chips select All / Missed (ground truth present but no prediction) / Worst IoU. Click a card to enlarge it. Keyboard `←/→` pages the strip by four images at a time; once a card is enlarged, the same keys instead step one image at a time through the current filter, and `Esc` (or "Back to strip") returns to the strip positioned on that card. The overlay shows ground truth as a dashed outline and predictions filled blue with confidence scores. IoU values per image are computed with the export's polygon convention and cached in `<run_dir>/predictions_iou.json`, rebuilt when `predictions.json` changes.

### Keyboard

`Enter` accept · `Esc` clear · `Z` undo click · `←/→` step · `Shift+←/→` ±10 · `Space` play/pause · `K` / `Shift+K`
next / previous keyframe of the active object · `G` its next empty frame · `O` / `Shift+O` next / previous object · `N` new object · `Del` delete or restore frame · `[` / `]` mark the first / last frame to clear · `-` / `=` shrink / grow the brush (Brush tool) · `M` mask overlay · mouse wheel zoom ·
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
   the mark disappears, and a new prompt lands on the active object with its class.
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
14. Upload footage from the Clips page over the SSH forward. Set stride/fps, then choose (or drop) several files
    at once, mixing a video, a zip and an invalid file (e.g. a `.txt`): uploading must start immediately with no
    button click, the invalid file must be skipped with a note naming it, and the rest must upload one at a time
    in the order chosen, each finished one becoming a clip whose name, frame count and fps are right. Watch the
    percentage move for the running file and the waiting files listed by name; remove a waiting file with its ×
    button and confirm it never uploads. Let one file fail (e.g. by editing `ingest.max_upload_bytes` down) and
    confirm it shows its error with a Dismiss button while the rest still upload. Press "Cancel upload" mid-transfer:
    it must stop without an error message and the queue must move on to the next file. With several files still
    queued, press "Cancel all": the running transfer and every waiting file must clear. Reload the page during an
    upload and confirm the transfer stops, no partial clip appears, and `data/uploads/` is empty.
15. Upload a file larger than `ingest.max_upload_bytes`: the GUI must refuse it before the transfer starts, and
    `curl -X POST --data-binary @big.mp4 .../clips/upload?name=big.mp4` must answer 413.
16. Brush: pick the Brush tool on a masked frame, paint a missing corner (left-drag) and erase a spill (right-drag);
    the transport, the other tools, Propagate and Cleanup are disabled until you Save (Enter) or Discard (Esc). After
    Save the frame shows a keyframe diamond; propagate from it and check the run stops at the next keyframe. Erase the
    whole mask on another frame and Save: the frame becomes empty. The brush works with the engine unloaded.
17. Mask colours (right sidebar): pick a new Class colour for the active object's class — the fill on this frame,
    the timeline lane and text-prompt candidates all repaint; reopen the clip (or the project's other clips) and
    the colour stuck. Then, under Outline, untick "Outline follows class colour" and pick a contrasting colour, drag
    Thickness from 0 (no outline) to 12; the stored mask, the frames arriving during a propagation and text-prompt
    candidates all follow it, with no white halo outside the mask. Reload the page: the outline setting is kept
    (it is this browser only). Reset restores the outline to the class colour at 5 px, without touching the class
    colour itself.
18. Named datasets: export twice with the name "Cups v2"; the folders are `cups-v2` and `cups-v2-2`; rename one on the
    Train page; the Runs table shows the new name.
19. (server) With two objects on a clip, `POST /propagate` with `objects: "all"` writes both objects' masks in one
    run; with `[2]` only object 2's; an object that disappears is reported under `lost` while the other continues.
20. Class selection: on a project with three classes, untick one on the Export page (the counts follow the
    included clips) and export all four formats; `yolo_seg/data.yaml` lists the remaining classes as 0 and 1,
    COCO categories are 1 and 2, the mask PNG values are 1 and 2, and `export.json` maps them back with
    `project_class_id`. Untick every class: Start says "Choose at least one class." Put a small object on a
    large one: the mask PNG shows the small one, and `overlap_px` is non-zero. Make an augmented copy of an
    export with COCO: `augment.cutouts` is `per_object`; of one without COCO: the panel notes the fallback and
    it is `per_class_region`.
21. Several objects: on one clip label a cup (#1), press `N`, choose another class and label it (#2). The Objects
    panel lists both with frame counts; the timeline shows one lane per project class (a third, unused
    class shows an empty lane too), the active object's class tall; clicking a lane or its name selects that class's
    object, and `O` / `Shift+O` cycle. Hide #1 with its eye: it leaves the frame and its lane dims once every object
    of its class is hidden. Only the active object carries the outline; brushing #2 keeps #1 visible underneath. With
    #1 active, pick #2's class from the Class select: it starts a new object of that class (a third object) rather
    than relabelling #1 — use "Change class of #1" and **Change** instead to actually relabel it.
22. Propagate "All objects" from a frame where both have masks: both classes' lanes fill in one run. Then
    "Selected: #2" from another of its keyframes: only #2's class lane changes. Let one object leave the frame: the
    panel names it as lost while the other continues; start from a frame where only #1 has a mask with "All
    objects": #2 is named as skipped.
23. Shift-drag a stretch and clear it with "Active object", then again with "All objects"; check the note counts the
    right masks each time. Relabel #2 during a propagation with "Change class of #2" and **Change** (allowed); try to
    delete #1 during a propagation and during an export (refused), then delete it (the dialog names its frame count)
    and check #2 becomes active.
24. Text-prompt "cup" on a New object row with several cups in view: "Accept all as new objects" creates one object
    per candidate, all in the New object row's class — the timeline keeps showing a single lane for it, labelled
    "cup ×n". Add classes until the project has seven, labelling one object in each: the timeline stops growing at
    six lanes and scrolls with the wheel, and `O` onto an off-screen object's class scrolls its lane into view.
