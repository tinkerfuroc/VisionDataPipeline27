# Using VisionDataPipeline27

The web GUI's workflow: import footage, label objects with SAM3, export a dataset, train YOLO segmentation models.
Server API details are in [api.md](api.md); the manual QA checklist is in [manual-qa.md](manual-qa.md).

## Demo data

With the server running, `uv run python scripts/seed_demo.py` creates a project "demo" with a synthetic 40-frame clip
(a disc moving across the frame), labels frame 0 with one click and propagates forward, then prints the URL to open.
`--no-label` imports only.

## Classes

The Clips page's Classes section lists the project's classes, each with a swatch and an "Add class" form. The
swatch is a colour picker: changing it saves the class's colour to the project (all of its clips) a moment after
you stop dragging, and new exports' mask images pick it up too — an export already on disk keeps the colours it
was made with. The same picker is on the Label page's right sidebar (its "Mask colours" section).
`PATCH /api/projects/{id}/classes/{class_id}` (body `{"color": "#rrggbb"}`) does the save; an unknown class or
project is a 404, a malformed colour a 422.

## Uploading footage

The Clips page uploads a video file or a `.zip` of images and imports it in one step. The browser sends each
file as the raw request body (not multipart), one request per file, to
`POST /api/projects/{p}/clips/upload?name=<file>&stride=<n>&fps=<n>` → `202 {job_id}`. `stride` applies to
both kinds; `fps` only to a zip, because a video's frame rate comes from its container. A name is accepted if
it ends in one of the video extensions in `vdp.api.clips.VIDEO_SUFFIXES` or in `.zip`; a zip may hold
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
is queued or running) aborts the running file and clears the rest. Each file is sent in 8 MB pieces, so a
dropped connection only pauses it: it continues by itself, and after a reload the page offers to continue when
you choose the same file again (unfinished uploads are kept for 24 hours; "Cancel upload" discards one). The
same uploads work from a phone: see the Clips page's Upload from a phone button.
Peak disk use during an import is the upload plus the JPEG frames and
thumbnails it writes.

## Label page: several objects

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
one lane per class that has an object in this clip (the active object's class tall), so a project with many classes
stays readable; a class with no object here gets a lane only while it is the class chosen for a new object (an empty
lane, its name with no count suffix). A class with more than one object gets one lane summarising all of them,
named "Class ×n". A name gutter sits on the left, and the playhead spans all lanes; past six lanes the lanes scroll
with the mouse wheel. Clicking a lane (or its name) selects that class: its lowest-id object if it has one, else it
starts a new object of it, same as the class select — clicking the class already active does nothing. Hovering a
frame lists every class's state on it. The Propagate panel runs **All objects** (every object with a mask on the
current frame; the others are named as skipped) or **Selected** only; lost objects are named one by one. Cleanup's
"Clear masks" acts on the **Active object** (default) or **All objects**. With a range marked, Cleanup also offers
**Delete frames** / **Restore frames** for the whole range (counting how many frames each would change); the mark
stays afterwards, so the other button undoes it at once. `K`, `G` and "has a mask here" follow the
active object's own lane, not its class's.

## Hiding confusing areas (regions)

Some footage has areas that only confuse SAM3: a second machine in the corner, a reflection, a label that looks like
the part. The **Regions** tool (in the tool rail; it needs no engine) lets you take them out. Two shapes are available
per frame range: a **crop** (keep only this rectangle) and **blocked boxes** (exclude these). SAM3 never sees an
excluded area, no label pixels are written there (a mask that reaches into one is trimmed), and exports are cropped
to the crop and greyed over the blocked boxes. A click on an excluded spot is ignored with a note. The outlines show
on the frame with every tool.

Regions apply to a **frame range**, because a camera or a part moves: press **Add region from here** on a frame no
range covers, **New range from here** to split the range at the current frame (the new range starts as a copy), or
**Set start to this frame** / **Set end to this frame** to change a range's bounds. Type the range's numeric Start and
End (press Enter or leave the field to apply), or use the buttons. Choose the **Blocked box** or **Crop** radio, then
drag on the frame to draw; drag a box to move it and its corners to resize it. To remove things: the × beside a blocked
box's top-right corner deletes it, **Clear** empties the range's crop and boxes, and **Remove range** deletes the
range. The panel also lists the range's crop and blocked boxes with **Delete** (and **Remove crop**) buttons, handy for
boxes whose × is hard to reach at the frame edge; hovering a row outlines that box on the frame. Edits are a draft ("Unsaved changes") until you press **Save regions**; **Discard** drops them. Leaving the
tool, going back to the clips list or closing the page asks first while a draft is open. A running propagation keeps
the regions it started with.

Saving regions does not rewrite existing masks: labels already on disk are cut at export, not changed. Re-propagate
or re-prompt a frame to get a mask trimmed to the regions.

## Keyboard

`Enter` accept · `Esc` clear · `Z` undo click · `←/→` step · `Shift+←/→` ±10 · `Space` play/pause · `K` / `Shift+K`
next / previous keyframe of the active object · `G` its next empty frame · `O` / `Shift+O` next / previous object · `N` new object · `Del` delete or restore frame · `[` / `]` mark the first / last frame to clear · `-` / `=` shrink / grow the brush (Brush tool) · `M` mask overlay · mouse wheel zoom ·
middle-drag pan · right click = negative point. `Enter`, `Esc` and `Z` act on the brush while it is selected (Save, Discard, Undo stroke) instead of accept / clear / undo click.

## Exporting

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

**Speed:** an export runs on `export.workers` processes, the same default as augmentation (all CPU cores but 4;
`1` runs in-process). Both the pass that reads the masks to plan the export and the pass that writes every format
run per frame on that pool; an export of fewer than 64 frames runs in-process. On a real 3-clip, 1,042-frame export
in all four formats it went from about 260 s to about 70 s on one process (`workers: 1`: the mask work is done on
each mask's bounding box, and every format is written in one pass over the frames) and to about 12-15 s with the worker
pool on a 20-core machine. Times depend on the machine; the output is byte-identical.

## Training

`uv sync` installs Ultralytics (YOLO26 and YOLO11 segmentation, and YOLO26 detection). Train from the Train page
(header → Train) or from the terminal:

    uv run vdp train --list                                  # exports that have a yolo_seg dataset (shows each export's folder id, name, project, image counts and classes)
    uv run vdp train --export <folder_id> --gpu 1            # yolo26s/m-seg and yolo11s/m-seg, one after another
    uv run vdp train --export <folder_id> --model yolo26s-seg --epochs 150 --imgsz 960 --batch 8
    uv run vdp train --export <folder_id> --model yolo26s     # a detection (bounding boxes) model

The `-seg` models learn masks; the plain YOLO26 models (`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`, labelled
"boxes" in the Train page's model list) learn bounding boxes only. A detection run trains on the export's `yolo_det`
dataset when it has one (one box per object), else on its `yolo_seg` one, whose polygons Ultralytics turns into boxes
(one box per contour, so an object split into pieces gives a box per piece); `run.json` records which as
`data_format` (and the task, `segment` or `detect`, as `task`). Detection runs report box metrics only: the Runs
table's mAP50 / mAP50-95 columns show box mAP for them (tagged "box", best value bolded among box models only), their
charts show box mAP and the box, cls and l1 losses, and their validation predictions are drawn as boxes. Continue and
Fine-tune always train the source run's model, so a box run continues as a box run and a mask run as a mask run.

Each run lives in `data/runs/<run_id>/`: `run.json` (request and status), `worker.log`, Ultralytics' own output under
`train/` (`weights/best.pt`, `last.pt`, `results.csv`, plots) and `predictions.json` (best.pt on every val image, conf
≥ 0.25). Pretrained checkpoints are cached in `data/train_weights/`. A run is a separate process
(`python -m vdp.train.worker`) with `CUDA_VISIBLE_DEVICES` set to its GPU, so a crash or out-of-memory error ends
that run only.

Before its first batch, Ultralytics scans a new export's labels (opening every image to verify it) and saves the
result as `labels/train.cache` and `val.cache` in the export's `yolo_seg/` folder; on a 30 000-image export that takes
about a minute, while a later run on the same export loads the cache in under a second. So when an export or
augmentation job finishes, the server does that scan in the background at low priority
(`python -m vdp.train.warm_cache <export_dir>`, one export at a time, output in the export's `warm_cache.log`) for its
`yolo_seg` folder and, when the export has one, its `yolo_det` folder (what detection runs read); a run started
before it is done simply scans as before. A cache that
cannot be loaded (say, cut short by a run stopped while saving it) is deleted and rebuilt, by the warm-up and by the
next run, instead of failing that run. Turn the warm-up off with `train.warm_label_cache: false` in the config.

Training never lowers the batch size on its own: Ultralytics would silently halve it (up to three times) when the
first epoch runs out of GPU memory, but here the run stops instead, with status `error` and the message "Out of GPU
memory on cuda:N at batch B (image size S)…". Start it again with a smaller batch or image size, on another GPU,
or with "Pause SAM3 while training" ticked if SAM3 shares that GPU.

"Estimate VRAM", under the Batch field, measures how much GPU memory the chosen model (the first picked one, or
pick it in "Estimate VRAM for") takes at the current image size on the chosen GPU, for batch 1, 2, 4, 8 and 12: a
short background process runs one real training step per batch size on random images, with as many objects per
image as the selected export's busiest training image can produce after mosaic (8 when no export is selected). It
never takes the last 1 GiB of free memory, so SAM3 keeps room to work. Each batch is shown as "about X GiB" and
marked fits, tight (within 10 % of the GPU's free memory now) or doesn't fit; a batch that did not fit in the memory
the measurement had shows "unknown — re-measure" once more is free. A "With SAM3 paused" column appears when SAM3 is
loaded on that GPU; click a row to use that batch. The numbers include a margin for what a whole run holds beyond
one step (calibrated against this machine's past runs: about 10 % plus 0.4 GiB). Answers are remembered per model,
image size, objects per image and GPU model in `data/train_weights/vram_estimates.json`, so asking again is
instant; "Re-measure" measures anew. The button is off while that GPU is training or exporting, and a run that
starts on the GPU stops a measurement first.

The server keeps one queue per GPU: one run at a time per GPU, GPUs in parallel. GPUs are ranked fastest first by
estimated compute throughput — SM count × cores per SM (from compute capability) × max SM clock (from NVML), not a
real benchmark but a much better speed proxy than compute capability or VRAM alone; VRAM only breaks a throughput
tie. The default GPU is the fastest one that is not already training — SAM3 sharing it does not disqualify it, and
the GUI's "shares memory with SAM3" warning still applies when it is chosen. The Train page's New-runs form has one
GPU choice for every model started together (listed best first, each labelled with its estimated TFLOPS and free
VRAM from NVML, and the default marked), plus a "Different GPU per model" checkbox that reveals a per-model override
seeded from that choice. `vdp train` without `--gpu` picks the same way and prints which GPU it chose and why;
`--gpu` overrides it, as before. Cancel stops a run after SIGTERM (SIGKILL after 30 s) and keeps `last.pt`. Resume
continues a cancelled, interrupted or errored run from its last finished epoch, on a GPU you choose (default the
fastest idle one): click "Resume" in the Runs table (or select the run) to open its "Stopped run" section in the
right sidebar, pick a GPU there (same best-first list and labels as New-runs, with the SAM3 and busy notes) and
click "Resume on cuda:N". The worker itself needs no GPU-specific change to move a resumed run: it always trains as
device 0 under a fresh `CUDA_VISIBLE_DEVICES` set to the newly chosen GPU, and Ultralytics' checkpoint-saved device
("0") is relative to that, not to any physical GPU. A run that was training when the server stopped is shown as
"interrupted" and can be resumed; queued runs are queued again.

"Mark complete" finishes a cancelled or interrupted run with its best epoch so far instead of training further: it
needs `best.pt` on disk already (a run cancelled or interrupted before its first improving epoch has none, and is
not eligible; neither is one that ended in `error`, since its `best.pt`, if any, is not trusted). Click "Mark
complete" in the Runs table (or select the run) to open the same "Stopped run" section, pick a GPU there and click
"Mark complete on cuda:N". It queues on that GPU like any run, so it shows in the GPU queues and can be cancelled;
the worker then skips training entirely, strips `best.pt`'s optimizer state (`last.pt` is untouched -- the same
finalising step Ultralytics itself does at the end of a normal run) and runs the validation-predictions gallery on
that GPU, before marking the run `done` (`completed_early: true` in `run.json`, and ONNX export becomes available)
with that best epoch's metrics. `POST /api/train/runs/{run_id}/complete` (optional `{"gpu": N}` body) does the same
from the API; a 409 means the run is not eligible.

When a run is headed for SAM3's GPU, the "shares memory with SAM3" warning (in New runs and in the Stopped run
section) has a "Pause SAM3 while training" checkbox, off by default, which also says roughly how much VRAM it frees.
Checked, the server unloads SAM3 before the run starts and reloads it by itself once that GPU has no training queued
or running; meanwhile prompting and propagation are unavailable, and the banner at the top offers "Reload now" (which
brings the memory pressure back while training still runs). With the box checked, starting is refused while a
propagation is running, or while SAM3 is still loading (untick the box, or wait for it to finish).

`vdp train` runs in the foreground and owns its runs; Ctrl-C,
SIGTERM and SIGHUP (a closed terminal, `kill`) all cancel them the same way. Do not start it on a GPU the server is
also training on. Do not set `CUDA_VISIBLE_DEVICES` for the server process itself — GPU indexes (from `vdp doctor`,
the Train page and `vdp train --gpu`) are passed to each worker as-is, so setting it on the server would remap or
hide devices under it. A run started by `vdp train` shows on the Train page's GPU queues as "running in another
process (vdp train)"; it can only be cancelled from the terminal it was started in.

Defaults: 100 epochs, early stop after 30 epochs without improvement, 960 px, batch 8, COCO-pretrained weights,
Ultralytics' default augmentation. Metrics come from `results.csv`; the comparison table ranks runs by the best mask
mAP50-95 epoch. ONNX export (`best.onnx`, static, square at the run's image size, 960 × 960 by default) runs on the run's GPU queue.

### Shorter epochs: images per epoch

On a large export one epoch takes a long time (28 000 images at 960 px are about 10 minutes on an RTX 3080 Ti), so
validation, the choice of `best.pt` and early stopping only happen that often. Two optional fields under the training
params make epochs shorter (empty = all images, as before; `vdp train --images-per-epoch N --val-images-per-epoch M`
from the terminal):

- **Images per epoch** (at least the batch size): each epoch trains on the next N images of the train set. The whole
  set is shuffled (with the run's seed) and handed out in consecutive slices, one per epoch, so every image is seen
  once before any is seen again; then it is shuffled anew. Which slice an epoch gets depends only on the seed, the
  epoch number and the sizes, so Resume and Continue pick up exactly where training was (Continue keeps the source's
  epoch counter, on its new export's images; Fine-tune starts again at epoch 1). Augmentation is unchanged (a mosaic
  still pulls its three extra images from the whole set). Ultralytics' warmup (3 epochs) and "close mosaic" (mosaic
  off for the last 10 epochs) are measured in epochs; they are scaled by train images / N so they still cover as many
  images as with whole epochs. Warmup is capped at 30 % of the run's epochs (but never below Ultralytics' own 3), so a
  short run does not warm up for most of its epochs; close mosaic never exceeds the run's epochs (with many short
  epochs, mosaic may be off for a large share of them, just as for the last 10 whole passes). Optimizer "auto" picks
  by the number of optimizer steps the sampled epochs actually make. Continuing a run onto an export no bigger than
  its images per epoch trains whole epochs again, with Ultralytics' unscaled close mosaic. Patience stays in (short) epochs, which is the point; the LR
  schedule follows the epochs as before.
- **Validation images per epoch**: each epoch validates on one fixed subset of M val images, chosen once with the
  run's seed, so the scores stay comparable from epoch to epoch. Clips keep their share of the val split (a clip is
  the `<clip_id>_` prefix of an image name), and every class present in the val split is in at least one chosen image.
  The subset is linked into the run folder (`val_subset/`, with its own label cache; the run's `data.yaml` points at
  it), so it never touches the export. The final evaluation of `best.pt` at the end of training (and "Mark complete")
  still runs on the whole val split (a run stopped during that evaluation runs it on Resume); those are the metrics in the Runs table and the run detail ("Final, best.pt
  (epoch N) on all … val images"), saved in the run folder as `final_val.json`. The per-epoch charts are labelled
  "(val subset of M)", and the validation-predictions gallery still covers every val image.

Under the fields, a live estimate reads e.g. "1 epoch = 7,000 images (≈ 2 min) · 100 epochs ≈ 25 full passes" and
"Validation: 400 of 1,643 images per epoch (≈ 5 s)"; Start stays off (with that note) while images per epoch is below
the batch size. The times come from the newest run of the same model and image
size on the same GPU: every run records its measured training and validation images per second in `speed.json`
(after each epoch); without such a run the times are left out. A run trained this way says so in its detail ("Each
epoch trains on the next 7,000 of 28,000 images (warmup 12 epochs, mosaic off for the last 40). …"). Training stays
on one GPU per run, as always.

## Train from a previous run on new data

Any run can be trained further on a *different* export, as long as that export has exactly the same classes (same
names, same order) as the run being started from. This always creates a new run — the source run is never modified
— in one of two modes:

- **Continue** — a true resume on the new export: from the source run's own `last.pt`, keeping its epoch counter,
  optimizer state, LR schedule and training hyper-parameters, and training the remaining epochs on the newly chosen
  export. Only a run that stopped before finishing (cancelled, interrupted or errored, with a `last.pt` on disk —
  the same rule as "Resume") is eligible; a finished run's `last.pt` has been stripped by Ultralytics and cannot be
  resumed.
- **Fine-tune** — a new training that starts from the source run's `best.pt` weights instead of the pretrained
  `<model>.pt`, with a fresh epoch schedule and whatever params you set (epochs, image size, batch, patience,
  workers, seed, augmentation…), exactly like starting a new run. Any run with a `best.pt` is eligible, finished ones
  included.

From the Train page: pick the source run from the New-runs form's "Start from" select (only continuable or
fine-tunable runs are listed, newest first) — or click "Train on new data" next to a stopped run, or (any run with
a `best.pt`) in the run detail header, which does the same thing and scrolls the New-runs form into view. Choose Continue or
Fine-tune (whichever the source run is not eligible for is shown disabled, with the reason); pick the export to
train on next — one whose classes do not match is disabled in the list, with " — different classes" appended, and
the Start button itself is disabled with a note if the current choice mismatches. For Continue, the params fields
are shown read-only (the source run's own settings are kept, not what is on screen); for Fine-tune they are editable
as normal. Pick a GPU and click "Continue on new data" / "Fine-tune on new data". The Runs table marks the new run
with a small note ("continued from …" / "fine-tuned from …"), and the run detail header names the epoch training
switched to the new data at.

From the API: `POST /api/train/runs/{run_id}/from-checkpoint` with `{"export_id": "<id>", "mode": "continue" |
"finetune", "gpu": N, "params": {...}}` (`params` is honoured only for `"finetune"`; a `"continue"` run always keeps
the source's own params) → `{"run_id": "<new id>"}` (201). 404 for an unknown run or export, 409 when the source run
is not eligible for the chosen mode or the export's classes do not match, 400 for an unknown GPU.

From the CLI: `uv run vdp train --export <new_export_id> --from-run <run_id> --mode continue|finetune [--gpu N]
[params…]` trains one run (the source's own model) in the foreground exactly like `vdp train` does for a new run;
`--model` cannot be combined with `--from-run`. As with the API, `[params…]` (`--epochs`, `--imgsz`, …) only affects
`--mode finetune` — a `continue` run always keeps the source's own params.

**Manual QA:** train a run to completion, cancel a second run partway through (so it has a `last.pt`), then
Continue the cancelled run and Fine-tune the finished one onto a different export with the same classes; confirm
both queue as new runs, the source runs are untouched on the Train page, and Continue's metrics chart carries over
the source run's earlier epochs.

## Augmentation

**Presets:** `uv run vdp train --export <id> --augment strong` (or `default`, on the Train page select Default / Strong / Custom). The **Strong** preset raises geometric, photometric and copy-paste augmentation above Ultralytics' defaults: degrees 15, shear 5, perspective 0.0005, flip up 0.5, flip horizontal 0.5, scale 0.6, translate 0.2, HSV hue 0.02, saturation 0.7, value 0.5, mosaic 1.0, mixup 0.1, copy_paste 0.3, plus RandomBrightnessContrast, MotionBlur, GaussianBlur, GaussNoise, ImageCompression and CLAHE, and lighting simulation (below). The Custom preset on the GUI allows per-field adjustment, with checkboxes for the blur/noise/JPEG/contrast set and for lighting.

`albumentations` is a dependency, which also lets Ultralytics enable its own built-in low-probability Blur, MedianBlur, ToGray and CLAHE transforms, independent of the preset above — so a **Default** run is not pixel-identical to a stock Ultralytics run.

**Lighting:** low light and mild over-exposure, for footage shot in different light from the training clips. Exposure is changed in linear light, as a camera would: low light is 1 to 3.3 stops under, with sensor noise (grain that grows with brightness, a noise floor, some colour speckle) and sometimes a warm tungsten cast; over-exposure is 0.5 to 1.1 stops over, with blown highlights and slightly lifted shadows. Only pixels change, so labels stay as they are. At training time (the Strong preset, or the Custom checkbox "Low light, over-exposure") it relights 30 % of training images on the fly, 60 % of those darkened and 40 % over-exposed; validation images are never relit. Strong runs started before lighting existed did not have it, so a new Strong run is not directly comparable to an older one. To bake it into a dataset instead, see "Lighting variants" under augmented copies below.

**Background pools:** `uv run vdp backgrounds fetch dtd` (CLI, ~625 MB) or `POST /api/backgrounds/dtd/fetch` (web job) downloads the Describable Textures Dataset (5,640 images, one category per folder) to `data/backgrounds/dtd/`. The `--force` flag redownloads even if images are present.

**Augmented copies:** From the Train page, when an export has the `masks` format, an "Augmented copy" panel creates a new export with real train frames plus `per_image × (frame count)` synthetic composites named `syn_*`, seeded by a "Seed" field (0 by default, so two copies of the same export can differ). Each composite pastes `1–3` (user configurable) cutouts cut from real frames (one cutout per object, rasterised from the source's COCO train annotations when the source has the COCO format; without it, one cutout per class region of the mask images, so touching objects of one class are cut together — the panel says so) onto random background textures, scaled, rotated, positioned and colour-matched. With COCO, shared pixels are resolved per class (the smaller object wins within a class; across classes the mask image decides) and holes come from the mask image, except a holed object lying on a larger object of the same class (the mask image can't tell those two apart, so the hole still shows the host's pixels — a known remaining limit). `export.json`'s `augment.cutouts` records which (`per_object` / `per_class_region`); the same seed on an export made before this change may give different composites. Val frames are unchanged (images hard-linked, labels copied verbatim per format). The source export is never modified and cannot itself be augmented if it is already derived. The copy carries over whichever of `yolo_seg`, `yolo_det`, `coco` and `masks` the source export has — not always all four. The copy's `export.json` names its source in a `derived_from` field (`null` for an ordinary export).

**Lighting variants:** the Augmented copy panel's "Lighting variants per image" (0–3) adds that many relit copies of every real train frame (`lit_<seed>_<frame>_<k>`), low light or mildly over-exposed in the same 60/40 split as at training time and reproducible per seed. Each copy keeps its source frame's labels in every format (in COCO, its image entry and annotations are duplicated). Composites per image may be 0 when lighting variants are above 0: such a lighting-only copy needs no background textures. `export.json` records `augment.lighting_variants` and `augment.lighting_images`, the copy is named "`<source> +K lighting`" (after "`×N augmented`" when it also has composites), and the sample strip shows the relit frames alongside any composites. `overlap_px` in a copy's `masks` entry does not count the lighting copies.

**Limits:** two optional fields in the Augmented copy panel bound the size or the duration of a copy. "At most training images" caps the whole train split (real + composite + lighting frames); "Stop after (min)" stops the run after that many whole minutes (1 to 1440 in the panel; 10 seconds to 24 hours via the API). Both fields start empty again when another export is selected. Whichever comes first wins. Real frames are always kept and the val split is never touched, so a cap at or below the export's own train frame count is refused (400, "this export already has N training frames; set a limit above that, or leave it empty"). Under a cap the room is shared between composites and lighting copies in proportion to what was asked for. Cutting the cutouts counts toward the time limit. When the time limit hits, queued work is dropped and work already running finishes, so a run can overshoot it slightly. A time-limited run depends on how fast the machine is, so it is not exactly repeatable even with the same seed; an image-capped one is. The panel reports what happened ("Stopped at the 10-minute limit: ..." / "Capped at N training images: ..."), also whenever the copy itself is selected later (the Train page selects it as soon as it is made), `export.json` records it in `augment.limits`, and the copy's id and name carry the limit (`cap<N>` / `t<seconds>`, " ≤N images" / " ≤M min"). With neither field set the output is unchanged.

**Speed:** augmentation runs on a process pool using all CPU cores but 4, so the server and labeling stay responsive; the workers run at lower priority with one OpenCV thread each, so a training run keeps its CPU share. Set `augment.workers` (and `export.workers` for the dataset export) in the config to change that (`1` runs in-process, with no worker processes). On a real 798-frame export (3 composites and 2 lighting copies per frame) it took 20 s instead of 170 s, with byte-identical output.

**Prediction strip:** On the Train page, when a run finishes, a "Validation predictions" gallery shows four images at a time, scrollable. Filter chips select All / Missed (ground truth present but no prediction) / Worst IoU. Click a card to enlarge it. Keyboard `←/→` pages the strip by four images at a time; once a card is enlarged, the same keys instead step one image at a time through the current filter, and `Esc` (or "Back to strip") returns to the strip positioned on that card. The overlay shows ground truth as a dashed outline and predictions filled blue with confidence scores. IoU values per image are computed with the export's polygon convention and cached in `<run_dir>/predictions_iou.json`, rebuilt when `predictions.json` changes.
