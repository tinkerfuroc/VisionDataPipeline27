# Augmentation and the prediction strip — design

Date: 2026-09-24. Status: approved in chat, spec written for the implementation plan.
Amends `2026-09-24-yolo-seg-training-design.md` (training): the Train page's preview becomes a strip, training gains
augmentation presets, and a new step derives an augmented copy of an export with objects pasted onto texture
backgrounds.

## Goal

1. Make validation predictions browsable: a row of about four images at a time, scrollable, filterable to the misses
   and the worst frames, each openable at full size.
2. Raise data augmentation well above Ultralytics' defaults, in particular by cutting each labelled object out by its
   mask and pasting it onto unrelated backgrounds, so the model stops relying on the few benches it was filmed on.

## Decisions taken with the user

- Preview: **strip + enlarge + filter chips** (All / Missed / Worst IoU).
- Backgrounds: **DTD textures** (Describable Textures Dataset, 5,640 images, one 625 MB archive), downloaded on request.
- Composites reach training as an **offline augmented export, derived from an existing export** (which must include
  the `masks` format). The source export is never modified.
- Online augmentation: **geometry, photometric and in-dataset copy-paste/mixup** turned up, offered as **presets on the
  Train page** (default / strong / custom).

## Facts verified before writing this spec (2026-09-24)

- `https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz` answers 200 after one redirect,
  Content-Length 625,239,812. Its images are `dtd/images/<category>/<name>.jpg` (47 categories).
- Ultralytics 8.4.161 accepts `train(augmentations=[...])`: a list of Albumentations transforms replacing its built-in
  low-probability set (`cfg/__init__.py` allows the custom key; `data/augment.py` passes it to `Albumentations`).
- Ultralytics' default hyper-parameters: `degrees 0, shear 0, perspective 0, flipud 0, fliplr 0.5, scale 0.5,
  translate 0.1, hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, mosaic 1.0, mixup 0, copy_paste 0 (mode flip)`.
- The `masks` export format is semantic: one 8-bit PNG per frame, `0` background, `class_id + 1` for the class. Two
  instances of one class in a frame are one region there. A derived export therefore cuts **one region per class per
  frame**; for this project (one object per clip) that is exactly the object.

## A. Prediction strip

### Server

`GET /api/train/runs/{id}/predictions?offset=&limit=&filter=all|missed|worst` (default `all`, limit ≤ 50):

- Each item gains `index` (0-based position in the name-sorted list, i.e. frame order), and `iou`: the IoU of the union
  of predicted polygons against the union of ground-truth polygons, rasterised at image size with the export's
  polygon convention (`polygons_to_mask`). Both empty → `null`; one empty → `0.0`.
- `missed` = frames whose ground truth is not empty and which have no prediction, in frame order.
- `worst` = frames with ground truth, ascending IoU (ties by index); frames with `iou == null` excluded.
- The response gains `counts: {all, missed, worst}` (sizes of each filtered list).
- IoUs are computed once per run and cached in `<run_dir>/predictions_iou.json` (`{name: iou}`), rebuilt when
  `predictions.json` is newer than the cache. Existing runs get them on first request.

### Web

The "Validation predictions" section becomes a gallery:

- Above: filter chips `All 252 · Missed 11 · Worst IoU`, the GT / prediction toggles (legend swatches as today).
- A horizontal strip of cards, four visible at the default width (`grid-auto-columns: calc(25% - gap)`), scroll-snap,
  horizontal overflow scrolls with trackpad/shift-wheel; ‹ › buttons move by four; `←/→` move by four when focus is not
  in a form field. Pages of 20 are fetched as the strip nears its end. Caption: `#<index+1> · IoU 0.94`, or
  `#<index+1> · missed` (text in ink, a small danger-coloured ✕ as the marker). A counter `13–16 of 252`.
- Clicking a card opens the large view (today's preview) in place of the strip, with ‹ › stepping one image through the
  current filter, `←/→` stepping, `Esc` or a "Back to strip" button returning to the strip at that card.
- Overlay (card and large): ground truth dashed ink outline, predictions filled series-slot-1 blue at 35 % with the
  score label (unchanged). Cards omit score labels (too small).
- Changing run or filter resets to the start. A run that is not `done` shows the existing placeholder.

## B1. Background pools

- Stored under `data/backgrounds/<pool>/` (images only, any sub-folders). One pool now: `dtd`.
- `vsp backgrounds fetch dtd` (CLI, foreground with a progress line) and `POST /api/backgrounds/dtd/fetch` (a
  `JobManager` job, kind `backgrounds`, progress `done/total` in bytes then files, cancellable) download the archive to
  `data/backgrounds/.dtd.tar.gz.part`, then extract only `dtd/images/**/*.jpg`, `*.jpeg` and `*.png` into
  `data/backgrounds/dtd/`, then delete the archive. A cancelled or failed fetch removes the partial archive and the
  partial pool directory. A pool that already has images is left alone unless `force`. This job and an augment job (B2)
  share one exclusive key (`augment-data`): only one of either may run at a time; the second attempt gets `409
  {job_id}` naming the one already running.
- `GET /api/backgrounds` → `[{name, images, fetching: job_id | null}]` for the known pools (`dtd`), `images` 0 when
  absent; `fetching` reports only a running *fetch* job, not a running augment job holding the same exclusive key.
- Extraction guards against unsafe tar members (absolute paths, `..`, backslashes, links): only regular files whose
  normalised path is under `dtd/images/` and ends in `.jpg`, `.jpeg` or `.png` are written.

## B2. Augmented copy of an export

### Request

`POST /api/exports/{id}/augment` body `{per_image: 1–10 (3), max_objects: 1–3 (2), seed: int (0), pool: "dtd"}` →
`202 {job_id}` (kind `augment`, exclusive key `augment-data` -- shared with a background fetch, B1). 400 when the
source is itself an augmented copy (`derived_from` is not null), has no `yolo_seg` or no `masks` format, has no
labelled train frame, names an unknown background pool, or the pool has no images; 404 unknown export.

### Output

A new export `<source_id>-aug<per_image>x-<4 hex>` under `data/exports/`:

- **val** = the source's val frames and labels, unchanged (images hard-linked; per-format label files copied verbatim from the source; `coco/annotations/instances_val.json` and `yolo_det/data.yaml` and `yolo_seg/data.yaml` copied verbatim).
- **train** = every real source train frame, plus `per_image × (real train frames)` composites named `syn_<seed>_<n:06d>`.
- The same formats as the source (`yolo_seg`, `yolo_det`, `coco`, `masks` as present), written by the existing exporters
  from an `ExportPlan` built from the source export instead of the store.
- `export.json` as a normal export plus `derived_from: <source_id>` and `augment: {per_image, max_objects, seed, pool,
  synthetic_images}`. `project`, `classes`, `clips` are copied from the source.
- Cancel or failure removes the new export directory (as exports do). The source export is never written.

### Composite recipe (pure function, seeded)

For composite `n` with `rng = np.random.default_rng([seed, n])`:

1. **Canvas**: the size of a random real train frame (portrait and landscape both occur). Background: a random image
   from the pool, a random crop with the canvas' aspect ratio covering 50–100 % of the image's short side, resized to the
   canvas (area interpolation).
2. **Objects**: `k = rng.integers(1, max_objects + 1)` cutouts, each from a random real train frame's **class region**
   of its `masks` image (same-class objects in one frame are cut as one region; a known limitation for multi-instance
   classes), cleaned with the export's `min_area_px`, cropped to its bounding box with a 4 px margin.
3. Per cutout: scale so the object's longer side is 15–70 % of the canvas' shorter side; rotate by U(−180°, 180°) with
   the mask rotated identically (nearest-neighbour for the mask, linear for the pixels, replicated border for pixels so
   no black edges bleed into the mask); horizontal flip with p 0.5.
4. **Placement**: random position such that at most 20 % of the object's pixels fall outside the canvas; up to 20 tries
   to avoid overlapping an already placed object by more than 10 % of the smaller one; a cutout that finds no place is
   dropped (at least one object always placed: the first uses no overlap test).
5. **Colour match**: per channel, move the object's mean toward the mean of the background under its box by a factor
   U(0, 0.5), and scale its contrast by U(0.8, 1.2); clip to 0–255.
6. **Blend**: alpha = the mask, Gaussian-blurred with σ 1.5 px, so edges are soft; the label mask is the hard mask.
7. **Whole-image finish**: with p 0.3 a Gaussian blur σ U(0.5, 1.5); with p 0.3 Gaussian noise σ U(2, 8); pixel
   values rounded and clipped to 0–255, saved as JPEG quality 90.
8. The label masks are cleaned with `min_area_px`; later objects occlude earlier ones (an occluded pixel belongs to the
   top object only). A cutout reduced below `min_area_px` by occlusion or cropping is dropped from the labels.

### Train page

Under the export picker, when the chosen export has the `masks` format: an **Augmented copy** section — "Composites per
image" (1–10, 3), "Objects per image" (1–3, 2), the pool line ("DTD textures · 5,640 images" or a **Download DTD
textures (625 MB)** button with progress and Cancel), and **Make augmented copy** with progress and Cancel. On success
the new export is selected. For an export with `derived_from`, the main area shows a **Composite samples** strip (the
gallery card component, ground truth only) of 12 random synthetic train images (`GET
/api/exports/{id}/samples?limit=12&seed=`), with a "Shuffle" button.

`GET /api/exports` gains `formats: [names]` and `derived_from: id | null` per export. `GET
/api/exports/{id}/images/{split}/{name}` serves train images too (split `train` | `val`).

## B3. Augmentation presets

`TrainParams` gains `augment: "default" | "strong" | "custom"` (default `"default"`) and `aug: AugParams | null`
(required when `custom`, ignored otherwise). `AugParams` (all floats, bounds):

| field | bounds | default preset | strong preset |
|---|---|---|---|
| degrees | 0–180 | 0 | 15 |
| shear | 0–45 | 0 | 5 |
| perspective | 0–0.001 | 0 | 0.0005 |
| flipud | 0–1 | 0 | 0.5 |
| fliplr | 0–1 | 0.5 | 0.5 |
| scale | 0–0.9 | 0.5 | 0.6 |
| translate | 0–0.9 | 0.1 | 0.2 |
| hsv_h | 0–0.1 | 0.015 | 0.02 |
| hsv_s | 0–1 | 0.7 | 0.7 |
| hsv_v | 0–1 | 0.4 | 0.5 |
| mosaic | 0–1 | 1.0 | 1.0 |
| mixup | 0–1 | 0 | 0.1 |
| copy_paste | 0–1 | 0 | 0.3 |
| photometric | bool | false | true |

`default` passes nothing (Ultralytics' own values). `strong`/`custom` pass the fields to `model.train(...)`
(`copy_paste_mode="flip"`), and `photometric: true` passes `augmentations=[RandomBrightnessContrast(0.25, 0.25, p=0.5),
MotionBlur(blur_limit=(3, 9), p=0.2), GaussianBlur(sigma_limit=(0.3, 1.5), p=0.1), GaussNoise(std_range=(0.02, 0.06),
p=0.2), ImageCompression(quality_range=(60, 95), p=0.3), CLAHE(p=0.1)]`. `albumentations` becomes a dependency.

The worker builds the arguments with a pure `train_overrides(params) -> dict` (tested); the Albumentations objects are
built in the worker only. The Train sidebar gets an **Augmentation** select (Default / Strong / Custom) and, for Custom,
the fields above prefilled from Strong. The runs table gains an **Aug** column (`default`, `strong`, `custom`).
`vsp train` gains `--augment default|strong` (custom is GUI/API only).

## Error handling

- Fetch: network failure → job error with the message; the partial files are removed. No disk-space pre-check beyond
  what the OS reports.
- Augment: a source export that is itself an augmented copy, missing its `masks` format, missing its `yolo_seg`
  format, having no labelled train frame, or when the pool is empty → 400 before the job starts; an unreadable source
  image during the job → job error, output removed.
- Training on a derived export needs nothing new: it is a normal yolo_seg export.

## Testing

- Composite: seeded determinism; the label mask equals the pasted hard mask (after occlusion); objects stay ≥ 80 %
  inside; the canvas has the chosen frame size; with `max_objects 1` exactly one object.
- Derive: on a tiny synthetic export — val images/labels byte-identical to the source's yolo_seg val; train count =
  real × (1 + per_image); every source format present and loadable (yolo labels parse, COCO loads with pycocotools);
  manifest fields; cancel removes the output; source unchanged (hash of its tree before/after).
- Fetch: a local HTTP server serving a small tarball with good members and hostile ones (`../x.jpg`, absolute path,
  symlink) — only the good images land; cancel removes the partial; an existing pool is skipped unless force.
- Presets: `train_overrides` for default / strong / custom; bounds validation.
- IoU: `mask_iou` cases (both empty, one empty, partial overlap); filters and counts; cache rebuilt when predictions
  change.
- Web (vitest): gallery paging, filter switching, enlarge and back, keyboard; augment panel states; preset select.
- After the build: a real run, yolo26s-seg, `strong`, on a ×3 augmented copy of `20260924-040124-8336a8`, compared with
  the existing yolo26s-seg run (mask mAP50-95 0.782).

## Out of scope

Other background pools (user photos, COCO), per-instance cutouts for multi-instance classes, Poisson blending or
shadow synthesis, augmenting the val split, hyper-parameter search.
