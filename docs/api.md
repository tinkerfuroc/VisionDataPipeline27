# Server API notes

Interactive OpenAPI docs are served at `/docs` while `vdp serve` runs. This page covers the behaviour the schema
does not show: status codes, job semantics and the multi-object model.

## Health, jobs and conflicts

`/api/health` shows `engine.loaded` and, if loading failed, the reason. It also reports `upload.max_bytes`, the
size cap the GUI checks before it starts an upload. Interactive API docs: `/docs`.
Importing, browsing and frame cleanup work without a GPU; prompting and propagation return 503 until the
engine is loaded, and 409 (with the running `job_id`) while a propagation job holds the GPU. A propagate
request also returns 409 with `job_id: "interactive"` if a prompt/accept/clear-range call is in flight;
retry. Deleting a clip also returns 409 (with the running `job_id`) while a GPU job runs, so a delete can
never race a propagation's writes. Cancelling an import job removes the partial clip it had started.

## Pausing SAM3 for training

`POST /api/train/runs`, `POST /api/train/runs/{id}/from-checkpoint`, `.../resume` and `.../complete` accept an
optional `pause_sam3: true` (default `false`). When at least one of the request's runs goes to SAM3's GPU and the
engine is loaded, the server unloads SAM3 after validating the request and before queueing anything; otherwise the
flag is ignored (no error). While a propagation job holds the GPU it returns 409 with that `job_id` instead, the
engine stays loaded and nothing is queued. While SAM3 is loading (at startup, or reloading after a pause) it
returns 409 "SAM3 is still loading; wait for it to finish, or untick Pause SAM3" and queues nothing, unless none of
the runs goes to SAM3's GPU. While paused, `/api/health` shows `engine.paused: true` (with
`loaded: false`, `error: null` and `device` still set), and prompting and propagation return 503 with the reason
"SAM3 is paused for training". `GET /api/train/status` keeps reporting `sam3_gpu` (the GPU SAM3 reloads onto) and
adds `sam3_vram_bytes`, what the model took when it loaded (`null` when unknown), so a client can say roughly how
much pausing frees.

SAM3 reloads by itself, in the background, once nothing is queued or running on its GPU any more (done, error or
cancelled). `POST /api/engine/load` → `202 {started}` reloads it now and ends the pause; it is a no-op
(`started: false`) when the engine is already loaded or loading. The pause is not persisted: after a restart the
engine loads at startup as usual. A load that fails (e.g. out of memory while training holds the GPU) hands its
partly allocated memory back and shows in `engine.error`; `POST /api/engine/load` retries it.

`POST /api/projects/{id}/export` starts an export job; body `{clips, formats, options, name}` → `202
{job_id}`. `options.classes` is a list of project class ids to export (null or absent = every class); an
empty list, a repeated id or an unknown class id is a 400. It returns 400 for an invalid request (unknown
format, no clips, a bad option) and 409 (with the running `job_id`) while a propagation job runs, since
propagation rewrites the very masks an export reads.

## VRAM estimate

`POST /api/train/vram-estimate` body `{model, imgsz, gpu, export_id, remeasure}` (`export_id` optional, `remeasure`
optional, default `false`) → `202 {job_id}`, a job of kind `vram_estimate` whose result is `{model, imgsz, gpu,
gpu_name, total_bytes, free_bytes, estimates: [{batch, bytes, over_bytes}], measured_at, cached, objects_per_image,
objects_from_export}` for batch 1, 2, 4, 8 and 12. `bytes` is the estimated VRAM a run takes at that batch
(calibration margin included). The probe may only use the GPU's free memory at its start minus 1 GiB (left for
SAM3 and other processes); `bytes: null` means that batch needed more than that (and so did every larger one), and
`over_bytes` is that limit with the margin: the batch needs more than `over_bytes`, which says nothing about whether
it fits in more memory than that. `free_bytes` is the GPU's free memory read when the answer is given (NVML; `null`
when unknown). While it measures, the job emits `{type: "progress", done, total, batch}` per batch size.

The probe's synthetic labels carry `objects_per_image` objects per image: with `export_id`, the most objects in any
one of that export's train images times 3 (mosaic tiles four images into one), rounded up to a power of two, at least
8 and at most 300 (`objects_from_export: true`); without it, 8.

Answers are cached per model, image size, objects per image and GPU name in
`data/train_weights/vram_estimates.json` (an entry from another probe version, or a malformed one, is ignored). The
endpoint is always a job: a cached answer's job finishes at once (`cached: true`) without measuring, unless
`remeasure` is true, or the answer has batches that did not fit and the GPU now has clearly more free memory than the
probe had (then it is measured again). A measurement runs `python -m vdp.train.vram_probe` at low priority on that
GPU (output in `data/train_weights/vram_probe.log`) and takes about half a minute plus any checkpoint download; only
one runs at a time (409 with its `job_id` otherwise). Cancelling the job, or stopping the server, kills the probe.

It returns 400 for an unknown model or GPU, 404 for an unknown export, 422 for an image size outside 320-1920 or not a
multiple of 32, 503 when Ultralytics is not installed (as the other train endpoints), and 409 while that GPU has a
training run or an ONNX export queued or running (the measurement would be wrong and could make the run run out of
memory) or while SAM3 is loading onto it. A probe never overlaps a run: before the scheduler starts anything on a GPU
it stops a probe there and waits for its process to exit, and a probe also stops when a `vdp train` process starts
training on its GPU or SAM3 starts loading onto it; the job then ends with an error and nothing is cached.
`GET /api/train/status` reports `sam3_loaded` (false while SAM3 is paused or loading), so a client knows whether
pausing it would free `sam3_vram_bytes` now.

## Augmented copies

`POST /api/exports/{id}/augment` starts an augment job; body `{per_image, max_objects, seed, pool, lighting}` plus the
optional limits `max_train_images` (integer 1 to 10 000 000, a cap on the whole train split: real + composite + lighting) and
`time_limit_s` (10 to 86 400 seconds, measured from the job start) → `202 {job_id}`. Leaving a limit out (or `null`)
means no limit. A `time_limit_s` outside 10 to 86 400 or a `max_train_images` outside 1 to 10 000 000 is a 422 (schema bounds). A
`max_train_images` at or below the export's train frame count is a 400: "this export already has N training frames; set a limit above that, or leave it empty". When
both limits are set, whichever is reached first ends the run. At the time limit queued work is dropped and running
work finishes; cancelling still ends the job as cancelled.

The finished export's `export.json` has `augment.limits`, and the job result carries the same object as `limits`:

    {"max_train_images": 2000 | null, "time_limit_s": 600 | null,
     "requested": {"composites": 2394, "lighting": 1596},
     "planned":   {"composites": 721,  "lighting": 481},     # after the image cap
     "produced":  {"composites": 721,  "lighting": 481},     # after the time limit
     "stopped_by": "images" | "time" | null}

`stopped_by` is `"time"` when the deadline cut the run short, otherwise `"images"` when the cap reduced the plan,
otherwise `null`. `GET /api/exports` gives each export's block as `augment_limits` (`null` for an ordinary export),
so a client can say how a derived copy was cut long after its job is gone.

## Several objects per clip

A clip can hold more than one labelled object (`clip.objects`, each with its own `id`, `class_id`, keyframes and
`propagated_ranges`; ids are never reused). The web Label page edits them one active object at a time (see [Label page: several objects](usage.md#label-page-several-objects)).

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
- `POST /api/clips/{id}/frames/delete-range` and `POST /api/clips/{id}/frames/restore-range` take
  `{"start", "end"}` (inclusive) and delete or restore every frame in that range (frames already in that state are
  left alone; files are never removed) → 200 with the clip detail; 400 when `start > end` or a frame is outside the
  clip; 404 for an unknown clip.
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

## Clip regions

`PUT /api/clips/{clip_id}/regions` replaces the clip's regions, the per-frame-range crop and blocked boxes that SAM3
never sees:

```json
{"regions": [{"start": 0, "end": 49, "crop": [x0, y0, x1, y1], "blocks": [[x0, y0, x1, y1]]}]}
```

`start` and `end` are inclusive frame indices, `crop` is `null` or a rectangle in pixels, and `blocks` may be empty.
The empty list clears all regions. Responses: `200` with the same body as `GET /api/clips/{id}` (the clip's `regions`
holds the saved list); `400` when the regions are invalid, the message naming the range (for example
`region 3–8 overlaps region 0–5`); `404` for an unknown clip; `422` for a malformed body. Saved masks are not
rewritten (they are cut at export), and a propagation that is already running keeps the regions it started with.

## Login

When a team password is set (`vdp set-password`), every route needs the `vdp_session` cookie except `GET /login`,
`GET /favicon.ico`, `POST /api/auth/login` and `POST /api/auth/logout`. Without it, `/api/*` and the docs answer
`401 {"detail": "login required"}`, pages redirect to `/login?next=…`, and WebSockets are accepted and closed with
code 4401. `POST /api/auth/login` takes a form (`password`, `next`) and answers `303` (to `next` with the cookie,
or back to `/login?error=wrong`); after 5 wrong passwords from one address in 5 minutes it answers `429` with
`Retry-After`. `/api/health` reports `auth.enabled`.

## Resumable uploads

Used by the GUI (Clips page and the phone page `/upload`), so a lost connection or a locked phone never restarts a
file. `POST /api/projects/{id}/uploads` with `{name, size, stride, fps}` → `201 {upload_id, chunk_size, received}`
(400 for a file that is not a video or `.zip`, 413 over `ingest.max_upload_bytes`). Then
`PUT /api/uploads/{upload_id}?offset=N` with the raw bytes of one chunk (at most 8 MiB) → `{received}`: the offset
must equal what the server has (a retried chunk that already arrived is accepted and ignored; any other mismatch is
`409 {received}`). The final chunk starts the import and adds `job_id`. `GET /api/uploads/{upload_id}` reports
progress, `GET /api/projects/{id}/uploads` lists unfinished uploads, `DELETE /api/uploads/{upload_id}` cancels one
(409 while its import runs). Unfinished uploads survive a server restart and expire after 24 hours without activity.
`GET /upload?project=<id>` redirects to the phone page.
