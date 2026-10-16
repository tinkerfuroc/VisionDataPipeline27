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

`POST /api/projects/{id}/export` starts an export job; body `{clips, formats, options, name}` → `202
{job_id}`. `options.classes` is a list of project class ids to export (null or absent = every class); an
empty list, a repeated id or an unknown class id is a 400. It returns 400 for an invalid request (unknown
format, no clips, a bad option) and 409 (with the running `job_id`) while a propagation job runs, since
propagation rewrites the very masks an export reads.

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
