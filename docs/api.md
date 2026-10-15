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

## Login

When a team password is set (`vdp set-password`), every route needs the `vdp_session` cookie except `GET /login`,
`GET /favicon.ico`, `POST /api/auth/login` and `POST /api/auth/logout`. Without it, `/api/*` and the docs answer
`401 {"detail": "login required"}`, pages redirect to `/login?next=…`, and WebSockets are accepted and closed with
code 4401. `POST /api/auth/login` takes a form (`password`, `next`) and answers `303` (to `next` with the cookie,
or back to `/login?error=wrong`); after 5 wrong passwords from one address in 5 minutes it answers `429` with
`Retry-After`. `/api/health` reports `auth.enabled`.
