# Plan A: Several Objects per Clip (Server) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the server stores, labels, propagates and serves any number of objects per clip.

**Architecture:** the storage layout already keys masks by object id; this plan removes the one-object rule, gives
every labelling function an explicit object id, changes the engine to track several seeds in one session, and
rewrites the propagation runner around per-object "tracks" that are grouped by anchor frame for each engine call.
The HTTP API gains object routes; three temporary legacy shims keep today's web app working until plan B.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, numpy, HF transformers SAM3 tracker, pytest (uv-managed env:
run everything with `uv run`).

**Spec:** `docs/superpowers/specs/2026-09-26-multi-object-labeling-and-export-selection-design.md` (§4–§7, §13–§14
plan A rows).

## Global Constraints

- Python tooling only through uv: `uv run pytest ...`, `uv run ruff check server`. Never pip/conda.
- Object ids are never reused within a clip: new id = `max(clip.next_object_id, max(existing ids, default=0) + 1)`,
  then `clip.next_object_id = new id + 1`. Old clip.json files (no `next_object_id`) load with 0.
- Library functions (`accept`, `set_mask`, `clear_range`, `run_propagation`, ...) take explicit object ids; there is
  no implicit "first object" anywhere in `server/vsp/labeling/`.
- Legacy shims (plan B removes them; each carries the comment `# legacy shim: removed in plan B`):
  1. `POST /accept` and `POST /mask` with the `object_id` key **absent** from the JSON body use the clip's first
     object, or create one with `class_id` when there is none, and apply `class_id` as a relabel (today's behaviour).
  2. `GET /api/clips/{id}/masks/{idx}` serves the first object's mask (404 when the clip has no object).
  3. Propagation `frame` events carry `has_mask` = any reported object has a mask, next to the new `objects` list.
- Every commit leaves `uv run pytest server/tests -q -m "not gpu"` and `uv run ruff check server` green, and the web
  tests (`cd web && node node_modules/.bin/vitest run`) untouched and green.
- Commit messages end with the two attribution lines:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6`.
- Stage only the files your task touches (`git add <paths>`), never `git add -A`.

## File map

| File | Change |
|---|---|
| `server/vsp/store/models.py` | `Clip.next_object_id` |
| `server/vsp/store/project.py` | `object_states`, `delete_object_masks` |
| `server/vsp/labeling/prompts.py` | `create_object` id rule, `OBJECT_ID` removed, `object_id` on preview/accept, accept-all |
| `server/vsp/labeling/objects.py` (new) | `find_object`, `relabel_object`, `delete_object` |
| `server/vsp/labeling/paint.py` | `set_mask(..., object_id=...)` |
| `server/vsp/labeling/frames.py` | `clear_range(..., objects=...)` |
| `server/vsp/engine/types.py`, `fake.py`, `sam3.py` | multi-seed `propagate` |
| `server/vsp/labeling/propagate.py` | multi-object runner |
| `server/vsp/api/labeling.py`, `api/clips.py`, `api/views.py` | object routes, bodies, shims, `object_states` |
| `server/tests/conftest.py` | `OBJ = 1` helper constant |
| tests | see each task |
| `README.md` | API/behaviour notes |

---

### Task 1: Store — object ids, per-object states, object mask removal

**Files:**
- Modify: `server/vsp/store/models.py`, `server/vsp/store/project.py`, `server/vsp/labeling/prompts.py` (only
  `create_object`)
- Test: `server/tests/test_store.py`, `server/tests/test_prompts.py`

**Interfaces:**
- Produces: `Clip.next_object_id: int = 0`; `create_object(clip, class_id) -> LabelObject` (no longer raises
  `Conflict`); `ProjectStore.object_states(clip, obj) -> list[str]`; `ProjectStore.delete_object_masks(clip_id,
  object_id) -> None`.

- [ ] **Step 1: Write the failing tests** (append to the named files)

```python
# test_store.py
from vsp.labeling.prompts import create_object


def test_object_states_are_per_object_and_deleted_frames_show_in_every_lane(store, make_clip):
    _, clip = make_clip(n_frames=6)
    a = create_object(clip, 0)
    b = create_object(clip, 0)
    clip.deleted_frames = [5]
    store.write_clip(clip)
    m = np.ones((60, 120), dtype=bool)
    store.write_mask(clip.id, a.id, 1, m)
    store.write_mask(clip.id, b.id, 2, m)
    store.write_mask(clip.id, b.id, 5, m)
    b.keyframes = [Keyframe(frame_idx=2, prompt=Prompt(type="paint", payload={}), created_at=now_iso())]
    assert store.object_states(clip, a) == ["empty", "masked", "empty", "empty", "empty", "deleted"]
    assert store.object_states(clip, b) == ["empty", "empty", "keyframe", "empty", "empty", "deleted"]
    assert store.frame_states(clip) == ["empty", "masked", "keyframe", "empty", "empty", "deleted"]


def test_delete_object_masks_removes_only_that_object(store, make_clip):
    _, clip = make_clip(n_frames=3)
    m = np.ones((60, 120), dtype=bool)
    store.write_mask(clip.id, 1, 0, m)
    store.write_mask(clip.id, 2, 0, m)
    store.delete_object_masks(clip.id, 2)
    assert store.mask_indices(clip.id, 1) == [0] and store.mask_indices(clip.id, 2) == []
    store.delete_object_masks(clip.id, 7)  # nothing there: no error
```

```python
# test_prompts.py  (replace the existing test that expects Conflict on a second object)
def test_object_ids_grow_and_are_never_reused(store, make_clip):
    _, clip = make_clip()
    assert [create_object(clip, 0).id for _ in range(3)] == [1, 2, 3]
    assert clip.next_object_id == 4
    clip.objects = [o for o in clip.objects if o.id != 3]  # deleting the newest must not free its id
    assert create_object(clip, 0).id == 4


def test_old_clip_files_without_next_object_id_continue_after_the_highest_id(store, make_clip):
    _, clip = make_clip()
    clip.objects = [LabelObject(id=1, class_id=0)]
    clip.next_object_id = 0  # what an old clip.json loads as
    assert create_object(clip, 0).id == 2
```

Imports needed in `test_store.py`: `numpy as np`, `from vsp.store import Keyframe, Prompt, now_iso` (check the
existing imports first). In `test_prompts.py`: `from vsp.store import LabelObject`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest server/tests/test_store.py server/tests/test_prompts.py -q`
Expected: FAIL (`object_states` / `delete_object_masks` missing, `create_object` raises Conflict).

- [ ] **Step 3: Implement**

`models.py` — in `Clip`, after `objects`:

```python
    # ids are never reused: a deleted object's stale mask URLs and job events can never reach a new object
    next_object_id: int = 0
```

Update the module docstring: `"""On-disk schema. A clip holds any number of objects; ids are unique within the clip
and never reused."""`

`prompts.py` — replace `create_object` (keep `OBJECT_ID` for now; Task 5 removes it):

```python
def create_object(clip: Clip, class_id: int) -> LabelObject:
    """Append a new object to `clip` (in memory; the caller writes the clip). Ids are never reused."""
    new_id = max(clip.next_object_id, max((o.id for o in clip.objects), default=0) + 1)
    obj = LabelObject(id=new_id, class_id=class_id)
    clip.objects.append(obj)
    clip.next_object_id = new_id + 1
    return obj
```

Remove the now-unused `Conflict` import only if nothing else in the file uses it.

`project.py` — refactor `frame_states` and add the two methods:

```python
    def _states(self, clip: Clip, objects: list) -> list[str]:
        """Priority: deleted > keyframe > masked > empty, over `objects`."""
        states = ["empty"] * clip.frame_count
        for obj in objects:
            for i in self.mask_indices(clip.id, obj.id):
                if 0 <= i < clip.frame_count:
                    states[i] = "masked"
        for obj in objects:
            for kf in obj.keyframes:
                if 0 <= kf.frame_idx < clip.frame_count:
                    states[kf.frame_idx] = "keyframe"
        for i in clip.deleted_frames:
            if 0 <= i < clip.frame_count:
                states[i] = "deleted"
        return states

    def frame_states(self, clip: Clip) -> list[str]:
        """Per-frame state of the whole clip (any object), for the Clips page and the timeline summary."""
        return self._states(clip, clip.objects)

    def object_states(self, clip: Clip, obj: LabelObject) -> list[str]:
        """Per-frame state of one object; a deleted frame is `deleted` in every object's lane."""
        return self._states(clip, [obj])

    def delete_object_masks(self, clip_id: str, object_id: int) -> None:
        shutil.rmtree(self.clip_dir(clip_id) / "masks" / str(int(object_id)), ignore_errors=True)
```

(`LabelObject` and `shutil` imports: check they are present.)

- [ ] **Step 4: Run the tests**

Run: `uv run pytest server/tests -q -m "not gpu"` — Expected: all pass. `uv run ruff check server` — clean.

- [ ] **Step 5: Commit**

```bash
git add server/vsp/store/models.py server/vsp/store/project.py server/vsp/labeling/prompts.py server/tests/test_store.py server/tests/test_prompts.py
git commit -m "Store: several objects per clip (never-reused ids, per-object states, object mask removal)" -m "Co-Authored-By: ..." -m "Claude-Session: ..."
```

---

### Task 2: Engine — track several seeds in one session

**Files:**
- Modify: `server/vsp/engine/types.py`, `server/vsp/engine/fake.py`, `server/vsp/engine/sam3.py`,
  `server/vsp/labeling/propagate.py` (only the one `engine.propagate(...)` call)
- Test: `server/tests/test_fake_engine.py`, `server/tests/test_engine_gpu.py`, `server/tests/test_api_labeling.py`
  (`SlowEngine`), `server/tests/test_api_export.py` (its `propagate` override), `server/tests/test_propagate.py`
  (any engine subclass there)

**Interfaces:**
- Produces: `Engine.propagate(frames, seeds: Mapping[int, np.ndarray], cancel_cb=None) -> Iterator[tuple[int,
  dict[int, np.ndarray | None]]]`. Every yielded dict has exactly the seeded ids as keys. `FakeEngine` gains
  `is_empty_obj: Callable[[np.ndarray, int], bool] | None = None` (per-object emptiness for tests) and
  `seed_calls: list[list[int]]` (sorted seeded ids per call).

- [ ] **Step 1: Update `types.py`** — the `propagate` protocol method:

```python
    def propagate(
        self,
        frames: list[np.ndarray] | PreparedChunk,
        seeds: Mapping[int, np.ndarray],
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, dict[int, np.ndarray | None]]]:
        """Track every seed (object id -> bool mask, all given for `frames[0]`) forward through `frames`.

        Yields `(offset, {object_id: mask or None})` with exactly the seeded ids as keys; None means no usable
        mask for that object on that frame. At least one seed; each non-empty and of the frames' shape, else
        ValueError. (The rest of the existing docstring -- laziness, the lock, `prepare()` -- stays as it is.)
        """
```

Import `Mapping` from `collections.abc`.

- [ ] **Step 2: Write the failing fake-engine tests** (in `test_fake_engine.py`; adapt the existing tests to pass
  `{1: _seed()}` and read `out[k][1]`, then add):

```python
def test_tracks_every_seed_independently():
    eng = FakeEngine(dx=2)
    a = _seed()
    b = np.zeros_like(a)
    b[40:50, 60:70] = True
    out = dict(eng.propagate(_frames(3), {1: a, 5: b}))
    assert set(out[2]) == {1, 5}
    assert np.array_equal(out[2][1][:, 4:], a[:, :-4]) and np.array_equal(out[2][5][:, 4:], b[:, :-4])
    assert eng.seed_calls == [[1, 5]]


def test_per_object_emptiness_hook():
    eng = FakeEngine(is_empty_obj=lambda frame, oid: oid == 5)
    b = _seed()
    out = dict(eng.propagate(_frames(2), {1: _seed(), 5: b}))
    assert out[1][1] is not None and out[1][5] is None


@pytest.mark.parametrize("seeds", [{}, {1: np.zeros((60, 120), dtype=bool)}, {1: np.ones((5, 5), dtype=bool)}])
def test_bad_seeds_raise_lazily(seeds):
    gen = FakeEngine().propagate(_frames(2), seeds)  # nothing raised yet
    with pytest.raises(ValueError):
        next(gen)
```

(Check `_frames`/`_seed` helpers' frame size in the file and use it for the `zeros((h, w))` seed.)

- [ ] **Step 3: Run to see them fail** — `uv run pytest server/tests/test_fake_engine.py -q` → FAIL.

- [ ] **Step 4: Implement `FakeEngine.propagate`**

```python
    def propagate(
        self,
        frames: list[np.ndarray] | PreparedChunk,
        seeds: Mapping[int, np.ndarray],
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, dict[int, np.ndarray | None]]]:
        self.propagate_calls.append(len(frames))
        self.seed_calls.append(sorted(seeds))
        if self.oom_if_more_than is not None and len(frames) > self.oom_if_more_than:
            raise ChunkOOMError(f"fake OOM: {len(frames)} frames > {self.oom_if_more_than}")
        if not seeds:
            raise ValueError("at least one seed is required")
        h, w = frames[0].shape[:2]
        for oid, seed in seeds.items():
            if seed.shape != (h, w):
                raise ValueError(f"seed {oid} shape {seed.shape} does not match frames {(h, w)}")
            if not seed.any():
                raise ValueError(f"seed {oid} must be non-empty")
        for offset, frame in enumerate(frames):
            if cancel_cb is not None and cancel_cb():
                return
            out: dict[int, np.ndarray | None] = {}
            for oid, seed in seeds.items():
                if (self.is_empty is not None and self.is_empty(frame)) or (
                    self.is_empty_obj is not None and self.is_empty_obj(frame, oid)
                ):
                    out[oid] = None
                    continue
                s = offset * self.dx
                shifted = seed.copy() if s == 0 else np.zeros_like(seed)
                if 0 < s < seed.shape[1]:
                    shifted[:, s:] = seed[:, :-s]
                out[oid] = None if self._is_degenerate(shifted) else shifted
            yield offset, out
```

The OOM check stays before the seed checks and before the first yield, as today (a generator: it all runs on the
first `next()`). Add `is_empty_obj` to `__init__` (stored as an attribute) and `self.seed_calls: list[list[int]] =
[]`; import `Mapping`.

- [ ] **Step 5: Implement `Sam3Engine`** (`sam3.py`)

`_binarize` gains a row:

```python
    def _binarize(self, pred_masks: torch.Tensor, h: int, w: int, row: int = 0) -> np.ndarray:
        full = torch.nn.functional.interpolate(
            pred_masks[row:row + 1].float(), size=(h, w), mode="bilinear", align_corners=False
        )
        return (full.sigmoid() > self.MASK_THRESHOLD)[0, 0].cpu().numpy()
```

`propagate` signature and checks:

```python
    def propagate(
        self,
        frames: list[np.ndarray] | PreparedChunk,
        seeds: Mapping[int, np.ndarray],
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, dict[int, np.ndarray | None]]]:
        ...  # docstring: say "every seed" instead of "seed_mask"; keep the lock/laziness text
        self._require_loaded()
        if isinstance(frames, _Prepared):
            prepared = frames
        else:
            if not frames:
                return
            prepared = self.prepare(frames)
        h, w = prepared.h, prepared.w
        if not seeds:
            raise ValueError("at least one seed is required")
        for oid, seed in seeds.items():
            if seed.shape != (h, w):
                raise ValueError(f"seed {oid} shape {seed.shape} does not match frames {(h, w)}")
            if not seed.any():
                raise ValueError(f"seed {oid} must be non-empty")
        video = prepared.pixel_values
        try:
            with self._lock, torch.inference_mode(), self._autocast():
                yield from self._propagate_locked(video, seeds, h, w, cancel_cb)
        ...  # the except/finally blocks are unchanged
```

`_propagate_locked`:

```python
    def _propagate_locked(self, video, seeds, h, w, cancel_cb):
        n = int(video.shape[0])
        session = self._new_session(video, h, w)
        # (keep the existing comment about encoding one frame ahead)
        self._encode_frame(session, 0)
        ids = [int(oid) for oid in seeds]
        for oid in ids:
            obj_idx = session.obj_id_to_idx(oid)
            seed = torch.from_numpy(seeds[oid].astype(np.float32))[None, None].to(self.device, dtype=self.torch_dtype)
            session.add_mask_inputs(obj_idx, 0, seed)
        session.obj_with_new_inputs = list(ids)
        for out in self._model.tracker_model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=0, max_frame_num_to_track=n
        ):
            if cancel_cb is not None and cancel_cb():
                return
            masks: dict[int, np.ndarray | None] = dict.fromkeys(ids)
            if out.pred_masks is not None and len(out.pred_masks) > 0:
                # one row per tracked object, in session order (out.object_ids); the tracker head runs once per
                # object per frame, so its cost grows with the object count while frame encoding is shared
                rows = list(out.object_ids) if out.object_ids is not None else ids
                for row, oid in enumerate(rows):
                    if oid not in masks or row >= len(out.pred_masks):
                        continue
                    m = self._binarize(out.pred_masks, h, w, row)
                    if not is_degenerate_mask(m, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX):
                        masks[oid] = m
            fr = int(out.frame_idx)
            yield fr, masks
            if fr + 1 < n:
                self._encode_frame(session, fr + 1)
```

`_OBJ_ID` stays for `predict_prompt` only.

- [ ] **Step 6: Adapt the single caller** in `propagate.py::_run_direction`:

```python
            with closing(engine.propagate(frames, {object_id: anchor_mask}, cancel_cb=cancel)) as masks:
                for offset, out in masks:
                    mask = out[object_id]
```

(the rest of the loop body unchanged; Task 3 replaces this function).

- [ ] **Step 7: Adapt test doubles** — every test subclass overriding `propagate(self, frames, seed_mask,
  cancel_cb=None)` becomes `propagate(self, frames, seeds, cancel_cb=None)` and passes `seeds` through. In
  `test_engine_gpu.py`, calls become `engine.propagate(frames, {1: truth[0]})` and each result `m` becomes `m[1]`
  (e.g. `got = {k: v[1] for k, v in engine.propagate(frames, {1: truth[0]})}`). Add a two-object GPU test next to
  the existing single-object accuracy test, reusing its fixture clip and ground truth:

```python
@pytest.mark.gpu
def test_two_objects_in_one_session_each_follow_their_own_seed(engine, ...):  # same fixtures as the single-object accuracy test
    # seed object 1 with the true mask, object 2 with the same mask shifted into the empty half of the frame
    # is not a real object; instead split the true mask into its left and right halves (two "objects" that
    # touch), propagate both, and check each output overlaps its own half's ground truth more than the other's
    ...
```

The implementer writes this test concretely from the existing single-object GPU accuracy test in the file: split
`truth[i]` at its bbox's horizontal centre into `left`/`right`, seed `{1: left[0], 2: right[0]}`, and assert for
every frame that `iou(out[1], left[i]) > iou(out[1], right[i])` and the same for object 2, plus that each returned
dict has keys `{1, 2}`. If the file has no ground-truth fixture that makes a split meaningful, report
DONE_WITH_CONCERNS with the reason instead of weakening the assertion.

- [ ] **Step 8: Run** — `uv run pytest server/tests -q -m "not gpu"` → pass; `uv run ruff check server` → clean;
  then `CUDA_VISIBLE_DEVICES=1 uv run pytest server/tests/test_engine_gpu.py -q -m gpu` (the RTX 2080 Ti; cuda:0 runs
  the SAM3 server) → pass, or report why it could not run.

- [ ] **Step 9: Commit** — "Engine: track several seeds in one propagate call".

---

### Task 3: Multi-object propagation runner

**Files:**
- Modify: `server/vsp/labeling/propagate.py`
- Test: `server/tests/test_propagate.py`, `server/tests/conftest.py` (add `OBJ = 1`)

**Interfaces:**
- Consumes: Task 2 engine API; `plan_indices(clip, obj, from_frame, direction)` (unchanged).
- Produces:
  - `resolve_participants(store, clip, from_frame, objects, *, strict) -> tuple[list[tuple[LabelObject, np.ndarray]], list[int]]`
  - `run_propagation(store, engine, clip_id, from_frame, direction, *, objects="all", chunk_size, min_chunk_size, lost_after, emit=None, cancel=None) -> PropagateResult`
  - `PropagateResult` fields: `processed, written, cleared, lost_at, cancelled, oom_retries,
    lost_by_direction: dict[str, dict[int, int]], lost: dict[int, int], skipped: list[int], objects: list[int]`.
  - Frame event: `{"type": "frame", "frame": idx, "objects": [{"id": oid, "has_mask": bool}, ...], "has_mask": bool,
    "done": n, "total": n}` (`has_mask` is legacy shim 3).
- `check_seed` is removed (replaced by `resolve_participants`); Task 5 updates its API caller. Until then
  `api/labeling.py` must keep importing something that exists: in this task, change its `propagate` route to call
  `resolve_participants(state.store, clip, body.from_frame, "all", strict=True)` and
  `plan_indices(clip, participants[0][0], ...)` for validation (Task 5 finishes the route).

Semantics (spec §6), exactly:

- Participants: requested objects with a mask on `from_frame`. `"all"` → objects without a seed go to `skipped`.
  An explicit id list: with `strict=True` an unknown id is `NotFound(f"object {oid} not found in clip {clip.id}")` and
  a listed object without a seed is `BadRequest(f"object {oid} has no mask on frame {from_frame}; ...")`; with
  `strict=False` (inside the job) both are skipped instead. A deleted `from_frame` is
  `BadRequest(f"frame {from_frame} is deleted; restore it or propagate from another frame")` either way. No
  participants: `BadRequest("nothing to propagate: accept a prompt first")` when the clip has no objects at all, else
  `BadRequest(f"frame {from_frame} has no mask; propagate from a masked frame or prompt here first")`.
- Per direction, each participant's frames are `plan_indices(...)`; all plans start at `from_frame ± 1` and are
  contiguous, so each is a prefix of the longest one, which is the direction's frame list. `total` = sum over
  directions of the longest plan's length.
- Tracks (one per participant per direction) carry: `limit` (its plan length), `pos` (index of its next frame in the
  direction's list), `anchor_idx`/`anchor_mask` (its last frame with a mask; starts at `from_frame`/seed),
  `empties`, `first_empty`, `finished`. A track is *active* while `not finished and pos < limit`.
- Loop: group the active tracks by `(pos, anchor_idx)`; run the group with the smallest `pos` (ties: the bigger group)
  on `chunk = frames[pos : min(pos + size, max limit of its members)]`, seeded `{id: anchor_mask}`, frames
  `prefetch.take(anchor_idx, chunk)`; schedule `prefetch.schedule(chunk[-1], next slice)` when the group has frames
  after this chunk.
- Per yielded frame: every member that is still active consumes it (`pos += 1`, the frame is added to its
  `covered` list); `None` deletes that object's stored mask on the frame (counted in `cleared`) and counts an empty;
  a mask resets the empties, becomes the anchor, and is written unless the frame is deleted (counted in `written`).
  After `lost_after` consecutive empties the track is finished and lost at its first empty. Members that are no
  longer active ignore their outputs. When no member is active, leave the engine call (break).
- `_record_range` per member with its covered frames on every exit path; it ignores an object that no longer
  exists.
- OOM (`ChunkOOMError`): exactly today's rule, per engine call (`yielded` frames = progress made).
- Stopped short (fewer frames yielded than the chunk has, while a member is still active): `cancel()` true →
  `cancelled`, stop the whole run; else `PropagationFailed` as today.
- `done`/`processed` = frames covered by at least one object: per direction `max(t.pos)`, summed over finished
  directions.
- `lost_by_direction[d][oid]`, `lost[oid]` (first direction in run order that lost it), `lost_at` = in the first
  direction (run order) with any loss, the lost frame nearest `from_frame`.

- [ ] **Step 1: Write the failing tests** — first, mechanically move every existing test in `test_propagate.py`
  to the new API: `OBJECT_ID` → `OBJ` (from `conftest`), `accept(...)` calls pass `object_id=None, class_id=0` for the
  first keyframe and `object_id=OBJ` for later ones (this needs Task 4's `accept` signature — **so in this task keep
  the existing `accept` calls as they are**; Task 4 moves them), result field reads unchanged (`lost_at` keeps its
  meaning for one object). Then add:

```python
def _two_objects(store, make_clip, eng, n_frames=30, white=(), a_kf=(0,), b_kf=(0,)):
    """Object 1 = the CLICK rect (x 20:40), object 2 = a rect lower down (x 60:80), both class 0."""
    _, clip = make_clip(n_frames=n_frames, white=white)
    clip = store.read_clip(clip.id)
    a, b = create_object(clip, 0), create_object(clip, 0)
    for obj, kfs, x in ((a, a_kf, 20), (b, b_kf, 60)):
        for kf in kfs:
            m = np.zeros((60, 120), dtype=bool)
            m[40:55, x:x + 20] = True
            store.write_mask(clip.id, obj.id, kf, m)
            obj.keyframes.append(Keyframe(frame_idx=kf, prompt=Prompt(type="paint", payload={}), created_at=now_iso()))
    store.write_clip(clip)
    return clip


def _xs(store, clip_id, oid, idx):
    mask = store.read_mask(clip_id, oid, idx)
    return None if mask is None else int(np.where(mask)[1].min())


def test_all_objects_share_one_engine_call_per_chunk(store, make_clip):
    eng = FakeEngine()
    clip = _two_objects(store, make_clip, eng)
    events = []
    res = run_propagation(store, eng, clip.id, 0, "forward", emit=events.append, **KW)
    assert (res.processed, res.written, res.objects, res.skipped) == (29, 58, [1, 2], [])
    assert eng.propagate_calls == [9, 9, 9, 6] and all(s == [1, 2] for s in eng.seed_calls)
    assert _xs(store, clip.id, 1, 9) == 20 + 18 and _xs(store, clip.id, 2, 9) == 60 + 18
    frame_events = [e for e in events if e["type"] == "frame"]
    assert frame_events[0]["objects"] == [{"id": 1, "has_mask": True}, {"id": 2, "has_mask": True}]
    assert frame_events[0]["has_mask"] is True and frame_events[-1]["done"] == 29
    c = store.read_clip(clip.id)
    assert [o.propagated_ranges for o in c.objects] == [[(1, 29)], [(1, 29)]]


def test_each_object_stops_at_its_own_next_keyframe(store, make_clip):
    eng = FakeEngine()
    clip = _two_objects(store, make_clip, eng, n_frames=20, a_kf=(0, 6), b_kf=(0,))
    before = store.mask_path(clip.id, 1, 6).read_bytes()
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 19 and res.written == 5 + 19
    assert store.mask_path(clip.id, 1, 6).read_bytes() == before
    assert store.mask_indices(clip.id, 1) == [0, 1, 2, 3, 4, 5, 6]
    assert store.mask_indices(clip.id, 2) == list(range(20))


def test_one_object_lost_while_the_other_continues(store, make_clip):
    # object 2 is invisible on frames 6..29; object 1 is always visible
    eng = FakeEngine(is_empty_obj=lambda frame, oid: oid == 2 and is_white(frame))
    clip = _two_objects(store, make_clip, eng, white=range(6, 30))
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.lost == {2: 6} and res.lost_by_direction == {"forward": {2: 6}} and res.lost_at == 6
    assert store.mask_indices(clip.id, 1) == list(range(30))
    assert store.mask_indices(clip.id, 2) == [0, 1, 2, 3, 4, 5]
    assert all(s == [1, 2] for s in eng.seed_calls[:1]) and eng.seed_calls[-1] == [1]


def test_anchors_diverge_into_two_groups_and_each_keeps_its_own_seed(store, make_clip):
    # object 2 is invisible on frame 8 only (lost_after=3, so it is not lost): its anchor stays at 7 while
    # object 1's moves on, so from the next chunk the two run as separate engine calls
    eng = FakeEngine(is_empty_obj=lambda frame, oid: oid == 2 and is_white(frame))
    clip = _two_objects(store, make_clip, eng, n_frames=20, white=(8,))
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.lost == {} and res.processed == 19
    assert [1] in eng.seed_calls and [2] in eng.seed_calls  # split after frame 8
    assert _xs(store, clip.id, 2, 8) is None
    # object 2's frame 9 is tracked from its frame-7 anchor prepended to the chunk (offset 1), like the single-object
    # rule after a missed frame: the fake shifts by dx per chunk offset, not per frame distance
    assert _xs(store, clip.id, 2, 9) == 60 + 2 * 7 + 2


def test_objects_without_a_seed_are_skipped_and_explicit_lists_are_strict(store, make_clip):
    eng = FakeEngine()
    clip = _two_objects(store, make_clip, eng, a_kf=(0,), b_kf=(5,))
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.objects == [1] and res.skipped == [2]
    clip = store.read_clip(clip.id)
    with pytest.raises(BadRequest, match="object 2 has no mask on frame 0"):
        resolve_participants(store, clip, 0, [2], strict=True)
    with pytest.raises(NotFound, match="object 9"):
        resolve_participants(store, clip, 0, [9], strict=True)
    parts, skipped = resolve_participants(store, clip, 0, [1, 9], strict=False)
    assert [o.id for o, _ in parts] == [1] and skipped == [9]


def test_selected_object_only_leaves_the_others_untouched(store, make_clip):
    eng = FakeEngine()
    clip = _two_objects(store, make_clip, eng)
    res = run_propagation(store, eng, clip.id, 0, "forward", objects=[2], **KW)
    assert res.objects == [2] and store.mask_indices(clip.id, 1) == [0]
    assert store.mask_indices(clip.id, 2) == list(range(30))


def test_both_directions_lost_at_prefers_the_first_direction_and_the_nearest_frame(store, make_clip):
    # frames 0..2 and 13..19 are white (nothing visible). Object 2 has a keyframe at 4, so backward it stops at 5
    # and never reaches the white frames; object 1 is lost in both directions.
    eng = FakeEngine(is_empty=is_white)
    clip = _two_objects(store, make_clip, eng, n_frames=20, white=(*range(0, 3), *range(13, 20)),
                        a_kf=(8,), b_kf=(4, 8))
    res = run_propagation(store, eng, clip.id, 8, "both", **KW)
    assert res.lost_by_direction == {"forward": {1: 13, 2: 13}, "backward": {1: 2}}
    assert res.lost == {1: 13, 2: 13} and res.lost_at == 13  # forward runs first
    assert store.mask_indices(clip.id, 2) == list(range(4, 13))


def test_two_objects_cancel_and_oom(store, make_clip):
    eng = FakeEngine(oom_if_more_than=5)
    clip = _two_objects(store, make_clip, eng)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.oom_retries >= 1 and res.processed == 29 and res.written == 58
    seen = []
    clip2 = _two_objects(store, make_clip, FakeEngine())
    res = run_propagation(store, FakeEngine(), clip2.id, 0, "forward", emit=seen.append,
                          cancel=lambda: len(seen) >= 4, **KW)
    assert res.cancelled
    idx1, idx2 = store.mask_indices(clip2.id, 1), store.mask_indices(clip2.id, 2)
    assert idx1 == idx2 and len(idx1) < 30


def test_an_object_deleted_before_the_job_starts_is_skipped(store, make_clip):
    eng = FakeEngine()
    clip = _two_objects(store, make_clip, eng)
    c = store.read_clip(clip.id)
    c.objects = [o for o in c.objects if o.id != 2]
    store.write_clip(c)
    res = run_propagation(store, eng, clip.id, 0, "forward", objects=[1, 2], **KW)
    assert res.objects == [1] and res.skipped == [2]
```

Imports to add: `from conftest import OBJ, is_white`, `from vsp.labeling.prompts import create_object`,
`from vsp.labeling.propagate import resolve_participants`, `from vsp.store import Keyframe, NotFound, Prompt,
now_iso` (check which module exports `NotFound`; `vsp.store` does for prompts.py). In `conftest.py` add
`OBJ = 1  # the id of a clip's first object`.

- [ ] **Step 2: Run them to see them fail** — `uv run pytest server/tests/test_propagate.py -q`.

- [ ] **Step 3: Implement** — replace the body of `propagate.py` below `plan_indices` with:

```python
def resolve_participants(
    store: ProjectStore,
    clip: Clip,
    from_frame: int,
    objects: Literal["all"] | list[int],
    *,
    strict: bool,
) -> tuple[list[tuple[LabelObject, np.ndarray]], list[int]]:
    """The objects a run from `from_frame` propagates, each with its seed mask, and the ids skipped.

    `strict` (the route, validating a request): an unknown id is NotFound and a listed object without a mask on
    `from_frame` is BadRequest. Not strict (the job, which re-reads the clip): both are skipped instead, so an object
    deleted between the request and the job's start does not fail the run.
    """
    if not 0 <= from_frame < clip.frame_count:
        raise BadRequest(f"frame {from_frame} is out of range 0..{clip.frame_count - 1}")
    if from_frame in clip.deleted_frames:
        raise BadRequest(f"frame {from_frame} is deleted; restore it or propagate from another frame")
    by_id = {o.id: o for o in clip.objects}
    explicit = objects != "all"
    wanted = list(by_id) if not explicit else list(dict.fromkeys(objects))
    participants: list[tuple[LabelObject, np.ndarray]] = []
    skipped: list[int] = []
    for oid in wanted:
        obj = by_id.get(oid)
        if obj is None:
            if strict:
                raise NotFound(f"object {oid} not found in clip {clip.id}")
            skipped.append(oid)
            continue
        seed = store.read_mask(clip.id, oid, from_frame)
        if seed is None:
            if strict and explicit:
                raise BadRequest(
                    f"object {oid} has no mask on frame {from_frame}; propagate it from a frame where it has one"
                )
            skipped.append(oid)
            continue
        participants.append((obj, seed))
    if not participants:
        if not clip.objects:
            raise BadRequest("nothing to propagate: accept a prompt first")
        raise BadRequest(f"frame {from_frame} has no mask; propagate from a masked frame or prompt here first")
    return participants, skipped


def _record_range(store: ProjectStore, clip_id: str, object_id: int, indices: list[int]) -> None:
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        obj = next((o for o in clip.objects if o.id == object_id), None)
        if obj is None:  # deleted while the run was going (the API forbids it, but never crash the run on it)
            return
        obj.propagated_ranges = add_range(obj.propagated_ranges, min(indices), max(indices))
        store.write_clip(clip)


@dataclass
class _Track:
    """One object's progress through one direction's frame list."""

    obj_id: int
    limit: int  # how many frames of the direction's list this object may cover (up to its next keyframe)
    anchor_idx: int  # its last frame with a mask: the chunk it runs in is seeded there
    anchor_mask: np.ndarray
    pos: int = 0  # index of its next frame in the direction's list
    empties: int = 0
    first_empty: int | None = None
    finished: bool = False  # lost

    @property
    def active(self) -> bool:
        return not self.finished and self.pos < self.limit


def _run_direction(
    store: ProjectStore,
    engine: Engine,
    clip: Clip,
    tracks: list[_Track],
    frames_list: list[int],
    direction: str,
    *,
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    base: int,
    total: int,
    result: PropagateResult,
    emit: Emit,
    cancel: Callable[[], bool],
    prefetch: ChunkPrefetcher,
) -> int:
    """Run one direction: engine calls over chunks, one per group of objects sharing an anchor frame.

    Each object keeps the single-object rule of old: it is seeded on its own last masked frame, which is prepended
    to the chunk. Objects whose anchors (and positions) coincide share one engine call -- the usual case; an object
    that missed a frame falls behind into its own group until it is found again or lost. Returns the (possibly
    OOM-backed-off) chunk size, so a following direction starts from it.
    """
    size = max(min_chunk_size, chunk_size)
    while True:
        live = [t for t in tracks if t.active]
        if not live:
            return size
        groups: dict[tuple[int, int], list[_Track]] = {}
        for t in live:
            groups.setdefault((t.pos, t.anchor_idx), []).append(t)
        (pos, anchor_idx), members = min(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1])))
        end = max(t.limit for t in members)
        chunk = frames_list[pos:min(pos + size, end)]
        # Deletes/restores stay allowed during a run: re-read every chunk so a frame restored mid-run gets its mask
        # written starting from the next chunk.
        deleted = set(store.read_clip(clip.id).deleted_frames)
        frames = prefetch.take(anchor_idx, chunk)
        # Predict this group's next chunk and start reading it now (see ChunkPrefetcher: a wrong prediction -- an
        # OOM back-off, another group running next, an anchor that is not the chunk's last frame -- only costs a
        # rebuild from the frames the prefetcher still holds).
        after = frames_list[pos + size:min(pos + 2 * size, end)]
        if after:
            prefetch.schedule(chunk[-1], after)
        seeds = {t.obj_id: t.anchor_mask for t in members}
        covered: dict[int, list[int]] = {t.obj_id: [] for t in members}
        yielded = 0
        oom_error: ChunkOOMError | None = None
        try:
            with closing(engine.propagate(frames, seeds, cancel_cb=cancel)) as outputs:
                for offset, masks in outputs:
                    if offset == 0:
                        continue  # the anchor's own output is discarded
                    if offset != yielded + 1 or offset > len(chunk):
                        raise PropagationFailed(f"engine yielded offset {offset}, expected {yielded + 1}")
                    if set(masks) != set(seeds):
                        raise PropagationFailed(f"engine yielded objects {sorted(masks)}, expected {sorted(seeds)}")
                    yielded = offset
                    idx = chunk[offset - 1]
                    report = []
                    for t in members:
                        if not t.active:
                            continue  # bounded or lost earlier in this chunk: its outputs are ignored
                        mask = masks[t.obj_id]
                        t.pos += 1
                        covered[t.obj_id].append(idx)
                        if mask is None:
                            if store.delete_mask(clip.id, t.obj_id, idx):
                                result.cleared += 1
                            t.empties += 1
                            t.first_empty = idx if t.first_empty is None else t.first_empty
                        else:
                            t.empties, t.first_empty = 0, None
                            t.anchor_idx, t.anchor_mask = idx, mask
                            if idx not in deleted:
                                store.write_mask(clip.id, t.obj_id, idx, mask)
                                result.written += 1
                        report.append({"id": t.obj_id, "has_mask": mask is not None and idx not in deleted})
                        if t.empties >= lost_after:
                            t.finished = True
                            result.lost_by_direction.setdefault(direction, {})[t.obj_id] = t.first_empty
                    done = base + max(t.pos for t in tracks)
                    emit({
                        "type": "frame", "frame": idx, "objects": report,
                        "has_mask": any(r["has_mask"] for r in report),  # legacy shim: removed in plan B
                        "done": done, "total": total,
                    })
                    if not any(t.active for t in members):
                        break  # leaving the `with` closes the generator and frees the GPU lock
        except ChunkOOMError as e:
            oom_error = e
        finally:
            # Frames landed in this call must be recorded on EVERY exit -- normal end, lost, cancel, OOM, or any
            # other exception escaping the loop -- before it propagates.
            for oid, idxs in covered.items():
                if idxs:
                    _record_range(store, clip.id, oid, idxs)
        del frames

        if oom_error is not None:
            result.oom_retries += 1
            if size <= min_chunk_size:
                if yielded == 0:
                    raise PropagationFailed(
                        f"CUDA out of memory even at chunk size {size}; lower propagate.min_chunk_size "
                        "or free GPU memory (check nvidia-smi)"
                    ) from oom_error
            else:
                size = max(min_chunk_size, size // 2)
            emit({"type": "log", "message": f"out of memory; retrying with chunk size {size}"})
            oom_error = None  # drop the traceback so the failed chunk's tensors are freed before the retry
            continue

        # Cancellation is latched from whether the engine actually stopped short of the chunk while cancel() is
        # true -- not from re-polling cancel() after a call that completed (see the single-object history).
        if yielded < len(chunk) and any(t.active for t in members):
            if cancel():
                result.cancelled = True
                return size
            if yielded:
                raise PropagationFailed(
                    f"the engine stopped after {yielded} of {len(chunk)} frames without being cancelled"
                )
            raise PropagationFailed("the engine returned no frames for a non-empty chunk")


def run_propagation(
    store: ProjectStore,
    engine: Engine,
    clip_id: str,
    from_frame: int,
    direction: Literal["forward", "backward", "both"],
    *,
    objects: Literal["all"] | list[int] = "all",
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    emit: Emit | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PropagateResult:
    """Propagate `objects` (default: every object with a mask on `from_frame`) from `from_frame` in `direction`.

    (keep the existing paragraph about `emit` and `cancel` running under the GPU lock)
    """
    emit = emit or (lambda event: None)
    cancel = cancel or (lambda: False)
    clip = store.read_clip(clip_id)
    participants, skipped = resolve_participants(store, clip, from_frame, objects, strict=False)
    directions = ["forward", "backward"] if direction == "both" else [direction]
    plans = {d: {obj.id: plan_indices(clip, obj, from_frame, d) for obj, _ in participants} for d in directions}
    frames_lists = {d: max(plans[d].values(), key=len) for d in directions}

    result = PropagateResult(skipped=skipped, objects=[obj.id for obj, _ in participants])
    total = sum(len(f) for f in frames_lists.values())
    emit({"type": "progress", "done": 0, "total": total})
    size = chunk_size
    base = 0
    with ChunkPrefetcher(store, clip, engine) as prefetch:
        for d in directions:
            if result.cancelled:
                break
            tracks = [_Track(obj.id, len(plans[d][obj.id]), from_frame, seed) for obj, seed in participants]
            size = _run_direction(
                store, engine, clip, tracks, frames_lists[d], d,
                chunk_size=size, min_chunk_size=min_chunk_size, lost_after=lost_after,
                base=base, total=total, result=result, emit=emit, cancel=cancel, prefetch=prefetch,
            )
            covered = max((t.pos for t in tracks), default=0)
            result.processed += covered
            base += covered
    for d in directions:
        for oid, frame in result.lost_by_direction.get(d, {}).items():
            result.lost.setdefault(oid, frame)
    for d in directions:
        lost_here = result.lost_by_direction.get(d)
        if lost_here:
            result.lost_at = min(lost_here.values(), key=lambda f: abs(f - from_frame))
            break
    return result
```

`PropagateResult` becomes:

```python
@dataclass
class PropagateResult:
    processed: int = 0  # frames covered by at least one object
    written: int = 0  # masks written, over all objects
    cleared: int = 0
    lost_at: int | None = None  # the frame the page seeks to (spec §6)
    cancelled: bool = False
    oom_retries: int = 0
    lost_by_direction: dict[str, dict[int, int]] = field(default_factory=dict)
    lost: dict[int, int] = field(default_factory=dict)
    skipped: list[int] = field(default_factory=list)
    objects: list[int] = field(default_factory=list)
```

Delete `check_seed` and the old `_run_direction`. Imports: add `LabelObject` (already), `NotFound` from
`..store`. Update the module docstring: "Keyframe-capped propagation of one or more objects with anchor-prepended
chunks (spec 5.3, multi-object spec §6)...".

`api/labeling.py` `propagate` route, for now:

```python
    participants, _ = resolve_participants(state.store, clip, body.from_frame, "all", strict=True)
    directions = ["forward", "backward"] if body.direction == "both" else [body.direction]
    for d in directions:  # validate now so a bad request is a 400, not a failed job
        plan_indices(clip, participants[0][0], body.from_frame, d)
```

and drop its `if not clip.objects` block (resolve_participants raises the same message).

- [ ] **Step 4: Run** — `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server` → green. If an
  existing single-object test's exact numbers change, stop and report it (for one object the new runner must
  reproduce the old results exactly: chunk sizes, `propagate_calls`, writes, ranges, `lost_at`).

- [ ] **Step 5: Commit** — "Propagation: several objects per run, grouped by anchor frame".

---

### Task 4: Labelling operations take explicit object ids

**Files:**
- Modify: `server/vsp/labeling/prompts.py`, `server/vsp/labeling/paint.py`, `server/vsp/labeling/frames.py`
- Create: `server/vsp/labeling/objects.py`
- Test: `server/tests/test_prompts.py`, `server/tests/test_paint.py`, `server/tests/test_frames.py`,
  `server/tests/test_propagate.py` (move its `accept` calls), create `server/tests/test_objects.py`

**Interfaces:**
- Produces:
  - `preview(store, engine, clip, frame, prompt, *, object_id: int | None = None) -> list[Detection]`
  - `accept(store, engine, clip_id, frame, prompt, *, object_id: int | None, class_id: int | None = None, candidates: list[int] | None = None) -> tuple[Clip, list[int]]` (the clip and the ids written to)
  - `set_mask(store, clip_id, frame, mask, *, object_id: int | None, class_id: int | None = None) -> tuple[Clip, int]`
  - `clear_range(store, clip_id, start, end, *, objects: Literal["all"] | list[int] = "all") -> Clip`
  - `objects.find_object(clip, object_id) -> LabelObject` (NotFound), `relabel_object(store, clip_id, object_id, class_id) -> Clip`, `delete_object(store, clip_id, object_id) -> Clip`

Rules:
- `object_id=None` creates a new object and requires `class_id` (`BadRequest("class_id is required to create a new
  object")`); an int writes to that object (`NotFound(f"object {object_id} not found in clip {clip_id}")`) and
  `class_id` must then be None (`BadRequest("class_id cannot be given with object_id; relabel the object instead")`).
  An unknown `class_id` is `BadRequest(f"unknown class id {class_id}")` (as today).
- `accept` with `candidates` (only with `object_id=None`, else `BadRequest("candidates needs object_id null")`):
  each listed candidate (valid index, no duplicates) becomes its own new object of `class_id`, each with a keyframe
  on `frame` whose payload records its candidate index; returns their ids in order. Without `candidates`, the single
  `candidate` of today (keyword kept, default 0) is used.
- `_effective_box(store, clip, frame, prompt, object_id)`: the derived box comes from `object_id`'s mask on `frame`
  when `object_id` is not None and the prompt is clicks without a box; otherwise as today without a derived box.
  `accept` passes its `object_id` so preview and accept agree.
- `set_mask` with an empty mask and `object_id=None`: `BadRequest("the mask is empty; paint something first")`.
- `relabel_object`: clip lock; unknown class → BadRequest; writes the clip.
- `delete_object`: clip lock; removes the entry, then `store.delete_object_masks`; writes the clip. (The API layer
  adds the job checks.)
- `clear_range(objects=[...])`: unknown ids → NotFound; only those objects' masks, keyframes and ranges in the
  range are cleared.

- [ ] **Step 1: Write failing tests** — move existing calls to the new signatures (first accept on a clip:
  `object_id=None, class_id=0`; later: `object_id=OBJ`; tests that relabelled via `class_id` on a later accept now
  call `relabel_object`); `accept` returns `(clip, ids)` so existing `out = accept(...)` becomes
  `out, _ = accept(...)`; the same for `set_mask`. Then add to `test_objects.py`:

```python
import numpy as np
import pytest
from conftest import OBJ
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import BadRequest
from vsp.labeling.frames import clear_range
from vsp.labeling.objects import delete_object, relabel_object
from vsp.labeling.paint import set_mask
from vsp.labeling.prompts import PromptSpec, accept, preview
from vsp.store import NotFound

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])
TEXT = PromptSpec(type="text", phrase="part")


def test_a_second_object_gets_its_own_masks(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(classes=("cup", "bowl"))
    _, (a,) = accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    out, (b,) = accept(store, eng, clip.id, 0, PromptSpec(type="clicks", points=[(90, 30, 1)]), object_id=None, class_id=1)
    assert (a, b) == (1, 2) and [(o.id, o.class_id) for o in out.objects] == [(1, 0), (2, 1)]
    assert store.read_mask(clip.id, 1, 0)[30, 30] and not store.read_mask(clip.id, 1, 0)[30, 90]
    assert store.read_mask(clip.id, 2, 0)[30, 90]


def test_object_and_class_rules(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip()
    with pytest.raises(BadRequest, match="class_id is required"):
        accept(store, eng, clip.id, 0, CLICK, object_id=None)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    with pytest.raises(BadRequest, match="relabel the object instead"):
        accept(store, eng, clip.id, 1, CLICK, object_id=OBJ, class_id=0)
    with pytest.raises(NotFound, match="object 5"):
        accept(store, eng, clip.id, 1, CLICK, object_id=5)


def test_accept_all_candidates_as_new_objects(store, make_clip):
    eng = FakeEngine(n_text_candidates=3)
    _, clip = make_clip()
    out, ids = accept(store, eng, clip.id, 2, TEXT, object_id=None, class_id=0, candidates=[0, 2])
    assert ids == [1, 2] and [o.keyframes[0].prompt.payload["candidate"] for o in out.objects] == [0, 2]
    with pytest.raises(BadRequest, match="candidates needs object_id null"):
        accept(store, eng, clip.id, 2, TEXT, object_id=1, candidates=[1])


def test_click_box_is_derived_from_the_chosen_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip()
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)  # object 1 around (30, 30)
    set_mask(store, clip.id, 0, _rect(60, 120, x=70, w=30), object_id=None, class_id=0)  # object 2 at x 70..100
    clip = store.read_clip(clip.id)
    click = PromptSpec(type="clicks", points=[(80, 30, 1)])
    (det,) = preview(store, eng, clip, 0, click, object_id=2)
    assert det.mask[30, 95] and not det.mask[30, 25]  # box from object 2's mask, not object 1's
    (det,) = preview(store, eng, clip, 0, click, object_id=None)
    assert not det.mask[30, 95]  # no derived box: the fake's 20x20 square around the click


def test_relabel_and_delete_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(classes=("cup", "bowl"))
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    assert relabel_object(store, clip.id, 2, 1).objects[1].class_id == 1
    with pytest.raises(BadRequest, match="unknown class id 9"):
        relabel_object(store, clip.id, 2, 9)
    out = delete_object(store, clip.id, 1)
    assert [o.id for o in out.objects] == [2] and store.mask_indices(clip.id, 1) == []
    with pytest.raises(NotFound):
        delete_object(store, clip.id, 1)


def test_clear_range_per_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip()
    for _ in range(2):
        accept(store, eng, clip.id, 3, CLICK, object_id=None, class_id=0)
    clear_range(store, clip.id, 0, 5, objects=[2])
    assert store.mask_indices(clip.id, 1) == [3] and store.mask_indices(clip.id, 2) == []
    with pytest.raises(NotFound):
        clear_range(store, clip.id, 0, 5, objects=[7])
```

with a local `_rect(h, w, x, w_)` helper returning a bool mask with `[20:40, x:x + w_] = True` (check the frame size
from `make_clip`: 60×120). If `NotFound` is not exported by `vsp.store`, import it from where `prompts.py` gets it.

- [ ] **Step 2: Run to see them fail.**

- [ ] **Step 3: Implement** the signatures and rules above. `objects.py`:

```python
"""Object-level operations: find, relabel, delete (multi-object spec §7)."""

from __future__ import annotations

from ..store import Clip, LabelObject, NotFound, ProjectStore
from .errors import BadRequest


def find_object(clip: Clip, object_id: int) -> LabelObject:
    for obj in clip.objects:
        if obj.id == object_id:
            return obj
    raise NotFound(f"object {object_id} not found in clip {clip.id}")


def check_class(store: ProjectStore, clip: Clip, class_id: int) -> None:
    if class_id not in {c.id for c in store.read_project(clip.project_id).classes}:
        raise BadRequest(f"unknown class id {class_id}")


def relabel_object(store: ProjectStore, clip_id: str, object_id: int, class_id: int) -> Clip:
    store.clip_dir(clip_id)  # raises NotFound before the lock table grows an entry for a bogus id
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        obj = find_object(clip, object_id)
        check_class(store, clip, class_id)
        obj.class_id = class_id
        store.write_clip(clip)
        return clip


def delete_object(store: ProjectStore, clip_id: str, object_id: int) -> Clip:
    store.clip_dir(clip_id)
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        find_object(clip, object_id)
        clip.objects = [o for o in clip.objects if o.id != object_id]
        store.write_clip(clip)  # the entry goes first: a crash between the two leaves orphan files, not a
        store.delete_object_masks(clip_id, object_id)  # listed object without masks
        return clip
```

A shared helper `resolve_target(store, clip, object_id, class_id) -> LabelObject` in `objects.py` implements the
None/int + class rules and is used by both `accept` and `set_mask` (inside their clip lock). `accept`'s
`candidates` path computes the detections once, validates every index first (nothing written on a bad index), then
creates the objects and writes their masks and keyframes under one lock.

- [ ] **Step 4: Run** — full non-GPU suite and ruff → green. **The API still calls the old signatures**: update
  `api/labeling.py` call sites in this task just enough to compile and keep today's behaviour (the legacy shim
  lives in Task 5; here, pass `object_id=clip.objects[0].id if clip.objects else None` and move a given `class_id`
  on an existing object into a `relabel_object` call first). Mark those lines `# legacy shim: removed in plan B`.

- [ ] **Step 5: Commit** — "Labelling: explicit object ids for accept, paint, clear; relabel and delete objects".

---

### Task 5: HTTP API — object routes, per-object masks, propagate/clear bodies

**Files:**
- Modify: `server/vsp/api/labeling.py`, `server/vsp/api/clips.py`, `server/vsp/api/views.py`,
  `server/vsp/labeling/prompts.py` (remove `OBJECT_ID`)
- Test: `server/tests/test_api_labeling.py`, `server/tests/test_api_core.py`

**Interfaces (HTTP):**
- `POST /api/clips/{id}/prompt` body adds `object_id: int | null = null`.
- `POST /accept` body: `frame, prompt, object_id: int | null, class_id: int | null, candidate: int = 0,
  candidates: list[int] | null`. Response: `clip_detail(...)` plus `"object_ids": [...]`.
- `POST /mask` body: `frame, mask_png_b64, object_id: int | null, class_id: int | null`. Response adds
  `"object_id"`.
- Legacy shim 1 (both routes): `object_id` **absent** (`"object_id" not in body.model_fields_set`) → today's
  behaviour: first object, or create with `class_id`; a `class_id` on an existing object relabels it first.
- `PATCH /api/clips/{id}/objects/{object_id}` body `{class_id: int}` → `clip_detail`. Clip lock only (no GPU lease).
- `DELETE /api/clips/{id}/objects/{object_id}` → `clip_detail`. Inside `gpu_lease(state)`: if
  `state.jobs.active("export")` → `JobBusy`; then `delete_object`. (A running propagation makes `gpu_lease` itself
  fail with 409.)
- `GET /api/clips/{id}/objects/{object_id}/masks/{idx}` → the PNG, `Cache-Control: no-cache`; 404 for an unknown
  object or a missing mask.
- Legacy shim 2: `GET /api/clips/{id}/masks/{idx}` serves the first object's mask; 404 when the clip has none.
- `POST /propagate` body adds `objects: Literal["all"] | list[int] = "all"`; the route validates with
  `resolve_participants(..., strict=True)` and `plan_indices` for every participant and direction, and passes
  `objects` to `run_propagation`.
- `POST /clear-range` body adds `objects: Literal["all"] | list[int] = "all"`.
- `clip_detail` adds `"object_states": {str(o.id): store.object_states(clip, o) for o in clip.objects}`.

- [ ] **Step 1: Write failing API tests** (in `test_api_labeling.py`; reuse its `_client`/`_clip` helpers and
  `wait_job`):

```python
def test_objects_over_http(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        r = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        assert r.status_code == 200 and r.json()["object_ids"] == [1]
        r = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": {"type": "clicks", "points": [[90, 30, 1]]}, "object_id": None, "class_id": 0})
        assert r.json()["object_ids"] == [2]
        detail = r.json()
        assert set(detail["object_states"]) == {"1", "2"} and detail["object_states"]["2"][0] == "keyframe"
        assert client.get(f"/api/clips/{cid}/objects/2/masks/0").headers["content-type"] == "image/png"
        assert client.get(f"/api/clips/{cid}/objects/9/masks/0").status_code == 404
        r = client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK, "object_id": 1, "class_id": 0})
        assert r.status_code == 400
        assert client.patch(f"/api/clips/{cid}/objects/2", json={"class_id": 0}).status_code == 200
        r = client.delete(f"/api/clips/{cid}/objects/2")
        assert r.status_code == 200 and [o["id"] for o in r.json()["clip"]["objects"]] == [1]


def test_prompt_preview_uses_the_given_object(tmp_path):
    ...  # accept object 1 at (30,30), paint object 2 at x 70..100 via /mask with object_id null; /prompt a click at
         # (80,30) with object_id 2 -> the candidate's bbox spans past x 95; with object_id null it does not


def test_legacy_bodies_without_object_id_keep_one_object(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "class_id": 0})
        r = client.post(f"/api/clips/{cid}/accept", json={"frame": 3, "prompt": CLICK})
        assert [o["id"] for o in r.json()["clip"]["objects"]] == [1]
        assert client.get(f"/api/clips/{cid}/masks/3").status_code == 200


def test_propagate_objects_body_and_events(tmp_path):
    ...  # two objects seeded on frame 0; POST /propagate {"objects": [2]} -> job result objects == [2], only object 2
         # gets masks; {"objects": [9]} -> 404; {"objects": [1]} from a frame where object 1 has no mask -> 400;
         # a WebSocket/job event of type "frame" has "objects" and the legacy "has_mask"


def test_delete_object_is_refused_while_a_propagation_or_an_export_runs(tmp_path):
    ...  # SlowEngine propagation running -> DELETE object -> 409 with job_id; after it ends, start an export job
         # whose engine/exporter is slowed the way test_api_export.py already does it -> DELETE object -> 409;
         # PATCH during the propagation -> 200


def test_clear_range_by_object(tmp_path):
    ...  # two objects on frame 3; POST /clear-range {"start":0,"end":5,"objects":[2]} -> only object 2 cleared
```

The `...` bodies are for the implementer to write concretely following the neighbouring tests in the file (the
assertions are listed in each comment and are required). Move the existing `/masks/{idx}` assertions in
`test_api_labeling.py` and `test_api_core.py` to the per-object route, except one kept in
`test_legacy_bodies_without_object_id_keep_one_object` for shim 2.

- [ ] **Step 2: Run to see them fail.**

- [ ] **Step 3: Implement** the routes and bodies above; remove `OBJECT_ID` from `prompts.py` and its import in
  `api/clips.py` (the legacy mask route uses `clip.objects[0].id`). Error mapping: `NotFound` → 404, `BadRequest`
  → 400, `JobBusy` → 409 (already mapped in `app.py`; check).

- [ ] **Step 4: Run** — full non-GPU suite, ruff, and the web tests (`cd web && node node_modules/.bin/vitest run`)
  → green.

- [ ] **Step 5: Commit** — "API: object routes, per-object masks, propagate and clear-range by object".

---

### Task 6: Docs and end-to-end check

**Files:** Modify `README.md` (the API table / labelling notes and the Manual QA checklist).

- [ ] **Step 1:** In the README's API or labelling section, document: several objects per clip (server side), the
  object routes, the `objects` field on propagate and clear-range, the per-object mask route, and that the web page
  still shows one object until plan B. Add to the Manual QA list: "18. (server) With two objects on a clip,
  `POST /propagate` with `objects: \"all\"` writes both objects' masks in one run; with `[2]` only object 2's; an
  object that disappears is reported under `lost` while the other continues."
- [ ] **Step 2:** Run `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server`, the web tests, and
  `CUDA_VISIBLE_DEVICES=1 uv run pytest server/tests -q -m gpu` (report if the GPU is unavailable).
- [ ] **Step 3: Commit** — "Docs: several objects per clip (server)".
