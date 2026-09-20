# Vision Sampler — Plan 2 of 4: Labeling, Jobs and HTTP API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything the GUI will call: prompt preview/accept, keyframe-capped forward/backward propagation with OOM back-off and lost-object stop, frame cleanup, background jobs with a WebSocket event stream, and the FastAPI server started by `vsp serve`.

**Architecture:** `labeling/` orchestrates the existing `engine` and `store` and is pure Python (no HTTP). `jobs/` runs blocking work on threads and records an append-only event list per job. `api/` is thin FastAPI routers over both, built by `create_app(settings, engine=…)` so every test injects `FakeEngine` and runs without a GPU. One GPU job at a time; while it runs, prompt/accept/propagate return 409 instead of blocking on the engine lock.

**Tech Stack:** Python 3.11, uv, FastAPI + Starlette TestClient (httpx), uvicorn, Pydantic v2, OpenCV, pytest. Builds on Plan 1 (`vsp.store`, `vsp.ingest`, `vsp.engine`).

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md` — read §4 (HTTP API table), §5.2–§5.4, §9. **Also read** the "Carried forward from Plan 1" section of `docs/superpowers/plans/2026-09-20-vsp-roadmap.md`.

## Global Constraints

- Environment is managed **only with uv** from the repository root (`uv sync`, `uv run …`). No conda, no `requirements.txt`, no bare `pip install`.
- Pins stay: `torch==2.14.0+cu126`, `torchvision==0.29.0+cu126`, `transformers==5.17.0`. `pyproject.toml`/`uv.lock` change **only in Task 1** (adding the web dependencies).
- `vision_train/`, `data/`, `*.pt`, `*.safetensors` are never committed. Stage by explicit path; never `git add -A` / `git add .`.
- `store` is the only module that writes `data/projects/`. `labeling` never touches paths directly except reading a frame JPEG through `store.frame_path`.
- `engine` stays HTTP- and disk-agnostic. **Do not modify `server/vsp/engine/sam3.py` or `fake.py` in this plan.**
- `Engine.propagate()` is a generator that holds the GPU lock until exhausted or closed: **always** wrap it in `contextlib.closing`, cancel only through `cancel_cb`, and never call another engine method from the thread consuming it.
- Vision-cache keys are clip-scoped: `frame_key=(clip.id, frame_idx)`.
- The engine returns `None` for masks under 64 px or over 45 % of the frame; an absent mask PNG means "object not present". Never write an empty mask.
- Single object per clip (v1): the object id is always `1`; the rule is enforced in `labeling/prompts.py` and nowhere else.
- Keyframes are never overwritten by propagation. Propagation ranges stop at the neighbouring keyframe (exclusive) or the clip boundary.
- Deleted frames are fed to the tracker (temporal continuity) but their masks are not written.
- Every test in this plan runs without a GPU (uses `FakeEngine`), except the two GPU tests added in Task 2. GPU tests need `HF_HUB_OFFLINE=1`.
- The installed ruff's default rules include import sorting, pyupgrade and simplify: write sorted imports, `collections.abc` for `Callable`/`Iterator`, and FastAPI parameters in `Annotated[...]` form (no function calls in argument defaults). Purely cosmetic lint-driven edits to this plan's listings are allowed and must be listed in the report.
- The server binds `127.0.0.1` by default.
- Commit after every task with the message given. Use a heredoc so the message is multi-line, ending with your environment's `Co-Authored-By:` line and `Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL`.

## File Structure (this plan)

```
pyproject.toml, uv.lock                  + fastapi, uvicorn[standard], python-multipart; dev: httpx
server/vsp/store/project.py              + encode_mask_png() (public; write_mask uses it)
server/vsp/labeling/
  __init__.py
  errors.py        LabelingError, BadRequest, Conflict, PropagationFailed
  ranges.py        add_range(), subtract_range()  (inclusive int ranges; pure)
  prompts.py       PromptSpec, load_frame(), padded_box(), create_object(), preview(), accept()
  propagate.py     plan_indices(), PropagateResult, run_propagation()
  frames.py        delete_frame(), restore_frame(), prune_from(), clear_range()
server/vsp/jobs/
  __init__.py
  manager.py       Job, JobManager, JobBusy, JobNotFound
server/vsp/api/
  __init__.py
  app.py           AppState, create_app(), exception handlers, health, static mount
  deps.py          get_state(), require_engine(), require_gpu_idle()
  views.py         clip_detail(), clip_summary(), candidate_view()
  projects.py      /api/projects…
  clips.py         /api/clips… (detail, delete, frame/thumb/mask files, imports)
  jobs.py          /api/jobs/{id}, DELETE, WS /ws/jobs/{id}
  labeling.py      prompt, accept, propagate, frame cleanup routes
server/vsp/cli.py                         + `vsp serve`
server/tests/
  conftest.py, test_ranges.py, test_prompts.py, test_propagate.py, test_frames.py,
  test_jobs.py, test_api_core.py, test_api_labeling.py
  test_engine_gpu.py                      + one multi-click refinement test
```

Deviation from the spec's API table, decided here: classes are added with `POST /api/projects/{p}/classes` instead of `PATCH /api/projects/{p}` (one verb, one meaning). The spec's "second object → 409" row has no route that can trigger it in v1 (accept always targets the single object); the guard exists as `create_object()` and is unit-tested.

---

### Task 1: Dependencies, labeling errors, range helpers, mask PNG encoder, shared test fixtures

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `server/vsp/store/project.py`, `server/vsp/store/__init__.py`
- Create: `server/vsp/labeling/__init__.py`, `server/vsp/labeling/errors.py`, `server/vsp/labeling/ranges.py`, `server/tests/conftest.py`
- Test: `server/tests/test_ranges.py`, `server/tests/test_store.py` (one added test)

**Interfaces:**
- Consumes: `ProjectStore`, `import_clip`, `FrameDirSource` from Plan 1.
- Produces:
  - `vsp.labeling.errors`: `LabelingError(Exception)`, `BadRequest(LabelingError)`, `Conflict(LabelingError)`, `PropagationFailed(LabelingError)`
  - `vsp.labeling.ranges`: `add_range(ranges: list[tuple[int,int]], lo: int, hi: int) -> list[tuple[int,int]]`, `subtract_range(ranges, lo, hi) -> list[tuple[int,int]]` — ranges are **inclusive**, results sorted and non-overlapping, adjacent ranges merged by `add_range`.
  - `vsp.store.encode_mask_png(mask: np.ndarray) -> bytes` (1-bit PNG; raises `StoreError` on an empty mask)
  - pytest fixtures in `server/tests/conftest.py`: `store` (`ProjectStore` in `tmp_path`), `make_clip(n_frames=12, white=(), classes=("part",)) -> tuple[Project, Clip]` (frames are 120×60 grey JPEGs; indices in `white` are pure white), and the helper function `is_white(frame) -> bool`.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, extend `dependencies` with these three lines (keep the existing entries and order otherwise):

```toml
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",
```

and change the dev group to:

```toml
dev = ["pytest>=8", "ruff>=0.6", "httpx>=0.27"]
```

Run: `UV_HTTP_TIMEOUT=600 uv sync`
Expected: installs `fastapi`, `starlette`, `uvicorn`, `python-multipart`, `httpx`, `websockets` (and friends); `torch` stays `2.14.0+cu126`.

Run: `uv run python -c "import fastapi, uvicorn, httpx, multipart, torch; print(fastapi.__version__, torch.__version__)"`
Expected: a fastapi version `>= 0.115` and `2.14.0+cu126`.

- [ ] **Step 2: Write the failing tests**

`server/tests/test_ranges.py`:

```python
from vsp.labeling.ranges import add_range, subtract_range


def test_add_to_empty():
    assert add_range([], 3, 7) == [(3, 7)]


def test_add_merges_overlapping_and_adjacent():
    assert add_range([(1, 4)], 5, 9) == [(1, 9)]  # adjacent
    assert add_range([(1, 4), (10, 12)], 3, 10) == [(1, 12)]  # bridges both
    assert add_range([(1, 4)], 6, 9) == [(1, 4), (6, 9)]  # gap of one frame stays separate


def test_add_accepts_reversed_bounds():
    assert add_range([], 9, 5) == [(5, 9)]


def test_add_keeps_result_sorted():
    assert add_range([(20, 25)], 1, 3) == [(1, 3), (20, 25)]


def test_subtract_middle_splits():
    assert subtract_range([(1, 10)], 4, 6) == [(1, 3), (7, 10)]


def test_subtract_edges_and_whole():
    assert subtract_range([(1, 10)], 1, 3) == [(4, 10)]
    assert subtract_range([(1, 10)], 8, 99) == [(1, 7)]
    assert subtract_range([(1, 10)], 0, 99) == []


def test_subtract_untouched_and_multiple():
    assert subtract_range([(1, 3), (8, 9)], 5, 6) == [(1, 3), (8, 9)]
    assert subtract_range([(1, 5), (7, 12)], 4, 8) == [(1, 3), (9, 12)]
```

Append to `server/tests/test_store.py` (add `encode_mask_png` to its existing `from vsp.store import (...)` list):

```python
def test_encode_mask_png_matches_what_write_mask_stores(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    mask = np.zeros((48, 64), dtype=bool)
    mask[5:15, 8:20] = True
    data = encode_mask_png(mask)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    store.write_mask(clip.id, 1, 0, mask)
    assert store.mask_path(clip.id, 1, 0).read_bytes() == data
    with pytest.raises(StoreError, match="empty mask"):
        encode_mask_png(np.zeros((4, 4), dtype=bool))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_ranges.py server/tests/test_store.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.labeling'` and `ImportError: cannot import name 'encode_mask_png'`.

- [ ] **Step 4: Implement**

`server/vsp/labeling/__init__.py`: empty file.

`server/vsp/labeling/errors.py`:

```python
"""Errors raised by the labeling layer. The API maps them to HTTP status codes."""

from __future__ import annotations


class LabelingError(Exception):
    pass


class BadRequest(LabelingError):
    """The request cannot be carried out as stated (HTTP 400)."""


class Conflict(LabelingError):
    """The request conflicts with the clip's current state (HTTP 409)."""


class PropagationFailed(LabelingError):
    """Propagation could not complete (HTTP 500 / job error)."""
```

`server/vsp/labeling/ranges.py`:

```python
"""Inclusive integer ranges, kept sorted and non-overlapping. Pure functions."""

from __future__ import annotations

Range = tuple[int, int]


def add_range(ranges: list[Range], lo: int, hi: int) -> list[Range]:
    """Union with [lo, hi]; overlapping and adjacent ranges are merged."""
    if lo > hi:
        lo, hi = hi, lo
    out: list[Range] = []
    for a, b in sorted(ranges):
        if b < lo - 1 or a > hi + 1:
            out.append((a, b))
        else:
            lo, hi = min(lo, a), max(hi, b)
    out.append((lo, hi))
    return sorted(out)


def subtract_range(ranges: list[Range], lo: int, hi: int) -> list[Range]:
    """Remove [lo, hi] from every range."""
    if lo > hi:
        lo, hi = hi, lo
    out: list[Range] = []
    for a, b in sorted(ranges):
        if b < lo or a > hi:
            out.append((a, b))
            continue
        if a < lo:
            out.append((a, lo - 1))
        if b > hi:
            out.append((hi + 1, b))
    return out
```

In `server/vsp/store/project.py`, add this module-level function directly above `class ProjectStore:`:

```python
def encode_mask_png(mask: np.ndarray) -> bytes:
    """1-bit PNG bytes for a boolean mask. An empty mask is an error: absent file = not present."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise StoreError("refusing to write an empty mask; an absent file means 'not present'")
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).convert("1", dither=Image.Dither.NONE).save(buf, format="PNG")
    return buf.getvalue()
```

and replace the body of `ProjectStore.write_mask` with:

```python
    def write_mask(self, clip_id: str, object_id: int, idx: int, mask: np.ndarray) -> None:
        atomic_write_bytes(self.mask_path(clip_id, object_id, idx), encode_mask_png(mask))
```

In `server/vsp/store/__init__.py`, import `encode_mask_png` from `.project` and add `"encode_mask_png"` to `__all__` (keep `__all__` sorted).

`server/tests/conftest.py`:

```python
"""Shared fixtures. Frames are 120x60 so FakeEngine's 20x20 click masks (400 px) pass the area bounds."""

from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest
from vsp.ingest import FrameDirSource, import_clip
from vsp.store import ProjectStore

FRAME_W, FRAME_H = 120, 60


def is_white(frame: np.ndarray) -> bool:
    """FakeEngine `is_empty` hook: pure-white frames mean 'object not visible'."""
    return bool(frame[0, 0, 0] > 250)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "data")


@pytest.fixture
def make_clip(store, tmp_path):
    def _make(n_frames: int = 12, white=(), classes=("part",)):
        src = tmp_path / f"src_{uuid.uuid4().hex[:6]}"
        src.mkdir()
        for i in range(n_frames):
            value = 255 if i in white else 90
            img = np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)
            assert cv2.imwrite(str(src / f"{i:04d}.png"), img)
        project = store.create_project("t")
        for name in classes:
            store.add_class(project.id, name)
        clip = import_clip(store, project.id, FrameDirSource(src))
        return store.read_project(project.id), clip

    return _make
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_ranges.py server/tests/test_store.py -q`
Expected: `23 passed` (7 range tests + 16 store tests).

- [ ] **Step 6: Whole suite, lint, commit**

Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q`
Expected: `All checks passed!` and `79 passed` (71 + 7 + 1), GPU tests not skipped.

```bash
git add pyproject.toml uv.lock server/vsp/labeling server/vsp/store/project.py server/vsp/store/__init__.py \
        server/tests/conftest.py server/tests/test_ranges.py server/tests/test_store.py
git commit -F - <<'EOF'
feat(labeling): web dependencies, range helpers, public mask PNG encoder, shared fixtures

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL
EOF
```

---

### Task 2: Prompt preview and accept

**Files:**
- Create: `server/vsp/labeling/prompts.py`
- Test: `server/tests/test_prompts.py`; modify `server/tests/test_engine_gpu.py` (one added test)

**Interfaces:**
- Consumes: `ProjectStore` (`read_clip`, `read_project`, `write_clip`, `write_mask`, `read_mask`, `frame_path`, `lock`), `Clip`, `Keyframe`, `LabelObject`, `Prompt`, `NotFound`, `now_iso` from `vsp.store`; `Engine`, `Detection` from `vsp.engine.types`; `bbox_of` from `vsp.engine.masks`; `BadRequest`, `Conflict` from Task 1; fixtures `store`, `make_clip` from `conftest.py`.
- Produces (`vsp.labeling.prompts`):
  - `OBJECT_ID = 1`, `BOX_FROM_MASK_PAD = 10`
  - `PromptSpec(type: "text"|"clicks"|"box", phrase: str | None = None, points: list[tuple[int,int,int]] = [], box: tuple[int,int,int,int] | None = None)` — pydantic model; validation: text needs a non-blank phrase, clicks needs ≥ 1 point, box needs a box, point labels are 0 or 1. A `box` prompt may also carry refinement points.
  - `load_frame(store, clip: Clip, idx: int) -> np.ndarray` (BGR; `BadRequest` when out of range)
  - `padded_box(mask, pad: int, width: int, height: int) -> tuple[int,int,int,int]`
  - `create_object(clip: Clip, class_id: int) -> LabelObject` (raises `Conflict` if the clip already has an object)
  - `preview(store, engine, clip: Clip, frame: int, prompt: PromptSpec) -> list[Detection]` — persists nothing
  - `accept(store, engine, clip_id: str, frame: int, prompt: PromptSpec, *, class_id: int | None = None, candidate: int = 0) -> Clip`

Behaviour to implement (spec §5.2): the server is stateless between preview and accept — `accept` recomputes the mask from the prompt. A `clicks` prompt with no explicit box on a frame where the object already has a mask is a **correction**: the engine is called with a box derived from the existing mask (padded by 10 px, clamped to the image) plus the clicks, and the keyframe payload records `box_from_mask`. Accepting on a frame that already is a keyframe replaces that keyframe. The first accept needs `class_id`; later accepts may omit it (or pass one to re-class the object).

- [ ] **Step 1: Write the failing tests**

`server/tests/test_prompts.py`:

```python
import numpy as np
import pytest
from pydantic import ValidationError
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import BadRequest, Conflict
from vsp.labeling.prompts import OBJECT_ID, PromptSpec, accept, create_object, padded_box, preview

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])  # FakeEngine -> rect x 20:40, y 20:40 (400 px)


def test_promptspec_validation():
    with pytest.raises(ValidationError):
        PromptSpec(type="text", phrase="  ")
    with pytest.raises(ValidationError):
        PromptSpec(type="clicks")
    with pytest.raises(ValidationError):
        PromptSpec(type="box")
    with pytest.raises(ValidationError):
        PromptSpec(type="clicks", points=[(1, 1, 2)])
    assert PromptSpec(type="box", box=(1, 2, 30, 40), points=[(5, 5, 0)]).points == [(5, 5, 0)]


def test_preview_persists_nothing(store, make_clip):
    _, clip = make_clip()
    dets = preview(store, FakeEngine(), clip, 0, CLICK)
    assert len(dets) == 1 and dets[0].mask[30, 30]
    assert store.read_clip(clip.id).objects == []
    assert store.mask_indices(clip.id, OBJECT_ID) == []


def test_preview_text_returns_ranked_candidates(store, make_clip):
    _, clip = make_clip()
    dets = preview(store, FakeEngine(n_text_candidates=3), clip, 0, PromptSpec(type="text", phrase=" part "))
    assert len(dets) == 3 and dets[0].phrase == "part"


def test_first_accept_creates_object_keyframe_and_mask(store, make_clip):
    project, clip = make_clip()
    out = accept(store, FakeEngine(), clip.id, 3, CLICK, class_id=project.classes[0].id)
    assert [o.id for o in out.objects] == [OBJECT_ID]
    obj = out.objects[0]
    assert obj.class_id == project.classes[0].id
    assert [k.frame_idx for k in obj.keyframes] == [3]
    assert obj.keyframes[0].prompt.type == "clicks"
    assert obj.keyframes[0].prompt.payload["points"] == [[30, 30, 1]]
    assert "box_from_mask" not in obj.keyframes[0].prompt.payload
    mask = store.read_mask(clip.id, OBJECT_ID, 3)
    assert mask[30, 30] and mask.sum() == 400
    assert store.read_clip(clip.id).objects[0].keyframes[0].frame_idx == 3  # persisted


def test_first_accept_needs_a_valid_class(store, make_clip):
    _, clip = make_clip()
    with pytest.raises(BadRequest, match="class_id"):
        accept(store, FakeEngine(), clip.id, 0, CLICK)
    with pytest.raises(BadRequest, match="unknown class"):
        accept(store, FakeEngine(), clip.id, 0, CLICK, class_id=99)
    assert store.read_clip(clip.id).objects == []


def test_text_accept_uses_the_chosen_candidate(store, make_clip):
    project, clip = make_clip()
    eng = FakeEngine(n_text_candidates=3)
    out = accept(store, eng, clip.id, 0, PromptSpec(type="text", phrase="part"), class_id=0, candidate=1)
    mask = store.read_mask(clip.id, OBJECT_ID, 0)
    assert mask[10, 40] and not mask[10, 10]  # candidate 1 sits at x 35:55, candidate 0 at x 5:25
    assert out.objects[0].keyframes[0].prompt.payload["candidate"] == 1
    with pytest.raises(BadRequest, match="candidate"):
        accept(store, eng, clip.id, 1, PromptSpec(type="text", phrase="part"), candidate=7)


def test_later_accepts_add_sorted_keyframes_and_replace_same_frame(store, make_clip):
    _, clip = make_clip()
    eng = FakeEngine()
    accept(store, eng, clip.id, 8, CLICK, class_id=0)
    accept(store, eng, clip.id, 2, CLICK)  # class optional now
    out = accept(store, eng, clip.id, 8, PromptSpec(type="box", box=(50, 10, 90, 40)))
    assert [k.frame_idx for k in out.objects[0].keyframes] == [2, 8]
    assert out.objects[0].keyframes[1].prompt.type == "box"
    assert store.read_mask(clip.id, OBJECT_ID, 8)[20, 70]


def test_accept_can_reclass_the_object(store, make_clip):
    _, clip = make_clip(classes=("a", "b"))
    eng = FakeEngine()
    accept(store, eng, clip.id, 0, CLICK, class_id=0)
    assert accept(store, eng, clip.id, 1, CLICK, class_id=1).objects[0].class_id == 1


def test_correction_on_a_masked_frame_uses_a_box_from_the_existing_mask(store, make_clip):
    _, clip = make_clip()
    eng = FakeEngine()
    accept(store, eng, clip.id, 0, CLICK, class_id=0)  # mask x 20:40, y 20:40
    fix = PromptSpec(type="clicks", points=[(30, 30, 1), (22, 22, 0)])
    out = accept(store, eng, clip.id, 0, fix)
    payload = out.objects[0].keyframes[0].prompt.payload
    assert payload["box_from_mask"] == [10, 10, 50, 50]  # 20:40 padded by 10
    mask = store.read_mask(clip.id, OBJECT_ID, 0)
    assert mask[12, 12] and not mask[22, 22]  # FakeEngine fills the box, negative click carves a hole


def test_no_usable_mask_is_a_bad_request(store, make_clip):
    _, clip = make_clip()
    tiny = PromptSpec(type="box", box=(5, 5, 8, 8))  # 9 px < 64 px -> engine returns None
    assert preview(store, FakeEngine(), clip, 0, tiny) == []
    with pytest.raises(BadRequest, match="no usable mask"):
        accept(store, FakeEngine(), clip.id, 0, tiny, class_id=0)


def test_frame_out_of_range_and_deleted_frame(store, make_clip):
    _, clip = make_clip(n_frames=4)
    with pytest.raises(BadRequest, match="out of range"):
        preview(store, FakeEngine(), clip, 4, CLICK)
    clip.deleted_frames = [2]
    store.write_clip(clip)
    with pytest.raises(BadRequest, match="deleted"):
        accept(store, FakeEngine(), clip.id, 2, CLICK, class_id=0)


def test_single_object_guard(store, make_clip):
    _, clip = make_clip()
    create_object(clip, 0)
    with pytest.raises(Conflict, match="one object per clip"):
        create_object(clip, 0)


def test_padded_box_is_clamped():
    m = np.zeros((60, 120), dtype=bool)
    m[0:10, 100:120] = True
    assert padded_box(m, 10, 120, 60) == (90, 0, 120, 20)
```

Append to `server/tests/test_engine_gpu.py` (it already imports `numpy as np` and defines `_disc`, `_iou`, `R`, `H`, `W`, and the `engine`/`video` fixtures):

```python
def test_many_click_refinement_with_a_box_stays_accurate(engine, video):
    """Plan 1 verified box + 2 clicks only; the labeling layer sends a box-from-mask plus every click."""
    frames, truth = video
    cx, cy = 120, 200
    box = (cx - R - 10, cy - R - 10, cx + R + 10, cy + R + 10)
    points = [
        (cx, cy, 1), (cx - 20, cy, 1), (cx + 20, cy - 10, 1),
        (cx - R - 25, cy, 0), (cx + R + 25, cy, 0), (cx, cy + R + 25, 0),
    ]
    det = engine.predict_prompt(frames[0], box=box, points=points, frame_key=("t", 0))
    assert det is not None
    assert _iou(det.mask, truth[0]) > 0.85


def test_many_clicks_without_a_box_stay_accurate(engine, video):
    """The fresh-prompt path: several positive and negative clicks, no box at all (added at the user's request)."""
    frames, truth = video
    cx, cy = 120, 200
    points = [
        (cx, cy, 1), (cx - 20, cy, 1), (cx + 20, cy - 10, 1),
        (cx - R - 25, cy, 0), (cx + R + 25, cy, 0), (cx, cy + R + 25, 0),
    ]
    det = engine.predict_prompt(frames[0], points=points, frame_key=("t", 0))
    assert det is not None
    assert _iou(det.mask, truth[0]) > 0.85
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_prompts.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.labeling.prompts'`

- [ ] **Step 3: Implement**

`server/vsp/labeling/prompts.py`:

```python
"""Prompt preview and accept (spec 5.2). Stateless: accept recomputes the mask from the prompt."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..engine.masks import bbox_of
from ..engine.types import Detection, Engine
from ..store import Clip, Keyframe, LabelObject, NotFound, ProjectStore, Prompt, now_iso
from .errors import BadRequest, Conflict

OBJECT_ID = 1  # v1: one object per clip
BOX_FROM_MASK_PAD = 10


class PromptSpec(BaseModel):
    type: Literal["text", "clicks", "box"]
    phrase: str | None = None
    points: list[tuple[int, int, int]] = Field(default_factory=list)  # (x, y, label); 1 = positive, 0 = negative
    box: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2

    @model_validator(mode="after")
    def _check(self) -> PromptSpec:
        if self.type == "text" and not (self.phrase and self.phrase.strip()):
            raise ValueError("a text prompt needs a non-blank phrase")
        if self.type == "clicks" and not self.points:
            raise ValueError("a clicks prompt needs at least one point")
        if self.type == "box" and self.box is None:
            raise ValueError("a box prompt needs a box")
        if any(label not in (0, 1) for _, _, label in self.points):
            raise ValueError("point labels must be 0 (negative) or 1 (positive)")
        return self


def load_frame(store: ProjectStore, clip: Clip, idx: int) -> np.ndarray:
    if not 0 <= idx < clip.frame_count:
        raise BadRequest(f"frame {idx} is out of range 0..{clip.frame_count - 1}")
    img = cv2.imread(str(store.frame_path(clip.id, idx)), cv2.IMREAD_COLOR)
    if img is None:
        raise NotFound(f"frame {idx} of clip {clip.id} is missing on disk")
    return img


def padded_box(mask: np.ndarray, pad: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_of(mask)
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def create_object(clip: Clip, class_id: int) -> LabelObject:
    """The single place the v1 one-object-per-clip rule is enforced."""
    if clip.objects:
        raise Conflict("v1 supports one object per clip; import the footage again to label another component")
    obj = LabelObject(id=OBJECT_ID, class_id=class_id)
    clip.objects.append(obj)
    return obj


def _effective_box(
    store: ProjectStore, clip: Clip, frame: int, prompt: PromptSpec
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    """(box to send to the engine, box derived from the existing mask or None)."""
    if prompt.type == "clicks" and prompt.box is None and clip.objects:
        existing = store.read_mask(clip.id, clip.objects[0].id, frame)
        if existing is not None:
            derived = padded_box(existing, BOX_FROM_MASK_PAD, clip.width, clip.height)
            return derived, derived
    return prompt.box, None


def preview(store: ProjectStore, engine: Engine, clip: Clip, frame: int, prompt: PromptSpec) -> list[Detection]:
    image = load_frame(store, clip, frame)
    key = (clip.id, frame)
    if prompt.type == "text":
        return engine.predict_text(image, prompt.phrase.strip(), frame_key=key)
    box, _ = _effective_box(store, clip, frame, prompt)
    det = engine.predict_prompt(image, points=list(prompt.points), box=box, frame_key=key)
    return [det] if det is not None else []


def accept(
    store: ProjectStore,
    engine: Engine,
    clip_id: str,
    frame: int,
    prompt: PromptSpec,
    *,
    class_id: int | None = None,
    candidate: int = 0,
) -> Clip:
    clip = store.read_clip(clip_id)
    if frame in clip.deleted_frames:
        raise BadRequest(f"frame {frame} is deleted; restore it before labeling")
    _, derived = _effective_box(store, clip, frame, prompt)
    detections = preview(store, engine, clip, frame, prompt)
    if not detections:
        raise BadRequest("the prompt produced no usable mask (nothing found, or the mask is too small or too large)")
    if not 0 <= candidate < len(detections):
        raise BadRequest(f"candidate {candidate} does not exist; the prompt produced {len(detections)}")
    mask = detections[candidate].mask

    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        if clip.objects:
            obj = clip.objects[0]
        else:
            if class_id is None:
                raise BadRequest("class_id is required for the first accept on a clip")
            obj = None
        if class_id is not None:
            known = {c.id for c in store.read_project(clip.project_id).classes}
            if class_id not in known:
                raise BadRequest(f"unknown class id {class_id}")
        if obj is None:
            obj = create_object(clip, class_id)
        elif class_id is not None:
            obj.class_id = class_id
        store.write_mask(clip.id, obj.id, frame, mask)
        payload = prompt.model_dump(mode="json", exclude_none=True)
        payload["candidate"] = candidate
        if derived is not None:
            payload["box_from_mask"] = list(derived)
        keyframe = Keyframe(frame_idx=frame, prompt=Prompt(type=prompt.type, payload=payload), created_at=now_iso())
        obj.keyframes = sorted(
            [k for k in obj.keyframes if k.frame_idx != frame] + [keyframe], key=lambda k: k.frame_idx
        )
        store.write_clip(clip)
        return clip
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_prompts.py -q`
Expected: `13 passed`

Run: `HF_HUB_OFFLINE=1 uv run pytest server/tests/test_engine_gpu.py -q -k many_click`
Expected: `2 passed`. **If this fails**, that is a real finding about `multimask_max_pt_num` with many clicks: do not change the threshold, do not touch `sam3.py`; report BLOCKED with the measured IoU (print it) and the IoU for the same box with only the three positive clicks.

- [ ] **Step 5: Whole suite, lint, commit**

Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q`
Expected: clean; `94 passed`.

```bash
git add server/vsp/labeling/prompts.py server/tests/test_prompts.py server/tests/test_engine_gpu.py
git commit -F - <<'EOF'
feat(labeling): stateless prompt preview/accept with box-from-mask corrections

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL
EOF
```

---

### Task 3: Propagation runner

**Files:**
- Create: `server/vsp/labeling/propagate.py`
- Test: `server/tests/test_propagate.py`

**Interfaces:**
- Consumes: `load_frame`, `OBJECT_ID`, `accept`, `PromptSpec` (Task 2); `add_range` (Task 1); `BadRequest`, `PropagationFailed` (Task 1); `ChunkOOMError`, `Engine` from `vsp.engine.types`; `ProjectStore`, `Clip`, `LabelObject`; fixtures `store`, `make_clip`, helper `is_white`.
- Produces (`vsp.labeling.propagate`):
  - `plan_indices(clip: Clip, obj: LabelObject, from_frame: int, direction: "forward"|"backward") -> list[int]` — ordered frame indices to process, excluding `from_frame`, stopping before the neighbouring keyframe; includes deleted frames; `BadRequest` if `from_frame` is not a keyframe.
  - `PropagateResult` dataclass: `processed: int = 0`, `written: int = 0`, `cleared: int = 0`, `lost_at: int | None = None`, `cancelled: bool = False`, `oom_retries: int = 0`; method `as_dict() -> dict`.
  - `run_propagation(store, engine, clip_id: str, from_frame: int, direction: "forward"|"backward"|"both", *, chunk_size: int, min_chunk_size: int, lost_after: int, emit: Callable[[dict], None] | None = None, cancel: Callable[[], bool] | None = None) -> PropagateResult`
  - Events passed to `emit`: `{"type": "progress", "done": int, "total": int}` once at the start; `{"type": "frame", "frame": int, "has_mask": bool, "done": int, "total": int}` per processed frame; `{"type": "log", "message": str}` on an OOM retry.

Algorithm (spec §5.3) — **anchor-prepended chunks**: every engine call gets `[anchor frame] + next N frames`, seeded with the anchor's mask at offset 0; the anchor's own output is discarded. The anchor starts as the keyframe and advances to the most recent frame that produced a real mask (kept **in memory**, so it also works across deleted frames, whose masks are not written). A `None` result deletes any stale propagated mask on that frame and counts towards `lost_after`; reaching it stops the run with `lost_at` = first frame of the empty streak. `ChunkOOMError` halves the chunk size (floor `min_chunk_size`) and retries from the first unprocessed frame; at the floor it raises `PropagationFailed`. `clip.json` (`propagated_ranges`) is updated once per chunk, under the clip lock.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_propagate.py`:

```python
import numpy as np
import pytest
from conftest import is_white
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import BadRequest, PropagationFailed
from vsp.labeling.prompts import OBJECT_ID, PromptSpec, accept
from vsp.labeling.propagate import plan_indices, run_propagation

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])  # seed rect x 20:40; FakeEngine shifts +2 px per offset
KW = {"chunk_size": 8, "min_chunk_size": 4, "lost_after": 3}


def _x0(store, clip_id, idx):
    mask = store.read_mask(clip_id, OBJECT_ID, idx)
    return None if mask is None else int(np.where(mask)[1].min())


def _seeded(store, make_clip, eng, n_frames=30, keyframes=(0,), white=()):
    _, clip = make_clip(n_frames=n_frames, white=white)
    for i, kf in enumerate(keyframes):
        accept(store, eng, clip.id, kf, CLICK, class_id=0 if i == 0 else None)
    return clip


def test_plan_indices_are_capped_at_neighbouring_keyframes(store, make_clip):
    clip = _seeded(store, make_clip, FakeEngine(), n_frames=20, keyframes=(5, 12))
    clip = store.read_clip(clip.id)
    obj = clip.objects[0]
    assert plan_indices(clip, obj, 5, "forward") == list(range(6, 12))
    assert plan_indices(clip, obj, 12, "forward") == list(range(13, 20))
    assert plan_indices(clip, obj, 12, "backward") == list(range(11, 5, -1))
    assert plan_indices(clip, obj, 5, "backward") == [4, 3, 2, 1, 0]
    with pytest.raises(BadRequest, match="not a keyframe"):
        plan_indices(clip, obj, 7, "forward")


def test_forward_writes_every_frame_in_anchor_prepended_chunks(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    events = []
    res = run_propagation(store, eng, clip.id, 0, "forward", emit=events.append, **KW)
    assert (res.processed, res.written, res.lost_at, res.cancelled) == (29, 29, None, False)
    assert store.mask_indices(clip.id, OBJECT_ID) == list(range(30))
    assert _x0(store, clip.id, 7) == 20 + 2 * 7  # continuous motion across the 8-frame chunk seam...
    assert _x0(store, clip.id, 9) == 20 + 2 * 9  # ...because each chunk is seeded by the previous frame's mask
    assert eng.propagate_calls == [9, 9, 9, 6]  # 29 frames in chunks of 8, each +1 anchor frame
    assert events[0] == {"type": "progress", "done": 0, "total": 29}
    frames = [e["frame"] for e in events if e["type"] == "frame"]
    assert frames == list(range(1, 30)) and events[-1]["done"] == 29
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 29)]


def test_keyframes_are_never_overwritten(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=20, keyframes=(0, 10))
    before = store.mask_path(clip.id, OBJECT_ID, 10).read_bytes()
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 9
    assert store.mask_path(clip.id, OBJECT_ID, 10).read_bytes() == before
    assert _x0(store, clip.id, 10) == 20  # still the accepted mask, not seed + 20
    assert store.mask_indices(clip.id, OBJECT_ID) == list(range(11))


def test_backward_and_both(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=16, keyframes=(8,))
    res = run_propagation(store, eng, clip.id, 8, "backward", **KW)
    assert res.processed == 8 and _x0(store, clip.id, 5) == 20 + 2 * 3  # 3 frames away from the keyframe
    res = run_propagation(store, eng, clip.id, 8, "both", **KW)
    assert res.processed == 15
    assert store.mask_indices(clip.id, OBJECT_ID) == list(range(16))
    # the keyframe itself is not a propagated frame, so the two passes stay separate ranges
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(0, 7), (9, 15)]


def test_not_a_keyframe_and_nothing_to_propagate(store, make_clip):
    eng = FakeEngine()
    _, empty = make_clip()
    with pytest.raises(BadRequest, match="accept a prompt"):
        run_propagation(store, eng, empty.id, 0, "forward", **KW)
    clip = _seeded(store, make_clip, eng)
    with pytest.raises(BadRequest, match="not a keyframe"):
        run_propagation(store, eng, clip.id, 3, "forward", **KW)


def test_deleted_frames_feed_the_tracker_but_are_not_written(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=12)
    c = store.read_clip(clip.id)
    c.deleted_frames = [5]
    store.write_clip(c)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 11 and res.written == 10
    assert 5 not in store.mask_indices(clip.id, OBJECT_ID)
    assert _x0(store, clip.id, 6) == 20 + 2 * 6  # continuity kept through the deleted frame


def test_lost_object_stops_early_and_clears_stale_masks(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=30, white=range(6, 30))
    stale = np.zeros((60, 120), dtype=bool)
    stale[0:20, 0:20] = True
    store.write_mask(clip.id, OBJECT_ID, 7, stale)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.lost_at == 6 and res.processed == 8 and res.cleared == 1  # frames 1..8; 6, 7, 8 empty
    assert store.mask_indices(clip.id, OBJECT_ID) == [0, 1, 2, 3, 4, 5]


def test_anchor_skips_empty_frames_at_a_chunk_seam(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=10, white=(4, 5))
    res = run_propagation(store, eng, clip.id, 0, "forward", chunk_size=4, min_chunk_size=4, lost_after=5)
    assert res.lost_at is None and res.processed == 9
    assert store.mask_indices(clip.id, OBJECT_ID) == [0, 1, 2, 3, 6, 7, 8, 9]
    # chunk 2 is [anchor=3, 5, 6, 7, 8]: frame 6 is offset 2 from the frame-3 mask (x0 = 26) -> 26 + 4
    assert _x0(store, clip.id, 6) == 30


def test_oom_halves_the_chunk_and_continues(store, make_clip):
    eng = FakeEngine(oom_if_more_than=5)
    clip = _seeded(store, make_clip, eng, n_frames=13)
    events = []
    res = run_propagation(store, eng, clip.id, 0, "forward", emit=events.append, **KW)
    assert res.oom_retries == 1 and res.processed == 12
    assert eng.propagate_calls == [9, 5, 5, 5]  # 8+1 fails, then chunks of 4 (+1 anchor)
    assert any(e["type"] == "log" and "out of memory" in e["message"] for e in events)


def test_oom_at_the_floor_fails(store, make_clip):
    eng = FakeEngine(oom_if_more_than=2)
    clip = _seeded(store, make_clip, eng, n_frames=13)
    with pytest.raises(PropagationFailed, match="chunk size 4"):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)


def test_cancel_keeps_landed_masks(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    seen = []
    res = run_propagation(
        store, eng, clip.id, 0, "forward",
        emit=lambda e: seen.append(e) if e["type"] == "frame" else None,
        cancel=lambda: len(seen) >= 3, **KW,
    )
    assert res.cancelled and res.processed == 3
    assert store.mask_indices(clip.id, OBJECT_ID) == [0, 1, 2, 3]
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 3)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_propagate.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.labeling.propagate'`

- [ ] **Step 3: Implement**

`server/vsp/labeling/propagate.py`:

```python
"""Keyframe-capped propagation with anchor-prepended chunks (spec 5.3)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from ..engine.types import ChunkOOMError, Engine
from ..store import Clip, LabelObject, ProjectStore
from .errors import BadRequest, PropagationFailed
from .prompts import load_frame
from .ranges import add_range

Emit = Callable[[dict], None]


@dataclass
class PropagateResult:
    processed: int = 0
    written: int = 0
    cleared: int = 0
    lost_at: int | None = None
    cancelled: bool = False
    oom_retries: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def plan_indices(clip: Clip, obj: LabelObject, from_frame: int, direction: Literal["forward", "backward"]) -> list[int]:
    keyframes = sorted(k.frame_idx for k in obj.keyframes)
    if from_frame not in keyframes:
        raise BadRequest(f"frame {from_frame} is not a keyframe; accept a prompt there first")
    if direction == "forward":
        stop = min((k for k in keyframes if k > from_frame), default=clip.frame_count)
        return list(range(from_frame + 1, stop))
    stop = max((k for k in keyframes if k < from_frame), default=-1)
    return list(range(from_frame - 1, stop, -1))


def _record_range(store: ProjectStore, clip_id: str, indices: list[int]) -> None:
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        obj = clip.objects[0]
        obj.propagated_ranges = add_range(obj.propagated_ranges, min(indices), max(indices))
        store.write_clip(clip)


def _run_direction(
    store: ProjectStore,
    engine: Engine,
    clip: Clip,
    object_id: int,
    from_frame: int,
    seed: np.ndarray,
    indices: list[int],
    *,
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    total: int,
    result: PropagateResult,
    emit: Emit,
    cancel: Callable[[], bool],
) -> None:
    deleted = set(clip.deleted_frames)
    anchor_idx, anchor_mask = from_frame, seed
    pending = list(indices)
    size = max(min_chunk_size, chunk_size)
    empties, first_empty = 0, None

    while pending and not result.cancelled and result.lost_at is None:
        chunk = pending[:size]
        frames = [load_frame(store, clip, anchor_idx)] + [load_frame(store, clip, i) for i in chunk]
        done_in_chunk: list[int] = []
        oom = False
        try:
            with closing(engine.propagate(frames, anchor_mask, cancel_cb=cancel)) as masks:
                for offset, mask in masks:
                    if offset == 0:
                        continue  # the anchor's own output is discarded
                    idx = chunk[offset - 1]
                    done_in_chunk.append(idx)
                    result.processed += 1
                    if mask is None:
                        if store.delete_mask(clip.id, object_id, idx):
                            result.cleared += 1
                        empties += 1
                        first_empty = idx if first_empty is None else first_empty
                    else:
                        empties, first_empty = 0, None
                        anchor_idx, anchor_mask = idx, mask
                        if idx not in deleted:
                            store.write_mask(clip.id, object_id, idx, mask)
                            result.written += 1
                    emit({
                        "type": "frame", "frame": idx,
                        "has_mask": mask is not None and idx not in deleted,
                        "done": result.processed, "total": total,
                    })
                    if empties >= lost_after:
                        result.lost_at = first_empty
                        break  # leaving the `with` closes the generator and frees the GPU lock
        except ChunkOOMError as e:
            oom = True
            if size <= min_chunk_size:
                raise PropagationFailed(
                    f"CUDA out of memory even at chunk size {size}; lower propagate.min_chunk_size "
                    "or free GPU memory (check nvidia-smi)"
                ) from e
            size = max(min_chunk_size, size // 2)
            result.oom_retries += 1
            emit({"type": "log", "message": f"out of memory; retrying with chunk size {size}"})

        if done_in_chunk:
            _record_range(store, clip.id, done_in_chunk)
            pending = pending[len(done_in_chunk):]
        if cancel():
            result.cancelled = True
        elif not done_in_chunk and not oom:
            raise PropagationFailed("the engine returned no frames for a non-empty chunk")


def run_propagation(
    store: ProjectStore,
    engine: Engine,
    clip_id: str,
    from_frame: int,
    direction: Literal["forward", "backward", "both"],
    *,
    chunk_size: int,
    min_chunk_size: int,
    lost_after: int,
    emit: Emit | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PropagateResult:
    emit = emit or (lambda event: None)
    cancel = cancel or (lambda: False)
    clip = store.read_clip(clip_id)
    if not clip.objects:
        raise BadRequest("nothing to propagate: accept a prompt first")
    obj = clip.objects[0]
    directions = ["forward", "backward"] if direction == "both" else [direction]
    plans = [plan_indices(clip, obj, from_frame, d) for d in directions]
    seed = store.read_mask(clip_id, obj.id, from_frame)
    if seed is None:
        raise BadRequest(f"keyframe {from_frame} has no mask on disk")

    result = PropagateResult()
    total = sum(len(p) for p in plans)
    emit({"type": "progress", "done": 0, "total": total})
    for indices in plans:
        if result.cancelled:
            break
        result.lost_at = None  # a lost forward pass must not suppress the backward pass
        _run_direction(
            store, engine, clip, obj.id, from_frame, seed, indices,
            chunk_size=chunk_size, min_chunk_size=min_chunk_size, lost_after=lost_after,
            total=total, result=result, emit=emit, cancel=cancel,
        )
    return result
```

Note on `direction="both"`: `lost_at` reports the last direction that lost the object (backward wins if both do). The per-frame events carry the full story.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_propagate.py -q`
Expected: `11 passed`. If `test_lost_object…` or `test_anchor_skips…` fails on the white-frame detection, check that frames read back from JPEG are still `> 250` at `[0, 0, 0]` (they are flat white, so they should be) — fix the fixture, not the assertion.

- [ ] **Step 5: Whole suite, lint, commit**

Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q`
Expected: clean; `105 passed`.

```bash
git add server/vsp/labeling/propagate.py server/tests/test_propagate.py
git commit -F - <<'EOF'
feat(labeling): keyframe-capped propagation with anchor-prepended chunks, OOM back-off, lost-object stop

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL
EOF
```

---

### Task 4: Frame cleanup operations

**Files:**
- Create: `server/vsp/labeling/frames.py`
- Test: `server/tests/test_frames.py`

**Interfaces:**
- Consumes: `ProjectStore`, `Clip`; `subtract_range` (Task 1); `BadRequest` (Task 1); `accept`, `PromptSpec`, `OBJECT_ID` (Task 2); `run_propagation` (Task 3) in tests.
- Produces (`vsp.labeling.frames`), each returning the updated, persisted `Clip` and validating indices (`BadRequest` when out of range):
  - `delete_frame(store, clip_id, idx)`, `restore_frame(store, clip_id, idx)` — idempotent; `deleted_frames` stays sorted and unique; **no files are removed**.
  - `prune_from(store, clip_id, from_idx)` — marks `from_idx..frame_count-1` deleted.
  - `clear_range(store, clip_id, start, end)` — inclusive; removes the object's mask PNGs on **non-keyframes** in the range and subtracts the range from `propagated_ranges`; keyframes and their masks are untouched.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_frames.py`:

```python
import pytest
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import BadRequest
from vsp.labeling.frames import clear_range, delete_frame, prune_from, restore_frame
from vsp.labeling.prompts import OBJECT_ID, PromptSpec, accept
from vsp.labeling.propagate import run_propagation

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])


def test_delete_and_restore_are_idempotent_and_keep_files(store, make_clip):
    _, clip = make_clip(n_frames=6)
    delete_frame(store, clip.id, 4)
    delete_frame(store, clip.id, 1)
    out = delete_frame(store, clip.id, 4)
    assert out.deleted_frames == [1, 4]
    assert store.frame_path(clip.id, 4).is_file()
    assert store.frame_states(out)[4] == "deleted"
    out = restore_frame(store, clip.id, 4)
    assert restore_frame(store, clip.id, 4).deleted_frames == [1] == out.deleted_frames
    assert store.read_clip(clip.id).deleted_frames == [1]


def test_prune_from_marks_the_tail(store, make_clip):
    _, clip = make_clip(n_frames=6)
    delete_frame(store, clip.id, 0)
    assert prune_from(store, clip.id, 3).deleted_frames == [0, 3, 4, 5]


def test_index_validation(store, make_clip):
    _, clip = make_clip(n_frames=6)
    for call in (
        lambda: delete_frame(store, clip.id, 6),
        lambda: restore_frame(store, clip.id, -1),
        lambda: prune_from(store, clip.id, 9),
        lambda: clear_range(store, clip.id, 2, 6),
        lambda: clear_range(store, clip.id, 4, 2),
    ):
        with pytest.raises(BadRequest):
            call()


def test_clear_range_spares_keyframes_and_trims_ranges(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=12)
    accept(store, eng, clip.id, 0, CLICK, class_id=0)
    accept(store, eng, clip.id, 6, CLICK)
    run_propagation(store, eng, clip.id, 0, "forward", chunk_size=8, min_chunk_size=4, lost_after=3)
    run_propagation(store, eng, clip.id, 6, "forward", chunk_size=8, min_chunk_size=4, lost_after=3)
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 5), (7, 11)]
    out = clear_range(store, clip.id, 4, 8)
    assert store.mask_indices(clip.id, OBJECT_ID) == [0, 1, 2, 3, 6, 9, 10, 11]  # keyframe 6 kept
    assert out.objects[0].propagated_ranges == [(1, 3), (9, 11)]
    assert [k.frame_idx for k in out.objects[0].keyframes] == [0, 6]


def test_clear_range_without_an_object_is_a_no_op(store, make_clip):
    _, clip = make_clip(n_frames=6)
    assert clear_range(store, clip.id, 0, 5).objects == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_frames.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.labeling.frames'`

- [ ] **Step 3: Implement**

`server/vsp/labeling/frames.py`:

```python
"""Frame cleanup (spec 5.4). Deleting a frame only flags it; no file is ever removed."""

from __future__ import annotations

from ..store import Clip, ProjectStore
from .errors import BadRequest
from .ranges import subtract_range


def _check(clip: Clip, idx: int) -> None:
    if not 0 <= idx < clip.frame_count:
        raise BadRequest(f"frame {idx} is out of range 0..{clip.frame_count - 1}")


def _set_deleted(store: ProjectStore, clip_id: str, indices: range, deleted: bool) -> Clip:
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        for idx in (indices.start, indices.stop - 1):
            _check(clip, idx)
        current = set(clip.deleted_frames)
        current = current | set(indices) if deleted else current - set(indices)
        clip.deleted_frames = sorted(current)
        store.write_clip(clip)
        return clip


def delete_frame(store: ProjectStore, clip_id: str, idx: int) -> Clip:
    return _set_deleted(store, clip_id, range(idx, idx + 1), True)


def restore_frame(store: ProjectStore, clip_id: str, idx: int) -> Clip:
    return _set_deleted(store, clip_id, range(idx, idx + 1), False)


def prune_from(store: ProjectStore, clip_id: str, from_idx: int) -> Clip:
    clip = store.read_clip(clip_id)
    _check(clip, from_idx)
    return _set_deleted(store, clip_id, range(from_idx, clip.frame_count), True)


def clear_range(store: ProjectStore, clip_id: str, start: int, end: int) -> Clip:
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        _check(clip, start)
        _check(clip, end)
        if start > end:
            raise BadRequest(f"start {start} is after end {end}")
        for obj in clip.objects:
            keyframes = {k.frame_idx for k in obj.keyframes}
            for idx in range(start, end + 1):
                if idx not in keyframes:
                    store.delete_mask(clip.id, obj.id, idx)
            obj.propagated_ranges = subtract_range(obj.propagated_ranges, start, end)
        store.write_clip(clip)
        return clip
```

- [ ] **Step 4: Run tests, whole suite, lint, commit**

Run: `uv run pytest server/tests/test_frames.py -q` → `5 passed`
Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q` → clean; `110 passed`.

```bash
git add server/vsp/labeling/frames.py server/tests/test_frames.py
git commit -F - <<'EOF'
feat(labeling): delete/restore/prune frames and clear-range that spares keyframes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL
EOF
```

---

### Task 5: Job manager

**Files:**
- Create: `server/vsp/jobs/__init__.py`, `server/vsp/jobs/manager.py`
- Test: `server/tests/test_jobs.py`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces (`from vsp.jobs import …`):
  - `TERMINAL = frozenset({"done", "error", "cancelled"})`
  - `JobBusy(Exception)` with attribute `job_id: str`; `JobNotFound(KeyError)`
  - `Job`: attributes `id: str`, `kind: str`, `exclusive: str | None`, `status: "pending"|"running"|"done"|"error"|"cancelled"`, `result: dict | None`, `error: str | None`, `created_at: str`; methods `emit(event: dict) -> None` (appends a copy with `"seq"` = its index), `events_since(n: int) -> list[dict]`, `request_cancel() -> None`, `cancel_requested() -> bool`, `join(timeout: float | None = None) -> None`, `snapshot() -> dict` (`id, kind, status, result, error, created_at, n_events`).
  - `JobManager`: `start(kind: str, fn: Callable[[Job], dict | None], *, exclusive: str | None = None) -> Job` (raises `JobBusy` when a non-terminal job holds the same `exclusive` key), `get(job_id) -> Job` (raises `JobNotFound`), `cancel(job_id) -> Job`, `running(exclusive: str) -> str | None`.
  - Ordering guarantee relied on by the WebSocket route: the final event (`{"type": "done" | "cancelled", **result}` or `{"type": "error", "message": …}`) is appended **before** `status` becomes terminal.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_jobs.py`:

```python
import threading

import pytest
from vsp.jobs import TERMINAL, JobBusy, JobManager, JobNotFound


def test_job_runs_and_finishes_with_result_and_events():
    jobs = JobManager()

    def work(job):
        job.emit({"type": "progress", "done": 1})
        return {"clip_id": "abc"}

    job = jobs.start("import", work)
    job.join(5)
    assert job.status == "done" and job.result == {"clip_id": "abc"} and job.error is None
    events = job.events_since(0)
    assert [e["type"] for e in events] == ["progress", "done"]
    assert [e["seq"] for e in events] == [0, 1]
    assert events[-1]["clip_id"] == "abc"
    assert job.events_since(1) == events[1:]
    snap = job.snapshot()
    assert snap["status"] == "done" and snap["n_events"] == 2 and snap["kind"] == "import"
    assert jobs.get(job.id) is job


def test_exception_becomes_error_status():
    jobs = JobManager()

    def boom(job):
        raise ValueError("bad video")

    job = jobs.start("import", boom)
    job.join(5)
    assert job.status == "error" and job.error == "ValueError: bad video"
    assert job.events_since(0)[-1] == {"type": "error", "message": "ValueError: bad video", "seq": 0}


def test_cancel_is_cooperative():
    jobs = JobManager()
    started = threading.Event()

    def work(job):
        started.set()
        while not job.cancel_requested():
            threading.Event().wait(0.01)
        return {"processed": 3}

    job = jobs.start("propagate", work, exclusive="gpu")
    assert started.wait(5)
    assert jobs.running("gpu") == job.id
    jobs.cancel(job.id)
    job.join(5)
    assert job.status == "cancelled" and job.result == {"processed": 3}
    assert job.events_since(0)[-1]["type"] == "cancelled"
    assert jobs.running("gpu") is None


def test_exclusive_key_allows_one_job_at_a_time():
    jobs = JobManager()
    release = threading.Event()
    first = jobs.start("propagate", lambda job: release.wait(5) and None, exclusive="gpu")
    with pytest.raises(JobBusy) as busy:
        jobs.start("propagate", lambda job: None, exclusive="gpu")
    assert busy.value.job_id == first.id
    other = jobs.start("import", lambda job: None)  # no exclusive key -> not blocked
    other.join(5)
    release.set()
    first.join(5)
    assert first.status in TERMINAL
    jobs.start("propagate", lambda job: None, exclusive="gpu").join(5)  # free again


def test_final_event_is_visible_before_terminal_status():
    jobs = JobManager()
    job = jobs.start("x", lambda job: {"n": 1})
    job.join(5)
    # whenever status is terminal, the last event must already be the terminal one
    assert job.status in TERMINAL and job.events_since(0)[-1]["type"] == job.status


def test_unknown_job():
    with pytest.raises(JobNotFound):
        JobManager().get("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_jobs.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.jobs'`

- [ ] **Step 3: Implement**

`server/vsp/jobs/manager.py`:

```python
"""Thread-backed background jobs with an append-only event list (spec: start / poll / cancel / stream)."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

TERMINAL = frozenset({"done", "error", "cancelled"})


class JobBusy(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id} is still running")
        self.job_id = job_id


class JobNotFound(KeyError):
    pass


class Job:
    def __init__(self, kind: str, exclusive: str | None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.exclusive = exclusive
        self.status = "pending"
        self.result: dict | None = None
        self.error: str | None = None
        self.created_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def emit(self, event: dict) -> None:
        with self._lock:
            self._events.append({**event, "seq": len(self._events)})

    def events_since(self, n: int) -> list[dict]:
        with self._lock:
            return list(self._events[n:])

    def request_cancel(self) -> None:
        self._cancel.set()

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            n_events = len(self._events)
        return {
            "id": self.id, "kind": self.kind, "status": self.status, "result": self.result,
            "error": self.error, "created_at": self.created_at, "n_events": n_events,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Job], dict | None], *, exclusive: str | None = None) -> Job:
        with self._lock:
            if exclusive is not None:
                for other in self._jobs.values():
                    if other.exclusive == exclusive and other.status not in TERMINAL:
                        raise JobBusy(other.id)
            job = Job(kind, exclusive)
            self._jobs[job.id] = job
        job._thread = threading.Thread(target=self._run, args=(job, fn), name=f"job-{kind}-{job.id}", daemon=True)
        job._thread.start()
        return job

    @staticmethod
    def _run(job: Job, fn: Callable[[Job], dict | None]) -> None:
        job.status = "running"
        try:
            result = fn(job) or {}
        except Exception as e:  # a job must never take the server down
            job.error = f"{type(e).__name__}: {e}"
            job.emit({"type": "error", "message": job.error})
            job.status = "error"
            return
        job.result = result
        final = "cancelled" if job.cancel_requested() else "done"
        job.emit({"type": final, **result})  # event first, status second: streamers rely on this order
        job.status = final

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.request_cancel()
        return job

    def running(self, exclusive: str) -> str | None:
        with self._lock:
            for job in self._jobs.values():
                if job.exclusive == exclusive and job.status not in TERMINAL:
                    return job.id
        return None
```

`server/vsp/jobs/__init__.py`:

```python
from .manager import TERMINAL, Job, JobBusy, JobManager, JobNotFound

__all__ = ["TERMINAL", "Job", "JobBusy", "JobManager", "JobNotFound"]
```

- [ ] **Step 4: Run tests, whole suite, lint, commit**

Run: `uv run pytest server/tests/test_jobs.py -q` → `6 passed`
Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q` → clean; `116 passed`.

```bash
git add server/vsp/jobs server/tests/test_jobs.py
git commit -F - <<'EOF'
feat(jobs): thread-backed JobManager with event log, cooperative cancel and exclusive keys

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL
EOF
```

---

### Task 6: API core — app factory, health, projects, clips, imports, jobs, `vsp serve`

**Files:**
- Create: `server/vsp/api/__init__.py`, `server/vsp/api/app.py`, `server/vsp/api/deps.py`, `server/vsp/api/views.py`, `server/vsp/api/projects.py`, `server/vsp/api/clips.py`, `server/vsp/api/jobs.py`
- Modify: `server/vsp/cli.py` (add `serve`)
- Test: `server/tests/test_api_core.py`

**Interfaces:**
- Consumes: `Settings`, `REPO_ROOT` (`vsp.config`); `ProjectStore`, `NotFound`, `StoreError`, `encode_mask_png`; `import_clip`, `FrameDirSource`, `VideoFileSource`, `IngestError`, `resolve_allowed`; `Engine`, `EngineNotLoaded`; `JobManager`, `JobBusy`, `JobNotFound`, `TERMINAL`; `BadRequest`, `Conflict`, `PropagationFailed`; `OBJECT_ID`.
- Produces:
  - `vsp.api.app.AppState(settings, store, engine, jobs)` dataclass and `create_app(settings: Settings, *, engine: Engine | None = None, load_engine: bool = True) -> FastAPI`. With `engine=None` a `Sam3Engine` is built from `settings.sam3`. With `load_engine=True` the engine's `load()` runs on a background thread at startup (failures are swallowed; they are visible through `/api/health`).
  - `vsp.api.deps`: `get_state(request) -> AppState`, `require_engine(state) -> Engine` (raises `EngineNotLoaded`), `require_gpu_idle(state) -> None` (raises `JobBusy`), and the alias `State = Annotated[AppState, Depends(get_state)]`.
  - `vsp.api.views`: `clip_summary(state, clip) -> dict`, `clip_detail(state, clip) -> dict` (`{"clip": …, "states": [...], "classes": [...]}`), `candidate_view(det) -> dict`.
  - Error mapping (JSON body `{"detail": str}`): `NotFound`/`JobNotFound` → 404; `StoreError`, `IngestError`, `BadRequest` → 400; `Conflict` → 409; `JobBusy` → 409 with extra key `job_id`; `EngineNotLoaded` → 503; `PropagationFailed` → 500.
  - Routes: `GET /api/health`; `GET|POST /api/projects`; `GET /api/projects/{p}`; `POST /api/projects/{p}/classes`; `GET /api/projects/{p}/clips`; `POST /api/projects/{p}/clips/import-frames`; `POST /api/projects/{p}/clips/import-video`; `GET|DELETE /api/clips/{c}`; `GET /api/clips/{c}/frames/{i}`; `GET /api/clips/{c}/thumbs/{i}`; `GET /api/clips/{c}/masks/{i}`; `GET|DELETE /api/jobs/{j}`; `WS /ws/jobs/{j}`.
  - CLI: `vsp serve [--host H] [--port N]`.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_api_core.py`:

```python
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine
from vsp.engine.types import EngineNotLoaded


def _settings(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    s.ingest.allowed_roots = [str(tmp_path)]
    return s


def wait_job(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error", "cancelled"):
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {snap}")


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        yield c


def _frames_dir(tmp_path, n=5):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(n):
        assert cv2.imwrite(str(src / f"f_{i}.png"), np.full((60, 120, 3), 90, np.uint8))
    return src


def _project(client, classes=("part",)):
    p = client.post("/api/projects", json={"name": "Board"}).json()
    for name in classes:
        assert client.post(f"/api/projects/{p['id']}/classes", json={"name": name}).status_code == 201
    return p["id"]


def test_health_with_a_loaded_engine(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["engine"] == {"loaded": True, "device": None, "error": None}


def test_health_reports_a_background_load_failure(tmp_path):
    class Broken:
        device = None
        load_error = None

        def is_loaded(self):
            return False

        def load(self):
            self.load_error = "no CUDA device"
            raise EngineNotLoaded(self.load_error)

    with TestClient(create_app(_settings(tmp_path), engine=Broken(), load_engine=True)) as c:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and c.get("/api/health").json()["engine"]["error"] is None:
            time.sleep(0.02)
        assert c.get("/api/health").json()["engine"] == {"loaded": False, "device": None, "error": "no CUDA device"}


def test_projects_and_classes(client):
    assert client.get("/api/projects").json() == []
    pid = _project(client, classes=("connector", "capacitor"))
    got = client.get(f"/api/projects/{pid}").json()
    assert [c["name"] for c in got["classes"]] == ["connector", "capacitor"]
    assert [p["id"] for p in client.get("/api/projects").json()] == [pid]
    dup = client.post(f"/api/projects/{pid}/classes", json={"name": "connector"})
    assert dup.status_code == 400 and "already exists" in dup.json()["detail"]
    assert client.get("/api/projects/nope-000000").status_code == 404
    assert client.get("/api/projects/..%2Fx").status_code in (400, 404)


def test_import_frames_then_browse_the_clip(client, tmp_path):
    pid = _project(client)
    src = _frames_dir(tmp_path)
    r = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(src), "fps": 12, "stride": 2})
    assert r.status_code == 202
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done"
    clip_id = snap["result"]["clip_id"]

    listing = client.get(f"/api/projects/{pid}/clips").json()
    assert [c["id"] for c in listing] == [clip_id]
    assert listing[0]["frame_count"] == 3 and listing[0]["labeled_frames"] == 0

    detail = client.get(f"/api/clips/{clip_id}").json()
    assert detail["clip"]["fps"] == 12 and detail["states"] == ["empty"] * 3
    assert [c["name"] for c in detail["classes"]] == ["part"]

    frame = client.get(f"/api/clips/{clip_id}/frames/0")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/clips/{clip_id}/thumbs/0").status_code == 200
    assert client.get(f"/api/clips/{clip_id}/frames/3").status_code == 404
    assert client.get(f"/api/clips/{clip_id}/masks/0").status_code == 404

    assert client.delete(f"/api/clips/{clip_id}").status_code == 204
    assert client.get(f"/api/clips/{clip_id}").status_code == 404


def test_import_frames_is_confined_to_allowed_roots(client, tmp_path):
    pid = _project(client)
    outside = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": "/etc"})
    assert outside.status_code == 400 and "outside" in outside.json()["detail"]
    missing = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(tmp_path / "nope")})
    assert missing.status_code == 400
    assert client.post("/api/projects/nope-000000/clips/import-frames", json={"path": str(tmp_path)}).status_code == 404


def test_import_video_upload(client, tmp_path):
    pid = _project(client)
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (120, 60))
    for _ in range(10):
        writer.write(np.full((60, 120, 3), 90, np.uint8))
    writer.release()
    with path.open("rb") as fh:
        r = client.post(
            f"/api/projects/{pid}/clips/import-video", files={"file": ("bench.avi", fh, "video/avi")}, data={"stride": "2"}
        )
    assert r.status_code == 202
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done"
    detail = client.get(f"/api/clips/{snap['result']['clip_id']}").json()
    assert detail["clip"]["frame_count"] == 5 and detail["clip"]["source"] == {"type": "video", "name": "bench.avi"}
    assert list((tmp_path / "data" / "tmp").iterdir()) == []  # the upload was removed after import


def test_a_bad_upload_becomes_a_job_error(client):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-video", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "error" and "cannot open" in snap["error"]


def test_job_endpoints_and_websocket_replay(client, tmp_path):
    assert client.get("/api/jobs/nope").status_code == 404
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(_frames_dir(tmp_path))})
    job_id = r.json()["job_id"]
    wait_job(client, job_id)
    events = []
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        try:
            while True:
                events.append(ws.receive_json())
        except WebSocketDisconnect:
            pass
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[-1]["type"] == "done" and events[0]["type"] == "progress"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_api_core.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.api'`

- [ ] **Step 3: Implement deps and views**

`server/vsp/api/__init__.py`: empty file.

`server/vsp/api/deps.py`:

```python
"""Request-scoped access to the application state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Request

from ..config import Settings
from ..engine.types import EngineNotLoaded
from ..jobs import JobBusy, JobManager
from ..store import ProjectStore

GPU = "gpu"  # exclusive key: one GPU job at a time


@dataclass
class AppState:
    settings: Settings
    store: ProjectStore
    engine: Any  # an Engine; Sam3Engine additionally has load(), device, load_error
    jobs: JobManager


def get_state(request: Request) -> AppState:
    return request.app.state.vsp


State = Annotated[AppState, Depends(get_state)]


def require_engine(state: AppState):
    if not state.engine.is_loaded():
        reason = getattr(state.engine, "load_error", None) or "the SAM3 engine is still loading"
        raise EngineNotLoaded(reason)
    return state.engine


def require_gpu_idle(state: AppState) -> None:
    """Fail fast with 409 instead of blocking a worker thread on the engine lock."""
    running = state.jobs.running(GPU)
    if running is not None:
        raise JobBusy(running)
```

`server/vsp/api/views.py`:

```python
"""JSON shapes shared by several routers."""

from __future__ import annotations

import base64

from ..engine.types import Detection
from ..store import Clip, encode_mask_png
from .deps import AppState


def clip_summary(state: AppState, clip: Clip) -> dict:
    states = state.store.frame_states(clip)
    body = clip.model_dump(mode="json", exclude={"original_names"})
    body["labeled_frames"] = sum(1 for s in states if s in ("masked", "keyframe"))
    return body


def clip_detail(state: AppState, clip: Clip) -> dict:
    project = state.store.read_project(clip.project_id)
    return {
        "clip": clip.model_dump(mode="json"),
        "states": state.store.frame_states(clip),
        "classes": [c.model_dump(mode="json") for c in project.classes],
    }


def candidate_view(det: Detection) -> dict:
    return {
        "score": round(float(det.score), 4),
        "bbox": list(det.bbox),
        "area": int(det.mask.sum()),
        "phrase": det.phrase,
        "mask_png_b64": base64.b64encode(encode_mask_png(det.mask)).decode("ascii"),
    }
```

- [ ] **Step 4: Implement the routers**

`server/vsp/api/projects.py`:

```python
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .deps import State
from .views import clip_summary

router = APIRouter(prefix="/api/projects", tags=["projects"])


class NameBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@router.get("")
def list_projects(state: State) -> list[dict]:
    return [p.model_dump(mode="json") for p in state.store.list_projects()]


@router.post("", status_code=201)
def create_project(body: NameBody, state: State) -> dict:
    return state.store.create_project(body.name.strip()).model_dump(mode="json")


@router.get("/{project_id}")
def get_project(project_id: str, state: State) -> dict:
    return state.store.read_project(project_id).model_dump(mode="json")


@router.post("/{project_id}/classes", status_code=201)
def add_class(project_id: str, body: NameBody, state: State) -> dict:
    return state.store.add_class(project_id, body.name).model_dump(mode="json")


@router.get("/{project_id}/clips")
def list_clips(project_id: str, state: State) -> list[dict]:
    state.store.read_project(project_id)
    return [clip_summary(state, c) for c in state.store.list_clips(project_id)]
```

`server/vsp/api/clips.py`:

```python
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..ingest import FrameDirSource, IngestError, VideoFileSource, import_clip, resolve_allowed
from ..jobs import Job
from ..labeling.prompts import OBJECT_ID
from ..store import NotFound
from .deps import AppState, State
from .views import clip_detail

router = APIRouter(tags=["clips"])


class ImportFramesBody(BaseModel):
    path: str
    fps: float = Field(default=30.0, gt=0)
    stride: int = Field(default=1, ge=1)


def _start_import(state: AppState, project_id: str, source, stride: int, cleanup: Path | None = None) -> dict:
    cfg = state.settings.ingest

    def work(job: Job) -> dict:
        try:
            clip = import_clip(
                state.store, project_id, source, stride=stride, jpeg_quality=cfg.jpeg_quality,
                thumb_width=cfg.thumb_width, progress=lambda n: job.emit({"type": "progress", "done": n}),
            )
            return {"clip_id": clip.id, "frame_count": clip.frame_count}
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

    return {"job_id": state.jobs.start("import", work).id}


@router.post("/api/projects/{project_id}/clips/import-frames", status_code=202)
def import_frames(project_id: str, body: ImportFramesBody, state: State) -> dict:
    state.store.read_project(project_id)
    path = resolve_allowed(body.path, state.settings.ingest.allowed_roots)
    return _start_import(state, project_id, FrameDirSource(path, fps=body.fps), body.stride)


@router.post("/api/projects/{project_id}/clips/import-video", status_code=202)
def import_video(
    project_id: str,
    state: State,
    file: Annotated[UploadFile, File()],
    stride: Annotated[int, Form(ge=1)] = 1,
) -> dict:
    state.store.read_project(project_id)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename or "upload").name)[:80] or "upload"
    tmp_dir = state.settings.project.data_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}_{name}"
    with tmp.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        source = VideoFileSource(tmp, name=name)
    except IngestError as e:
        # Report an unreadable upload as a failed job so the UI has one code path, and remove the upload.
        tmp.unlink(missing_ok=True)
        error = IngestError(str(e))  # `e` is unbound once this block ends; the job runs later on another thread

        def fail(job: Job) -> dict:
            raise error

        return {"job_id": state.jobs.start("import", fail).id}
    return _start_import(state, project_id, source, stride, cleanup=tmp)


@router.get("/api/clips/{clip_id}")
def get_clip(clip_id: str, state: State) -> dict:
    return clip_detail(state, state.store.read_clip(clip_id))


@router.delete("/api/clips/{clip_id}", status_code=204)
def delete_clip(clip_id: str, state: State) -> Response:
    state.store.delete_clip(clip_id)
    return Response(status_code=204)


def _image(path: Path, media_type: str) -> FileResponse:
    if not path.is_file():
        raise NotFound(f"{path.name} not found")
    return FileResponse(path, media_type=media_type)


@router.get("/api/clips/{clip_id}/frames/{idx}")
def get_frame(clip_id: str, idx: int, state: State) -> FileResponse:
    return _image(state.store.frame_path(clip_id, idx), "image/jpeg")


@router.get("/api/clips/{clip_id}/thumbs/{idx}")
def get_thumb(clip_id: str, idx: int, state: State) -> FileResponse:
    return _image(state.store.thumb_path(clip_id, idx), "image/jpeg")


@router.get("/api/clips/{clip_id}/masks/{idx}")
def get_mask(clip_id: str, idx: int, state: State) -> FileResponse:
    return _image(state.store.mask_path(clip_id, OBJECT_ID, idx), "image/png")
```

Design note: the uploaded video is probed inside the request (cheap), and a probe failure is turned into a *failed job* rather than a 400, so the UI handles every import outcome through the job stream.

`server/vsp/api/jobs.py`:

```python
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket

from ..jobs import TERMINAL, JobNotFound
from .deps import State

router = APIRouter(tags=["jobs"])
_POLL_SECONDS = 0.05


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, state: State) -> dict:
    return state.jobs.get(job_id).snapshot()


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str, state: State) -> dict:
    return state.jobs.cancel(job_id).snapshot()


@router.websocket("/ws/jobs/{job_id}")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        job = websocket.app.state.vsp.jobs.get(job_id)
    except JobNotFound:
        await websocket.close(code=4404, reason="unknown job")
        return
    sent = 0
    while True:
        terminal = job.status in TERMINAL  # read status BEFORE events: the final event precedes the status flip
        for event in job.events_since(sent):
            await websocket.send_json(event)
            sent += 1
        if terminal:
            break
        await asyncio.sleep(_POLL_SECONDS)
    await websocket.close()
```

- [ ] **Step 5: Implement the app factory**

`server/vsp/api/app.py`:

```python
"""FastAPI application factory. Tests inject FakeEngine; `vsp serve` builds the real Sam3Engine."""

from __future__ import annotations

import contextlib
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import REPO_ROOT, Settings
from ..engine.types import EngineNotLoaded
from ..ingest import IngestError
from ..jobs import JobBusy, JobManager, JobNotFound
from ..labeling.errors import BadRequest, Conflict, PropagationFailed
from ..store import NotFound, ProjectStore, StoreError
from . import clips, jobs, projects
from .deps import AppState, State

log = logging.getLogger(__name__)

_STATUS = [  # most specific first
    (NotFound, 404), (JobNotFound, 404), (Conflict, 409), (EngineNotLoaded, 503), (PropagationFailed, 500),
    (BadRequest, 400), (IngestError, 400), (StoreError, 400),
]


def _build_engine(settings: Settings):
    from ..engine.sam3 import Sam3Engine  # imports torch; keep it out of test collection paths

    cfg = settings.sam3
    return Sam3Engine(
        cfg.model_dir, device=cfg.device, dtype=cfg.dtype,
        text_threshold=cfg.text_threshold, score_mode=cfg.score_mode,
    )


def create_app(settings: Settings, *, engine=None, load_engine: bool = True) -> FastAPI:
    state = AppState(
        settings=settings,
        store=ProjectStore(settings.project.data_root),
        engine=engine if engine is not None else _build_engine(settings),
        jobs=JobManager(),
    )

    def _load() -> None:
        with contextlib.suppress(EngineNotLoaded):  # the reason is kept in engine.load_error for /api/health
            state.engine.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_engine and hasattr(state.engine, "load"):
            threading.Thread(target=_load, name="engine-load", daemon=True).start()
        yield

    app = FastAPI(title="vision-sampler", version=__version__, lifespan=lifespan)
    app.state.vsp = state

    def _handler(status: int):
        async def handle(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc.args[0]) if exc.args else str(exc)})

        return handle

    for exc_type, status in _STATUS:
        app.add_exception_handler(exc_type, _handler(status))

    @app.exception_handler(JobBusy)
    async def _busy(request: Request, exc: JobBusy) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "job_id": exc.job_id})

    @app.get("/api/health")
    def health(st: State) -> dict:
        eng = st.engine
        return {
            "status": "ok",
            "version": __version__,
            "engine": {
                "loaded": bool(eng.is_loaded()),
                "device": getattr(eng, "device", None),
                "error": getattr(eng, "load_error", None),
            },
        }

    app.include_router(projects.router)
    app.include_router(clips.router)
    app.include_router(jobs.router)

    dist = REPO_ROOT / "web" / "dist"  # built by Plan 3; absent until then
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="spa")
    return app
```

- [ ] **Step 6: Add `vsp serve`**

In `server/vsp/cli.py`: after the `convert-weights` parser definition add

```python
    p_serve = sub.add_parser("serve", help="run the HTTP API (and the web UI when web/dist exists)")
    p_serve.add_argument("--host", default=None, help="bind address (default: server.host from the config)")
    p_serve.add_argument("--port", type=int, default=None, help="port (default: server.port from the config)")
```

and, before the final `parser.error(...)` line, add

```python
    if args.cmd == "serve":
        return _serve(settings, args.host, args.port)
```

and add this function above `main`:

```python
def _serve(settings: Settings, host: str | None, port: int | None) -> int:
    import uvicorn

    from .api.app import create_app

    host = host or settings.server.host
    port = port or settings.server.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding {host} exposes an unauthenticated labeling server to the network")
    print(f"data_root: {settings.project.data_root}")
    print(f"model_dir: {settings.sam3.model_dir}")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
    return 0
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_api_core.py -q`
Expected: `8 passed`, no warnings.

- [ ] **Step 8: Smoke-test the real server once**

Run (it loads the real model in the background; stop it with Ctrl-C / `kill`):

```bash
HF_HUB_OFFLINE=1 uv run vsp serve --port 28765 &
SERVER=$!
# --noproxy: this machine exports http_proxy; never send localhost traffic through it
for i in $(seq 1 60); do curl -s --noproxy '*' http://127.0.0.1:28765/api/health | grep -q '"loaded":true' && break; sleep 1; done
curl -s --noproxy '*' http://127.0.0.1:28765/api/health
kill $SERVER
```

Expected final line: `{"status":"ok","version":"0.1.0","engine":{"loaded":true,"device":"cuda:0","error":null}}`.
Record the output in your report. If `loaded` never becomes true, include the `error` field in the report and stop.

- [ ] **Step 9: Whole suite, lint, commit**

Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q` → clean; `124 passed`.

```bash
git add server/vsp/api server/vsp/cli.py server/tests/test_api_core.py
git commit -F - <<'EOF'
feat(api): app factory, health, projects, clips, imports, job polling + WebSocket, vsp serve

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL
EOF
```

---

### Task 7: API labeling routes and the full-flow test

**Files:**
- Create: `server/vsp/api/labeling.py`
- Modify: `server/vsp/api/app.py` (include the router)
- Test: `server/tests/test_api_labeling.py`
- Modify: `README.md` (add a "Run the server" section)

**Interfaces:**
- Consumes: `State`, `AppState`, `GPU`, `require_engine`, `require_gpu_idle` (`vsp.api.deps`); `clip_detail`, `candidate_view` (`vsp.api.views`); `PromptSpec`, `preview`, `accept` (Task 2); `run_propagation` (Task 3); `delete_frame`, `restore_frame`, `prune_from`, `clear_range` (Task 4); `Job` (Task 5).
- Produces these routes (all return the `clip_detail` shape unless noted):
  - `POST /api/clips/{c}/prompt` body `{"frame": int, "prompt": PromptSpec}` → `{"candidates": [candidate_view…]}`; 503 if the engine is not loaded; 409 (`job_id`) while a GPU job runs.
  - `POST /api/clips/{c}/accept` body `{"frame", "prompt", "class_id": int | null, "candidate": int = 0}`; same 503/409 rules.
  - `POST /api/clips/{c}/propagate` body `{"from_frame": int, "direction": "forward"|"backward"|"both"}` → 202 `{"job_id"}`; validates synchronously (a non-keyframe is a 400, not a failed job); exclusive key `GPU`.
  - `POST /api/clips/{c}/frames/{i}/delete`, `POST /api/clips/{c}/frames/{i}/restore`, `POST /api/clips/{c}/prune` body `{"from": int}`, `POST /api/clips/{c}/clear-range` body `{"start": int, "end": int}`. Delete/restore/prune are allowed while a GPU job runs (the runner refreshes `deleted_frames` per chunk); **clear-range returns 409 with `job_id` while a GPU job runs**, because it deletes masks and trims `propagated_ranges` that the runner is writing (ruling from the Task 3 review).
  - Propagate job result also carries `lost_by_direction` (added in the Task 3 fix round).
  - Propagate job result: `PropagateResult.as_dict()`; events as defined in Task 3.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_api_labeling.py`:

```python
import base64
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect
from test_api_core import _frames_dir, _project, _settings, wait_job
from vsp.api.app import create_app
from vsp.engine.fake import FakeEngine

CLICK = {"type": "clicks", "points": [[30, 30, 1]]}


class SlowEngine(FakeEngine):
    def propagate(self, frames, seed_mask, cancel_cb=None):
        for item in super().propagate(frames, seed_mask, cancel_cb):
            time.sleep(0.03)
            yield item


def _client(tmp_path, engine):
    return TestClient(create_app(_settings(tmp_path), engine=engine, load_engine=False))


def _clip(client, tmp_path, n=20):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-frames", json={"path": str(_frames_dir(tmp_path, n))})
    return wait_job(client, r.json()["job_id"])["result"]["clip_id"]


def _mask(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")) > 127


def test_full_labeling_flow(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)

        cand = client.post(f"/api/clips/{cid}/prompt", json={"frame": 5, "prompt": CLICK}).json()["candidates"]
        assert len(cand) == 1 and cand[0]["area"] == 400 and _mask(cand[0]["mask_png_b64"])[30, 30]
        assert client.get(f"/api/clips/{cid}").json()["states"][5] == "empty"  # preview persisted nothing

        first = client.post(f"/api/clips/{cid}/accept", json={"frame": 5, "prompt": CLICK, "class_id": 0})
        assert first.status_code == 200 and first.json()["states"][5] == "keyframe"
        assert client.get(f"/api/clips/{cid}/masks/5").headers["content-type"] == "image/png"

        job = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 5, "direction": "both"})
        assert job.status_code == 202
        snap = wait_job(client, job.json()["job_id"])
        assert snap["status"] == "done" and snap["result"]["processed"] == 19 and snap["result"]["lost_at"] is None
        states = client.get(f"/api/clips/{cid}").json()["states"]
        assert states.count("masked") == 19 and states[5] == "keyframe"

        fixed = client.post(
            f"/api/clips/{cid}/accept",
            json={"frame": 12, "prompt": {"type": "clicks", "points": [[44, 30, 1], [40, 22, 0]]}},
        ).json()
        kf = [k for k in fixed["clip"]["objects"][0]["keyframes"] if k["frame_idx"] == 12][0]
        assert "box_from_mask" in kf["prompt"]["payload"]  # a correction, not a fresh prompt
        again = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 12, "direction": "forward"})
        assert wait_job(client, again.json()["job_id"])["result"]["processed"] == 7  # 13..19

        assert client.post(f"/api/clips/{cid}/frames/3/delete").json()["states"][3] == "deleted"
        assert client.post(f"/api/clips/{cid}/frames/3/restore").json()["states"][3] == "masked"
        assert client.post(f"/api/clips/{cid}/prune", json={"from": 18}).json()["states"][18:] == ["deleted"] * 2
        cleared = client.post(f"/api/clips/{cid}/clear-range", json={"start": 10, "end": 14}).json()
        assert cleared["states"][10:15] == ["empty", "empty", "keyframe", "empty", "empty"]


def test_validation_errors_are_400_not_failed_jobs(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=6)
        assert client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).status_code == 400
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "class_id": 0})
        nk = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 3, "direction": "forward"})
        assert nk.status_code == 400 and "not a keyframe" in nk.json()["detail"]
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK, "class_id": 9}).status_code == 400
        tiny = {"type": "box", "box": [1, 1, 4, 4]}
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": tiny}).json() == {"candidates": []}
        no = client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": tiny})
        assert no.status_code == 400 and "no usable mask" in no.json()["detail"]
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": {"type": "clicks"}}).status_code == 422
        assert client.post(f"/api/clips/{cid}/frames/99/delete").status_code == 400
        assert client.post("/api/clips/nope/prompt", json={"frame": 0, "prompt": CLICK}).status_code == 404


def test_gpu_is_exclusive_and_jobs_can_be_cancelled_and_streamed(tmp_path):
    with _client(tmp_path, SlowEngine()) as client:
        cid = _clip(client, tmp_path, n=40)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "class_id": 0})
        job_id = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).json()["job_id"]

        busy = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"})
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK}).status_code == 409
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 1, "prompt": CLICK}).status_code == 409
        assert client.post(f"/api/clips/{cid}/frames/30/delete").status_code == 200  # flagging a frame is safe mid-run
        blocked = client.post(f"/api/clips/{cid}/clear-range", json={"start": 1, "end": 3})
        assert blocked.status_code == 409 and blocked.json()["job_id"] == job_id  # it mutates what the runner writes

        events = []
        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            try:
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["type"] == "frame" and event["frame"] == 5:
                        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
            except WebSocketDisconnect:
                pass
        assert [e["seq"] for e in events] == list(range(len(events)))
        assert events[0]["type"] == "progress" and events[-1]["type"] == "cancelled"
        snap = wait_job(client, job_id)
        assert snap["status"] == "cancelled" and 5 <= snap["result"]["processed"] < 39
        landed = client.get(f"/api/clips/{cid}").json()["states"].count("masked")
        assert landed == snap["result"]["written"]  # cancel keeps what landed
        ok = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK})
        assert ok.status_code == 200  # GPU is free again


def test_engine_not_loaded_is_503_but_browsing_still_works(tmp_path):
    class NotReady(FakeEngine):
        load_error = "no CUDA device with compute capability >= 8.0"

        def is_loaded(self):
            return False

    with _client(tmp_path, NotReady()) as client:
        cid = _clip(client, tmp_path, n=4)  # import needs no engine
        r = client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK})
        assert r.status_code == 503 and "compute capability" in r.json()["detail"]
        assert client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 0, "direction": "forward"}).status_code == 503
        assert client.get(f"/api/clips/{cid}").status_code == 200
        assert client.post(f"/api/clips/{cid}/frames/1/delete").status_code == 200


@pytest.mark.parametrize("body", [{"from_frame": 0, "direction": "sideways"}, {"direction": "forward"}])
def test_propagate_body_validation(tmp_path, body):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path, n=4)
        assert client.post(f"/api/clips/{cid}/propagate", json=body).status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_api_labeling.py -q`
Expected: failures with `404 Not Found` on `/api/clips/{c}/prompt` (the router does not exist yet).

- [ ] **Step 3: Implement**

`server/vsp/api/labeling.py`:

```python
"""Prompt, accept, propagate and frame-cleanup routes (spec 4 API table, 5.2-5.4)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..jobs import Job
from ..labeling import frames as frame_ops
from ..labeling.errors import BadRequest
from ..labeling.prompts import PromptSpec, accept, preview
from ..labeling.propagate import plan_indices, run_propagation
from .deps import GPU, State, require_engine, require_gpu_idle
from .views import candidate_view, clip_detail

router = APIRouter(prefix="/api/clips/{clip_id}", tags=["labeling"])


class PromptBody(BaseModel):
    frame: int = Field(ge=0)
    prompt: PromptSpec


class AcceptBody(PromptBody):
    class_id: int | None = None
    candidate: int = Field(default=0, ge=0)


class PropagateBody(BaseModel):
    from_frame: int = Field(ge=0)
    direction: Literal["forward", "backward", "both"]


class PruneBody(BaseModel):
    from_: int = Field(alias="from", ge=0)


class RangeBody(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)


@router.post("/prompt")
def prompt_preview(clip_id: str, body: PromptBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    require_gpu_idle(state)
    return {"candidates": [candidate_view(d) for d in preview(state.store, engine, clip, body.frame, body.prompt)]}


@router.post("/accept")
def prompt_accept(clip_id: str, body: AcceptBody, state: State) -> dict:
    state.store.read_clip(clip_id)
    engine = require_engine(state)
    require_gpu_idle(state)
    clip = accept(
        state.store, engine, clip_id, body.frame, body.prompt, class_id=body.class_id, candidate=body.candidate
    )
    return clip_detail(state, clip)


@router.post("/propagate", status_code=202)
def propagate(clip_id: str, body: PropagateBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    if not clip.objects:
        raise BadRequest("nothing to propagate: accept a prompt first")
    directions = ["forward", "backward"] if body.direction == "both" else [body.direction]
    for d in directions:  # validate now so a bad request is a 400, not a failed job
        plan_indices(clip, clip.objects[0], body.from_frame, d)
    cfg = state.settings.propagate

    def work(job: Job) -> dict:
        result = run_propagation(
            state.store, engine, clip_id, body.from_frame, body.direction,
            chunk_size=cfg.chunk_size, min_chunk_size=cfg.min_chunk_size, lost_after=cfg.lost_after,
            emit=job.emit, cancel=job.cancel_requested,
        )
        return result.as_dict()

    return {"job_id": state.jobs.start("propagate", work, exclusive=GPU).id}


@router.post("/frames/{idx}/delete")
def delete_frame(clip_id: str, idx: int, state: State) -> dict:
    return clip_detail(state, frame_ops.delete_frame(state.store, clip_id, idx))


@router.post("/frames/{idx}/restore")
def restore_frame(clip_id: str, idx: int, state: State) -> dict:
    return clip_detail(state, frame_ops.restore_frame(state.store, clip_id, idx))


@router.post("/prune")
def prune(clip_id: str, body: PruneBody, state: State) -> dict:
    return clip_detail(state, frame_ops.prune_from(state.store, clip_id, body.from_))


@router.post("/clear-range")
def clear_range(clip_id: str, body: RangeBody, state: State) -> dict:
    state.store.read_clip(clip_id)
    require_gpu_idle(state)  # it deletes masks and trims ranges a running propagation is writing
    return clip_detail(state, frame_ops.clear_range(state.store, clip_id, body.start, body.end))
```

In `server/vsp/api/app.py`: change `from . import clips, jobs, projects` to `from . import clips, jobs, labeling, projects` and add `app.include_router(labeling.router)` directly after `app.include_router(clips.router)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_api_labeling.py -q`
Expected: `6 passed`, no warnings. If `test_gpu_is_exclusive…` is flaky, the cause is timing in the test (the job finished before the second request): raise `n` in that test's `_clip(...)` call, never add sleeps to production code.

- [ ] **Step 5: README**

Append to `README.md`:

```markdown
## Run the server

    HF_HUB_OFFLINE=1 uv run vsp serve        # http://127.0.0.1:28000 ; the model loads in the background
    curl -s --noproxy '*' http://127.0.0.1:28000/api/health

`/api/health` shows `engine.loaded` and, if loading failed, the reason. Interactive API docs: `/docs`.
Importing, browsing and frame cleanup work without a GPU; prompting and propagation return 503 until the
engine is loaded, and 409 (with the running `job_id`) while a propagation job holds the GPU.
```

- [ ] **Step 6: Whole suite, lint, commit**

Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q` → clean; `130 passed`, GPU tests not skipped, no warnings.

```bash
git add server/vsp/api/labeling.py server/vsp/api/app.py server/tests/test_api_labeling.py README.md
git commit -F - <<'EOF'
feat(api): prompt, accept, propagate and frame-cleanup routes with GPU exclusivity

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL
EOF
```

---

## Done criteria for Plan 2

- `uv run ruff check server` clean; `HF_HUB_OFFLINE=1 uv run pytest -q` → 130 passed, none skipped, no warnings.
- `uv run vsp serve` starts, `/api/health` reaches `engine.loaded: true` on this machine, and `/docs` lists every route in the spec's API table except export (Plan 4).
- A full flow works over HTTP with the real engine: import frames → click prompt → accept → propagate both → correct mid-clip → propagate → clear range. (The automated version of this runs against `FakeEngine`; doing it once by hand against the real engine is Plan 3's manual QA, where there is a GUI to do it with.)
- `git ls-files | grep -c vision_train` prints `0`.

---

## As executed (2026-09-20)

All seven tasks landed on `plan2-labeling-api`. The per-task `passed` totals above were written before execution and
are superseded: review-driven fix rounds added tests, so the branch ends at **167 passed** (12 on GPU), not 130.
Changes made during execution that go beyond this plan's text — the hardened propagation runner (mid-chunk OOM
bookkeeping, offset validation, `lost_by_direction`, per-chunk `deleted_frames` refresh, cancel latch), interactive GPU
leases in `JobManager` (replacing `require_gpu_idle`), the lease on `DELETE /api/clips/{c}`, server-side box/point
validation, cancellable imports with honest `cancelled` status, `add_class` under a project lock and
`Cache-Control: no-cache` on masks — are summarised for Plan 3 in the roadmap's "Carried forward from Plan 2" section.
