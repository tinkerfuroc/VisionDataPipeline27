# Propagation chunk prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the ~0.5 s pause between propagation chunks by reading and preprocessing the next chunk on a CPU thread while the current chunk is on the GPU.

**Architecture:** The engine contract gains `prepare(frames) -> PreparedChunk`, a CPU-only, lock-free step that `propagate()` accepts in place of raw frames. A new `ChunkPrefetcher` in the labeling layer owns one worker thread and a small cache of raw frames (the current and the scheduled chunk only): the chunk loop `take()`s the chunk it needs and immediately `schedule()`s the chunk it expects to need next. A request that differs from the schedule (OOM back-off, an anchor that is not the chunk's last frame) is built synchronously from the cache, reading only what is missing.

**Tech Stack:** Python 3.11, numpy, torch + transformers (SAM3), `concurrent.futures.ThreadPoolExecutor`, pytest. Python runs through uv only (`uv run pytest`, `uv run ruff check`).

**Spec:** No separate spec file. The design was agreed in chat on 2026-09-22 and is summarised in the Architecture paragraph above and in the "Design notes" section below; the roadmap entry "Propagation streaming (done, 2026-09-21)" in `docs/superpowers/plans/2026-09-20-vsp-roadmap.md` records the measured pause this plan removes.

## Global Constraints

- Python is managed with uv only: `uv run pytest server/tests -q`, `uv run ruff check server`. No pip, no conda.
- Never stage or commit anything under `vision_train/`, `data/`, `web/node_modules/`, `web/dist/`, or any `*.pt` / `*.safetensors`. Stage by explicit path only. Do not push; no remote is configured.
- Commit messages end with the two lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL`.
- The engine's `propagate()` MUST keep accepting a raw `list[np.ndarray]` (the GPU tests and the test doubles in `server/tests/test_propagate.py` call it that way).
- `prepare()` MUST NOT take the engine's GPU lock and MUST NOT touch the GPU: it is called from another thread while a `propagate()` generator is open.
- Frames on disk never change while a clip exists (deletion is soft, in `clip.deleted_frames`), so cached raw frames cannot go stale. The prefetcher must still never hold more than the current and the scheduled chunk of raw frames.
- No prefetch thread may outlive `run_propagation()` on any exit path: done, lost, cancelled, OOM failure, any exception.
- `ruff check server` must pass. Do not run `ruff format` on whole files (the project does not enforce formatting and it would rewrite unrelated lines).
- The GPU tests (`server/tests/test_engine_gpu.py`, marker `gpu`) run for real on this machine (RTX 3080 Ti). A development server may be running on port 28000 holding ~3 GiB of VRAM; the tests fit beside it. Do not stop that server.

## Design notes

- Per 25-frame chunk at 1080p the boundary pause is: ~0.2 s reading JPEGs, ~0.19 s processor resize/normalise (measured), ~0.05 s session build and GPU copy, ~0.1 s re-encoding the anchor. The first two overlap the previous chunk under this plan; the anchor re-encode stays (it needs the new session).
- `FakeEngine.prepare()` returns the frame list itself, so every existing test double that overrides `propagate(self, frames, ...)` and indexes `frames` keeps working unchanged.
- The prefetcher predicts the next chunk as "this chunk completes and its last frame becomes the anchor": `schedule(chunk[-1], pending[size:2*size])`.
- Memory: one extra prepared chunk on the CPU (25 × 3 × 1008² fp32 ≈ 300 MB) plus at most two chunks of raw frames. No extra GPU memory.
- Not in scope: carrying the anchor's features across tracker sessions; prefetching more than one chunk ahead; prefetching the second direction's first chunk during the first direction of a "both" run.

---

### Task 1: Engine contract and FakeEngine.prepare

**Files:**
- Modify: `server/vsp/engine/types.py`
- Modify: `server/vsp/engine/fake.py`
- Test: `server/tests/test_fake_engine.py`

**Interfaces:**
- Produces: `PreparedChunk` (type alias for `object`) and `Engine.prepare(self, frames: list[np.ndarray]) -> PreparedChunk`; `Engine.propagate(self, frames: list[np.ndarray] | PreparedChunk, seed_mask, cancel_cb=None)`.
- Produces: `FakeEngine.prepare(frames) -> list[np.ndarray]` (a copy of the list) and `FakeEngine.prepare_calls: list[int]` (chunk lengths, like `propagate_calls`).

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_fake_engine.py`:

```python
def test_prepare_returns_the_frames_and_propagate_accepts_them():
    eng = FakeEngine()
    frames = [np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(4)]
    seed = np.zeros((40, 60), dtype=bool)
    seed[10:20, 5:15] = True
    prepared = eng.prepare(frames)
    assert isinstance(prepared, list) and len(prepared) == 4 and prepared is not frames
    assert eng.prepare_calls == [4]
    raw = [(o, None if m is None else m.copy()) for o, m in eng.propagate(frames, seed)]
    via_prepared = list(eng.propagate(prepared, seed))
    assert [o for o, _ in raw] == [o for o, _ in via_prepared]
    for (_, a), (_, b) in zip(raw, via_prepared):
        assert (a is None) == (b is None)
        if a is not None:
            assert np.array_equal(a, b)
    assert eng.propagate_calls == [4, 4]
```

Check the file's existing imports; it already imports `numpy as np` and `FakeEngine` (add either if missing).

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_fake_engine.py -q -k prepare`
Expected: FAIL with `AttributeError: 'FakeEngine' object has no attribute 'prepare'`

- [ ] **Step 3: Extend the contract**

In `server/vsp/engine/types.py`, add after `EngineNotLoaded`:

```python
PreparedChunk = object
"""Opaque, engine-specific output of `Engine.prepare()`; only the engine that produced it reads it."""
```

In `class Engine(Protocol)`, add before `propagate`:

```python
    def prepare(self, frames: list[np.ndarray]) -> PreparedChunk:
        """CPU-only preprocessing of a chunk for `propagate()`.

        Lock-free and thread-safe: it may run on another thread while a `propagate()` generator
        from this engine is open, so a caller can prepare the next chunk while the current one is
        on the GPU. Never touches the GPU. The result is opaque; pass it to `propagate()` in place
        of the raw frame list.
        """
        ...
```

Change the `propagate` signature to `frames: list[np.ndarray] | PreparedChunk,` and add this sentence to the end of its docstring's first paragraph: `\`frames\` is either the raw BGR frames or the result of \`prepare()\` on them.` Add to the last paragraph: `The one exception is \`prepare()\`, which never blocks on the lock.`

- [ ] **Step 4: Implement FakeEngine.prepare**

In `server/vsp/engine/fake.py`, add `self.prepare_calls: list[int] = []` next to `self.propagate_calls` in `__init__`, and add the method before `propagate`:

```python
    def prepare(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        # The fake's "prepared chunk" is the frame list itself, so test doubles that override
        # `propagate` and index `frames` keep working.
        self.prepare_calls.append(len(frames))
        return list(frames)
```

Change `propagate`'s annotation to `frames: list[np.ndarray] | PreparedChunk` and import `PreparedChunk` from `.types`. Inside `propagate`, nothing else changes: the fake only ever receives a list.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest server/tests/test_fake_engine.py server/tests/test_propagate.py -q && uv run ruff check server`
Expected: all pass; `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add server/vsp/engine/types.py server/vsp/engine/fake.py server/tests/test_fake_engine.py
git commit -m "Add Engine.prepare(): CPU-only chunk preprocessing that propagate() accepts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

---

### Task 2: Sam3Engine.prepare and propagate(PreparedChunk)

**Files:**
- Modify: `server/vsp/engine/sam3.py` (the `propagate` method, around lines 275-317)
- Test: `server/tests/test_engine_gpu.py`

**Interfaces:**
- Consumes: `PreparedChunk` from Task 1.
- Produces: `Sam3Engine.prepare(frames) -> _Prepared` where `_Prepared` is a module-private dataclass `(pixel_values: torch.Tensor on the CPU, h: int, w: int)`; `Sam3Engine.propagate` accepts a raw list or a `_Prepared`.

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_engine_gpu.py` (the file already imports `contextlib`; add `import threading` at the top with the other imports):

```python
def test_a_prepared_chunk_propagates_like_raw_frames(engine, video):
    frames, truth = video
    raw = dict(engine.propagate(frames, truth[0]))
    got = dict(engine.propagate(engine.prepare(frames), truth[0]))
    assert sorted(got) == sorted(raw) == list(range(len(frames)))
    for i in raw:
        assert (raw[i] is None) == (got[i] is None)
        if raw[i] is not None:
            assert _iou(raw[i], got[i]) > 0.98


def test_prepare_runs_on_another_thread_while_a_propagation_is_open(engine, video):
    """prepare() must not take the GPU lock: the chunk loop calls it from a worker thread while
    the current chunk's generator (which holds the lock) is open."""
    frames, truth = video
    with contextlib.closing(engine.propagate(frames[:4], truth[0])) as masks:
        next(masks)  # the lock is now held by this thread
        done = threading.Event()
        out = {}

        def worker():
            out["prepared"] = engine.prepare(frames[:3])
            done.set()

        threading.Thread(target=worker, daemon=True).start()
        assert done.wait(15), "prepare() blocked behind the GPU lock"
    assert out["prepared"].pixel_values.shape[0] == 3
    assert out["prepared"].pixel_values.device.type == "cpu"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest server/tests/test_engine_gpu.py -q -k "prepared or prepare_runs"`
Expected: FAIL with `AttributeError: 'Sam3Engine' object has no attribute 'prepare'`

- [ ] **Step 3: Implement prepare and accept it in propagate**

In `server/vsp/engine/sam3.py`, import `dataclass` from `dataclasses` (if not already imported) and `PreparedChunk` from `.types`. Add a module-level dataclass near the other module constants (after `_OBJ_ID`):

```python
@dataclass
class _Prepared:
    """A chunk after the processor: the tensor `propagate()` builds its tracker session from, still on the CPU."""

    pixel_values: torch.Tensor  # (n, 3, S, S)
    h: int
    w: int
```

Add the method right before `propagate` (inside the `# --- propagate` section):

```python
    def prepare(self, frames: list[np.ndarray]) -> _Prepared:
        """Resize and normalise a chunk on the CPU. Lock-free, so the chunk loop can run it on a
        worker thread while the previous chunk's `propagate()` generator holds the GPU lock."""
        processor = self._processor  # a local: release() may null the attribute concurrently
        if processor is None:
            raise EngineNotLoaded("SAM3 engine is not loaded")
        if not frames:
            raise ValueError("cannot prepare an empty chunk")
        h, w = frames[0].shape[:2]
        rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        return _Prepared(processor(images=rgb, return_tensors="pt")["pixel_values"], h, w)
```

Check how `_require_loaded()` raises `EngineNotLoaded` and reuse its exact message text if it has one, so the two paths report the same error.

Rewrite the head of `propagate` so it reads:

```python
    def propagate(
        self,
        frames: list[np.ndarray] | PreparedChunk,
        seed_mask: np.ndarray,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        """...existing docstring, plus this sentence at the end of the first paragraph:
        `frames` is either the raw BGR frames or the result of `prepare()` on them; the
        preprocessing happens here in the former case.
        """
        self._require_loaded()
        if isinstance(frames, _Prepared):
            prepared = frames
        else:
            if not frames:
                return
            prepared = self.prepare(frames)
        h, w = prepared.h, prepared.w
        if seed_mask.shape != (h, w):
            raise ValueError(f"seed_mask shape {seed_mask.shape} does not match frames {(h, w)}")
        if not seed_mask.any():
            raise ValueError("seed_mask must be non-empty")
        video = prepared.pixel_values
        try:
            with self._lock, torch.inference_mode(), self._autocast():
                yield from self._propagate_locked(video, seed_mask, h, w, cancel_cb)
        ...unchanged from here (the except/finally block)...
```

Delete the old `rgb = ...` and `video = self._processor(...)` lines; they moved into `prepare`.

- [ ] **Step 4: Run the GPU tests and lint**

Run: `uv run pytest server/tests/test_engine_gpu.py -q && uv run ruff check server`
Expected: all GPU tests pass (15 tests); `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add server/vsp/engine/sam3.py server/tests/test_engine_gpu.py
git commit -m "Sam3Engine.prepare(): preprocess a chunk off the GPU lock

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

---

### Task 3: ChunkPrefetcher

**Files:**
- Create: `server/vsp/labeling/prefetch.py`
- Test: `server/tests/test_prefetch.py`

**Interfaces:**
- Consumes: `Engine.prepare` (Task 1), `load_frame(store, clip, idx)` from `vsp.labeling.prompts`, `ProjectStore.frame_path`.
- Produces: `ChunkPrefetcher(store, clip, engine)` context manager with `schedule(anchor_idx: int, indices: list[int]) -> None`, `take(anchor_idx: int, indices: list[int]) -> PreparedChunk`, `close() -> None`, and read-only `reads: int` (frames read from disk so far) and `cached: list[int]` (sorted frame indices held), both for tests.

- [ ] **Step 1: Write the failing tests**

Create `server/tests/test_prefetch.py`:

```python
import threading

import pytest
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import NotFound
from vsp.labeling.prefetch import ChunkPrefetcher


def _white(frame) -> bool:
    return bool(frame[0, 0, 0] > 250)


def test_take_returns_the_scheduled_chunk_and_reuses_the_anchor(store, make_clip):
    _, clip = make_clip(n_frames=10, white=(2,))
    eng = FakeEngine()
    with ChunkPrefetcher(store, clip, eng) as pf:
        pf.schedule(0, [1, 2, 3])
        chunk = pf.take(0, [1, 2, 3])
        assert [_white(f) for f in chunk] == [False, False, True, False]  # anchor first, then 1, 2, 3
        assert pf.reads == 4 and eng.prepare_calls == [4]
        pf.schedule(3, [4, 5, 6])
        chunk = pf.take(3, [4, 5, 6])
        assert len(chunk) == 4 and pf.reads == 7  # the anchor 3 came from the cache


def test_a_request_that_differs_from_the_schedule_reads_only_what_is_missing(store, make_clip):
    _, clip = make_clip(n_frames=10)
    eng = FakeEngine()
    with ChunkPrefetcher(store, clip, eng) as pf:
        assert len(pf.take(0, [1, 2, 3, 4])) == 5  # nothing scheduled: built synchronously
        assert pf.reads == 5 and eng.prepare_calls == [5]
        pf.schedule(4, [5, 6, 7, 8])
        chunk = pf.take(4, [5, 6])  # an OOM back-off retries a prefix of the same frames
        assert len(chunk) == 3 and pf.reads == 9 and eng.prepare_calls == [5, 5, 3]
        chunk = pf.take(3, [5, 6])  # the anchor is not the previous chunk's last frame
        assert len(chunk) == 3 and pf.reads == 9


def test_only_the_current_and_the_scheduled_chunk_are_kept(store, make_clip):
    _, clip = make_clip(n_frames=12)
    with ChunkPrefetcher(store, clip, FakeEngine()) as pf:
        pf.take(0, [1, 2])
        pf.schedule(2, [3, 4])
        pf.take(2, [3, 4])
        pf.schedule(4, [5, 6])
        pf.take(4, [5, 6])
        assert pf.cached == [2, 3, 4, 5, 6]


def test_a_missing_frame_surfaces_at_take(store, make_clip):
    _, clip = make_clip(n_frames=6)
    store.frame_path(clip.id, 4).unlink()
    with ChunkPrefetcher(store, clip, FakeEngine()) as pf:
        pf.schedule(2, [3, 4, 5])
        with pytest.raises(NotFound, match="frame 4"):
            pf.take(2, [3, 4, 5])
        assert len(pf.take(2, [3])) == 2  # the prefetcher is still usable afterwards


def test_close_stops_the_worker_thread(store, make_clip):
    _, clip = make_clip(n_frames=6)
    with ChunkPrefetcher(store, clip, FakeEngine()) as pf:
        pf.schedule(0, [1, 2, 3])
    assert not [t for t in threading.enumerate() if t.name.startswith("vsp-prefetch")]
```

Check `make_clip` in `server/tests/conftest.py`: it returns `(project, clip)` and takes `n_frames` and `white`. Check `NotFound` lives in `vsp.labeling.errors` (it is what `load_frame` raises for a missing file); if `load_frame` raises a different class, use that one.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest server/tests/test_prefetch.py -q`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'vsp.labeling.prefetch'`

- [ ] **Step 3: Implement the prefetcher**

Create `server/vsp/labeling/prefetch.py`:

```python
"""Read and preprocess the next propagation chunk on a CPU thread while the current one is on the GPU."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress

import numpy as np

from ..engine.types import Engine, PreparedChunk
from ..store import Clip, ProjectStore
from .prompts import load_frame

Key = tuple[int, tuple[int, ...]]  # (anchor frame, chunk frames)


class ChunkPrefetcher:
    """One chunk ahead: `schedule()` the chunk the loop expects to need next, `take()` the one it actually needs.

    Raw frames are cached for the current and the scheduled chunk only. A request that differs from the
    schedule (an OOM retry on a prefix of the same frames, an anchor that is not the chunk's last frame) is
    built synchronously and reads only the frames it does not hold; only the preprocessing is redone. Frames
    on disk never change while a clip exists (deletion is soft), so cached pixels cannot go stale.

    Use as a context manager: `close()` joins the worker, so no thread outlives the propagation run.
    """

    def __init__(self, store: ProjectStore, clip: Clip, engine: Engine) -> None:
        self._store, self._clip, self._engine = store, clip, engine
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vsp-prefetch")
        self._lock = threading.Lock()  # guards _frames and reads; the worker writes both
        self._frames: dict[int, np.ndarray] = {}
        self._taken: Key | None = None
        self._scheduled: tuple[Key, Future[PreparedChunk]] | None = None
        self.reads = 0

    # ------------------------------------------------------------------ API
    def schedule(self, anchor_idx: int, indices: list[int]) -> None:
        """Start loading and preparing `[anchor] + indices` in the background. Replaces any earlier schedule."""
        self._settle()
        key: Key = (anchor_idx, tuple(indices))
        self._retain(set(self._taken or ()) | {anchor_idx, *indices})
        self._scheduled = (key, self._pool.submit(self._build, key))

    def take(self, anchor_idx: int, indices: list[int]) -> PreparedChunk:
        """Return the prepared chunk `[anchor] + indices`, from the schedule when it matches, else built now."""
        key: Key = (anchor_idx, tuple(indices))
        scheduled = self._scheduled
        self._scheduled = None
        if scheduled is not None and scheduled[0] == key:
            future = scheduled[1]
        else:
            if scheduled is not None:
                with suppress(Exception):  # a failed load of a chunk nobody needs is not an error
                    scheduled[1].result()
            future = None
        self._taken = key
        return future.result() if future is not None else self._build(key)

    def close(self) -> None:
        self._settle()
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._frames.clear()

    def __enter__(self) -> ChunkPrefetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def cached(self) -> list[int]:
        with self._lock:
            return sorted(self._frames)

    # ------------------------------------------------------------ internals
    def _build(self, key: Key) -> PreparedChunk:
        anchor_idx, indices = key
        frames = [self._load(anchor_idx)] + [self._load(i) for i in indices]
        return self._engine.prepare(frames)

    def _load(self, idx: int) -> np.ndarray:
        with self._lock:
            frame = self._frames.get(idx)
        if frame is None:
            frame = load_frame(self._store, self._clip, idx)
            with self._lock:
                self._frames[idx] = frame
                self.reads += 1
        return frame

    def _retain(self, wanted: set[int]) -> None:
        with self._lock:
            for idx in [i for i in self._frames if i not in wanted]:
                del self._frames[idx]

    def _settle(self) -> None:
        """Wait for a still-running schedule so the cache is quiescent before it is edited or the pool closed."""
        if self._scheduled is not None:
            with suppress(Exception):
                self._scheduled[1].result()
            self._scheduled = None
```

Note for the implementer: `take()` reads the cache from the main thread only after the worker has finished (`future.result()` or `_settle()`), and `schedule()` edits the cache only after `_settle()`; the lock exists because `_load` on the worker writes while `cached` may be read from a test. Keep it that simple; do not add a second worker.

- [ ] **Step 4: Run the tests and lint**

Run: `uv run pytest server/tests/test_prefetch.py -q && uv run ruff check server`
Expected: 5 passed; `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add server/vsp/labeling/prefetch.py server/tests/test_prefetch.py
git commit -m "Add ChunkPrefetcher: load and prepare the next chunk on a worker thread

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

---

### Task 4: Use the prefetcher in the chunk loop, measure, document

**Files:**
- Modify: `server/vsp/labeling/propagate.py` (`_run_direction`, `run_propagation`)
- Test: `server/tests/test_propagate.py`
- Modify: `docs/superpowers/plans/2026-09-20-vsp-roadmap.md` (append a section)

**Interfaces:**
- Consumes: `ChunkPrefetcher` (Task 3), `Engine.propagate(PreparedChunk)` (Tasks 1-2).
- Produces: `run_propagation(...)` keeps its signature; `_run_direction` gains a keyword-only `prefetch: ChunkPrefetcher` parameter.

- [ ] **Step 1: Write the failing tests**

Append to `server/tests/test_propagate.py` (add `import threading` and `import vsp.labeling.prefetch as prefetch_mod` to the imports):

```python
def test_each_frame_is_read_from_disk_once_per_run(store, make_clip, monkeypatch):
    reads: list[int] = []
    real = prefetch_mod.load_frame

    def counting(store_, clip_, idx):
        reads.append(idx)
        return real(store_, clip_, idx)

    monkeypatch.setattr(prefetch_mod, "load_frame", counting)
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.written == 29
    assert sorted(reads) == list(range(30))  # anchors are not re-read at chunk seams
    assert eng.prepare_calls == [9, 9, 9, 6] and eng.propagate_calls == [9, 9, 9, 6]


def test_no_prefetch_thread_survives_the_run(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    seen = []

    def cancel():
        return len(seen) >= 3

    run_propagation(store, eng, clip.id, 0, "forward", emit=seen.append, cancel=cancel, **KW)
    assert not [t for t in threading.enumerate() if t.name.startswith("vsp-prefetch")]

    eng = FakeEngine(oom_if_more_than=2)
    clip = _seeded(store, make_clip, eng)
    with pytest.raises(PropagationFailed):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert not [t for t in threading.enumerate() if t.name.startswith("vsp-prefetch")]
```

Check the existing OOM test around line 132 for the exact `FakeEngine(oom_if_more_than=2)` + `KW` combination that raises `PropagationFailed` (min_chunk_size 4 + anchor = 5 frames > 2 at the floor) and match it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest server/tests/test_propagate.py -q -k "read_from_disk_once or prefetch_thread"`
Expected: the first FAILS because anchors are re-read (`sorted(reads)` has duplicates) and `prepare_calls == []`; the second passes vacuously now and turns into a real guard after the change.

- [ ] **Step 3: Wire the prefetcher into the loop**

In `server/vsp/labeling/propagate.py`, import `ChunkPrefetcher` from `.prefetch`. In `_run_direction`, add a keyword-only parameter `prefetch: ChunkPrefetcher,` after `cancel`. Replace the two lines

```python
        chunk = pending[:size]
        frames = [load_frame(store, clip, anchor_idx)] + [load_frame(store, clip, i) for i in chunk]
```

with

```python
        chunk = pending[:size]
        frames = prefetch.take(anchor_idx, chunk)
        # Predict the next chunk (this one completes and its last frame becomes the anchor) and start reading and
        # preprocessing it now, so that work overlaps this chunk's GPU time instead of following it. A wrong
        # prediction costs only the preprocessing: the prefetcher re-reads nothing it already holds.
        if len(pending) > size:
            prefetch.schedule(chunk[-1], pending[size:2 * size])
```

Keep `del frames` where it is. Remove the now-unused `load_frame` import if nothing else in the module uses it (check with ruff).

In `run_propagation`, wrap the direction loop:

```python
    result = PropagateResult()
    total = sum(len(p) for p in plans)
    emit({"type": "progress", "done": 0, "total": total})
    size = chunk_size
    with ChunkPrefetcher(store, clip, engine) as prefetch:
        for d, indices in zip(directions, plans):
            if result.cancelled:
                break
            size = _run_direction(
                store, engine, clip, obj.id, from_frame, seed, indices, d,
                chunk_size=size, min_chunk_size=min_chunk_size, lost_after=lost_after,
                total=total, result=result, emit=emit, cancel=cancel, prefetch=prefetch,
            )
```

Update the module docstring's first line to mention the prefetch, e.g. `"""Keyframe-capped propagation with anchor-prepended chunks (spec 5.3); the next chunk is read and preprocessed while the current one runs."""`.

- [ ] **Step 4: Run the whole Python suite and lint**

Run: `uv run pytest server/tests -q && uv run ruff check server`
Expected: all pass (the two new tests included; 27 + 2 in `test_propagate.py`); `All checks passed!`

- [ ] **Step 5: Measure on the running server**

The development server on http://127.0.0.1:28000 must be restarted to pick up the change. Find its PID with `ss -ltnp | grep 28000`, `kill <pid>`, then start it again from the repo root with `nohup uv run vsp serve > <scratchpad>/serve-prefetch.log 2>&1 &` and wait until `curl -s --noproxy '*' http://127.0.0.1:28000/api/health` reports `"loaded":true`. Then run this script with `uv run python` and paste its last three output lines into your report:

```python
import time
import httpx

c = httpx.Client(base_url="http://127.0.0.1:28000", trust_env=False, timeout=30)
p = c.get("/api/projects").json()[0]
clips = c.get(f"/api/projects/{p['id']}/clips").json()
clip = next(k for k in clips if k["objects"])
kf = sorted(k["frame_idx"] for k in clip["objects"][0]["keyframes"])[0]
jid = c.post(f"/api/clips/{clip['id']}/propagate", json={"from_frame": kf, "direction": "forward"}).json()["job_id"]
t0 = time.perf_counter(); seen = 0; arrivals = []
while True:
    s = c.get(f"/api/jobs/{jid}").json()
    if s["n_events"] > seen:
        arrivals.append((round(time.perf_counter() - t0, 2), s["n_events"] - seen))
        seen = s["n_events"]
    if s["status"] in ("done", "error", "cancelled"):
        break
    time.sleep(0.05)
gaps = [b[0] - a[0] for a, b in zip(arrivals, arrivals[1:])]
print("status", s["status"], s["result"])
print("arrivals", arrivals)
print("events", seen, "max gap after the first frame %.2fs" % max(gaps[1:]), "median gap %.2fs" % sorted(gaps)[len(gaps) // 2])
```

Before this plan the same script measured a 0.56 s gap at the chunk boundary (frame 24) against a 0.20-0.25 s per-frame rate. Expected now: the boundary gap is within about one frame step of the median (≤ ~0.45 s). If it is not, report the numbers; do not tune anything.

- [ ] **Step 6: Record the result in the roadmap**

Append to `docs/superpowers/plans/2026-09-20-vsp-roadmap.md`:

```markdown

## Chunk prefetch (done, 2026-09-22, branch `prefetch-chunks`)

Closes the deferred item under "Propagation streaming": the next chunk's frames are read and preprocessed on a worker
thread while the current chunk is on the GPU. `Engine.prepare()` is the CPU-only, lock-free half of `propagate()`;
`ChunkPrefetcher` (`server/vsp/labeling/prefetch.py`) owns one worker and a cache of raw frames limited to the
current and the scheduled chunk, so anchors are never re-read and a wrong prediction (OOM back-off, an anchor that is
not the chunk's last frame) only redoes the preprocessing. Measured on the demo clip: chunk-boundary gap <MEASURED> s
against a median frame step of <MEASURED> s (was 0.56 s against 0.22 s).

Still deferred: the anchor is re-encoded in every new tracker session (~one frame step per chunk); the second
direction of a "both" run starts with a synchronous chunk; one chunk of look-ahead only.
```

Replace both `<MEASURED>` with the numbers from Step 5.

- [ ] **Step 7: Commit**

```bash
git add server/vsp/labeling/propagate.py server/tests/test_propagate.py docs/superpowers/plans/2026-09-20-vsp-roadmap.md
git commit -m "Prefetch the next propagation chunk while the current one is on the GPU

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

## As executed (2026-09-22)

Four tasks, each with a spec + quality review, then an Opus whole-branch review, one fix round and a scoped re-review.
- Task 3's implementer found a bug in this plan's reference code: `set(self._taken or ())` unpacked the `(anchor, indices)` key instead of the frame indices, so the eviction kept the wrong frames; the plan's own tests caught it. `NotFound` lives in `vsp.store`, not `vsp.labeling.errors`.
- Task 4 measured on the demo clip (640×480): chunk-boundary gap 0.56 s → 0.46 s against a 0.25 s per-frame step. The residual is the anchor re-encode in the new tracker session, kept by design. The saving grows with resolution (reading and preprocessing a 1080p chunk is ~0.4 s).
- Final review (approve, six minor findings, all fixed): `propagate()` docstring paragraphing; `oom_error = None` before the retry so the failed chunk's tensors are freed before the next `take()`; `take()` now evicts to its own chunk (cache = current chunk after `take()`, current + scheduled after `schedule()`), which means an OOM back-off drops the already-prefetched next chunk and reads it again later, an accepted cost; mismatch waits documented; anchor comment corrected to "last frame with a mask"; `PreparedChunk: TypeAlias = Any`.
- Two subagent commits had their Co-Authored-By line rewritten to the required literal before the merge.
