import threading

import numpy as np
import pytest
import vsp.labeling.prefetch as prefetch_mod
from conftest import is_white
from vsp.engine.fake import FakeEngine
from vsp.engine.types import ChunkOOMError
from vsp.labeling.errors import BadRequest, PropagationFailed
from vsp.labeling.frames import restore_frame
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


# ---------------------------------------------------------------- fix round 1: stub engines


class OomAfter(FakeEngine):
    """Delegates to FakeEngine.propagate but raises ChunkOOMError after yielding `k` real frames
    (offsets 0..k), on each of its first `n_failures` calls; afterwards behaves normally."""

    def __init__(self, k, n_failures=1, **kw):
        super().__init__(**kw)
        self.k = k
        self.n_failures = n_failures
        self.calls = 0

    def propagate(self, frames, seed_mask, cancel_cb=None):
        self.calls += 1
        if self.calls <= self.n_failures:
            gen = super().propagate(frames, seed_mask, cancel_cb)
            for _ in range(self.k + 1):
                yield next(gen)
            gen.close()
            raise ChunkOOMError(f"fake OOM after {self.k} real frames")
        yield from super().propagate(frames, seed_mask, cancel_cb)


class OomProgressThenStarve(FakeEngine):
    """First call OOMs after landing one real frame (progress). Every later call OOMs before
    yielding any frame at all (no progress)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def propagate(self, frames, seed_mask, cancel_cb=None):
        self.calls += 1
        self.propagate_calls.append(len(frames))
        if self.calls == 1:
            yield 0, seed_mask.copy()
            shifted = np.zeros_like(seed_mask)
            shifted[:, self.dx:] = seed_mask[:, :-self.dx]
            yield 1, shifted
        raise ChunkOOMError("fake OOM")


class OnlySeed(FakeEngine):
    """Yields only the anchor's own offset-0 output, then stops -- no real frame ever lands."""

    def propagate(self, frames, seed_mask, cancel_cb=None):
        self.propagate_calls.append(len(frames))
        yield 0, seed_mask.copy()


class StopsAfterTwo(FakeEngine):
    """Yields the anchor plus exactly 2 real frames of a longer chunk, then stops without OOM or
    cancellation -- exercises the "stopped short with partial progress" PropagationFailed message."""

    def propagate(self, frames, seed_mask, cancel_cb=None):
        self.propagate_calls.append(len(frames))
        gen = super().propagate(frames, seed_mask, cancel_cb)
        for _ in range(3):  # offset 0 (anchor, discarded) + 2 real frames
            yield next(gen)
        gen.close()


class Skips(FakeEngine):
    """Yields offsets 0, 1, 3 -- skipping 2 -- to exercise offset validation."""

    def propagate(self, frames, seed_mask, cancel_cb=None):
        self.propagate_calls.append(len(frames))
        for offset in (0, 1, 3):
            if offset >= len(frames):
                return
            if offset == 0:
                yield 0, seed_mask.copy()
                continue
            shifted = np.zeros_like(seed_mask)
            s = offset * self.dx
            if s < seed_mask.shape[1]:
                shifted[:, s:] = seed_mask[:, :-s]
            yield offset, shifted


# ---------------------------------------------------------------- fix round 1: new tests


def test_mid_chunk_oom_resumes_at_the_first_unprocessed_frame(store, make_clip):
    eng = OomAfter(3)
    clip = _seeded(store, make_clip, eng, n_frames=13)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 12 and res.oom_retries == 1
    assert store.mask_indices(clip.id, OBJECT_ID) == list(range(13))
    for k in (1, 3, 4, 8, 12):
        assert _x0(store, clip.id, k) == 20 + 2 * k


def test_mid_chunk_oom_records_the_landed_frames(store, make_clip):
    eng = OomAfter(3)
    clip = _seeded(store, make_clip, eng, n_frames=13)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 12)]

    # OOMs on every call, at the floor, but always after landing >= 1 frame: not fatal.
    eng2 = OomAfter(1, n_failures=100)
    clip2 = _seeded(store, make_clip, eng2, n_frames=8)
    res2 = run_propagation(store, eng2, clip2.id, 0, "forward", chunk_size=4, min_chunk_size=4, lost_after=3)
    assert res2.processed == 7
    assert store.mask_indices(clip2.id, OBJECT_ID) == list(range(8))


def test_oom_at_the_floor_without_progress_keeps_what_landed(store, make_clip):
    eng = OomProgressThenStarve()
    clip = _seeded(store, make_clip, eng, n_frames=13)
    with pytest.raises(PropagationFailed, match="chunk size 4"):
        run_propagation(store, eng, clip.id, 0, "forward", chunk_size=4, min_chunk_size=4, lost_after=3)
    assert store.mask_indices(clip.id, OBJECT_ID) == [0, 1]
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 1)]


def test_both_reports_a_lost_forward_pass(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=20, keyframes=(10,), white=range(13, 20))
    res = run_propagation(store, eng, clip.id, 10, "both", **KW)
    assert res.lost_at == 13
    assert res.lost_by_direction == {"forward": 13}
    for i in range(10):
        assert store.read_mask(clip.id, OBJECT_ID, i) is not None


def test_both_when_both_directions_lose(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=20, keyframes=(10,), white=(0, 1, 2, 13, 14, 15, 16, 17, 18, 19))
    res = run_propagation(store, eng, clip.id, 10, "both", **KW)
    assert res.lost_by_direction == {"forward": 13, "backward": 2}
    assert res.lost_at == 13  # forward wins when both lose


def test_engine_returning_only_the_seed_raises_propagation_failed(store, make_clip):
    eng = OnlySeed()
    clip = _seeded(store, make_clip, eng, n_frames=10)
    with pytest.raises(PropagationFailed, match="returned no frames"):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)


def test_engine_stopping_mid_chunk_without_cancel_names_frames_landed(store, make_clip):
    eng = StopsAfterTwo()
    clip = _seeded(store, make_clip, eng, n_frames=13)
    with pytest.raises(PropagationFailed, match="stopped after 2 of 8 frames without being cancelled"):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.mask_indices(clip.id, OBJECT_ID) == [0, 1, 2]


def test_cancel_before_the_first_frame_of_a_chunk(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    res = run_propagation(store, eng, clip.id, 0, "forward", cancel=lambda: True, **KW)
    assert res.cancelled and res.processed == 0
    assert store.mask_indices(clip.id, OBJECT_ID) == [0]


def test_cancel_after_the_last_frame_is_not_reported_as_cancelled(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=5)  # forward pending: 1..4, one chunk of 4
    seen = []
    res = run_propagation(
        store, eng, clip.id, 0, "forward",
        emit=lambda e: seen.append(e) if e["type"] == "frame" else None,
        cancel=lambda: len(seen) >= 4, **KW,
    )
    assert res.cancelled is False
    assert res.processed == 4


def test_nothing_to_propagate_returns_an_empty_result(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=10, keyframes=(0,))
    events = []
    res = run_propagation(store, eng, clip.id, 0, "backward", emit=events.append, **KW)
    assert res.processed == 0 and eng.propagate_calls == []
    assert events == [{"type": "progress", "done": 0, "total": 0}]

    eng2 = FakeEngine()
    clip2 = _seeded(store, make_clip, eng2, n_frames=10, keyframes=(9,))
    res2 = run_propagation(store, eng2, clip2.id, 9, "forward", **KW)
    assert res2.processed == 0 and eng2.propagate_calls == []

    eng3 = FakeEngine()
    clip3 = _seeded(store, make_clip, eng3, n_frames=10, keyframes=(4, 5))
    res3 = run_propagation(store, eng3, clip3.id, 4, "forward", **KW)
    assert res3.processed == 0 and eng3.propagate_calls == []

    eng4 = FakeEngine()
    clip4 = _seeded(store, make_clip, eng4, n_frames=1, keyframes=(0,))
    res4 = run_propagation(store, eng4, clip4.id, 0, "both", **KW)
    assert res4.processed == 0 and eng4.propagate_calls == []


def test_missing_seed_mask_is_a_bad_request(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    store.delete_mask(clip.id, OBJECT_ID, 0)
    with pytest.raises(BadRequest, match="has no mask"):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)


def test_propagate_result_as_dict(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert set(res.as_dict().keys()) == {
        "processed", "written", "cleared", "lost_at", "cancelled", "oom_retries", "lost_by_direction",
    }


def test_out_of_order_engine_offsets_are_rejected(store, make_clip):
    eng = Skips()
    clip = _seeded(store, make_clip, eng, n_frames=10)
    with pytest.raises(PropagationFailed, match="expected 2"):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_mask(clip.id, OBJECT_ID, 1) is not None
    assert store.read_mask(clip.id, OBJECT_ID, 3) is None


def test_restoring_a_frame_mid_run_gets_its_mask_from_the_next_chunk(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=13)
    c = store.read_clip(clip.id)
    c.deleted_frames = [10]
    store.write_clip(c)

    def on_event(e):
        if e["type"] == "frame" and e["frame"] == 2:
            restore_frame(store, clip.id, 10)

    res = run_propagation(
        store, eng, clip.id, 0, "forward", emit=on_event, chunk_size=4, min_chunk_size=4, lost_after=99
    )
    assert res.processed == 12
    assert store.read_mask(clip.id, OBJECT_ID, 10) is not None


def test_backed_off_chunk_size_carries_into_the_backward_pass(store, make_clip):
    eng = FakeEngine(oom_if_more_than=5)
    clip = _seeded(store, make_clip, eng, n_frames=30, keyframes=(15,))
    res = run_propagation(store, eng, clip.id, 15, "both", **KW)
    assert res.oom_retries == 1
    assert all(n <= 5 for n in eng.propagate_calls[1:])


def test_propagated_ranges_after_a_lost_stop_cover_the_processed_frames(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=30, white=range(6, 30))
    stale = np.zeros((60, 120), dtype=bool)
    stale[0:20, 0:20] = True
    store.write_mask(clip.id, OBJECT_ID, 7, stale)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 8)]


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
