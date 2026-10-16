import threading

import numpy as np
import pytest
import vdp.labeling.prefetch as prefetch_mod
from conftest import OBJ, is_white
from vdp.engine.fake import FakeEngine
from vdp.engine.types import ChunkOOMError
from vdp.labeling.errors import BadRequest, PropagationFailed
from vdp.labeling.frames import restore_frame
from vdp.labeling.paint import set_mask
from vdp.labeling.prompts import PromptSpec, accept, create_object
from vdp.labeling.propagate import plan_indices, resolve_participants, run_propagation
from vdp.store import Keyframe, NotFound, Prompt, RegionRange, now_iso

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])  # seed rect x 20:40; FakeEngine shifts +2 px per offset
KW = {"chunk_size": 8, "min_chunk_size": 4, "lost_after": 3}


def _x0(store, clip_id, idx):
    mask = store.read_mask(clip_id, OBJ, idx)
    return None if mask is None else int(np.where(mask)[1].min())


def _seeded(store, make_clip, eng, n_frames=30, keyframes=(0,), white=()):
    _, clip = make_clip(n_frames=n_frames, white=white)
    for i, kf in enumerate(keyframes):
        accept(store, eng, clip.id, kf, CLICK, object_id=None if i == 0 else OBJ, class_id=0 if i == 0 else None)
    return clip


def test_plan_indices_are_capped_at_neighbouring_keyframes(store, make_clip):
    clip = _seeded(store, make_clip, FakeEngine(), n_frames=20, keyframes=(5, 12))
    clip = store.read_clip(clip.id)
    obj = clip.objects[0]
    assert plan_indices(clip, obj, 5, "forward") == list(range(6, 12))
    assert plan_indices(clip, obj, 12, "forward") == list(range(13, 20))
    assert plan_indices(clip, obj, 12, "backward") == list(range(11, 5, -1))
    assert plan_indices(clip, obj, 5, "backward") == [4, 3, 2, 1, 0]
    # any frame may be a start: the range still runs up to the neighbouring keyframes
    assert plan_indices(clip, obj, 7, "forward") == list(range(8, 12))
    assert plan_indices(clip, obj, 7, "backward") == [6]
    assert plan_indices(clip, obj, 15, "backward") == [14, 13]
    for bad in (-1, 20):
        with pytest.raises(BadRequest, match="out of range"):
            plan_indices(clip, obj, bad, "forward")


def test_forward_writes_every_frame_in_anchor_prepended_chunks(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    events = []
    res = run_propagation(store, eng, clip.id, 0, "forward", emit=events.append, **KW)
    assert (res.processed, res.written, res.lost_at, res.cancelled) == (29, 29, None, False)
    assert store.mask_indices(clip.id, OBJ) == list(range(30))
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
    before = store.mask_path(clip.id, OBJ, 10).read_bytes()
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 9
    assert store.mask_path(clip.id, OBJ, 10).read_bytes() == before
    assert _x0(store, clip.id, 10) == 20  # still the accepted mask, not seed + 20
    assert store.mask_indices(clip.id, OBJ) == list(range(11))


def test_backward_and_both(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=16, keyframes=(8,))
    res = run_propagation(store, eng, clip.id, 8, "backward", **KW)
    assert res.processed == 8 and _x0(store, clip.id, 5) == 20 + 2 * 3  # 3 frames away from the keyframe
    res = run_propagation(store, eng, clip.id, 8, "both", **KW)
    assert res.processed == 15
    assert store.mask_indices(clip.id, OBJ) == list(range(16))
    # the keyframe itself is not a propagated frame, so the two passes stay separate ranges
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(0, 7), (9, 15)]


def test_no_mask_at_the_start_frame_and_nothing_to_propagate(store, make_clip):
    eng = FakeEngine()
    _, empty = make_clip()
    with pytest.raises(BadRequest, match="accept a prompt"):
        run_propagation(store, eng, empty.id, 0, "forward", **KW)
    clip = _seeded(store, make_clip, eng)
    with pytest.raises(BadRequest, match="frame 3 has no mask"):
        run_propagation(store, eng, clip.id, 3, "forward", **KW)
    assert eng.propagate_calls == []


def test_resumes_from_a_propagated_frame_and_still_stops_at_the_next_keyframe(store, make_clip):
    """Stop mid-run, then continue from the last masked frame: the resumed run seeds from that frame's
    propagated mask (not the keyframe's) and stops before the next keyframe like any other run."""
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=30, keyframes=(0, 20))
    seen = []
    res = run_propagation(store, eng, clip.id, 0, "forward", emit=seen.append, cancel=lambda: len(seen) >= 6, **KW)
    assert res.cancelled and store.mask_indices(clip.id, OBJ) == [0, 1, 2, 3, 4, 5, 20]

    res = run_propagation(store, eng, clip.id, 5, "forward", **KW)
    assert (res.processed, res.written, res.cancelled) == (14, 14, False)  # frames 6..19; keyframe 20 untouched
    assert store.mask_indices(clip.id, OBJ) == [*range(21)]
    assert _x0(store, clip.id, 6) == 20 + 2 * 6  # continuous with the first run: seeded by frame 5's mask
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 19)]

    with pytest.raises(BadRequest, match="frame 25 has no mask"):
        run_propagation(store, eng, clip.id, 25, "backward", **KW)

    # "both" from a propagated frame: forward to the next keyframe, backward to the previous one, neither touched
    store.write_mask(clip.id, OBJ, 25, store.read_mask(clip.id, OBJ, 20))
    res = run_propagation(store, eng, clip.id, 25, "both", **KW)
    assert (res.processed, res.written) == (8, 8)  # 26..29 forward, 24..21 backward
    assert store.mask_indices(clip.id, OBJ) == [*range(30)]
    # 25 itself was written by hand above, not by a run, so it is not a propagated frame
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 19), (21, 24), (26, 29)]


def test_deleted_frames_feed_the_tracker_but_are_not_written(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=12)
    c = store.read_clip(clip.id)
    c.deleted_frames = [5]
    store.write_clip(c)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 11 and res.written == 10
    assert 5 not in store.mask_indices(clip.id, OBJ)
    assert _x0(store, clip.id, 6) == 20 + 2 * 6  # continuity kept through the deleted frame


def test_lost_object_stops_early_and_clears_stale_masks(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=30, white=range(6, 30))
    stale = np.zeros((60, 120), dtype=bool)
    stale[0:20, 0:20] = True
    store.write_mask(clip.id, OBJ, 7, stale)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.lost_at == 6 and res.processed == 8 and res.cleared == 1  # frames 1..8; 6, 7, 8 empty
    assert store.mask_indices(clip.id, OBJ) == [0, 1, 2, 3, 4, 5]


def test_anchor_skips_empty_frames_at_a_chunk_seam(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=10, white=(4, 5))
    res = run_propagation(store, eng, clip.id, 0, "forward", chunk_size=4, min_chunk_size=4, lost_after=5)
    assert res.lost_at is None and res.processed == 9
    assert store.mask_indices(clip.id, OBJ) == [0, 1, 2, 3, 6, 7, 8, 9]
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
    assert store.mask_indices(clip.id, OBJ) == [0, 1, 2, 3]
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

    def propagate(self, frames, seeds, cancel_cb=None):
        self.calls += 1
        if self.calls <= self.n_failures:
            gen = super().propagate(frames, seeds, cancel_cb)
            for _ in range(self.k + 1):
                yield next(gen)
            gen.close()
            raise ChunkOOMError(f"fake OOM after {self.k} real frames")
        yield from super().propagate(frames, seeds, cancel_cb)


class OomProgressThenStarve(FakeEngine):
    """First call OOMs after landing one real frame (progress). Every later call OOMs before
    yielding any frame at all (no progress)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def propagate(self, frames, seeds, cancel_cb=None):
        self.calls += 1
        self.propagate_calls.append(len(frames))
        (k, seed_mask), = seeds.items()
        if self.calls == 1:
            yield 0, {k: seed_mask.copy()}
            shifted = np.zeros_like(seed_mask)
            shifted[:, self.dx:] = seed_mask[:, :-self.dx]
            yield 1, {k: shifted}
        raise ChunkOOMError("fake OOM")


class OnlySeed(FakeEngine):
    """Yields only the anchor's own offset-0 output, then stops -- no real frame ever lands."""

    def propagate(self, frames, seeds, cancel_cb=None):
        self.propagate_calls.append(len(frames))
        yield 0, {k: v.copy() for k, v in seeds.items()}


class StopsAfterTwo(FakeEngine):
    """Yields the anchor plus exactly 2 real frames of a longer chunk, then stops without OOM or
    cancellation -- exercises the "stopped short with partial progress" PropagationFailed message."""

    def propagate(self, frames, seeds, cancel_cb=None):
        self.propagate_calls.append(len(frames))
        gen = super().propagate(frames, seeds, cancel_cb)
        for _ in range(3):  # offset 0 (anchor, discarded) + 2 real frames
            yield next(gen)
        gen.close()


class Skips(FakeEngine):
    """Yields offsets 0, 1, 3 -- skipping 2 -- to exercise offset validation."""

    def propagate(self, frames, seeds, cancel_cb=None):
        self.propagate_calls.append(len(frames))
        (k, seed_mask), = seeds.items()
        for offset in (0, 1, 3):
            if offset >= len(frames):
                return
            if offset == 0:
                yield 0, {k: seed_mask.copy()}
                continue
            shifted = np.zeros_like(seed_mask)
            s = offset * self.dx
            if s < seed_mask.shape[1]:
                shifted[:, s:] = seed_mask[:, :-s]
            yield offset, {k: shifted}


# ---------------------------------------------------------------- fix round 1: new tests


def test_mid_chunk_oom_resumes_at_the_first_unprocessed_frame(store, make_clip):
    eng = OomAfter(3)
    clip = _seeded(store, make_clip, eng, n_frames=13)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 12 and res.oom_retries == 1
    assert store.mask_indices(clip.id, OBJ) == list(range(13))
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
    assert store.mask_indices(clip2.id, OBJ) == list(range(8))


def test_oom_at_the_floor_without_progress_keeps_what_landed(store, make_clip):
    eng = OomProgressThenStarve()
    clip = _seeded(store, make_clip, eng, n_frames=13)
    with pytest.raises(PropagationFailed, match="chunk size 4"):
        run_propagation(store, eng, clip.id, 0, "forward", chunk_size=4, min_chunk_size=4, lost_after=3)
    assert store.mask_indices(clip.id, OBJ) == [0, 1]
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 1)]


def test_both_reports_a_lost_forward_pass(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=20, keyframes=(10,), white=range(13, 20))
    res = run_propagation(store, eng, clip.id, 10, "both", **KW)
    assert res.lost_at == 13
    assert res.lost_by_direction == {"forward": {OBJ: 13}}
    for i in range(10):
        assert store.read_mask(clip.id, OBJ, i) is not None


def test_both_when_both_directions_lose(store, make_clip):
    eng = FakeEngine(is_empty=is_white)
    clip = _seeded(store, make_clip, eng, n_frames=20, keyframes=(10,), white=(0, 1, 2, 13, 14, 15, 16, 17, 18, 19))
    res = run_propagation(store, eng, clip.id, 10, "both", **KW)
    assert res.lost_by_direction == {"forward": {OBJ: 13}, "backward": {OBJ: 2}}
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
    assert store.mask_indices(clip.id, OBJ) == [0, 1, 2]


def test_cancel_before_the_first_frame_of_a_chunk(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    res = run_propagation(store, eng, clip.id, 0, "forward", cancel=lambda: True, **KW)
    assert res.cancelled and res.processed == 0
    assert store.mask_indices(clip.id, OBJ) == [0]


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
    store.delete_mask(clip.id, OBJ, 0)
    with pytest.raises(BadRequest, match="has no mask"):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)


def test_propagate_result_as_dict(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert set(res.as_dict().keys()) == {
        "processed", "written", "cleared", "lost_at", "cancelled", "oom_retries", "lost_by_direction",
        "lost", "skipped", "objects",
    }


def test_out_of_order_engine_offsets_are_rejected(store, make_clip):
    eng = Skips()
    clip = _seeded(store, make_clip, eng, n_frames=10)
    with pytest.raises(PropagationFailed, match="expected 2"):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_mask(clip.id, OBJ, 1) is not None
    assert store.read_mask(clip.id, OBJ, 3) is None


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
    assert store.read_mask(clip.id, OBJ, 10) is not None


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
    store.write_mask(clip.id, OBJ, 7, stale)
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
    assert not [t for t in threading.enumerate() if t.name.startswith("vdp-prefetch")]

    eng = FakeEngine(oom_if_more_than=2)
    clip = _seeded(store, make_clip, eng)
    with pytest.raises(PropagationFailed):
        run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert not [t for t in threading.enumerate() if t.name.startswith("vdp-prefetch")]


def test_propagation_seeds_from_a_painted_keyframe_and_stops_at_it(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=20)
    painted = np.zeros((clip.height, clip.width), dtype=bool)
    painted[20:40, 60:80] = True  # x0 = 60, well away from the click seed's x0 = 20
    set_mask(store, clip.id, 10, painted, object_id=OBJ)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 9  # frames 1..9: the painted keyframe bounds the run
    assert _x0(store, clip.id, 10) == 60  # and is untouched
    res = run_propagation(store, eng, clip.id, 10, "forward", **KW)
    assert res.processed == 9 and _x0(store, clip.id, 11) == 62  # seeded by the painted mask, shifted once


# ---------------------------------------------------------------- multi-object propagation


def _two_objects(store, make_clip, eng, n_frames=30, white=(), a_kf=(0,), b_kf=(0,)):
    """Object 1 = a rect at x 20:40, object 2 = a rect at x 50:70 (both rows 40:55), class 0. Object 2 starts at 50,
    not further right, so the fake's +2 px per frame never pushes it below min_area_px within 30 frames."""
    _, clip = make_clip(n_frames=n_frames, white=white)
    clip = store.read_clip(clip.id)
    a, b = create_object(clip, 0), create_object(clip, 0)
    for obj, kfs, x in ((a, a_kf, 20), (b, b_kf, 50)):
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
    assert _xs(store, clip.id, 1, 9) == 20 + 18 and _xs(store, clip.id, 2, 9) == 50 + 18
    frame_events = [e for e in events if e["type"] == "frame"]
    assert frame_events[0]["objects"] == [{"id": 1, "has_mask": True}, {"id": 2, "has_mask": True}]
    assert "has_mask" not in frame_events[0] and frame_events[-1]["done"] == 29
    assert events[0] == {"type": "progress", "done": 0, "total": 29}
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
    assert len(eng.prepare_calls) == len(eng.propagate_calls)
    # object 2's frame 9 is tracked from its frame-7 anchor prepended to the chunk (offset 1), like the single-object
    # rule after a missed frame: the fake shifts by dx per chunk offset, not per frame distance
    assert _xs(store, clip.id, 2, 9) == 50 + 2 * 7 + 2


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
    clip = _two_objects(store, make_clip, eng, n_frames=20, white=(*range(3), *range(13, 20)),
                        a_kf=(8,), b_kf=(4, 8))
    res = run_propagation(store, eng, clip.id, 8, "both", **KW)
    assert res.lost_by_direction == {"forward": {1: 13, 2: 13}, "backward": {1: 2}}
    assert res.lost == {1: 13, 2: 13} and res.lost_at == 13  # forward runs first
    assert res.processed <= 11 + 8  # total = 11 forward frames (9..19) + 8 backward (7..0)
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


def _regions(store, clip_id, *regions):
    clip = store.read_clip(clip_id)
    clip.regions = list(regions)
    store.write_clip(clip)


def test_propagated_masks_are_trimmed_and_frames_get_grey_input(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=12)
    _regions(store, clip.id, RegionRange(start=0, end=11, blocks=[(36, 0, 50, 60)]))
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    m = store.read_mask(clip.id, OBJ, 2)  # seed x 20:40 shifted +4 -> 24:44, blocked from 36 -> 24:36
    assert m is not None and not m[:, 36:].any() and m[30, 30]


def test_a_mask_emptied_by_trimming_counts_as_not_found(store, make_clip):
    eng = FakeEngine(dx=10)
    clip = _seeded(store, make_clip, eng, n_frames=12)
    # from offset 3 on the shifted rect (x 50:70 and beyond) lies wholly in the block
    _regions(store, clip.id, RegionRange(start=1, end=11, blocks=[(45, 0, 120, 60)]))
    result = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_mask(clip.id, OBJ, 1) is not None
    assert store.read_mask(clip.id, OBJ, 3) is None
    assert result.lost_by_direction.get("forward", {}).get(OBJ) is not None


def test_an_object_whose_seed_is_wholly_excluded_is_skipped(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=12)
    _regions(store, clip.id, RegionRange(start=0, end=0, blocks=[(15, 15, 45, 45)]))
    with pytest.raises(BadRequest):  # the only object is skipped: nothing to propagate
        run_propagation(store, eng, clip.id, 0, "forward", **KW)
    with pytest.raises(BadRequest, match="excluded"):
        resolve_participants(store, store.read_clip(clip.id), 0, [OBJ], strict=True)
