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
