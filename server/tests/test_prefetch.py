import threading

import pytest
from vsp.engine.fake import FakeEngine
from vsp.labeling.prefetch import ChunkPrefetcher
from vsp.store import NotFound


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
