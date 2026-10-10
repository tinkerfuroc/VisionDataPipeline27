import numpy as np
import pytest
from vdp.engine.fake import FakeEngine
from vdp.engine.types import ChunkOOMError, Engine


def _frames(n, h=60, w=120):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def _seed(h=60, w=120):
    m = np.zeros((h, w), dtype=bool)
    m[20:40, 10:30] = True
    return m


def test_satisfies_protocol():
    engine: Engine = FakeEngine()
    assert engine.is_loaded()


def test_propagate_shifts_seed_right_by_dx_per_frame():
    eng = FakeEngine(dx=3)
    out = list(eng.propagate(_frames(4), {1: _seed()}))
    assert [o for o, _ in out] == [0, 1, 2, 3]
    assert np.array_equal(out[0][1][1], _seed())
    _ys, xs = np.where(out[2][1][1])
    assert (xs.min(), xs.max() + 1) == (16, 36)  # 10 + 2*3
    assert eng.propagate_calls == [4]


def test_propagate_reports_none_where_is_empty_says_so():
    frames = _frames(4)
    frames[2][0, 0, 0] = 255
    eng = FakeEngine(is_empty=lambda fr: fr[0, 0, 0] == 255)
    out = dict(eng.propagate(frames, {1: _seed()}))
    assert out[2][1] is None and out[1][1] is not None and out[3][1] is not None


def test_propagate_oom_when_chunk_too_large():
    eng = FakeEngine(oom_if_more_than=5)
    with pytest.raises(ChunkOOMError):
        list(eng.propagate(_frames(6), {1: _seed()}))
    assert len(list(eng.propagate(_frames(5), {1: _seed()}))) == 5


def test_propagate_honours_cancel():
    eng = FakeEngine()
    seen = []

    def cancel():
        return len(seen) >= 2

    for off, _ in eng.propagate(_frames(10), {1: _seed()}, cancel_cb=cancel):
        seen.append(off)
    assert seen == [0, 1]


def test_predict_prompt_box_points_and_negative():
    eng = FakeEngine()
    img = _frames(1)[0]
    det = eng.predict_prompt(img, box=(10, 20, 50, 40))
    assert det.bbox == (10, 20, 50, 40) and det.mask[25, 30]
    det = eng.predict_prompt(img, points=[(60, 30, 1)])
    assert det.mask[30, 60] and not det.mask[0, 0]
    det = eng.predict_prompt(img, points=[(60, 30, 1), (60, 30, 0)])
    assert not det.mask[30, 60]  # negative click carves a hole
    assert eng.predict_prompt(img) is None


def test_predict_text_returns_ranked_candidates():
    eng = FakeEngine(n_text_candidates=3)
    dets = eng.predict_text(_frames(1)[0], "connector")
    assert len(dets) == 3
    assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)
    assert all(d.phrase == "connector" for d in dets)
    assert not (dets[0].mask & dets[1].mask).any()  # distinct objects


def test_predict_prompt_rejects_a_speck_box():
    eng = FakeEngine()
    img = _frames(1)[0]
    assert eng.predict_prompt(img, box=(10, 20, 15, 25)) is None  # 5x5 = 25 px < 64


def test_predict_prompt_rejects_a_frame_covering_box():
    eng = FakeEngine()
    img = _frames(1)[0]  # 60x120
    assert eng.predict_prompt(img, box=(0, 0, 120, 60)) is None


def test_propagate_yields_none_once_the_shifted_mask_shrinks_below_min_area():
    # seed is a 20x20 (400 px) block at columns [10,30); with dx=27 the visible sliver at
    # offset 4 (shift 108) is only 2 columns wide (40 px), well under min_area_px=64, but
    # still nonzero -- so this exercises the degenerate-mask filter, not the "shifted off
    # the frame entirely" all-zero case that was already handled.
    eng = FakeEngine(dx=27)
    out = dict(eng.propagate(_frames(5), {1: _seed()}))
    assert out[0][1] is not None and out[1][1] is not None and out[2][1] is not None and out[3][1] is not None
    assert out[4][1] is None


def test_tracks_every_seed_independently():
    eng = FakeEngine(dx=2)
    a = _seed()
    b = np.zeros_like(a)
    b[40:50, 60:70] = True
    c = np.zeros_like(a)
    c[5:20, 90:110] = True
    out = dict(eng.propagate(_frames(3), {1: a, 5: b, 7: c}))
    assert set(out[2]) == {1, 5, 7}
    assert np.array_equal(out[2][1][:, 4:], a[:, :-4]) and np.array_equal(out[2][5][:, 4:], b[:, :-4])
    assert eng.seed_calls == [[1, 5, 7]]


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


def test_fake_engine_knobs_accept_the_speck_box():
    eng = FakeEngine(min_area_px=1, max_area_frac=1.0)
    img = _frames(1)[0]
    det = eng.predict_prompt(img, box=(10, 20, 15, 25))
    assert det is not None and det.bbox == (10, 20, 15, 25)


def test_prepare_returns_the_frames_and_propagate_accepts_them():
    eng = FakeEngine()
    frames = [np.full((40, 60, 3), 90, dtype=np.uint8) for _ in range(4)]
    seed = np.zeros((40, 60), dtype=bool)
    seed[10:20, 5:15] = True
    prepared = eng.prepare(frames)
    assert isinstance(prepared, list) and len(prepared) == 4 and prepared is not frames
    assert eng.prepare_calls == [4]
    raw = [(o, {k: (None if m is None else m.copy()) for k, m in d.items()}) for o, d in eng.propagate(frames, {1: seed})]
    via_prepared = list(eng.propagate(prepared, {1: seed}))
    assert [o for o, _ in raw] == [o for o, _ in via_prepared]
    for (_, a), (_, b) in zip(raw, via_prepared):
        assert (a[1] is None) == (b[1] is None)
        if a[1] is not None:
            assert np.array_equal(a[1], b[1])
    assert eng.propagate_calls == [4, 4]
