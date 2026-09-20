import numpy as np
import pytest
from vsp.engine.fake import FakeEngine
from vsp.engine.types import ChunkOOMError, Engine


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
    out = list(eng.propagate(_frames(4), _seed()))
    assert [o for o, _ in out] == [0, 1, 2, 3]
    assert np.array_equal(out[0][1], _seed())
    _ys, xs = np.where(out[2][1])
    assert (xs.min(), xs.max() + 1) == (16, 36)  # 10 + 2*3
    assert eng.propagate_calls == [4]


def test_propagate_reports_none_where_is_empty_says_so():
    frames = _frames(4)
    frames[2][0, 0, 0] = 255
    eng = FakeEngine(is_empty=lambda fr: fr[0, 0, 0] == 255)
    out = dict(eng.propagate(frames, _seed()))
    assert out[2] is None and out[1] is not None and out[3] is not None


def test_propagate_oom_when_chunk_too_large():
    eng = FakeEngine(oom_if_more_than=5)
    with pytest.raises(ChunkOOMError):
        list(eng.propagate(_frames(6), _seed()))
    assert len(list(eng.propagate(_frames(5), _seed()))) == 5


def test_propagate_honours_cancel():
    eng = FakeEngine()
    seen = []

    def cancel():
        return len(seen) >= 2

    for off, _ in eng.propagate(_frames(10), _seed(), cancel_cb=cancel):
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
