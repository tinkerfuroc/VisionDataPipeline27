import base64

import numpy as np
import pytest
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import BadRequest
from vsp.labeling.frames import delete_frame
from vsp.labeling.paint import decode_mask_png, set_mask
from vsp.labeling.prompts import OBJECT_ID, PromptSpec, accept
from vsp.labeling.propagate import run_propagation
from vsp.store.project import encode_mask_png

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])
KW = {"chunk_size": 8, "min_chunk_size": 4, "lost_after": 3}


def _blob(h, w, rows=slice(10, 30), cols=slice(10, 40)):
    m = np.zeros((h, w), dtype=bool)
    m[rows, cols] = True
    return m


def test_decode_round_trips_the_store_encoding_and_rejects_junk():
    m = _blob(48, 64)
    b64 = base64.b64encode(encode_mask_png(m)).decode("ascii")
    assert np.array_equal(decode_mask_png(b64), m)
    with pytest.raises(BadRequest, match="not a PNG"):
        decode_mask_png(base64.b64encode(b"not a png").decode("ascii"))
    with pytest.raises(BadRequest, match="not a PNG"):
        decode_mask_png("%%%")


def test_first_save_needs_a_class_creates_the_object_and_a_paint_keyframe(store, make_clip):
    _, clip = make_clip(n_frames=6)
    m = _blob(clip.height, clip.width)
    with pytest.raises(BadRequest, match="class_id is required"):
        set_mask(store, clip.id, 2, m)
    with pytest.raises(BadRequest, match="unknown class"):
        set_mask(store, clip.id, 2, m, class_id=7)
    out = set_mask(store, clip.id, 2, m, class_id=0)
    obj = out.objects[0]
    assert obj.id == OBJECT_ID and obj.class_id == 0
    assert [(k.frame_idx, k.prompt.type, k.prompt.payload) for k in obj.keyframes] == [(2, "paint", {})]
    assert np.array_equal(store.read_mask(clip.id, OBJECT_ID, 2), m)
    assert store.frame_states(out)[2] == "keyframe"


def test_later_saves_replace_the_mask_and_keep_one_keyframe_entry(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=12)
    accept(store, eng, clip.id, 0, CLICK, class_id=0)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 11)]
    m = _blob(clip.height, clip.width)
    out = set_mask(store, clip.id, 5, m)  # a propagated frame becomes a painted keyframe
    obj = out.objects[0]
    assert [k.frame_idx for k in obj.keyframes] == [0, 5] and obj.keyframes[1].prompt.type == "paint"
    assert obj.propagated_ranges == [(1, 4), (6, 11)]
    m2 = _blob(clip.height, clip.width, cols=slice(0, 5))
    out = set_mask(store, clip.id, 5, m2)
    assert [k.frame_idx for k in out.objects[0].keyframes] == [0, 5]
    assert np.array_equal(store.read_mask(clip.id, OBJECT_ID, 5), m2)
    out = set_mask(store, clip.id, 0, m)  # painting over a prompt keyframe replaces its prompt record
    assert out.objects[0].keyframes[0].prompt.type == "paint"
    out = set_mask(store, clip.id, 1, m, class_id=0)  # a class on a later save is accepted, like accept()
    assert out.objects[0].class_id == 0


def test_an_empty_mask_removes_the_mask_and_the_keyframe_but_keeps_the_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=6)
    accept(store, eng, clip.id, 0, CLICK, class_id=0)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    empty = np.zeros((clip.height, clip.width), dtype=bool)
    out = set_mask(store, clip.id, 0, empty)
    assert out.objects[0].keyframes == [] and out.objects[0].class_id == 0
    assert store.read_mask(clip.id, OBJECT_ID, 0) is None
    assert store.frame_states(out)[0] == "empty"
    out = set_mask(store, clip.id, 3, empty)  # a propagated frame: mask gone, range trimmed
    assert store.read_mask(clip.id, OBJECT_ID, 3) is None
    assert out.objects[0].propagated_ranges == [(1, 2), (4, 5)]
    out = set_mask(store, clip.id, 3, empty)  # clearing an already empty frame is not an error
    assert out.objects[0].propagated_ranges == [(1, 2), (4, 5)]


def test_refused_saves_change_nothing(store, make_clip):
    _, clip = make_clip(n_frames=6)
    empty = np.zeros((clip.height, clip.width), dtype=bool)
    with pytest.raises(BadRequest, match="no object"):
        set_mask(store, clip.id, 1, empty, class_id=0)
    m = _blob(clip.height, clip.width)
    with pytest.raises(BadRequest, match="out of range"):
        set_mask(store, clip.id, 6, m, class_id=0)
    delete_frame(store, clip.id, 1)
    with pytest.raises(BadRequest, match="deleted"):
        set_mask(store, clip.id, 1, m, class_id=0)
    with pytest.raises(BadRequest, match=f"the clip is {clip.width}×{clip.height}"):
        set_mask(store, clip.id, 2, m[:-1, :], class_id=0)
    assert store.read_clip(clip.id).objects == []
    assert store.mask_indices(clip.id, OBJECT_ID) == []
