import base64

import numpy as np
import pytest
from conftest import OBJ
from vdp.engine.fake import FakeEngine
from vdp.labeling.errors import BadRequest
from vdp.labeling.frames import delete_frame
from vdp.labeling.objects import relabel_object
from vdp.labeling.paint import decode_mask_png, set_mask
from vdp.labeling.prompts import PromptSpec, accept
from vdp.labeling.propagate import run_propagation
from vdp.store.project import encode_mask_png

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
        set_mask(store, clip.id, 2, m, object_id=None)
    with pytest.raises(BadRequest, match="unknown class"):
        set_mask(store, clip.id, 2, m, object_id=None, class_id=7)
    out, oid = set_mask(store, clip.id, 2, m, object_id=None, class_id=0)
    assert oid == OBJ
    obj = out.objects[0]
    assert obj.id == OBJ and obj.class_id == 0
    assert [(k.frame_idx, k.prompt.type, k.prompt.payload) for k in obj.keyframes] == [(2, "paint", {})]
    assert np.array_equal(store.read_mask(clip.id, OBJ, 2), m)
    assert store.frame_states(out)[2] == "keyframe"


def test_later_saves_replace_the_mask_and_keep_one_keyframe_entry(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=12)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 11)]
    m = _blob(clip.height, clip.width)
    out, _ = set_mask(store, clip.id, 5, m, object_id=OBJ)  # a propagated frame becomes a painted keyframe
    obj = out.objects[0]
    assert [k.frame_idx for k in obj.keyframes] == [0, 5] and obj.keyframes[1].prompt.type == "paint"
    assert obj.propagated_ranges == [(1, 4), (6, 11)]
    m2 = _blob(clip.height, clip.width, cols=slice(0, 5))
    out, _ = set_mask(store, clip.id, 5, m2, object_id=OBJ)
    assert [k.frame_idx for k in out.objects[0].keyframes] == [0, 5]
    assert np.array_equal(store.read_mask(clip.id, OBJ, 5), m2)
    out, _ = set_mask(store, clip.id, 0, m, object_id=OBJ)  # painting over a prompt keyframe replaces its prompt record
    assert out.objects[0].keyframes[0].prompt.type == "paint"
    out, _ = set_mask(store, clip.id, 1, m, object_id=OBJ)
    out = relabel_object(store, clip.id, OBJ, 0)  # a class on a later save is a relabel now, not part of set_mask
    assert out.objects[0].class_id == 0


def test_an_empty_mask_removes_the_mask_and_the_keyframe_but_keeps_the_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=6)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    empty = np.zeros((clip.height, clip.width), dtype=bool)
    out, _ = set_mask(store, clip.id, 0, empty, object_id=OBJ)
    assert out.objects[0].keyframes == [] and out.objects[0].class_id == 0
    assert store.read_mask(clip.id, OBJ, 0) is None
    assert store.frame_states(out)[0] == "empty"
    out, _ = set_mask(store, clip.id, 3, empty, object_id=OBJ)  # a propagated frame: mask gone, range trimmed
    assert store.read_mask(clip.id, OBJ, 3) is None
    assert out.objects[0].propagated_ranges == [(1, 2), (4, 5)]
    out, _ = set_mask(store, clip.id, 3, empty, object_id=OBJ)  # clearing an already empty frame is not an error
    assert out.objects[0].propagated_ranges == [(1, 2), (4, 5)]


def test_refused_saves_change_nothing(store, make_clip):
    _, clip = make_clip(n_frames=6)
    empty = np.zeros((clip.height, clip.width), dtype=bool)
    with pytest.raises(BadRequest, match="the mask is empty"):
        set_mask(store, clip.id, 1, empty, object_id=None, class_id=0)
    m = _blob(clip.height, clip.width)
    with pytest.raises(BadRequest, match="out of range"):
        set_mask(store, clip.id, 6, m, object_id=None, class_id=0)
    delete_frame(store, clip.id, 1)
    with pytest.raises(BadRequest, match="deleted"):
        set_mask(store, clip.id, 1, m, object_id=None, class_id=0)
    with pytest.raises(BadRequest, match=f"the clip is {clip.width}×{clip.height}"):
        set_mask(store, clip.id, 2, m[:-1, :], object_id=None, class_id=0)
    assert store.read_clip(clip.id).objects == []
    assert store.mask_indices(clip.id, OBJ) == []


def test_decode_refuses_a_decompression_bomb_and_reads_alpha_only_masks():
    import io
    import struct
    import zlib

    from PIL import Image

    # A PNG whose header claims 100000×100000 pixels: Pillow refuses to decode it, and so must we (a 400, not a 500).
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    bomb = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 100_000, 100_000, 8, 0, 0, 0, 0))
    bomb += chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")
    with pytest.raises(BadRequest, match="not a PNG"):
        decode_mask_png(base64.b64encode(bomb).decode("ascii"))

    # A mask encoded only in the alpha channel (transparent background, opaque foreground): still a mask.
    img = Image.new("RGBA", (8, 4), (255, 255, 255, 0))
    for x in range(2, 6):
        img.putpixel((x, 1), (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    got = decode_mask_png(base64.b64encode(buf.getvalue()).decode("ascii"))
    expected = np.zeros((4, 8), dtype=bool)
    expected[1, 2:6] = True
    assert np.array_equal(got, expected)
