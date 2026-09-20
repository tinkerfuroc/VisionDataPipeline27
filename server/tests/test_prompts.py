import numpy as np
import pytest
from pydantic import ValidationError
from vsp.engine.fake import FakeEngine
from vsp.engine.masks import bbox_of
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
    _project, clip = make_clip()
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


def test_box_is_normalised_regardless_of_drag_direction():
    assert PromptSpec(type="box", box=(80, 30, 40, 10)).box == (40.0, 10.0, 80.0, 30.0)
    assert PromptSpec(type="box", box=(40, 10, 80, 30)).box == (40.0, 10.0, 80.0, 30.0)


def test_box_with_non_finite_coordinates_is_a_bad_request():
    with pytest.raises(BadRequest):
        PromptSpec(type="box", box=(float("nan"), 0, 10, 10))
    with pytest.raises(BadRequest):
        PromptSpec(type="box", box=(0, 0, float("inf"), 10))


def test_inverted_box_gives_the_same_mask_and_payload_stores_the_normalised_box(store, make_clip):
    _, clip = make_clip()
    eng = FakeEngine()
    normal = PromptSpec(type="box", box=(40, 10, 80, 30))
    inverted = PromptSpec(type="box", box=(80, 30, 40, 10))
    normal_mask = preview(store, eng, clip, 0, normal)[0].mask
    inverted_mask = preview(store, eng, clip, 0, inverted)[0].mask
    assert np.array_equal(normal_mask, inverted_mask)
    out = accept(store, eng, clip.id, 0, inverted, class_id=0)
    assert out.objects[0].keyframes[0].prompt.payload["box"] == [40, 10, 80, 30]


def test_box_partly_outside_the_frame_is_clamped_and_works(store, make_clip):
    _, clip = make_clip()
    eng = FakeEngine()
    prompt = PromptSpec(type="box", box=(100, 10, 140, 30))
    dets = preview(store, eng, clip, 0, prompt)
    assert len(dets) == 1
    assert bbox_of(dets[0].mask) == (100, 10, 120, 30)  # clamped to frame width 120


def test_box_fully_outside_the_frame_is_a_bad_request(store, make_clip):
    _, clip = make_clip()
    prompt = PromptSpec(type="box", box=(130, 10, 150, 30))
    with pytest.raises(BadRequest, match="empty or outside"):
        preview(store, FakeEngine(), clip, 0, prompt)


def test_zero_area_box_is_a_bad_request(store, make_clip):
    _, clip = make_clip()
    prompt = PromptSpec(type="box", box=(10, 10, 10, 10))
    with pytest.raises(BadRequest, match="empty or outside"):
        preview(store, FakeEngine(), clip, 0, prompt)


def test_point_outside_the_frame_is_a_bad_request(store, make_clip):
    _, clip = make_clip()
    with pytest.raises(BadRequest, match="outside the frame"):
        preview(store, FakeEngine(), clip, 0, PromptSpec(type="clicks", points=[(-1, 5, 1)]))
    with pytest.raises(BadRequest, match="outside the frame"):
        preview(store, FakeEngine(), clip, 0, PromptSpec(type="clicks", points=[(120, 5, 1)]))  # x == width


def test_accept_rechecks_deleted_inside_the_lock(store, make_clip, monkeypatch):
    """A delete racing between accept's first (unlocked) read and its locked re-read must not leave a
    frame that is both deleted and a keyframe."""
    _, clip = make_clip(n_frames=6)
    eng = FakeEngine()
    real_lock = store.lock

    def sneaky_lock(clip_id):
        c = store.read_clip(clip_id)
        c.deleted_frames = [0]
        store.write_clip(c)
        return real_lock(clip_id)

    monkeypatch.setattr(store, "lock", sneaky_lock)
    with pytest.raises(BadRequest, match="deleted"):
        accept(store, eng, clip.id, 0, CLICK, class_id=0)
    assert store.read_clip(clip.id).objects == []
    assert store.mask_indices(clip.id, OBJECT_ID) == []
