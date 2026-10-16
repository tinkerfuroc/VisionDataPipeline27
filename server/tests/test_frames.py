import pytest
from conftest import OBJ
from vdp.engine.fake import FakeEngine
from vdp.labeling.errors import BadRequest
from vdp.labeling.frames import clear_range, delete_frame, prune_from, restore_frame, set_range_deleted
from vdp.labeling.prompts import PromptSpec, accept
from vdp.labeling.propagate import run_propagation
from vdp.store import NotFound

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])


def test_delete_and_restore_are_idempotent_and_keep_files(store, make_clip):
    _, clip = make_clip(n_frames=6)
    delete_frame(store, clip.id, 4)
    delete_frame(store, clip.id, 1)
    out = delete_frame(store, clip.id, 4)
    assert out.deleted_frames == [1, 4]
    assert store.frame_path(clip.id, 4).is_file()
    assert store.frame_states(out)[4] == "deleted"
    out = restore_frame(store, clip.id, 4)
    assert restore_frame(store, clip.id, 4).deleted_frames == [1] == out.deleted_frames
    assert store.read_clip(clip.id).deleted_frames == [1]


def test_prune_from_marks_the_tail(store, make_clip):
    _, clip = make_clip(n_frames=6)
    delete_frame(store, clip.id, 0)
    assert prune_from(store, clip.id, 3).deleted_frames == [0, 3, 4, 5]


def test_range_delete_and_restore(store, make_clip):
    _, clip = make_clip(n_frames=8)
    delete_frame(store, clip.id, 1)
    assert set_range_deleted(store, clip.id, 2, 4, True).deleted_frames == [1, 2, 3, 4]
    assert set_range_deleted(store, clip.id, 0, 3, False).deleted_frames == [4]  # already-kept frames are fine
    assert store.frame_path(clip.id, 3).is_file()
    assert set_range_deleted(store, clip.id, 5, 5, True).deleted_frames == [4, 5]
    with pytest.raises(BadRequest, match="after"):
        set_range_deleted(store, clip.id, 4, 2, True)
    with pytest.raises(BadRequest, match="out of range"):
        set_range_deleted(store, clip.id, 6, 8, True)
    assert store.read_clip(clip.id).deleted_frames == [4, 5]  # a refused request changes nothing


def test_index_validation(store, make_clip):
    _, clip = make_clip(n_frames=6)
    for call in (
        lambda: delete_frame(store, clip.id, 6),
        lambda: restore_frame(store, clip.id, -1),
        lambda: prune_from(store, clip.id, 9),
        lambda: clear_range(store, clip.id, 2, 6),
        lambda: clear_range(store, clip.id, 4, 2),
    ):
        with pytest.raises(BadRequest):
            call()


def test_clear_range_removes_keyframes_too_and_trims_ranges(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=12)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    accept(store, eng, clip.id, 6, CLICK, object_id=OBJ)
    run_propagation(store, eng, clip.id, 0, "forward", chunk_size=8, min_chunk_size=4, lost_after=3)
    run_propagation(store, eng, clip.id, 6, "forward", chunk_size=8, min_chunk_size=4, lost_after=3)
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 5), (7, 11)]
    out = clear_range(store, clip.id, 4, 8)
    assert store.mask_indices(clip.id, OBJ) == [0, 1, 2, 3, 9, 10, 11]  # keyframe 6 cleared with the rest
    assert out.objects[0].propagated_ranges == [(1, 3), (9, 11)]
    assert [k.frame_idx for k in out.objects[0].keyframes] == [0]
    assert store.frame_states(out)[4:9] == ["empty"] * 5
    # clearing the last keyframe leaves the object (and its class) in place for the next accepted prompt
    out = clear_range(store, clip.id, 0, 0)
    assert out.objects[0].keyframes == [] and out.objects[0].class_id == 0
    assert store.mask_indices(clip.id, OBJ) == [1, 2, 3, 9, 10, 11]


def test_clear_range_without_an_object_is_a_no_op(store, make_clip):
    _, clip = make_clip(n_frames=6)
    assert clear_range(store, clip.id, 0, 5).objects == []


def test_unknown_clip_does_not_grow_the_lock_table(store):
    with pytest.raises(NotFound):
        delete_frame(store, "nope", 0)
    assert "nope" not in store._locks


def test_a_prompt_accepted_after_the_last_keyframe_was_cleared_lands_on_the_same_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=6)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    clear_range(store, clip.id, 0, 0)
    out, _ = accept(store, eng, clip.id, 3, CLICK, object_id=OBJ)  # no class_id: the object, and its class, are still there
    assert [o.id for o in out.objects] == [OBJ] and out.objects[0].class_id == 0
    assert [k.frame_idx for k in out.objects[0].keyframes] == [3]
    assert store.mask_indices(clip.id, OBJ) == [3]
