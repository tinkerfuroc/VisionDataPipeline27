import numpy as np
import pytest
from conftest import OBJ
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import BadRequest
from vsp.labeling.frames import clear_range
from vsp.labeling.objects import delete_object, relabel_object
from vsp.labeling.paint import set_mask
from vsp.labeling.prompts import PromptSpec, accept, preview
from vsp.store import NotFound

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])
TEXT = PromptSpec(type="text", phrase="part")


def _rect(x, width):
    m = np.zeros((60, 120), dtype=bool)
    m[20:40, x:x + width] = True
    return m


def test_a_second_object_gets_its_own_masks(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(classes=("cup", "bowl"))
    _, (a,) = accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    out, (b,) = accept(store, eng, clip.id, 0, PromptSpec(type="clicks", points=[(90, 30, 1)]), object_id=None, class_id=1)
    assert (a, b) == (1, 2) and [(o.id, o.class_id) for o in out.objects] == [(1, 0), (2, 1)]
    assert store.read_mask(clip.id, 1, 0)[30, 30] and not store.read_mask(clip.id, 1, 0)[30, 90]
    assert store.read_mask(clip.id, 2, 0)[30, 90]


def test_object_and_class_rules(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip()
    with pytest.raises(BadRequest, match="class_id is required"):
        accept(store, eng, clip.id, 0, CLICK, object_id=None)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    with pytest.raises(BadRequest, match="relabel the object instead"):
        accept(store, eng, clip.id, 1, CLICK, object_id=OBJ, class_id=0)
    with pytest.raises(NotFound, match="object 5"):
        accept(store, eng, clip.id, 1, CLICK, object_id=5)


def test_accept_all_candidates_as_new_objects(store, make_clip):
    eng = FakeEngine(n_text_candidates=3)
    _, clip = make_clip()
    out, ids = accept(store, eng, clip.id, 2, TEXT, object_id=None, class_id=0, candidates=[0, 2])
    assert ids == [1, 2] and [o.keyframes[0].prompt.payload["candidate"] for o in out.objects] == [0, 2]
    with pytest.raises(BadRequest, match="candidates needs object_id null"):
        accept(store, eng, clip.id, 2, TEXT, object_id=1, candidates=[1])


def test_click_box_is_derived_from_the_chosen_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip()
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)  # object 1 around (30, 30)
    set_mask(store, clip.id, 0, _rect(x=70, width=30), object_id=None, class_id=0)  # object 2 at x 70..100
    clip = store.read_clip(clip.id)
    click = PromptSpec(type="clicks", points=[(80, 30, 1)])
    (det,) = preview(store, eng, clip, 0, click, object_id=2)
    assert det.mask[30, 95] and not det.mask[30, 25]  # box from object 2's mask, not object 1's
    (det,) = preview(store, eng, clip, 0, click, object_id=None)
    assert not det.mask[30, 95]  # no derived box: the fake's 20x20 square around the click


def test_relabel_and_delete_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(classes=("cup", "bowl"))
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    accept(store, eng, clip.id, 0, CLICK, object_id=None, class_id=0)
    assert relabel_object(store, clip.id, 2, 1).objects[1].class_id == 1
    with pytest.raises(BadRequest, match="unknown class id 9"):
        relabel_object(store, clip.id, 2, 9)
    out = delete_object(store, clip.id, 1)
    assert [o.id for o in out.objects] == [2] and store.mask_indices(clip.id, 1) == []
    with pytest.raises(NotFound):
        delete_object(store, clip.id, 1)


def test_clear_range_per_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip()
    for _ in range(2):
        accept(store, eng, clip.id, 3, CLICK, object_id=None, class_id=0)
    clear_range(store, clip.id, 0, 5, objects=[2])
    assert store.mask_indices(clip.id, 1) == [3] and store.mask_indices(clip.id, 2) == []
    with pytest.raises(NotFound):
        clear_range(store, clip.id, 0, 5, objects=[7])
