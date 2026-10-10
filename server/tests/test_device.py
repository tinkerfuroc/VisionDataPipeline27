import pytest
from vdp.engine.device import pick_device


def test_auto_picks_first_bf16_capable():
    # This machine under PyTorch ordering: 3080 Ti (8.6) first, 2080 Ti (7.5) second.
    assert pick_device("auto", [(8, 6), (7, 5)]) == "cuda:0"
    # nvidia-smi ordering is the reverse; auto must still find the 8.x card.
    assert pick_device("auto", [(7, 5), (8, 6)]) == "cuda:1"


def test_auto_returns_none_without_capable_device():
    assert pick_device("auto", [(7, 5)]) is None
    assert pick_device("auto", []) is None


def test_explicit_index_is_honoured_even_if_not_bf16():
    assert pick_device("cuda:1", [(8, 6), (7, 5)]) == "cuda:1"


def test_explicit_index_out_of_range():
    with pytest.raises(ValueError, match="cuda:3"):
        pick_device("cuda:3", [(8, 6)])


def test_garbage_rejected():
    with pytest.raises(ValueError):
        pick_device("gpu0", [(8, 6)])
