"""CPU-only test for Sam3Engine.load()'s exception-wrapping contract. Needs neither a GPU nor the
checkpoint: pick_device/cuda_capabilities are monkeypatched so load() fails on the missing model
directory instead."""

import numpy as np
import pytest
import vdp.engine.sam3
from vdp.engine.sam3 import Sam3Engine
from vdp.engine.types import EngineNotLoaded


def test_load_wraps_unexpected_exceptions_as_engine_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(vdp.engine.sam3, "pick_device", lambda *a, **k: "cuda:0")
    monkeypatch.setattr(vdp.engine.sam3, "cuda_capabilities", lambda: [(8, 6)])
    engine = Sam3Engine(tmp_path / "nope")

    with pytest.raises(EngineNotLoaded):
        engine.load()

    assert isinstance(engine.load_error, str) and engine.load_error
    assert not engine.is_loaded()

    with pytest.raises(EngineNotLoaded) as excinfo:
        engine.predict_text(np.zeros((8, 8, 3), np.uint8), "x")
    assert engine.load_error in str(excinfo.value)
