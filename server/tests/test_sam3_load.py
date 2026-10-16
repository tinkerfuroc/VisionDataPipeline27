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


def test_a_failed_load_returns_the_cuda_cache(tmp_path, monkeypatch):
    """An OOM half-way through moving the weights must not leave their blocks reserved in torch's
    caching allocator while SAM3 is not loaded (e.g. a Reload now while training holds the GPU)."""
    monkeypatch.setattr(vdp.engine.sam3, "pick_device", lambda *a, **k: "cuda:0")
    monkeypatch.setattr(vdp.engine.sam3, "cuda_capabilities", lambda: [(8, 6)])
    emptied = []
    monkeypatch.setattr(vdp.engine.sam3.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vdp.engine.sam3.torch.cuda, "empty_cache", lambda: emptied.append(1))
    engine = Sam3Engine(tmp_path / "nope")
    with pytest.raises(EngineNotLoaded):
        engine.load()
    assert emptied and engine._processor is None and engine._tracker_processor is None


class _ReleaseWinsLock:
    """Stands in for the engine lock when a concurrent release() gets it first: the engine is unloaded
    by the time the caller holds it."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def __enter__(self):
        self.engine._model = self.engine._processor = self.engine._tracker_processor = None
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_an_engine_call_that_loses_the_lock_to_release_is_engine_not_loaded(tmp_path):
    engine = Sam3Engine(tmp_path / "nope")
    engine._model = engine._processor = engine._tracker_processor = object()  # "loaded"
    engine._lock = _ReleaseWinsLock(engine)
    image = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(EngineNotLoaded):
        engine.predict_text(image, "x")
    engine._model = object()
    with pytest.raises(EngineNotLoaded):
        engine.predict_prompt(image, points=[(1, 1, 1)])
