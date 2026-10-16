"""Shared fixtures. Frames are 120x60 so FakeEngine's 20x20 click masks (400 px) pass the area bounds."""

from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest
from vdp.ingest import FrameDirSource, import_clip
from vdp.store import ProjectStore

# Registers train_fixtures.py's fixtures (e.g. `server`) for every test module without an explicit
# `from train_fixtures import server` -- that import trips ruff's F811 in tests that also take
# `server` as a parameter (see the comment in test_api_export.py).
pytest_plugins = ["train_fixtures"]

FRAME_W, FRAME_H = 120, 60
OBJ = 1  # the id of a clip's first object


def is_white(frame: np.ndarray) -> bool:
    """FakeEngine `is_empty` hook: pure-white frames mean 'object not visible'."""
    return bool(frame[0, 0, 0] > 250)


@pytest.fixture(autouse=True)
def _no_label_cache_warmup(monkeypatch):
    """Export and augment jobs never start a real Ultralytics label-cache warm-up in tests; a test that wants one
    passes `spawn=` to LabelCacheWarmer, or a fake `warmer=` to create_app."""
    from vdp.train import warm_cache

    monkeypatch.setattr(warm_cache, "default_spawn", lambda export_dir: None)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "data")


@pytest.fixture
def make_clip(store, tmp_path):
    def _make(n_frames: int = 12, white=(), classes=("part",)):
        src = tmp_path / f"src_{uuid.uuid4().hex[:6]}"
        src.mkdir()
        for i in range(n_frames):
            value = 255 if i in white else 90
            img = np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)
            assert cv2.imwrite(str(src / f"{i:04d}.png"), img)
        project = store.create_project("t")
        for name in classes:
            store.add_class(project.id, name)
        clip = import_clip(store, project.id, FrameDirSource(src))
        return store.read_project(project.id), clip

    return _make
