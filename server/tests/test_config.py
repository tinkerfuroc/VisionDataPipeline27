from pathlib import Path

import pytest
from pydantic import ValidationError
from vsp.config import Settings


def test_defaults():
    s = Settings.load(None)
    assert s.server.host == "127.0.0.1"
    assert s.server.port == 28000
    assert s.sam3.device == "auto"
    assert s.sam3.dtype == "bfloat16"
    assert s.sam3.model_dir == Path("./vision_train/sam3_checkpoint_hf")
    assert s.propagate.chunk_size == 24
    assert s.propagate.min_chunk_size == 4
    assert s.propagate.lost_after == 15
    assert s.ingest.allowed_roots == ["~"]
    assert s.export.val_fraction == 0.2


def test_load_yaml_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("server:\n  port: 9000\nsam3:\n  device: 'cuda:1'\n")
    s = Settings.load(p)
    assert s.server.port == 9000
    assert s.sam3.device == "cuda:1"
    assert s.server.host == "127.0.0.1"  # untouched default


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("propagate:\n  chunk_sise: 10\n")  # typo must not pass silently
    with pytest.raises(ValidationError):
        Settings.load(p)


def test_repo_default_yaml_is_valid():
    repo_root = Path(__file__).resolve().parents[2]
    s = Settings.load(repo_root / "configs" / "default.yaml")
    assert s.sam3.score_mode == "native"
