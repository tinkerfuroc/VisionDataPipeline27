import json

import numpy as np
from safetensors.numpy import save_file
from vdp.weights import (
    EXPECTED_TRACKER_NECK_TENSORS,
    rename_public_prefixes,
    verify_checkpoint,
)


def _make_checkpoint(d, *, neck=EXPECTED_TRACKER_NECK_TENSORS, tokenizer_bytes=200_000):
    d.mkdir(parents=True, exist_ok=True)
    tensors = {f"tracker_neck.fpn_layers.{i}.w": np.zeros((1,), dtype=np.float32) for i in range(neck)}
    tensors["detector_model.x"] = np.zeros((1,), dtype=np.float32)
    save_file(tensors, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"model_type": "sam3_video"}))
    (d / "processor_config.json").write_text("{}")
    (d / "tokenizer_config.json").write_text("{}")
    (d / "tokenizer.json").write_text("x" * tokenizer_bytes)
    return d


def test_rename_public_prefixes():
    sd = {"detector.backbone.w": 1, "tracker.sam_mask_decoder.w": 2, "other.w": 3}
    assert rename_public_prefixes(sd) == {
        "sam3_model.backbone.w": 1,
        "sam2_predictor.model.sam_mask_decoder.w": 2,
        "other.w": 3,
    }


def test_verify_ok(tmp_path):
    assert verify_checkpoint(_make_checkpoint(tmp_path / "ok")) == []


def test_verify_flags_stub_tokenizer(tmp_path):
    problems = verify_checkpoint(_make_checkpoint(tmp_path / "c", tokenizer_bytes=1884))
    assert len(problems) == 1 and "stub" in problems[0]


def test_verify_flags_missing_tracker_neck(tmp_path):
    problems = verify_checkpoint(_make_checkpoint(tmp_path / "c", neck=0))
    assert len(problems) == 1 and "tracker_neck" in problems[0] and "0" in problems[0]


def test_verify_flags_missing_files(tmp_path):
    d = _make_checkpoint(tmp_path / "c")
    (d / "model.safetensors").unlink()
    (d / "processor_config.json").unlink()
    problems = verify_checkpoint(d)
    assert any("model.safetensors" in p for p in problems)
    assert any("processor_config.json" in p for p in problems)


def test_verify_missing_dir(tmp_path):
    assert verify_checkpoint(tmp_path / "nope") == [f"{tmp_path / 'nope'} is not a directory"]
