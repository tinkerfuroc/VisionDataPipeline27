"""SAM3 checkpoint conversion (public sam3.pt -> HF format) and verification."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

EXPECTED_TRACKER_NECK_TENSORS = 22
_REQUIRED_FILES = ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json")
_OFFICIAL_FILES = (
    "tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json",
    "special_tokens_map.json", "processor_config.json",
)
_MIN_REAL_TOKENIZER_BYTES = 100_000  # the real file is ~3.6 MB; the offline stub is ~2 KB
# HF's converter was written against Meta-internal key prefixes; the public release differs.
_PUBLIC_TO_INTERNAL = (("detector.", "sam3_model."), ("tracker.", "sam2_predictor.model."))


def rename_public_prefixes(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        for public, internal in _PUBLIC_TO_INTERNAL:
            if key.startswith(public):
                key = internal + key[len(public):]
                break
        out[key] = value
    return out


def verify_checkpoint(model_dir: str | Path) -> list[str]:
    d = Path(model_dir)
    if not d.is_dir():
        return [f"{d} is not a directory"]
    problems = [f"missing file: {name}" for name in _REQUIRED_FILES if not (d / name).is_file()]
    tok = d / "tokenizer.json"
    if tok.is_file() and tok.stat().st_size < _MIN_REAL_TOKENIZER_BYTES:
        problems.append(
            f"tokenizer.json is a {tok.stat().st_size}-byte stub; copy the official tokenizer files over it"
        )
    weights = d / "model.safetensors"
    if weights.is_file():
        from safetensors import safe_open

        with safe_open(str(weights), framework="np") as f:
            n = sum(1 for k in f.keys() if k.startswith("tracker_neck."))  # noqa: SIM118
        if n != EXPECTED_TRACKER_NECK_TENSORS:
            problems.append(f"expected {EXPECTED_TRACKER_NECK_TENSORS} top-level tracker_neck.* tensors, found {n}")
    return problems


def convert_checkpoint(checkpoint: Path, converter: Path, out_dir: Path, official_files_from: Path) -> list[str]:
    """Run HF's convert_sam3_video_to_hf.py on the *public* sam3.pt, then repair tokenizer/config."""
    spec = importlib.util.spec_from_file_location("hf_sam3_video_converter", str(converter))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import converter script: {converter}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_loader = module.load_original_state_dict
    module.load_original_state_dict = lambda path: rename_public_prefixes(original_loader(path))
    module.convert_sam3_checkpoint(checkpoint_path=str(checkpoint), output_path=str(out_dir))

    for name in _OFFICIAL_FILES:
        src = Path(official_files_from) / name
        if src.is_file():
            shutil.copyfile(src, Path(out_dir) / name)
    cfg_path = Path(out_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["recondition_on_trk_masks"] = False  # official facebook/sam3 value; the converter defaults to true
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return verify_checkpoint(out_dir)
