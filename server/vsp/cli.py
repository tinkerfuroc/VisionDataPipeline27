"""`vsp` command line. Subcommands are added by later tasks/plans."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, Settings
from .engine.device import cuda_capabilities, pick_device
from .weights import convert_checkpoint, verify_checkpoint


def _load_settings(config: str | None) -> Settings:
    if config:
        return Settings.load(config)
    return Settings.load(DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None)


def _report(problems: list[str], target: str) -> int:
    if not problems:
        print(f"OK: {target}")
        return 0
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1


def _doctor(settings: Settings) -> int:
    import torch

    print(f"torch {torch.__version__} (CUDA build {torch.version.cuda})")
    caps = cuda_capabilities()
    for i, cap in enumerate(caps):
        gib = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)} cc={cap[0]}.{cap[1]} {gib:.1f} GiB")
    if not caps:
        print("FAIL: CUDA is not available to this torch build")
        return 1
    dev = pick_device(settings.sam3.device, caps)
    if dev is None:
        print("FAIL: no CUDA device with compute capability >= 8.0 (bfloat16)")
        return 1
    print(f"selected: {dev}")
    a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    ok = bool(torch.isfinite((a @ a).float()).all())
    q = torch.randn(1, 8, 256, 64, device=dev, dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize(dev)
    print(f"bfloat16 matmul + attention: {'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vsp")
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the torch build, GPU selection and bfloat16 support")
    sub.add_parser("verify-weights", help="check that sam3.model_dir is a complete converted checkpoint")
    p_conv = sub.add_parser("convert-weights", help="convert Meta's public sam3.pt to HF format")
    p_conv.add_argument("--checkpoint", required=True, help="path to sam3.pt")
    p_conv.add_argument(
        "--converter", required=True,
        help="path to convert_sam3_video_to_hf.py downloaded from the transformers GitHub repo",
    )
    p_conv.add_argument("--out", required=True, help="output directory (HF format)")
    p_conv.add_argument(
        "--official-files-from", required=True,
        help="directory holding the official tokenizer/processor files (the facebook/sam3 repo files)",
    )
    args = parser.parse_args(argv)
    settings = _load_settings(args.config)
    if args.cmd == "doctor":
        return _doctor(settings)
    if args.cmd == "verify-weights":
        return _report(verify_checkpoint(settings.sam3.model_dir), str(settings.sam3.model_dir))
    if args.cmd == "convert-weights":
        problems = convert_checkpoint(
            Path(args.checkpoint), Path(args.converter), Path(args.out), Path(args.official_files_from)
        )
        return _report(problems, args.out)
    parser.error(f"unknown command {args.cmd}")
    return 2
