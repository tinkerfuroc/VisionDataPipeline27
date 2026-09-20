"""`vsp` command line. Subcommands are added by later tasks/plans."""

from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG_PATH, Settings
from .engine.device import cuda_capabilities, pick_device


def _load_settings(config: str | None) -> Settings:
    if config:
        return Settings.load(config)
    return Settings.load(DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None)


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
    args = parser.parse_args(argv)
    settings = _load_settings(args.config)
    if args.cmd == "doctor":
        return _doctor(settings)
    parser.error(f"unknown command {args.cmd}")
    return 2
