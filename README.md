# vision-sampler-pipeline

SAM3-driven video labeling for one user-chosen component per clip.
Design: `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`.

## Setup

    uv sync
    uv run vsp doctor          # torch build, GPU selection, bfloat16
    uv run vsp verify-weights  # converted SAM3 checkpoint is complete
    uv run pytest              # GPU tests skip themselves without CUDA + checkpoint

Python, torch (`2.14.0+cu126`) and transformers (`5.17.0`) are pinned in `pyproject.toml`.
Do not install torch from PyPI: its default build needs an NVIDIA driver >= 580.

## Verified on this machine

- Date: 2026-09-20
- GPU: NVIDIA GeForce RTX 3080 Ti, driver 535.183.01, torch 2.14.0+cu126, transformers 5.17.0
- GPU smoke tests: 9 passed
- Peak VRAM, 24-frame propagate at 640x480: 2.47 GiB
- Text prompt "red circle" on the synthetic scene: 1 detection(s)
