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
