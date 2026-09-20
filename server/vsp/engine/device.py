"""CUDA device selection. `auto` = first device with native bfloat16 (compute capability >= 8.0)."""

from __future__ import annotations

import re

_BF16_MIN_CAPABILITY = (8, 0)
_CUDA_RE = re.compile(r"^cuda:(\d+)$")


def pick_device(requested: str, capabilities: list[tuple[int, int]]) -> str | None:
    """Return a torch device string, or None when `auto` finds no bfloat16-capable GPU.

    `capabilities[i]` is the compute capability of `cuda:i` in PyTorch's ordering.
    """
    if requested == "auto":
        for idx, cap in enumerate(capabilities):
            if tuple(cap) >= _BF16_MIN_CAPABILITY:
                return f"cuda:{idx}"
        return None
    m = _CUDA_RE.match(requested)
    if not m:
        raise ValueError(f"sam3.device must be 'auto' or 'cuda:N', got {requested!r}")
    if int(m.group(1)) >= len(capabilities):
        raise ValueError(f"{requested} does not exist ({len(capabilities)} CUDA device(s) visible)")
    return requested


def cuda_capabilities() -> list[tuple[int, int]]:
    import torch

    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
