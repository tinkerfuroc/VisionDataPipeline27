# Vision Sampler — Plan 1 of 4: Backend Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless, tested Python package (`vsp`) that can import a video or frame directory into an on-disk project, and run SAM3 text/click/box prompting and single-object propagation on the GPU.

**Architecture:** `engine` (arrays in, masks out; no disk or HTTP knowledge), `ingest` (any source → `frames/NNNNNN.jpg`), and `store` (the only module that touches `data/projects/`) are independent units behind small interfaces. A deterministic `FakeEngine` implements the same `Engine` protocol as the real `Sam3Engine`, so everything above the engine is testable without a GPU. Later plans add labeling/API (Plan 2), the web GUI (Plan 3), and exporters (Plan 4) — see `2026-09-20-vsp-roadmap.md`.

**Tech Stack:** Python 3.11, uv, PyTorch 2.14.0+cu126, transformers 5.17.0 (HF SAM3), OpenCV (headless), Pillow, Pydantic v2, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md` — read §2–§6 and §11 before starting.

## Global Constraints

- Environment is managed **only with uv**: `uv sync`, `uv run …`, from the repository root. No conda, no `requirements.txt`, no bare `pip install`.
- `requires-python = ">=3.11,<3.12"`.
- Exact pins: `torch==2.14.0+cu126`, `torchvision==0.29.0+cu126`, `transformers==5.17.0`. `torch`/`torchvision` resolve from the explicit index `https://download.pytorch.org/whl/cu126`. **Never** let torch resolve from PyPI (its default build needs driver ≥ 580; this machine has 535). If the PyTorch index is unreachable, retry, then use `uv sync --find-links https://mirrors.aliyun.com/pytorch-wheels/cu126/`.
- No system ffmpeg; video decoding is OpenCV only.
- The SAM3 device is chosen by `sam3.device: auto` = first CUDA device with compute capability ≥ 8.0. Never hard-code a CUDA index (PyTorch and `nvidia-smi` disagree on this machine).
- Converted checkpoint lives at `vision_train/sam3_checkpoint_hf/` (already present, 3.4 GB). `vision_train/`, `data/`, `*.pt`, `*.safetensors` must never be committed.
- `.gitignore` data rule is anchored: `/data/` (an unanchored `data/` is what broke the template).
- Single object per clip (v1). The store schema keeps `objects: [...]`; enforcement is Plan 2's job, not the store's.
- An absent mask PNG means "object not present in this frame". Never write an empty mask file.
- Files ported from the template carry a header crediting `tinkerfuroc/tk25_vision_train` (MIT, © 2025 tinker@furoc) at commit `5874ba0`.
- Package source is `server/vsp`, tests are `server/tests`, run as `uv run pytest`.
- Commit after every task with the message given in the task. End every commit message with the two attribution lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL`

## File Structure (this plan)

```
pyproject.toml, uv.lock, .python-version, .gitignore, README.md, THIRD_PARTY_NOTICES.md
configs/default.yaml
server/vsp/
  __init__.py            version
  config.py              Settings (pydantic) loaded from YAML
  cli.py                 `vsp doctor | verify-weights | convert-weights`
  weights.py             public→internal key rename, convert wrapper, checkpoint verification
  engine/
    __init__.py
    device.py            pick_device(), cuda_capabilities()
    types.py             Detection, Point, Engine protocol, ChunkOOMError, EngineNotLoaded
    masks.py             is_degenerate_mask(), bbox_of()
    fake.py              FakeEngine (deterministic, scriptable)
    sam3.py              Sam3Engine (ported + point prompts)
  store/
    __init__.py
    models.py            Project, ClassDef, Clip, ClipSource, LabelObject, Keyframe, Prompt
    project.py           ProjectStore, atomic writes, mask PNG I/O, frame_states()
  ingest/
    __init__.py
    source.py            IngestError, SourceFrame, FrameSource protocol
    frame_dir.py         natural_key(), resolve_allowed(), FrameDirSource
    video_file.py        VideoFileSource
    importer.py          import_clip()
server/tests/
  test_config.py, test_device.py, test_store.py, test_ingest.py,
  test_weights.py, test_masks.py, test_fake_engine.py, test_engine_gpu.py
```

---

### Task 1: Repository scaffold, config, device selection, `vsp doctor`

**Files:**
- Create: `.gitignore`, `.python-version`, `pyproject.toml`, `README.md`, `THIRD_PARTY_NOTICES.md`, `configs/default.yaml`
- Create: `server/vsp/__init__.py`, `server/vsp/config.py`, `server/vsp/cli.py`, `server/vsp/engine/__init__.py`, `server/vsp/engine/device.py`
- Test: `server/tests/test_config.py`, `server/tests/test_device.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `vsp.config.Settings.load(path: str | Path | None) -> Settings` with attributes `server.host/port`, `project.data_root: Path`, `ingest.allowed_roots: list[str]`, `ingest.default_stride: int`, `ingest.jpeg_quality: int`, `ingest.thumb_width: int`, `sam3.model_dir: Path`, `sam3.device: str`, `sam3.dtype: str`, `sam3.text_threshold: float`, `sam3.box_threshold: float`, `sam3.score_mode: "native"|"per_query"`, `propagate.chunk_size/min_chunk_size/lost_after: int`, `export.val_fraction: float`, `export.min_area_px: int`.
  - `vsp.engine.device.pick_device(requested: str, capabilities: list[tuple[int, int]]) -> str | None`
  - `vsp.engine.device.cuda_capabilities() -> list[tuple[int, int]]`
  - Console script `vsp` → `vsp.cli:main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Initialise git and ignore rules FIRST (the repo contains 6.9 GB of checkpoints)**

```bash
cd /home/rvsa/projects/cindy/vision-sampler-pipeline
git init && git symbolic-ref HEAD refs/heads/main   # git 2.25 here has no `init -b`
```

Create `.gitignore`:

```gitignore
/data/
/vision_train/
/.template/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.egg-info/
web/node_modules/
web/dist/
*.pt
*.safetensors
```

Run: `git status --short | head`
Expected: only `.gitignore` and `docs/` listed. `vision_train/` must NOT appear. If it appears, stop and fix `.gitignore`.

- [ ] **Step 2: Create the uv project files**

`.python-version`:

```
3.11
```

`pyproject.toml` (this exact file was validated with `uv sync` on this machine):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vsp"
version = "0.1.0"
description = "SAM3 video labeling pipeline with an interactive web GUI."
requires-python = ">=3.11,<3.12"
license = { text = "MIT" }
dependencies = [
    "pydantic>=2.5",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "opencv-python-headless>=4.10,<6",
    "pillow>=10",
    "safetensors>=0.4",
    "torch==2.14.0+cu126",
    "torchvision==0.29.0+cu126",
    "transformers==5.17.0",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
vsp = "vsp.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["server/vsp"]

[tool.uv]
# torch "+cu126" wheels do not exist for macOS; restrict the lock to this platform.
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }

[tool.pytest.ini_options]
testpaths = ["server/tests"]
markers = ["gpu: needs CUDA and the converted SAM3 checkpoint"]

[tool.ruff]
line-length = 110
target-version = "py311"
```

`server/vsp/__init__.py`:

```python
__version__ = "0.1.0"
```

`server/vsp/engine/__init__.py`: empty file.

- [ ] **Step 3: Sync the environment**

Run: `UV_HTTP_TIMEOUT=600 uv sync`
Expected: ends with `+ vsp==0.1.0 (from file://…)`. Wheels are already in `~/.cache/uv`, so this takes seconds.

Run: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.14.0+cu126 True`

- [ ] **Step 4: Write the failing tests**

`server/tests/test_config.py`:

```python
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
```

`server/tests/test_device.py`:

```python
import pytest

from vsp.engine.device import pick_device


def test_auto_picks_first_bf16_capable():
    # This machine under PyTorch ordering: 3080 Ti (8.6) first, 2080 Ti (7.5) second.
    assert pick_device("auto", [(8, 6), (7, 5)]) == "cuda:0"
    # nvidia-smi ordering is the reverse; auto must still find the 8.x card.
    assert pick_device("auto", [(7, 5), (8, 6)]) == "cuda:1"


def test_auto_returns_none_without_capable_device():
    assert pick_device("auto", [(7, 5)]) is None
    assert pick_device("auto", []) is None


def test_explicit_index_is_honoured_even_if_not_bf16():
    assert pick_device("cuda:1", [(8, 6), (7, 5)]) == "cuda:1"


def test_explicit_index_out_of_range():
    with pytest.raises(ValueError, match="cuda:3"):
        pick_device("cuda:3", [(8, 6)])


def test_garbage_rejected():
    with pytest.raises(ValueError):
        pick_device("gpu0", [(8, 6)])
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_config.py server/tests/test_device.py -q`
Expected: collection errors — `ModuleNotFoundError: No module named 'vsp.config'` / `vsp.engine.device`.

- [ ] **Step 6: Implement config, device, default YAML**

`server/vsp/config.py`:

```python
"""Typed settings loaded from configs/default.yaml. Unknown keys are errors."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerCfg(_Cfg):
    host: str = "127.0.0.1"
    port: int = 28000


class ProjectCfg(_Cfg):
    data_root: Path = Path("./data")


class IngestCfg(_Cfg):
    allowed_roots: list[str] = Field(default_factory=lambda: ["~"])
    default_stride: int = 1
    jpeg_quality: int = 95
    thumb_width: int = 160


class Sam3Cfg(_Cfg):
    model_dir: Path = Path("./vision_train/sam3_checkpoint_hf")
    device: str = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    text_threshold: float = 0.45
    box_threshold: float = 0.4
    score_mode: Literal["native", "per_query"] = "native"


class PropagateCfg(_Cfg):
    chunk_size: int = 24
    min_chunk_size: int = 4
    lost_after: int = 15


class ExportCfg(_Cfg):
    val_fraction: float = 0.2
    min_area_px: int = 64


class Settings(_Cfg):
    server: ServerCfg = Field(default_factory=ServerCfg)
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    sam3: Sam3Cfg = Field(default_factory=Sam3Cfg)
    propagate: PropagateCfg = Field(default_factory=PropagateCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
```

`server/vsp/engine/device.py`:

```python
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
```

`configs/default.yaml`:

```yaml
server:    {host: 127.0.0.1, port: 28000}
project:   {data_root: ./data}
ingest:    {allowed_roots: ["~"], default_stride: 1, jpeg_quality: 95, thumb_width: 160}
sam3:
  model_dir: ./vision_train/sam3_checkpoint_hf
  device: auto          # first CUDA device with compute capability >= 8.0; or "cuda:N"
  dtype: bfloat16
  text_threshold: 0.45
  box_threshold: 0.4
  score_mode: native    # "per_query" drops the presence multiplier for verbose prompts
propagate: {chunk_size: 24, min_chunk_size: 4, lost_after: 15}
export:    {val_fraction: 0.2, min_area_px: 64}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_config.py server/tests/test_device.py -q`
Expected: `9 passed`

- [ ] **Step 8: Implement the CLI with `doctor`**

`server/vsp/cli.py`:

```python
"""`vsp` command line. Subcommands are added by later tasks/plans."""

from __future__ import annotations

import argparse
from pathlib import Path

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
```

- [ ] **Step 9: Run `vsp doctor` on the real machine**

Run: `uv run vsp doctor`
Expected (device line is the important one):

```
torch 2.14.0+cu126 (CUDA build 12.6)
  cuda:0 NVIDIA GeForce RTX 3080 Ti cc=8.6 11.8 GiB
  cuda:1 NVIDIA GeForce RTX 2080 Ti cc=7.5 10.7 GiB
selected: cuda:0
bfloat16 matmul + attention: ok
```

- [ ] **Step 10: README and third-party notice**

`README.md`:

```markdown
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
```

`THIRD_PARTY_NOTICES.md`:

```markdown
# Third-party notices

## tinkerfuroc/tk25_vision_train

Portions of `server/vsp/engine/sam3.py` and `server/vsp/engine/masks.py` are ported from
https://github.com/tinkerfuroc/tk25_vision_train at commit 5874ba0.

MIT License

Copyright (c) 2025 tinker@furoc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 11: Lint and commit**

Run: `uv run ruff check server && uv run pytest -q`
Expected: `All checks passed!` and `9 passed`.

```bash
git add .gitignore .python-version pyproject.toml uv.lock README.md THIRD_PARTY_NOTICES.md configs server docs
git status --short | grep -c vision_train   # must print 0
git commit -m "feat: scaffold uv project, settings, GPU auto-selection and vsp doctor"
```

---

### Task 2: Project store (models, atomic writes, mask PNG I/O)

**Files:**
- Create: `server/vsp/store/__init__.py`, `server/vsp/store/models.py`, `server/vsp/store/project.py`
- Test: `server/tests/test_store.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (`from vsp.store import …`):
  - Models: `ClassDef(id:int, name:str, color:str)`, `Project(id, name, created_at, classes)`, `Prompt(type: "text"|"clicks"|"box", payload: dict)`, `Keyframe(frame_idx:int, prompt:Prompt, created_at:str)`, `LabelObject(id:int, class_id:int, keyframes:list[Keyframe], propagated_ranges:list[tuple[int,int]])`, `ClipSource(type: "video"|"frame_dir", name:str)`, `Clip(id, project_id, source, fps:float, stride:int, width:int, height:int, frame_count:int, created_at:str, original_names:dict[int,str], deleted_frames:list[int], objects:list[LabelObject])`
  - Errors: `StoreError`, `NotFound(StoreError)`
  - `atomic_write_bytes(path: Path, data: bytes) -> None`
  - `ProjectStore(data_root: Path)` with:
    `create_project(name:str) -> Project`, `list_projects() -> list[Project]`, `read_project(project_id) -> Project`, `write_project(Project) -> None`, `add_class(project_id, name) -> ClassDef`,
    `new_clip_dir(project_id) -> tuple[str, Path]`, `clip_dir(clip_id) -> Path`, `write_clip(Clip) -> None`, `read_clip(clip_id) -> Clip`, `list_clips(project_id) -> list[Clip]`, `delete_clip(clip_id) -> None`,
    `frame_path(clip_id, idx) -> Path`, `thumb_path(clip_id, idx) -> Path`, `mask_path(clip_id, object_id, idx) -> Path`,
    `write_mask(clip_id, object_id, idx, mask: np.ndarray) -> None`, `read_mask(clip_id, object_id, idx) -> np.ndarray | None`, `delete_mask(clip_id, object_id, idx) -> bool`, `mask_indices(clip_id, object_id) -> list[int]`,
    `frame_states(clip: Clip) -> list[str]` (each `"deleted" | "keyframe" | "masked" | "empty"`),
    `lock(clip_id) -> threading.RLock`

- [ ] **Step 1: Write the failing tests**

`server/tests/test_store.py`:

```python
import os

import numpy as np
import pytest

from vsp.store import (
    Clip,
    ClipSource,
    Keyframe,
    LabelObject,
    NotFound,
    ProjectStore,
    Prompt,
    StoreError,
    atomic_write_bytes,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "data")


def _clip(store, project_id, frame_count=5):
    clip_id, cdir = store.new_clip_dir(project_id)
    clip = Clip(
        id=clip_id, project_id=project_id, source=ClipSource(type="video", name="a.mp4"),
        fps=30.0, stride=1, width=64, height=48, frame_count=frame_count, created_at="2026-09-20T00:00:00+00:00",
    )
    store.write_clip(clip)
    return clip


def test_project_roundtrip_and_classes(store):
    p = store.create_project("Board A")
    assert p.id.startswith("board-a-")
    c1 = store.add_class(p.id, "connector")
    c2 = store.add_class(p.id, "capacitor")
    assert (c1.id, c2.id) == (0, 1)
    assert c1.color != c2.color
    again = store.read_project(p.id)
    assert [c.name for c in again.classes] == ["connector", "capacitor"]
    assert [x.id for x in store.list_projects()] == [p.id]


def test_duplicate_class_name_rejected(store):
    p = store.create_project("p")
    store.add_class(p.id, "connector")
    with pytest.raises(StoreError, match="already exists"):
        store.add_class(p.id, "connector")


def test_clip_roundtrip_preserves_int_keys(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    clip.original_names = {0: "frame_1.png", 1: "frame_10.png"}
    clip.deleted_frames = [3]
    clip.objects = [LabelObject(id=1, class_id=0, keyframes=[
        Keyframe(frame_idx=2, prompt=Prompt(type="box", payload={"xyxy": [1, 2, 3, 4]}), created_at="t")
    ], propagated_ranges=[(2, 4)])]
    store.write_clip(clip)
    got = store.read_clip(clip.id)
    assert got.original_names == {0: "frame_1.png", 1: "frame_10.png"}
    assert got.objects[0].propagated_ranges == [(2, 4)]
    assert got.objects[0].keyframes[0].prompt.payload == {"xyxy": [1, 2, 3, 4]}
    assert [c.id for c in store.list_clips(p.id)] == [clip.id]


def test_clip_lookup_by_id_and_not_found(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    assert store.clip_dir(clip.id).name == clip.id
    with pytest.raises(NotFound):
        store.read_clip("doesnotexist")


def test_list_clips_skips_half_imported_dirs(store):
    p = store.create_project("p")
    store.new_clip_dir(p.id)  # directory exists, clip.json not yet written
    assert store.list_clips(p.id) == []


def test_mask_roundtrip_and_absent(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 5:30] = True
    store.write_mask(clip.id, 1, 2, mask)
    got = store.read_mask(clip.id, 1, 2)
    assert got.dtype == bool and got.shape == (48, 64)
    assert np.array_equal(got, mask)
    assert store.read_mask(clip.id, 1, 3) is None
    assert store.mask_indices(clip.id, 1) == [2]
    assert store.delete_mask(clip.id, 1, 2) is True
    assert store.delete_mask(clip.id, 1, 2) is False


def test_empty_mask_is_never_written(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    with pytest.raises(StoreError, match="empty mask"):
        store.write_mask(clip.id, 1, 0, np.zeros((48, 64), dtype=bool))
    assert store.mask_indices(clip.id, 1) == []


def test_frame_states_priority(store):
    p = store.create_project("p")
    clip = _clip(store, p.id, frame_count=5)
    m = np.ones((48, 64), dtype=bool)
    for i in (1, 2, 3):
        store.write_mask(clip.id, 1, i, m)
    clip.objects = [LabelObject(id=1, class_id=0, keyframes=[
        Keyframe(frame_idx=1, prompt=Prompt(type="clicks", payload={}), created_at="t")])]
    clip.deleted_frames = [3]
    assert store.frame_states(clip) == ["empty", "keyframe", "masked", "deleted", "empty"]


def test_atomic_write_leaves_nothing_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "x" / "clip.json"
    atomic_write_bytes(target, b"old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert [f.name for f in target.parent.iterdir()] == ["clip.json"]  # no stray temp file


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "UPPER", ".hidden"])
def test_path_traversal_ids_rejected(store, bad):
    with pytest.raises(StoreError):
        store.read_project(bad)
    with pytest.raises(StoreError):
        store.clip_dir(bad)


def test_delete_clip(store):
    p = store.create_project("p")
    clip = _clip(store, p.id)
    store.delete_clip(clip.id)
    with pytest.raises(NotFound):
        store.read_clip(clip.id)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_store.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.store'`

- [ ] **Step 3: Implement the models**

`server/vsp/store/models.py`:

```python
"""On-disk schema. `objects` is a list for forward compatibility; v1 uses at most one."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassDef(BaseModel):
    id: int
    name: str
    color: str


class Project(BaseModel):
    id: str
    name: str
    created_at: str
    classes: list[ClassDef] = Field(default_factory=list)


class Prompt(BaseModel):
    type: Literal["text", "clicks", "box"]
    payload: dict


class Keyframe(BaseModel):
    frame_idx: int
    prompt: Prompt
    created_at: str


class LabelObject(BaseModel):
    id: int
    class_id: int
    keyframes: list[Keyframe] = Field(default_factory=list)
    propagated_ranges: list[tuple[int, int]] = Field(default_factory=list)


class ClipSource(BaseModel):
    type: Literal["video", "frame_dir"]
    name: str


class Clip(BaseModel):
    id: str
    project_id: str
    source: ClipSource
    fps: float
    stride: int
    width: int
    height: int
    frame_count: int
    created_at: str
    original_names: dict[int, str] = Field(default_factory=dict)
    deleted_frames: list[int] = Field(default_factory=list)
    objects: list[LabelObject] = Field(default_factory=list)
```

- [ ] **Step 4: Implement the store**

`server/vsp/store/project.py`:

```python
"""The only module that reads or writes data/projects/. All writes are atomic (temp file + rename)."""

from __future__ import annotations

import io
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .models import ClassDef, Clip, Project

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


class StoreError(Exception):
    pass


class NotFound(StoreError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_id(value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise StoreError(f"invalid id: {value!r}")
    return value


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ProjectStore:
    def __init__(self, data_root: Path) -> None:
        self.root = Path(data_root)
        self.projects_dir = self.root / "projects"
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------ projects
    def _project_dir(self, project_id: str) -> Path:
        return self.projects_dir / _check_id(project_id)

    def create_project(self, name: str) -> Project:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "project"
        project = Project(id=f"{slug}-{uuid.uuid4().hex[:6]}", name=name, created_at=now_iso())
        self.write_project(project)
        return project

    def write_project(self, project: Project) -> None:
        path = self._project_dir(project.id) / "project.json"
        atomic_write_bytes(path, project.model_dump_json(indent=2).encode())

    def read_project(self, project_id: str) -> Project:
        path = self._project_dir(project_id) / "project.json"
        if not path.is_file():
            raise NotFound(f"project {project_id!r} not found")
        return Project.model_validate_json(path.read_text())

    def list_projects(self) -> list[Project]:
        if not self.projects_dir.is_dir():
            return []
        found = [Project.model_validate_json(p.read_text()) for p in self.projects_dir.glob("*/project.json")]
        return sorted(found, key=lambda p: p.created_at)

    def add_class(self, project_id: str, name: str) -> ClassDef:
        project = self.read_project(project_id)
        name = name.strip()
        if not name:
            raise StoreError("class name must not be empty")
        if any(c.name == name for c in project.classes):
            raise StoreError(f"class {name!r} already exists")
        new_id = max((c.id for c in project.classes), default=-1) + 1
        cls = ClassDef(id=new_id, name=name, color=_PALETTE[new_id % len(_PALETTE)])
        project.classes.append(cls)
        self.write_project(project)
        return cls

    # --------------------------------------------------------------- clips
    def new_clip_dir(self, project_id: str) -> tuple[str, Path]:
        self.read_project(project_id)
        clip_id = uuid.uuid4().hex[:12]
        cdir = self._project_dir(project_id) / "clips" / clip_id
        for sub in ("frames", "thumbs", "masks"):
            (cdir / sub).mkdir(parents=True)
        return clip_id, cdir

    def clip_dir(self, clip_id: str) -> Path:
        _check_id(clip_id)
        matches = list(self.projects_dir.glob(f"*/clips/{clip_id}"))
        if not matches:
            raise NotFound(f"clip {clip_id!r} not found")
        return matches[0]

    def write_clip(self, clip: Clip) -> None:
        path = self._project_dir(clip.project_id) / "clips" / _check_id(clip.id) / "clip.json"
        atomic_write_bytes(path, clip.model_dump_json(indent=2).encode())

    def read_clip(self, clip_id: str) -> Clip:
        path = self.clip_dir(clip_id) / "clip.json"
        if not path.is_file():
            raise NotFound(f"clip {clip_id!r} not found")
        return Clip.model_validate_json(path.read_text())

    def list_clips(self, project_id: str) -> list[Clip]:
        base = self._project_dir(project_id) / "clips"
        found = [Clip.model_validate_json(p.read_text()) for p in base.glob("*/clip.json")]
        return sorted(found, key=lambda c: c.created_at)

    def delete_clip(self, clip_id: str) -> None:
        shutil.rmtree(self.clip_dir(clip_id))

    def lock(self, clip_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(clip_id, threading.RLock())

    # --------------------------------------------------------------- paths
    def frame_path(self, clip_id: str, idx: int) -> Path:
        return self.clip_dir(clip_id) / "frames" / f"{idx:06d}.jpg"

    def thumb_path(self, clip_id: str, idx: int) -> Path:
        return self.clip_dir(clip_id) / "thumbs" / f"{idx:06d}.jpg"

    def mask_path(self, clip_id: str, object_id: int, idx: int) -> Path:
        return self.clip_dir(clip_id) / "masks" / str(int(object_id)) / f"{idx:06d}.png"

    # --------------------------------------------------------------- masks
    def write_mask(self, clip_id: str, object_id: int, idx: int, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise StoreError("refusing to write an empty mask; an absent file means 'not present'")
        buf = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255).convert("1", dither=Image.Dither.NONE).save(buf, format="PNG")
        atomic_write_bytes(self.mask_path(clip_id, object_id, idx), buf.getvalue())

    def read_mask(self, clip_id: str, object_id: int, idx: int) -> np.ndarray | None:
        path = self.mask_path(clip_id, object_id, idx)
        if not path.is_file():
            return None
        return np.array(Image.open(path).convert("L")) > 127

    def delete_mask(self, clip_id: str, object_id: int, idx: int) -> bool:
        path = self.mask_path(clip_id, object_id, idx)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def mask_indices(self, clip_id: str, object_id: int) -> list[int]:
        mdir = self.clip_dir(clip_id) / "masks" / str(int(object_id))
        if not mdir.is_dir():
            return []
        return sorted(int(p.stem) for p in mdir.glob("[0-9]*.png"))

    def frame_states(self, clip: Clip) -> list[str]:
        """Per-frame state for the timeline. Priority: deleted > keyframe > masked > empty."""
        states = ["empty"] * clip.frame_count
        for obj in clip.objects:
            for i in self.mask_indices(clip.id, obj.id):
                if 0 <= i < clip.frame_count:
                    states[i] = "masked"
        for obj in clip.objects:
            for kf in obj.keyframes:
                if 0 <= kf.frame_idx < clip.frame_count:
                    states[kf.frame_idx] = "keyframe"
        for i in clip.deleted_frames:
            if 0 <= i < clip.frame_count:
                states[i] = "deleted"
        return states
```

`server/vsp/store/__init__.py`:

```python
from .models import ClassDef, Clip, ClipSource, Keyframe, LabelObject, Project, Prompt
from .project import NotFound, ProjectStore, StoreError, atomic_write_bytes, now_iso

__all__ = [
    "ClassDef", "Clip", "ClipSource", "Keyframe", "LabelObject", "Project", "Prompt",
    "NotFound", "ProjectStore", "StoreError", "atomic_write_bytes", "now_iso",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_store.py -q`
Expected: `15 passed` (11 test functions; the parametrized one contributes 5).

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check server && uv run pytest -q`

```bash
git add server/vsp/store server/tests/test_store.py
git commit -m "feat(store): project/clip models, atomic writes, 1-bit mask PNG I/O, frame states"
```

---

### Task 3: Ingest (frame directory + video file → clip)

**Files:**
- Create: `server/vsp/ingest/__init__.py`, `server/vsp/ingest/source.py`, `server/vsp/ingest/frame_dir.py`, `server/vsp/ingest/video_file.py`, `server/vsp/ingest/importer.py`
- Test: `server/tests/test_ingest.py`

**Interfaces:**
- Consumes: `ProjectStore.read_project`, `ProjectStore.new_clip_dir(project_id) -> (clip_id, Path)`, `ProjectStore.write_clip(Clip)`, `Clip`, `ClipSource`, `now_iso()` from Task 2.
- Produces (`from vsp.ingest import …`):
  - `IngestError(Exception)`
  - `SourceFrame(image: np.ndarray, original_name: str | None, src_path: Path | None)` (frozen dataclass; `image` is BGR uint8)
  - `FrameSource` protocol: attributes `type: "video"|"frame_dir"`, `name: str`, `fps: float`; method `frames() -> Iterator[SourceFrame]`
  - `natural_key(name: str) -> list`, `resolve_allowed(path: str | Path, allowed_roots: list[str]) -> Path`
  - `FrameDirSource(path: Path, fps: float = 30.0)`, `VideoFileSource(path: Path, name: str | None = None)`
  - `import_clip(store, project_id: str, source: FrameSource, *, stride: int = 1, jpeg_quality: int = 95, thumb_width: int = 160, progress: Callable[[int], None] | None = None) -> Clip`

- [ ] **Step 1: Write the failing tests**

`server/tests/test_ingest.py`:

```python
import cv2
import numpy as np
import pytest

from vsp.ingest import (
    FrameDirSource,
    IngestError,
    VideoFileSource,
    import_clip,
    natural_key,
    resolve_allowed,
)
from vsp.store import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "data")


def _img(value, w=64, h=48):
    img = np.full((h, w, 3), value, dtype=np.uint8)
    img[:, : w // 2] = 255 - value  # some structure so JPEG has content
    return img


def _write(path, value, **kw):
    assert cv2.imwrite(str(path), _img(value, **kw))


def test_natural_key_orders_numbers_numerically():
    names = ["frame_10.png", "frame_2.png", "frame_1.png"]
    assert sorted(names, key=natural_key) == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_frame_dir_natural_order_gaps_and_original_names(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for n, v in [(1, 10), (2, 60), (10, 200)]:  # gap between 2 and 10 is fine
        _write(src / f"frame_{n}.png", v)
    (src / "notes.txt").write_text("ignored")
    p = store.create_project("p")
    clip = import_clip(store, p.id, FrameDirSource(src, fps=12.0))
    assert clip.frame_count == 3
    assert (clip.width, clip.height, clip.fps) == (64, 48, 12.0)
    assert clip.source.type == "frame_dir"
    assert clip.original_names == {0: "frame_1.png", 1: "frame_2.png", 2: "frame_10.png"}
    # index 2 must be the image written as frame_10 (value 200 on the right half)
    got = cv2.imread(str(store.frame_path(clip.id, 2)))
    assert abs(int(got[0, -1, 0]) - 200) <= 3
    assert store.thumb_path(clip.id, 0).is_file()
    assert store.read_clip(clip.id).frame_count == 3


def test_jpeg_is_copied_byte_identical_png_is_reencoded(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _write(src / "0.jpg", 50)
    _write(src / "1.png", 90)
    p = store.create_project("p")
    clip = import_clip(store, p.id, FrameDirSource(src))
    assert store.frame_path(clip.id, 0).read_bytes() == (src / "0.jpg").read_bytes()
    assert store.frame_path(clip.id, 1).read_bytes()[:2] == b"\xff\xd8"  # JPEG magic


def test_stride_keeps_every_nth(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for n in range(6):
        _write(src / f"{n}.png", n * 40)
    p = store.create_project("p")
    clip = import_clip(store, p.id, FrameDirSource(src), stride=2)
    assert clip.frame_count == 3
    assert clip.stride == 2
    assert clip.original_names == {0: "0.png", 1: "2.png", 2: "4.png"}


def test_mixed_sizes_rejected_and_partial_clip_removed(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _write(src / "0.png", 10)
    _write(src / "1.png", 10, w=32, h=32)
    p = store.create_project("p")
    with pytest.raises(IngestError, match="mixed frame sizes"):
        import_clip(store, p.id, FrameDirSource(src))
    assert list((store.projects_dir / p.id / "clips").iterdir()) == []


def test_unreadable_image_rejected(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    (src / "0.png").write_bytes(b"not a png")
    p = store.create_project("p")
    with pytest.raises(IngestError, match="unreadable"):
        import_clip(store, p.id, FrameDirSource(src))


def test_empty_dir_and_missing_dir_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IngestError, match="no .jpg"):
        FrameDirSource(empty)
    with pytest.raises(IngestError, match="not a directory"):
        FrameDirSource(tmp_path / "nope")


def test_resolve_allowed(tmp_path):
    root = tmp_path / "ok"
    (root / "sub").mkdir(parents=True)
    assert resolve_allowed(root / "sub", [str(root)]) == (root / "sub").resolve()
    with pytest.raises(IngestError, match="outside"):
        resolve_allowed(tmp_path / "other", [str(root)])
    with pytest.raises(IngestError, match="outside"):
        resolve_allowed(root / "sub" / ".." / ".." / "other", [str(root)])


def test_video_import_counts_decoded_frames(store, tmp_path):
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (64, 48))
    assert writer.isOpened()
    for n in range(30):
        writer.write(_img(n * 8))
    writer.release()
    p = store.create_project("p")
    seen = []
    clip = import_clip(store, p.id, VideoFileSource(path), progress=seen.append)
    assert clip.frame_count == 30
    assert clip.source.type == "video" and clip.source.name == "v.avi"
    assert clip.fps == pytest.approx(15.0, abs=0.01)
    assert clip.original_names == {}
    assert seen[-1] == 30


def test_bad_video_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(IngestError, match="cannot open"):
        VideoFileSource(bad)


def test_invalid_stride(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _write(src / "0.png", 1)
    p = store.create_project("p")
    with pytest.raises(IngestError, match="stride"):
        import_clip(store, p.id, FrameDirSource(src), stride=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_ingest.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.ingest'`

- [ ] **Step 3: Implement sources**

`server/vsp/ingest/source.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol

import numpy as np


class IngestError(Exception):
    pass


@dataclass(frozen=True)
class SourceFrame:
    image: np.ndarray  # BGR uint8
    original_name: str | None
    src_path: Path | None


class FrameSource(Protocol):
    type: Literal["video", "frame_dir"]
    name: str
    fps: float

    def frames(self) -> Iterator[SourceFrame]: ...
```

`server/vsp/ingest/frame_dir.py`:

```python
"""Directory of numbered frames. Idea ported from the template's FolderSource; adds natural sort + validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import cv2

from .source import IngestError, SourceFrame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def natural_key(name: str) -> list:
    """'frame_2.png' < 'frame_10.png'. re.split alternates text/digits, so types align by position."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def resolve_allowed(path: str | Path, allowed_roots: list[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    for root in allowed_roots:
        r = Path(root).expanduser().resolve()
        if resolved == r or r in resolved.parents:
            return resolved
    raise IngestError(f"{resolved} is outside the allowed import roots")


class FrameDirSource:
    type = "frame_dir"

    def __init__(self, path: Path, fps: float = 30.0) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise IngestError(f"{self.path} is not a directory")
        self.files = sorted(
            (f for f in self.path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda f: natural_key(f.name),
        )
        if not self.files:
            raise IngestError(f"no .jpg/.jpeg/.png files in {self.path}")
        self.name = self.path.name
        self.fps = float(fps)

    def frames(self) -> Iterator[SourceFrame]:
        for f in self.files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                raise IngestError(f"unreadable image: {f.name}")
            yield SourceFrame(image=img, original_name=f.name, src_path=f)
```

`server/vsp/ingest/video_file.py`:

```python
"""Video decoding through OpenCV's bundled FFmpeg; no system ffmpeg needed."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from .source import IngestError, SourceFrame


class VideoFileSource:
    type = "video"

    def __init__(self, path: Path, name: str | None = None) -> None:
        self.path = Path(path)
        self.name = name or self.path.name
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise IngestError(f"cannot open video: {self.name}")
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        self.fps = float(fps) if fps and fps > 0 else 30.0

    def frames(self) -> Iterator[SourceFrame]:
        cap = cv2.VideoCapture(str(self.path))  # rotation metadata is auto-applied by default
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield SourceFrame(image=img, original_name=None, src_path=None)
        finally:
            cap.release()
```

- [ ] **Step 4: Implement the importer**

`server/vsp/ingest/importer.py`:

```python
"""Any FrameSource -> frames/NNNNNN.jpg + thumbs/ + clip.json. Failure removes the partial clip."""

from __future__ import annotations

import shutil
from typing import Callable

import cv2

from ..store import Clip, ClipSource, ProjectStore, now_iso
from .source import FrameSource, IngestError

_JPEG_SUFFIXES = {".jpg", ".jpeg"}


def import_clip(
    store: ProjectStore,
    project_id: str,
    source: FrameSource,
    *,
    stride: int = 1,
    jpeg_quality: int = 95,
    thumb_width: int = 160,
    progress: Callable[[int], None] | None = None,
) -> Clip:
    if stride < 1:
        raise IngestError("stride must be >= 1")
    clip_id, cdir = store.new_clip_dir(project_id)
    try:
        size: tuple[int, int] | None = None
        kept = 0
        names: dict[int, str] = {}
        for src_idx, fr in enumerate(source.frames()):
            if src_idx % stride:
                continue
            h, w = fr.image.shape[:2]
            if size is None:
                size = (w, h)
            elif (w, h) != size:
                label = fr.original_name or f"frame {src_idx}"
                raise IngestError(f"mixed frame sizes: {label} is {w}x{h}, expected {size[0]}x{size[1]}")
            dst = cdir / "frames" / f"{kept:06d}.jpg"
            if fr.src_path is not None and fr.src_path.suffix.lower() in _JPEG_SUFFIXES:
                shutil.copyfile(fr.src_path, dst)  # copy, never hardlink: store frames are immutable
            elif not cv2.imwrite(str(dst), fr.image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
                raise IngestError(f"failed to write {dst.name}")
            th = max(1, round(h * thumb_width / w))
            thumb = cv2.resize(fr.image, (thumb_width, th), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(cdir / "thumbs" / f"{kept:06d}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if fr.original_name:
                names[kept] = fr.original_name
            kept += 1
            if progress is not None:
                progress(kept)
        if kept == 0 or size is None:
            raise IngestError("source produced no frames")
        clip = Clip(
            id=clip_id, project_id=project_id, source=ClipSource(type=source.type, name=source.name),
            fps=float(source.fps), stride=stride, width=size[0], height=size[1], frame_count=kept,
            created_at=now_iso(), original_names=names,
        )
        store.write_clip(clip)
        return clip
    except BaseException:
        shutil.rmtree(cdir, ignore_errors=True)
        raise
```

`server/vsp/ingest/__init__.py`:

```python
from .frame_dir import FrameDirSource, natural_key, resolve_allowed
from .importer import import_clip
from .source import FrameSource, IngestError, SourceFrame
from .video_file import VideoFileSource

__all__ = [
    "FrameDirSource", "FrameSource", "IngestError", "SourceFrame", "VideoFileSource",
    "import_clip", "natural_key", "resolve_allowed",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_ingest.py -q`
Expected: `11 passed`. If `test_video_import_counts_decoded_frames` fails at `writer.isOpened()`, the OpenCV build lacks the MJPG encoder: change the test's container/codec to `"v.mp4"` + `"mp4v"` and re-run; do not change production code.

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check server && uv run pytest -q`

```bash
git add server/vsp/ingest server/tests/test_ingest.py
git commit -m "feat(ingest): frame-directory and video sources, import_clip with cleanup on failure"
```

---

### Task 4: Weights — verify and (re)convert the SAM3 checkpoint

Background the implementer needs: the user supplied Meta's raw `sam3.pt` in `vision_train/sam3_checkpoint/` together with the official `config.json`, tokenizer and processor files. It was already converted once to `vision_train/sam3_checkpoint_hf/` using Hugging Face's `convert_sam3_video_to_hf.py` from the transformers GitHub `main` branch (not shipped in the wheel). Two pitfalls were found and this task encodes both: (1) that script expects Meta-internal key prefixes `sam3_model.` / `sam2_predictor.model.`, but the public file uses `detector.` / `tracker.`; (2) run offline, the script writes a ~2 KB stub `tokenizer.json`. A correct conversion has exactly 22 tensors named `tracker_neck.*` at the top level.

**Files:**
- Create: `server/vsp/weights.py`
- Modify: `server/vsp/cli.py` (add two subcommands)
- Test: `server/tests/test_weights.py`

**Interfaces:**
- Consumes: `Settings.sam3.model_dir` (Task 1).
- Produces:
  - `vsp.weights.rename_public_prefixes(state_dict: dict) -> dict`
  - `vsp.weights.verify_checkpoint(model_dir: str | Path) -> list[str]` (empty list = OK; otherwise human-readable problems)
  - `vsp.weights.convert_checkpoint(checkpoint: Path, converter: Path, out_dir: Path, official_files_from: Path) -> list[str]` (returns `verify_checkpoint(out_dir)`)
  - `vsp.weights.EXPECTED_TRACKER_NECK_TENSORS = 22`
  - CLI: `vsp verify-weights`, `vsp convert-weights --checkpoint … --converter … --out … --official-files-from …`

- [ ] **Step 1: Write the failing tests**

`server/tests/test_weights.py`:

```python
import json

import numpy as np
from safetensors.numpy import save_file

from vsp.weights import EXPECTED_TRACKER_NECK_TENSORS, rename_public_prefixes, verify_checkpoint


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_weights.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.weights'`

- [ ] **Step 3: Implement `weights.py`**

`server/vsp/weights.py`:

```python
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
            n = sum(1 for k in f.keys() if k.startswith("tracker_neck."))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_weights.py -q`
Expected: `6 passed`

- [ ] **Step 5: Add the CLI subcommands**

In `server/vsp/cli.py`, add `from .weights import convert_checkpoint, verify_checkpoint` to the imports, and replace the body of `main` from `sub.add_parser("doctor", …)` through the end of the function with:

```python
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
```

and add this helper above `main`:

```python
def _report(problems: list[str], target: str) -> int:
    if not problems:
        print(f"OK: {target}")
        return 0
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1
```

- [ ] **Step 6: Verify the real converted checkpoint**

Run: `uv run vsp verify-weights`
Expected: `OK: vision_train/sam3_checkpoint_hf`

If it reports problems, reconvert (takes ~2 minutes, needs ~8 GB RAM):

```bash
mkdir -p .template
curl -L -o .template/convert_sam3_video_to_hf.py \
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/sam3_video/convert_sam3_video_to_hf.py
HF_HUB_OFFLINE=1 uv run vsp convert-weights \
  --checkpoint vision_train/sam3_checkpoint/sam3.pt \
  --converter .template/convert_sam3_video_to_hf.py \
  --out vision_train/sam3_checkpoint_hf \
  --official-files-from vision_train/sam3_checkpoint
```

Expected during conversion: `Unexpected keys (38)` (32 `rope_embeddings` + 6 `geometry_encoder.points_*`; benign) and **no** `Missing keys` block. The known-good converter script has sha256 `7dcf1fd3c8a1b437fbba46282b771fcb0a67bef4326cdbae96b9bc0c966cdc4a`; if upstream changed and conversion fails, report it rather than patching the script.

- [ ] **Step 7: Lint and commit**

Run: `uv run ruff check server && uv run pytest -q`

```bash
git add server/vsp/weights.py server/vsp/cli.py server/tests/test_weights.py
git commit -m "feat(weights): verify-weights and convert-weights for the public SAM3 checkpoint"
```

---

### Task 5: Engine contract, mask helpers, `FakeEngine`

**Files:**
- Create: `server/vsp/engine/types.py`, `server/vsp/engine/masks.py`, `server/vsp/engine/fake.py`
- Test: `server/tests/test_masks.py`, `server/tests/test_fake_engine.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `vsp.engine.types`: `Point = tuple[int, int, int]` (x, y, label; label 1 = positive, 0 = negative), `Detection(mask: np.ndarray[bool HxW], bbox: tuple[int,int,int,int] xyxy, score: float, phrase: str | None = None)`, `ChunkOOMError(RuntimeError)`, `EngineNotLoaded(RuntimeError)`, and the `Engine` protocol:
    - `is_loaded() -> bool`
    - `predict_text(image_bgr: np.ndarray, phrase: str, *, frame_key: tuple | None = None) -> list[Detection]` (best first, may be empty)
    - `predict_prompt(image_bgr: np.ndarray, *, points: Sequence[Point] = (), box: tuple[int,int,int,int] | None = None, frame_key: tuple | None = None) -> Detection | None`
    - `propagate(frames: list[np.ndarray], seed_mask: np.ndarray, cancel_cb: Callable[[], bool] | None = None) -> Iterator[tuple[int, np.ndarray | None]]` — yields `(offset, mask_or_None)` for offsets `0..len(frames)-1` in order; offset 0 is the seed frame; `None` means "object not found / degenerate". Raises `ChunkOOMError` on CUDA OOM.
  - `vsp.engine.masks`: `is_degenerate_mask(mask, *, max_area_frac: float, min_area_px: int) -> bool`, `bbox_of(mask) -> tuple[int,int,int,int]`
  - `vsp.engine.fake.FakeEngine(dx: int = 2, n_text_candidates: int = 2, oom_if_more_than: int | None = None, is_empty: Callable[[np.ndarray], bool] | None = None)` with attribute `propagate_calls: list[int]` (length of `frames` per call).

- [ ] **Step 1: Write the failing tests**

`server/tests/test_masks.py`:

```python
import numpy as np

from vsp.engine.masks import bbox_of, is_degenerate_mask


def _m(h=100, w=100):
    return np.zeros((h, w), dtype=bool)


def test_empty_and_tiny_masks_are_degenerate():
    m = _m()
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)
    m[0:7, 0:9] = True  # 63 px
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)
    m[0:8, 0:8] = True  # now >= 64 px
    assert not is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)


def test_frame_covering_mask_is_degenerate():
    m = _m()
    m[:50, :] = True  # 50 %
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64)


def test_bbox_of_is_xyxy_exclusive_max():
    m = _m()
    m[10:20, 30:45] = True
    assert bbox_of(m) == (30, 10, 45, 20)
```

`server/tests/test_fake_engine.py`:

```python
import numpy as np
import pytest

from vsp.engine.fake import FakeEngine
from vsp.engine.types import ChunkOOMError, Engine


def _frames(n, h=60, w=120):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def _seed(h=60, w=120):
    m = np.zeros((h, w), dtype=bool)
    m[20:40, 10:30] = True
    return m


def test_satisfies_protocol():
    engine: Engine = FakeEngine()
    assert engine.is_loaded()


def test_propagate_shifts_seed_right_by_dx_per_frame():
    eng = FakeEngine(dx=3)
    out = list(eng.propagate(_frames(4), _seed()))
    assert [o for o, _ in out] == [0, 1, 2, 3]
    assert np.array_equal(out[0][1], _seed())
    ys, xs = np.where(out[2][1])
    assert (xs.min(), xs.max() + 1) == (16, 36)  # 10 + 2*3
    assert eng.propagate_calls == [4]


def test_propagate_reports_none_where_is_empty_says_so():
    frames = _frames(4)
    frames[2][0, 0, 0] = 255
    eng = FakeEngine(is_empty=lambda fr: fr[0, 0, 0] == 255)
    out = dict(eng.propagate(frames, _seed()))
    assert out[2] is None and out[1] is not None and out[3] is not None


def test_propagate_oom_when_chunk_too_large():
    eng = FakeEngine(oom_if_more_than=5)
    with pytest.raises(ChunkOOMError):
        list(eng.propagate(_frames(6), _seed()))
    assert len(list(eng.propagate(_frames(5), _seed()))) == 5


def test_propagate_honours_cancel():
    eng = FakeEngine()
    seen = []

    def cancel():
        return len(seen) >= 2

    for off, _ in eng.propagate(_frames(10), _seed(), cancel_cb=cancel):
        seen.append(off)
    assert seen == [0, 1]


def test_predict_prompt_box_points_and_negative():
    eng = FakeEngine()
    img = _frames(1)[0]
    det = eng.predict_prompt(img, box=(10, 20, 50, 40))
    assert det.bbox == (10, 20, 50, 40) and det.mask[25, 30]
    det = eng.predict_prompt(img, points=[(60, 30, 1)])
    assert det.mask[30, 60] and not det.mask[0, 0]
    det = eng.predict_prompt(img, points=[(60, 30, 1), (60, 30, 0)])
    assert not det.mask[30, 60]  # negative click carves a hole
    assert eng.predict_prompt(img) is None


def test_predict_text_returns_ranked_candidates():
    eng = FakeEngine(n_text_candidates=3)
    dets = eng.predict_text(_frames(1)[0], "connector")
    assert len(dets) == 3
    assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)
    assert all(d.phrase == "connector" for d in dets)
    assert not (dets[0].mask & dets[1].mask).any()  # distinct objects
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_masks.py server/tests/test_fake_engine.py -q`
Expected: `ModuleNotFoundError` for `vsp.engine.masks` / `vsp.engine.fake`.

- [ ] **Step 3: Implement types and mask helpers**

`server/vsp/engine/types.py`:

```python
"""The contract between the labeling layer and any segmentation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, Sequence

import numpy as np

Point = tuple[int, int, int]  # (x, y, label); label 1 = positive, 0 = negative


@dataclass
class Detection:
    mask: np.ndarray  # bool, HxW
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (max exclusive)
    score: float
    phrase: str | None = None


class ChunkOOMError(RuntimeError):
    """CUDA ran out of memory while propagating a chunk; retry with fewer frames."""


class EngineNotLoaded(RuntimeError):
    pass


class Engine(Protocol):
    def is_loaded(self) -> bool: ...

    def predict_text(
        self, image_bgr: np.ndarray, phrase: str, *, frame_key: tuple | None = None
    ) -> list[Detection]: ...

    def predict_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        points: Sequence[Point] = (),
        box: tuple[int, int, int, int] | None = None,
        frame_key: tuple | None = None,
    ) -> Detection | None: ...

    def propagate(
        self,
        frames: list[np.ndarray],
        seed_mask: np.ndarray,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, np.ndarray | None]]: ...
```

`server/vsp/engine/masks.py`:

```python
"""Mask helpers. is_degenerate_mask is ported from tinkerfuroc/tk25_vision_train (MIT, (c) 2025
tinker@furoc), server/tk_vision/annotate/sam3.py @ 5874ba0."""

from __future__ import annotations

import numpy as np


def is_degenerate_mask(mask: np.ndarray, *, max_area_frac: float, min_area_px: int) -> bool:
    """True for empty/speck masks and for hallucinated frame-covering masks."""
    area = int(mask.sum())
    if area < min_area_px:
        return True
    return area > max_area_frac * mask.size


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
```

- [ ] **Step 4: Implement `FakeEngine`**

`server/vsp/engine/fake.py`:

```python
"""Deterministic Engine for tests: masks are rectangles, tracking is 'shift right by dx per frame'."""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

import numpy as np

from .masks import bbox_of
from .types import ChunkOOMError, Detection, Point


class FakeEngine:
    def __init__(
        self,
        dx: int = 2,
        n_text_candidates: int = 2,
        oom_if_more_than: int | None = None,
        is_empty: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        self.dx = dx
        self.n_text_candidates = n_text_candidates
        self.oom_if_more_than = oom_if_more_than
        self.is_empty = is_empty
        self.propagate_calls: list[int] = []

    def is_loaded(self) -> bool:
        return True

    def predict_text(self, image_bgr: np.ndarray, phrase: str, *, frame_key: tuple | None = None) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        dets = []
        for i in range(self.n_text_candidates):
            mask = np.zeros((h, w), dtype=bool)
            x0 = 5 + i * 30
            mask[5:25, x0:x0 + 20] = True
            dets.append(Detection(mask=mask, bbox=bbox_of(mask), score=0.9 - 0.1 * i, phrase=phrase))
        return dets

    def predict_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        points: Sequence[Point] = (),
        box: tuple[int, int, int, int] | None = None,
        frame_key: tuple | None = None,
    ) -> Detection | None:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        positives = [(x, y) for x, y, lbl in points if lbl == 1]
        if box is not None:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
        elif positives:
            x, y = positives[0]
            mask[max(0, y - 10):y + 10, max(0, x - 10):x + 10] = True
        else:
            return None
        for x, y, lbl in points:
            if lbl == 0:
                mask[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = False
        if not mask.any():
            return None
        return Detection(mask=mask, bbox=bbox_of(mask), score=0.95)

    def propagate(
        self,
        frames: list[np.ndarray],
        seed_mask: np.ndarray,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        self.propagate_calls.append(len(frames))
        if self.oom_if_more_than is not None and len(frames) > self.oom_if_more_than:
            raise ChunkOOMError(f"fake OOM: {len(frames)} frames > {self.oom_if_more_than}")
        for offset, frame in enumerate(frames):
            if cancel_cb is not None and cancel_cb():
                return
            if self.is_empty is not None and self.is_empty(frame):
                yield offset, None
                continue
            shifted = np.zeros_like(seed_mask)
            s = offset * self.dx
            if s == 0:
                shifted = seed_mask.copy()
            elif s < seed_mask.shape[1]:
                shifted[:, s:] = seed_mask[:, :-s]
            yield offset, (shifted if shifted.any() else None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_masks.py server/tests/test_fake_engine.py -q`
Expected: `10 passed`

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check server && uv run pytest -q`

```bash
git add server/vsp/engine server/tests/test_masks.py server/tests/test_fake_engine.py
git commit -m "feat(engine): Engine protocol, mask helpers and deterministic FakeEngine"
```

---

### Task 6: `Sam3Engine` on the GPU — **the verification gate**

**Status of what you are about to write.** `predict_text`, `predict_box`-style decoding and `propagate` are ports of template code that its authors ran against `transformers>=4.57`. The point/box path through the tracker (`predict_prompt`) is **new** and was written from source inspection of `transformers==5.17.0` (`Sam3TrackerVideoInferenceSession.add_point_inputs`, `Sam3TrackerVideoProcessor.add_inputs_to_inference_session(inference_session, frame_idx, obj_ids, input_points, input_labels, input_boxes, …)`, `Sam3TrackerVideoModel.forward(inference_session, frame_idx)` → `.pred_masks`, `.object_score_logits`). **None of it has been executed on this machine yet.** Steps 4–6 are the gate that proves it. If a call signature differs, read the installed source under `.venv/lib/python3.11/site-packages/transformers/models/sam3_tracker_video/` and `sam3_video/`, fix the call, and note the change in the commit message. Do not weaken the test assertions to get a pass.

**Files:**
- Create: `server/vsp/engine/sam3.py`
- Test: `server/tests/test_engine_gpu.py`

**Interfaces:**
- Consumes: `pick_device`, `cuda_capabilities` (Task 1); `Detection`, `Point`, `ChunkOOMError`, `EngineNotLoaded` (Task 5); `is_degenerate_mask`, `bbox_of` (Task 5); `verify_checkpoint` (Task 4).
- Produces: `vsp.engine.sam3.Sam3Engine(model_dir: str | Path, *, device: str = "auto", dtype: str = "bfloat16", text_threshold: float = 0.45, score_mode: str = "native")` implementing the `Engine` protocol, plus `load() -> None` (raises `EngineNotLoaded` with a reason), `release() -> None`, and attributes `device: str | None`, `load_error: str | None`.

- [ ] **Step 1: Write the GPU tests**

`server/tests/test_engine_gpu.py`:

```python
"""GPU smoke tests. Skipped without a bf16-capable CUDA device + the converted checkpoint.

The scene is synthetic (a bright disc moving over a noise background) so ground truth is exact.
"""

from pathlib import Path

import numpy as np
import pytest

from vsp.engine.device import cuda_capabilities, pick_device
from vsp.weights import verify_checkpoint

MODEL_DIR = Path(__file__).resolve().parents[2] / "vision_train" / "sam3_checkpoint_hf"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(bool(verify_checkpoint(MODEL_DIR)), reason="converted SAM3 checkpoint not available"),
    pytest.mark.skipif(pick_device("auto", cuda_capabilities()) is None, reason="no bf16-capable CUDA device"),
]

H, W, R = 480, 640, 45


def _disc(cx, cy):
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= R * R


def _frame(cx, cy, rng):
    img = rng.integers(60, 120, size=(H, W, 3), dtype=np.uint8)
    img[_disc(cx, cy)] = (40, 60, 230)  # BGR: a red disc
    return img


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max(1, (a | b).sum())


@pytest.fixture(scope="module")
def engine():
    from vsp.engine.sam3 import Sam3Engine

    eng = Sam3Engine(MODEL_DIR)
    eng.load()
    yield eng
    eng.release()


@pytest.fixture(scope="module")
def video():
    rng = np.random.default_rng(0)
    centers = [(120 + 6 * i, 200 + 2 * i) for i in range(20)]
    return [_frame(cx, cy, rng) for cx, cy in centers], [_disc(cx, cy) for cx, cy in centers]


def test_load_selects_bf16_device(engine):
    assert engine.is_loaded()
    assert engine.device is not None and engine.device.startswith("cuda:")


def test_box_prompt_segments_the_disc(engine, video):
    frames, truth = video
    cx, cy = 120, 200
    det = engine.predict_prompt(frames[0], box=(cx - R - 5, cy - R - 5, cx + R + 5, cy + R + 5), frame_key=("t", 0))
    assert det is not None
    assert _iou(det.mask, truth[0]) > 0.85


def test_single_positive_click_segments_the_disc(engine, video):
    frames, truth = video
    det = engine.predict_prompt(frames[0], points=[(120, 200, 1)], frame_key=("t", 0))
    assert det is not None
    assert det.mask.shape == (H, W) and det.mask.dtype == bool
    assert _iou(det.mask, truth[0]) > 0.85


def test_negative_click_is_accepted_together_with_box(engine, video):
    frames, truth = video
    det = engine.predict_prompt(
        frames[0], box=(60, 140, 180, 260), points=[(120, 200, 1), (20, 20, 0)], frame_key=("t", 0)
    )
    assert det is not None and _iou(det.mask, truth[0]) > 0.85


def test_text_prompt_runs_and_returns_ranked_detections(engine, video):
    frames, _ = video
    dets = engine.predict_text(frames[0], "red circle", frame_key=("t", 0))
    assert isinstance(dets, list)  # synthetic art may or may not match; the call must work
    assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)
    print(f"\n[text] 'red circle' -> {len(dets)} detection(s)")


def _track(engine, frames, truth):
    got = dict(engine.propagate(frames, truth[0]))
    assert sorted(got) == list(range(len(frames)))
    good = [i for i in range(1, len(frames)) if got[i] is not None and _iou(got[i], truth[i]) > 0.7]
    return len(good)


def test_forward_propagation_tracks_the_disc(engine, video):
    frames, truth = video
    assert _track(engine, frames, truth) >= 18  # of 19 non-seed frames


def test_backward_propagation_is_just_a_reversed_list(engine, video):
    frames, truth = video
    assert _track(engine, frames[::-1], truth[::-1]) >= 18


def test_peak_vram_fits_the_12gb_card(engine, video):
    import torch

    frames, truth = video
    torch.cuda.reset_peak_memory_stats(engine.device)
    list(engine.propagate(frames + frames[::-1][:4], truth[0]))  # 24 frames = default chunk_size
    peak = torch.cuda.max_memory_allocated(engine.device) / 2**30
    print(f"\n[vram] peak during 24-frame propagate: {peak:.2f} GiB")
    assert peak < 11.0
```

- [ ] **Step 2: Run to verify it fails for the right reason**

Run: `uv run pytest server/tests/test_engine_gpu.py -q`
Expected: errors with `ModuleNotFoundError: No module named 'vsp.engine.sam3'` (NOT "skipped" — if everything is skipped, `vsp verify-weights` or `vsp doctor` is failing; fix that first).

- [ ] **Step 3: Implement `Sam3Engine`**

`server/vsp/engine/sam3.py`:

```python
"""SAM3 engine on Hugging Face transformers.

Ported from tinkerfuroc/tk25_vision_train (MIT, (c) 2025 tinker@furoc),
server/tk_vision/annotate/sam3.py @ 5874ba0. Changes from the template:
  * device auto-selection; torch.autocast always gets device_type "cuda";
  * tracker_neck weights are synced from whichever key layout the checkpoint uses;
  * single-object API matching vsp.engine.types.Engine;
  * predict_prompt(): real point/box prompts through the tracker's prompt encoder
    (the template collapsed clicks into a padded box on the detector).
"""

from __future__ import annotations

import collections
import logging
import threading
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch

from .device import cuda_capabilities, pick_device
from .masks import bbox_of, is_degenerate_mask
from .types import ChunkOOMError, Detection, EngineNotLoaded, Point

log = logging.getLogger(__name__)

_FRAME_CACHE_CAPACITY = 8
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_OBJ_ID = 1


class Sam3Engine:
    MASK_THRESHOLD = 0.5  # on sigmoid(mask logits); SAM3 masks smear on low-confidence detections
    MAX_DETECTIONS = 8
    MAX_MASK_AREA_FRAC = 0.45
    MIN_MASK_AREA_PX = 64

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        text_threshold: float = 0.45,
        score_mode: str = "native",
    ) -> None:
        self.model_dir = str(Path(model_dir).resolve())
        self.requested_device = device
        self.torch_dtype = _DTYPES[dtype]
        self.text_threshold = float(text_threshold)
        self.score_mode = score_mode
        self.device: str | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._tracker_processor = None
        self._cache: collections.OrderedDict[tuple, tuple[torch.Tensor, object]] = collections.OrderedDict()

    # ---------------------------------------------------------------- lifecycle
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                self._load_locked()
                self.load_error = None
            except Exception as e:
                self.load_error = str(e)
                raise

    def _load_locked(self) -> None:
        device = pick_device(self.requested_device, cuda_capabilities())
        if device is None:
            raise EngineNotLoaded("no CUDA device with compute capability >= 8.0 (needed for bfloat16)")
        from transformers import Sam3Processor, Sam3TrackerVideoProcessor, Sam3VideoModel

        self._processor = Sam3Processor.from_pretrained(self.model_dir)
        self._tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_dir)
        model = Sam3VideoModel.from_pretrained(self.model_dir, dtype=self.torch_dtype)
        synced, expected = self._sync_tracker_neck(model)
        if synced != expected:
            raise EngineNotLoaded(
                f"tracker_neck: synced {synced}/{expected} weights from {self.model_dir}/model.safetensors; "
                "run `vsp verify-weights`"
            )
        self._model = model.to(device).eval()
        self.device = device
        log.info("Sam3Engine loaded on %s; vram=%.2f GiB", device, torch.cuda.memory_allocated(device) / 2**30)

    @staticmethod
    def _sync_tracker_neck(model) -> tuple[int, int]:
        """Copy tracker_neck weights straight from the safetensors file.

        HF's loader has been observed to drop these (suffix collision with the detector's neck),
        leaving the tracker FPN at random init. Copying is idempotent when they did load.
        Accepts both key layouts: top-level (official converter) and tracker_model.* (template).
        """
        from safetensors import safe_open

        params = dict(model.tracker_neck.named_parameters())
        synced = 0
        with safe_open(str(Path(model.config.name_or_path) / "model.safetensors"), framework="pt", device="cpu") as f:
            available = set(f.keys())
            for name, param in params.items():
                for key in (f"tracker_neck.{name}", f"tracker_model.tracker_neck.{name}"):
                    if key in available:
                        tensor = f.get_tensor(key)
                        if tensor.shape == param.shape:
                            with torch.no_grad():
                                param.copy_(tensor.to(param.dtype))
                            synced += 1
                        break
        return synced, len(params)

    def release(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._tracker_processor = None
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _require_loaded(self) -> None:
        if self._model is None:
            raise EngineNotLoaded(self.load_error or "Sam3Engine.load() has not been called")

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.torch_dtype)

    # ------------------------------------------------------------ vision cache
    def _vision_locked(self, image_bgr: np.ndarray, frame_key: tuple | None):
        """Returns (pixel_values [1,3,1008,1008], detector vision embeds). Caller holds the lock."""
        if frame_key is not None and frame_key in self._cache:
            self._cache.move_to_end(frame_key)
            return self._cache[frame_key]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        pixel_values = self._processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
        embeds = self._model.detector_model.get_vision_features(pixel_values=pixel_values)
        if frame_key is not None:
            self._cache[frame_key] = (pixel_values, embeds)
            while len(self._cache) > _FRAME_CACHE_CAPACITY:
                self._cache.popitem(last=False)
        return pixel_values, embeds

    # ------------------------------------------------------------------- text
    def predict_text(self, image_bgr: np.ndarray, phrase: str, *, frame_key: tuple | None = None) -> list[Detection]:
        self._require_loaded()
        h, w = image_bgr.shape[:2]
        with self._lock, torch.inference_mode(), self._autocast():
            _, embeds = self._vision_locked(image_bgr, frame_key)
            enc = self._processor.tokenizer(
                phrase, return_tensors="pt", padding="max_length", max_length=32, truncation=True
            )
            out = self._model.detector_model(
                vision_embeds=embeds,
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            scores, masks = self._decode(out, (h, w))
        dets: list[Detection] = []
        for idx in np.argsort(-scores):
            if len(dets) >= self.MAX_DETECTIONS:
                break
            score, mask = float(scores[idx]), masks[idx]
            if self.score_mode == "per_query" and score <= self.text_threshold:
                continue
            if is_degenerate_mask(mask, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX):
                continue
            dets.append(Detection(mask=mask, bbox=bbox_of(mask), score=score, phrase=phrase))
        return dets

    def _decode(self, out, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = size
        if self.score_mode == "per_query":  # ignore the presence head (verbose prompts)
            scores = out.pred_logits[0].detach().float().sigmoid()
            low = out.pred_masks[0].detach().float().sigmoid()
            full = torch.nn.functional.interpolate(low.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
            return scores.cpu().numpy(), (full.squeeze(0) > self.MASK_THRESHOLD).cpu().numpy()
        res = self._processor.post_process_instance_segmentation(
            out, threshold=self.text_threshold, mask_threshold=self.MASK_THRESHOLD, target_sizes=[(h, w)]
        )
        if not res or res[0].get("scores") is None or len(res[0]["scores"]) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0, h, w), dtype=bool)
        r = res[0]
        return r["scores"].detach().float().cpu().numpy(), r["masks"].detach().cpu().numpy().astype(bool)

    # ---------------------------------------------------------- points / box
    def _new_session(self, video: torch.Tensor, h: int, w: int):
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoInferenceSession,
        )

        n = int(video.shape[0])
        session = Sam3TrackerVideoInferenceSession(
            video=video, video_height=h, video_width=w,
            inference_device=self.device, inference_state_device="cpu", video_storage_device=self.device,
            dtype=self.torch_dtype,
            # The tracker inside Sam3VideoModel has no vision encoder; every frame must be pre-cached.
            max_vision_features_cache_size=max(n, 1),
        )
        return session

    def _cache_tracker_features(self, session, frame_idx: int, vision_embeds) -> None:
        feats, pos = self._model.get_vision_features_for_tracker(vision_embeds=vision_embeds)
        session.cache.cache_vision_features(frame_idx, {"vision_feats": feats, "vision_pos_embeds": pos})

    def _binarize(self, pred_masks: torch.Tensor, h: int, w: int) -> np.ndarray:
        full = torch.nn.functional.interpolate(pred_masks.float(), size=(h, w), mode="bilinear", align_corners=False)
        return (full.sigmoid() > self.MASK_THRESHOLD)[0, 0].cpu().numpy()

    def predict_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        points: Sequence[Point] = (),
        box: tuple[int, int, int, int] | None = None,
        frame_key: tuple | None = None,
    ) -> Detection | None:
        if not points and box is None:
            return None
        self._require_loaded()
        h, w = image_bgr.shape[:2]
        with self._lock, torch.inference_mode(), self._autocast():
            pixel_values, embeds = self._vision_locked(image_bgr, frame_key)
            session = self._new_session(pixel_values, h, w)
            self._cache_tracker_features(session, 0, embeds)
            kwargs = {}
            if points:
                kwargs["input_points"] = [[[[float(x), float(y)] for x, y, _ in points]]]
                kwargs["input_labels"] = [[[int(lbl) for _, _, lbl in points]]]
            if box is not None:
                kwargs["input_boxes"] = [[[float(v) for v in box]]]
            self._tracker_processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0, obj_ids=_OBJ_ID, **kwargs
            )
            out = self._model.tracker_model(inference_session=session, frame_idx=0)
            mask = self._binarize(out.pred_masks, h, w)
            score = float(out.object_score_logits.float().sigmoid().flatten()[0])
        if not mask.any():
            return None
        return Detection(mask=mask, bbox=bbox_of(mask), score=score)

    # ------------------------------------------------------------- propagate
    def propagate(
        self,
        frames: list[np.ndarray],
        seed_mask: np.ndarray,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        self._require_loaded()
        if not frames:
            return
        h, w = frames[0].shape[:2]
        if seed_mask.shape != (h, w):
            raise ValueError(f"seed_mask shape {seed_mask.shape} does not match frames {(h, w)}")
        if not seed_mask.any():
            raise ValueError("seed_mask must be non-empty")
        rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        video = self._processor(images=rgb, return_tensors="pt")["pixel_values"]
        try:
            with self._lock, torch.inference_mode(), self._autocast():
                yield from self._propagate_locked(video, seed_mask, h, w, cancel_cb)
        except torch.cuda.OutOfMemoryError as e:
            raise ChunkOOMError(str(e)) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise ChunkOOMError(str(e)) from e
            raise
        finally:
            torch.cuda.empty_cache()

    def _propagate_locked(self, video, seed_mask, h, w, cancel_cb):
        n = int(video.shape[0])
        session = self._new_session(video, h, w)
        for fr in range(n):
            px = session.get_frame(fr).unsqueeze(0)
            embeds = self._model.detector_model.get_vision_features(pixel_values=px)
            self._cache_tracker_features(session, fr, embeds)
        obj_idx = session.obj_id_to_idx(_OBJ_ID)
        seed = torch.from_numpy(seed_mask.astype(np.float32))[None, None].to(self.device, dtype=self.torch_dtype)
        session.add_mask_inputs(obj_idx, 0, seed)
        session.obj_with_new_inputs = [_OBJ_ID]
        for out in self._model.tracker_model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=0, max_frame_num_to_track=n
        ):
            if cancel_cb is not None and cancel_cb():
                return
            mask = None
            if out.pred_masks is not None and len(out.pred_masks) > 0:
                m = self._binarize(out.pred_masks, h, w)
                if not is_degenerate_mask(m, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX):
                    mask = m
            yield int(out.frame_idx), mask
```

- [ ] **Step 4: Run the gate**

Run: `uv run pytest server/tests/test_engine_gpu.py -q -s`
Expected: `8 passed`, with two printed lines: `[text] 'red circle' -> N detection(s)` and `[vram] peak during 24-frame propagate: X.XX GiB` (must be < 11.0).

Known places this can fail, and the fix for each (apply the fix; do not loosen assertions):

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'Sam3TrackerVideoProcessor' from 'transformers'` | Import it from `transformers.models.sam3_tracker_video.processing_sam3_tracker_video`. |
| `Sam3TrackerVideoProcessor.from_pretrained` fails on this directory (its `processor_config.json` names `Sam3VideoProcessor`) | Build it by hand: `from transformers import Sam3VideoProcessor`; `vp = Sam3VideoProcessor.from_pretrained(self.model_dir)`; `Sam3TrackerVideoProcessor(image_processor=vp.image_processor, video_processor=vp.video_processor)`. |
| `add_inputs_to_inference_session` raises about `original_size` | Pass `original_size=(h, w)`. |
| `tracker_model(...)` rejects `frame_idx=0` on a fresh session, or returns no masks | Drive it like propagation: `next(iter(self._model.tracker_model.propagate_in_video_iterator(inference_session=session, start_frame_idx=0, max_frame_num_to_track=1)))`. |
| Click/box IoU below 0.85 while propagation passes | The prompt path is wired but weak. Print `det.score` and save the mask to a PNG to look at it. If masks are garbage (not just loose), the tracker_neck features are wrong for the single-frame path — confirm `_cache_tracker_features` is fed detector embeds of the *same* `pixel_values`. |
| Propagation IoU fails, masks look random | `_sync_tracker_neck` did not run or synced 0 — check the log line and `vsp verify-weights`. |
| `ChunkOOMError` / peak ≥ 11 GiB | Another process is using the 3080 Ti (`nvidia-smi`). If the card is otherwise idle, lower `propagate.chunk_size` default to 16 in `config.py`, `configs/default.yaml`, the spec §11 and `test_config.py`, and say so in the commit. |

**If `predict_prompt` cannot be made to pass after working through the table:** stop and report to the user with the failing output. The agreed fallback (spec §5.2) is to port the template's `predict_clicks`/`predict_box` on the detector instead — that is a design change the user must confirm, not something to do silently.

- [ ] **Step 5: Run the whole suite**

Run: `uv run ruff check server && uv run pytest -q`
Expected: all tests pass (≈ 59), none skipped on this machine.

- [ ] **Step 6: Record the measured numbers**

Append to `README.md`, replacing each `<…>` with the value printed by `vsp doctor` and by Step 4 (these numbers are what Plan 2 uses to justify the default `chunk_size`):

```markdown
## Verified on this machine

- Date: <YYYY-MM-DD>
- GPU: <name printed after "selected:" by `vsp doctor`>, driver 535.183.01, torch 2.14.0+cu126, transformers 5.17.0
- GPU smoke tests: 8 passed
- Peak VRAM, 24-frame propagate at 640x480: <X.XX> GiB
- Text prompt "red circle" on the synthetic scene: <N> detection(s)
```

- [ ] **Step 7: Commit**

```bash
git add server/vsp/engine/sam3.py server/tests/test_engine_gpu.py README.md
git commit -m "feat(engine): Sam3Engine with text, point/box and propagation, verified on GPU"
```

---

## Done criteria for Plan 1

- `uv run vsp doctor` and `uv run vsp verify-weights` both exit 0.
- `uv run pytest -q` passes with the GPU tests **running**, not skipped.
- `git status` is clean and `git ls-files | grep -c vision_train` prints `0`.
- A Python session can do, end to end: `ProjectStore` → `import_clip` → `Sam3Engine.predict_prompt` on a stored frame → `store.write_mask` → `Sam3Engine.propagate` over stored frames.

Plan 2 (labeling, jobs, HTTP API) is written next, against the engine interface as it actually stands after Task 6.
