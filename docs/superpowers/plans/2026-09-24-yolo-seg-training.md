# YOLO-seg Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Ultralytics YOLO26/YOLO11 segmentation models on a `yolo_seg` export from a CLI and from a new Train page, with per-GPU queues, live curves, a comparison table, weight downloads and a val prediction preview.

**Architecture:** A new package `server/vsp/train/` holds request validation (`models.py`), run folders (`runs.py`), result parsing (`results.py`), the training subprocess (`worker.py`) and the per-GPU queue (`scheduler.py`). Each run is a `python -m vsp.train.worker <run_dir>` subprocess; the scheduler is the only writer of `run.json`. `server/vsp/api/train.py` exposes it, `vsp train` drives it from the terminal, and `web/src/pages/Train.tsx` is the GUI.

**Tech Stack:** Python 3.11, FastAPI, pydantic 2, ultralytics 8.4.x, torch 2.14+cu126 (pinned), React 18 + Tailwind + vitest, uv.

**Spec:** `docs/superpowers/specs/2026-09-24-yolo-seg-training-design.md`

## Global Constraints

- Environment is uv only: `uv sync`, `uv run pytest`, `uv run vsp …`. No conda, no `requirements.txt`, no bare `pip install`.
- `ultralytics>=8.4.161,<8.5`; `[tool.uv] override-dependencies = ["opencv-python; sys_platform == 'never'"]` (only `opencv-python-headless` stays installed). Torch stays `2.14.0+cu126`.
- Worker environment: `CUDA_VISIBLE_DEVICES=<gpu>`, `YOLO_AUTOINSTALL=False`, `PYTHONUNBUFFERED=1`, cwd `data/train_weights/`; Ultralytics `device=0`.
- Models: `yolo26{n,s,m,l,x}-seg`, `yolo11{n,s,m,l,x}-seg`; defaults `yolo26s-seg, yolo26m-seg, yolo11s-seg, yolo11m-seg`.
- Params: `epochs` 1–1000 (100), `imgsz` 320–1920 multiple of 32 (960), `batch` 1–64 (8), `patience` 0–1000 (30), `workers` 0–16 (8), `seed` ≥ 0 (0).
- Run statuses: `queued | running | done | error | cancelled | interrupted`; terminal = the last four. Worker exit codes: 0 done, 2 cancelled, anything else error.
- `run.json` is written atomically (temp + `os.replace`) and only by the scheduler (or the CLI's scheduler).
- Metrics come from `train/results.csv`, never from the `on_fit_epoch_end` callback.
- GUI: every control in a right-hand sidebar at the existing scale (`md:w-80` aside, `.btn`, `.field`, `.heading`, `.note`). Chart colours: dark categorical slots `#3987e5, #d95926, #199e70, #c98500` (validated on `#1E1F21`), 2 px lines, 1 px grid in `#4A4D52`, text in `ink`/`dim` tokens only, legend when ≥ 2 series, hover crosshair + tooltip.
- Python line length 110 (ruff). Run `uv run ruff check server` before each commit.

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | ultralytics + onnx deps, opencv override (already edited in the working tree; committed in Task 1) |
| `server/vsp/train/__init__.py` | package marker |
| `server/vsp/train/models.py` | model whitelist, `TrainParams`, `ExportInfo`, `find_export`, `list_exports`, `check_model` |
| `server/vsp/train/runs.py` | `RunStore` (run dirs + `run.json`), atomic JSON I/O, weight paths |
| `server/vsp/train/results.py` | `results.csv` parsing, best-epoch summary, ground-truth labels, log tail |
| `server/vsp/train/worker.py` | the subprocess: train / resume / predict / ONNX |
| `server/vsp/train/scheduler.py` | `Gpu`, `detect_gpus`, `default_gpu`, `TrainScheduler` |
| `server/vsp/api/train.py` | the HTTP routes |
| `server/vsp/api/deps.py`, `server/vsp/api/app.py` | wire the scheduler into `AppState` and the lifespan |
| `server/vsp/cli.py` | `vsp train` |
| `server/tests/fake_train_worker.py` | stand-in worker for tests |
| `server/tests/train_fixtures.py` | shared helpers: fake export, fake worker command, waiting |
| `server/tests/test_train_*.py` | tests |
| `web/src/api/types.ts`, `web/src/api/rest.ts` | client types and calls |
| `web/src/lib/chart.ts`, `web/src/lib/train.ts` | pure helpers |
| `web/src/hooks/usePoll.ts` | polling hook |
| `web/src/components/LineChart.tsx`, `MetricCharts.tsx`, `RunsTable.tsx`, `TrainSidebar.tsx`, `PredictionPreview.tsx` | Train page parts |
| `web/src/pages/Train.tsx`, `web/src/App.tsx`, `web/src/components/AppHeader.tsx` | page, route, header link |
| `README.md`, `docs/superpowers/plans/2026-09-20-vsp-roadmap.md` | docs |

---

### Task 1: Dependencies, `models.py` and `runs.py`

**Files:**
- Commit: `pyproject.toml`, `uv.lock` (already modified in the working tree — verify, do not re-edit)
- Create: `server/vsp/train/__init__.py`, `server/vsp/train/models.py`, `server/vsp/train/runs.py`
- Create: `server/tests/train_fixtures.py`
- Test: `server/tests/test_train_models.py`, `server/tests/test_train_runs.py`

**Interfaces:**
- Produces: `MODELS`, `DEFAULT_MODELS`, `TrainParams`, `ExportInfo(id, dir, project_name, created_at, classes, train_images, val_images)` with `.yolo_dir` and `.to_json()`, `check_model(model)`, `find_export(data_root, export_id) -> ExportInfo`, `list_exports(data_root) -> list[ExportInfo]`.
- Produces: `TERMINAL`, `now_iso()`, `atomic_write_json(path, obj)`, `read_json(path, default=None)`, `new_run_id(model, now=None)`, `best_weights(run_dir)`, `last_weights(run_dir)`, `onnx_weights(run_dir)`, `RunStore(root)` with `dir(run_id)`, `create(export, model, gpu, params, owner_pid) -> dict`, `read(run_id) -> dict`, `update(run_id, **fields) -> dict`, `list() -> list[dict]`, `delete(run_id)`.
- Produces (tests): `train_fixtures.make_export(data_root, export_id="20260101-000000-abcdef", n_train=2, n_val=2) -> Path`.

- [ ] **Step 1: Verify the dependency change**

Run: `git diff pyproject.toml` — expect the four new dependencies (`ultralytics>=8.4.161,<8.5`, `onnx>=1.17`, `onnxslim>=0.1.82`, `onnxruntime>=1.20`) and the `override-dependencies` block with its comment. Then:

Run: `uv sync && uv run python -c "import cv2, ultralytics; print(cv2.__version__, ultralytics.__version__)" && uv pip list | grep -i opencv`
Expected: `5.0.0 8.4.161` and a single line `opencv-python-headless`.

- [ ] **Step 2: Write the shared test fixture module**

`server/tests/train_fixtures.py`:

```python
"""Helpers shared by the training tests: a minimal yolo_seg export, the fake worker command, polling."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

FAKE_WORKER = [sys.executable, str(Path(__file__).with_name("fake_train_worker.py"))]
EXPORT_ID = "20260101-000000-abcdef"
SQUARE = "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"


def make_export(data_root: Path, export_id: str = EXPORT_ID, n_train: int = 2, n_val: int = 2) -> Path:
    """An export directory shaped like vsp's exporter output, with one class and a square on every image."""
    d = data_root / "exports" / export_id
    y = d / "yolo_seg"
    for split, n in (("train", n_train), ("val", n_val)):
        (y / "images" / split).mkdir(parents=True, exist_ok=True)
        (y / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((64, 96, 3), 90, np.uint8)
            cv2.rectangle(img, (24, 16), (72, 48), (200, 200, 200), -1)
            assert cv2.imwrite(str(y / "images" / split / f"clip_{i:06d}.jpg"), img)
            (y / "labels" / split / f"clip_{i:06d}.txt").write_text(SQUARE + "\n")
    (y / "data.yaml").write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: "Module"\n')
    manifest = {
        "export_id": export_id, "created_at": "2026-01-01T00:00:00+00:00",
        "project": {"id": "p1", "name": "Bench"},
        "classes": [{"id": 0, "name": "Module", "color": "#e6194b", "yolo_id": 0, "coco_id": 1, "mask_value": 1}],
        "formats": [{"name": "yolo_seg", "images": {"train": n_train, "val": n_val},
                     "annotations": {"train": n_train, "val": n_val}, "linked": True}],
    }
    (d / "export.json").write_text(json.dumps(manifest))
    return d


def wait_for(predicate, timeout: float = 15.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")
```

- [ ] **Step 3: Write the failing tests**

`server/tests/test_train_models.py`:

```python
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from train_fixtures import EXPORT_ID, make_export
from vsp.labeling.errors import BadRequest
from vsp.store import NotFound
from vsp.train.models import DEFAULT_MODELS, MODELS, TrainParams, check_model, find_export, list_exports


def test_model_whitelist():
    assert "yolo26s-seg" in MODELS and "yolo11x-seg" in MODELS and len(MODELS) == 10
    assert DEFAULT_MODELS == ("yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg")
    check_model("yolo26m-seg")
    with pytest.raises(BadRequest):
        check_model("yolov8s-seg")


def test_params_defaults_and_bounds():
    p = TrainParams()
    assert (p.epochs, p.imgsz, p.batch, p.patience, p.workers, p.seed) == (100, 960, 8, 30, 8, 0)
    for bad in ({"imgsz": 1000}, {"imgsz": 288}, {"epochs": 0}, {"batch": 65}, {"workers": 17}, {"x": 1}):
        with pytest.raises(ValidationError):
            TrainParams(**bad)


def test_find_export(tmp_path):
    make_export(tmp_path, n_train=3, n_val=2)
    info = find_export(tmp_path, EXPORT_ID)
    assert info.project_name == "Bench" and info.classes == ("Module",)
    assert (info.train_images, info.val_images) == (3, 2)
    assert info.yolo_dir == tmp_path / "exports" / EXPORT_ID / "yolo_seg"
    assert info.to_json()["id"] == EXPORT_ID and "dir" not in info.to_json()


def test_find_export_errors(tmp_path):
    with pytest.raises(NotFound):
        find_export(tmp_path, "nope")
    with pytest.raises(NotFound):
        find_export(tmp_path, "../etc")
    d = make_export(tmp_path)
    m = json.loads((d / "export.json").read_text())
    m["formats"] = [{"name": "coco", "images": {"train": 1, "val": 1}}]
    (d / "export.json").write_text(json.dumps(m))
    with pytest.raises(BadRequest):
        find_export(tmp_path, EXPORT_ID)


def test_list_exports_newest_first_and_skips_incomplete(tmp_path):
    make_export(tmp_path, "20260101-000000-aaaaaa")
    make_export(tmp_path, "20260102-000000-bbbbbb")
    (tmp_path / "exports" / "20260103-000000-cccccc").mkdir()  # an export still being written: no manifest
    assert [e.id for e in list_exports(tmp_path)] == ["20260102-000000-bbbbbb", "20260101-000000-aaaaaa"]
    assert list_exports(tmp_path / "missing") == []
```

`server/tests/test_train_runs.py`:

```python
from __future__ import annotations

import os
from datetime import datetime

import pytest
from train_fixtures import EXPORT_ID, make_export
from vsp.store import NotFound
from vsp.train.models import TrainParams, find_export
from vsp.train.runs import RunStore, atomic_write_json, new_run_id, read_json


def test_new_run_id_shape():
    rid = new_run_id("yolo26s-seg", datetime(2026, 9, 24, 12, 3, 1))
    assert rid.startswith("20260924-120301-yolo26s-seg-") and len(rid.split("-")[-1]) == 4


def test_atomic_json_roundtrip(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert read_json(tmp_path / "a.json") == {"x": 1}
    assert read_json(tmp_path / "missing.json", {}) == {}
    (tmp_path / "bad.json").write_text("{")
    assert read_json(tmp_path / "bad.json") is None
    assert not list(tmp_path.glob("*.tmp"))


def test_create_read_update_list_delete(tmp_path):
    export = find_export(tmp_path, make_export(tmp_path).name)
    store = RunStore(tmp_path / "runs")
    run = store.create(export, "yolo26s-seg", 1, TrainParams(epochs=5), owner_pid=os.getpid())
    assert run["status"] == "queued" and run["gpu"] == 1 and run["params"]["epochs"] == 5
    assert run["export_id"] == EXPORT_ID and run["export_dir"] == str(export.dir)
    assert run["project_name"] == "Bench" and run["resume"] is False and run["best"] is None
    assert store.read(run["id"]) == run
    assert store.update(run["id"], status="running")["status"] == "running"
    second = store.create(export, "yolo11m-seg", 0, TrainParams(), owner_pid=os.getpid())
    assert [r["id"] for r in store.list()][0] in (run["id"], second["id"])
    assert {r["id"] for r in store.list()} == {run["id"], second["id"]}
    store.delete(run["id"])
    with pytest.raises(NotFound):
        store.read(run["id"])
    with pytest.raises(NotFound):
        store.dir("../../etc")
    assert RunStore(tmp_path / "nothing").list() == []
```

- [ ] **Step 4: Run to see them fail**

Run: `uv run pytest server/tests/test_train_models.py server/tests/test_train_runs.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'vsp.train'`.

- [ ] **Step 5: Implement**

`server/vsp/train/__init__.py`:

```python
"""YOLO-seg training: run folders, the training subprocess and the per-GPU scheduler."""
```

`server/vsp/train/models.py`:

```python
"""Training request validation: the model whitelist, hyper-parameters and yolo_seg export lookup."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..labeling.errors import BadRequest
from ..store import NotFound

FAMILIES = ("yolo26", "yolo11")
SIZES = ("n", "s", "m", "l", "x")
MODELS = tuple(f"{family}{size}-seg" for family in FAMILIES for size in SIZES)
DEFAULT_MODELS = ("yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg")
_EXPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=960, ge=320, le=1920)
    batch: int = Field(default=8, ge=1, le=64)
    patience: int = Field(default=30, ge=0, le=1000)
    workers: int = Field(default=8, ge=0, le=16)
    seed: int = Field(default=0, ge=0)

    @field_validator("imgsz")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        if v % 32:
            raise ValueError("imgsz must be a multiple of 32")
        return v


@dataclass(frozen=True)
class ExportInfo:
    id: str
    dir: Path
    project_name: str
    created_at: str
    classes: tuple[str, ...]
    train_images: int
    val_images: int

    @property
    def yolo_dir(self) -> Path:
        return self.dir / "yolo_seg"

    def to_json(self) -> dict:
        return {
            "id": self.id, "project_name": self.project_name, "created_at": self.created_at,
            "classes": list(self.classes), "train_images": self.train_images, "val_images": self.val_images,
        }


def check_model(model: str) -> None:
    if model not in MODELS:
        raise BadRequest(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")


def find_export(data_root: Path, export_id: str) -> ExportInfo:
    if not _EXPORT_ID_RE.match(export_id):
        raise NotFound(f"export {export_id} not found")
    d = Path(data_root) / "exports" / export_id
    manifest_path = d / "export.json"
    if not manifest_path.is_file():
        raise NotFound(f"export {export_id} not found")
    manifest = json.loads(manifest_path.read_text())
    fmt = next((f for f in manifest.get("formats", []) if f.get("name") == "yolo_seg"), None)
    if fmt is None or not (d / "yolo_seg" / "data.yaml").is_file():
        raise BadRequest(f"export {export_id} has no yolo_seg dataset")
    classes = tuple(c["name"] for c in sorted(manifest.get("classes", []), key=lambda c: c["yolo_id"]))
    return ExportInfo(
        id=export_id, dir=d, project_name=manifest.get("project", {}).get("name", ""),
        created_at=manifest.get("created_at", ""), classes=classes,
        train_images=int(fmt["images"]["train"]), val_images=int(fmt["images"]["val"]),
    )


def list_exports(data_root: Path) -> list[ExportInfo]:
    """Every export with a yolo_seg dataset, newest first (export ids start with a timestamp)."""
    root = Path(data_root) / "exports"
    if not root.is_dir():
        return []
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        try:
            out.append(find_export(data_root, d.name))
        except (NotFound, BadRequest, KeyError, ValueError):  # half-written, not yolo_seg, or unreadable
            continue
    return out
```

`server/vsp/train/runs.py`:

```python
"""Run folders under data/runs/ and their run.json (spec: Storage). The scheduler is run.json's only writer."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

from ..store import NotFound
from .models import ExportInfo, TrainParams

TERMINAL = frozenset({"done", "error", "cancelled", "interrupted"})
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9-]+-[0-9a-f]{4}$")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def new_run_id(model: str, now: datetime | None = None) -> str:
    now = now or datetime.now()  # local time, like export ids
    return f"{now:%Y%m%d-%H%M%S}-{model}-{secrets.token_hex(2)}"


def best_weights(run_dir: Path) -> Path:
    return run_dir / "train" / "weights" / "best.pt"


def last_weights(run_dir: Path) -> Path:
    return run_dir / "train" / "weights" / "last.pt"


def onnx_weights(run_dir: Path) -> Path:
    return run_dir / "train" / "weights" / "best.onnx"


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def dir(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise NotFound(f"run {run_id} not found")
        return self.root / run_id

    def create(self, export: ExportInfo, model: str, gpu: int, params: TrainParams, *, owner_pid: int) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            run_id = new_run_id(model)
            try:
                (self.root / run_id).mkdir()
                break
            except FileExistsError:
                continue
        run = {
            "id": run_id, "created_at": now_iso(), "started_at": None, "finished_at": None,
            "export_id": export.id, "export_dir": str(export.dir), "project_name": export.project_name,
            "model": model, "gpu": gpu, "params": params.model_dump(),
            "status": "queued", "error": None, "pid": None, "owner_pid": owner_pid,
            "resume": False, "onnx": None, "best": None,
        }
        atomic_write_json(self.root / run_id / "run.json", run)
        return run

    def read(self, run_id: str) -> dict:
        run = read_json(self.dir(run_id) / "run.json")
        if run is None:
            raise NotFound(f"run {run_id} not found")
        return run

    def update(self, run_id: str, **fields) -> dict:
        run = self.read(run_id)
        run.update(fields)
        atomic_write_json(self.dir(run_id) / "run.json", run)
        return run

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        runs = [r for p in self.root.glob("*/run.json") if (r := read_json(p)) is not None]
        return sorted(runs, key=lambda r: (r["created_at"], r["id"]), reverse=True)

    def delete(self, run_id: str) -> None:
        shutil.rmtree(self.dir(run_id))
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest server/tests/test_train_models.py server/tests/test_train_runs.py -q && uv run ruff check server`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock server/vsp/train server/tests/train_fixtures.py server/tests/test_train_models.py server/tests/test_train_runs.py
git commit -m "Training: ultralytics dependency, request validation and run folders"
```

---

### Task 2: `results.py` and the worker's pure helpers

**Files:**
- Create: `server/vsp/train/results.py`, `server/vsp/train/worker.py`
- Test: `server/tests/test_train_results.py`, `server/tests/test_train_worker.py`

**Interfaces:**
- Consumes: `atomic_write_json`, `read_json`, `best_weights`, `last_weights`, `onnx_weights` (Task 1).
- Produces: `MASK_MAP = "metrics/mAP50-95(M)"`, `MASK_MAP50`, `BOX_MAP`, `BOX_MAP50`; `read_metrics(run_dir) -> list[dict]`, `best_row(rows) -> dict | None`, `summarize(rows) -> dict | None` with keys `epoch, mask_map50, mask_map50_95, box_map50, box_map50_95, train_seconds`; `read_ground_truth(label_path) -> list[{"class_id", "polygons"}]`; `tail(path, n) -> str`.
- Produces: `worker.CANCELLED_EXIT = 2`, `worker.PREDICT_CONF = 0.25`, `write_data_yaml(yolo_dir, out) -> Path`, `ProgressWriter(path, min_interval=1.0, clock=time.monotonic)` with `.state` and `.update(force=False, **fields)`, `instances_from_result(result) -> list[dict]`, `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_train_results.py`:

```python
from __future__ import annotations

import pytest
from vsp.train.results import best_row, read_ground_truth, read_metrics, summarize, tail

# Ultralytics right-aligns header names with spaces; YOLO26 adds l1/sem losses.
CSV = (
    "                  epoch,                 time,  train/box_loss,  train/seg_loss,  train/l1_loss,"
    "  metrics/mAP50(B),  metrics/mAP50-95(B),  metrics/mAP50(M),  metrics/mAP50-95(M)\n"
    "1,4.8,1.4,5.0,0.03,0.10,0.05,0.08,0.04\n"
    "2,9.1,1.2,4.1,0.02,0.60,0.40,0.55,0.35\n"
    "3,13.5,1.1,3.9,0.02,0.62,0.41,0.57,0.35\n"
)


def _run_dir(tmp_path, text=CSV):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "results.csv").write_text(text)
    return tmp_path


def test_read_metrics_strips_headers_and_types(tmp_path):
    rows = read_metrics(_run_dir(tmp_path))
    assert len(rows) == 3
    assert rows[0]["epoch"] == 1 and isinstance(rows[0]["epoch"], int)
    assert rows[1]["metrics/mAP50-95(M)"] == pytest.approx(0.35)
    assert rows[0]["train/l1_loss"] == pytest.approx(0.03)


def test_read_metrics_missing_and_partial(tmp_path):
    assert read_metrics(tmp_path) == []
    rows = read_metrics(_run_dir(tmp_path, CSV + "4,17"))  # a row caught mid-write
    assert [r["epoch"] for r in rows] == [1, 2, 3, 4]
    assert "metrics/mAP50-95(M)" not in rows[3]


def test_best_row_prefers_later_epoch_on_ties(tmp_path):
    rows = read_metrics(_run_dir(tmp_path))
    assert best_row(rows)["epoch"] == 3
    assert best_row([]) is None


def test_summarize(tmp_path):
    s = summarize(read_metrics(_run_dir(tmp_path)))
    assert s == {
        "epoch": 3, "mask_map50": pytest.approx(0.57), "mask_map50_95": pytest.approx(0.35),
        "box_map50": pytest.approx(0.62), "box_map50_95": pytest.approx(0.41), "train_seconds": pytest.approx(13.5),
    }
    assert summarize([]) is None


def test_read_ground_truth_one_instance_per_line(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n\n0 0.1 0.2\n")
    gt = read_ground_truth(p)
    assert len(gt) == 2  # the 1-point line is not a polygon
    assert gt[0] == {"class_id": 0, "polygons": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]]}
    assert read_ground_truth(tmp_path / "missing.txt") == []


def test_tail(tmp_path):
    p = tmp_path / "log"
    p.write_text("".join(f"line {i}\n" for i in range(50)))
    assert tail(p, 3) == "line 47\nline 48\nline 49"
    assert tail(tmp_path / "missing", 3) == ""
```

`server/tests/test_train_worker.py`:

```python
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import yaml
from train_fixtures import make_export
from vsp.train.worker import ProgressWriter, instances_from_result, write_data_yaml


def test_write_data_yaml_makes_path_absolute(tmp_path):
    yolo = make_export(tmp_path) / "yolo_seg"
    out = write_data_yaml(yolo, tmp_path / "data.yaml")
    data = yaml.safe_load(out.read_text())
    assert data["path"] == str(yolo.resolve())
    assert data["train"] == "images/train" and data["names"] == {0: "Module"}


def test_progress_writer_throttles_but_force_always_writes(tmp_path):
    t = [0.0]
    w = ProgressWriter(tmp_path / "p.json", min_interval=1.0, clock=lambda: t[0])
    w.update(force=True, phase="train", epoch=1, epochs=10, batches=100)
    w.update(batch=1)
    assert json.loads((tmp_path / "p.json").read_text())["batch"] == 0  # throttled
    t[0] = 1.5
    w.update(batch=2)
    assert json.loads((tmp_path / "p.json").read_text())["batch"] == 2
    w.update(force=True, phase="val")
    assert json.loads((tmp_path / "p.json").read_text())["phase"] == "val"


def test_instances_from_result():
    poly = np.array([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]], dtype=np.float32)
    tiny = np.array([[0.1, 0.2], [0.3, 0.2]], dtype=np.float32)
    result = SimpleNamespace(
        masks=SimpleNamespace(xyn=[poly, tiny]),
        boxes=SimpleNamespace(cls=np.array([0.0, 0.0]), conf=np.array([0.91234, 0.5])),
    )
    out = instances_from_result(result)
    assert len(out) == 1
    assert out[0]["class_id"] == 0 and out[0]["score"] == 0.9123
    assert out[0]["polygons"][0] == [0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
    assert instances_from_result(SimpleNamespace(masks=None, boxes=None)) == []
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest server/tests/test_train_results.py server/tests/test_train_worker.py -q`
Expected: `ModuleNotFoundError` for `vsp.train.results` / `vsp.train.worker`.

- [ ] **Step 3: Implement `results.py`**

```python
"""Reading a run: Ultralytics' results.csv, the best epoch, ground-truth labels and the worker log."""

from __future__ import annotations

import csv
from pathlib import Path

MASK_MAP = "metrics/mAP50-95(M)"
MASK_MAP50 = "metrics/mAP50(M)"
BOX_MAP = "metrics/mAP50-95(B)"
BOX_MAP50 = "metrics/mAP50(B)"


def read_metrics(run_dir: Path) -> list[dict]:
    """One dict per epoch row; header names are stripped (Ultralytics pads them), values are floats, epoch an int."""
    try:
        text = (Path(run_dir) / "train" / "results.csv").read_text()
    except FileNotFoundError:
        return []
    rows = []
    for raw in csv.DictReader(text.splitlines()):
        row: dict = {}
        for key, value in raw.items():
            if key is None or value is None:
                continue
            try:
                row[key.strip()] = float(value.strip())
            except ValueError:
                continue
        if "epoch" in row:
            row["epoch"] = int(row["epoch"])
            rows.append(row)
    return rows


def best_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(rows, key=lambda r: (r.get(MASK_MAP, -1.0), r["epoch"]))


def summarize(rows: list[dict]) -> dict | None:
    best = best_row(rows)
    if best is None:
        return None
    return {
        "epoch": best["epoch"], "mask_map50": best.get(MASK_MAP50), "mask_map50_95": best.get(MASK_MAP),
        "box_map50": best.get(BOX_MAP50), "box_map50_95": best.get(BOX_MAP), "train_seconds": rows[-1].get("time"),
    }


def read_ground_truth(label_path: Path) -> list[dict]:
    """YOLO-seg label lines (one contour per line, as vsp exports them) as preview instances."""
    try:
        text = Path(label_path).read_text()
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least three points
            continue
        out.append({"class_id": int(parts[0]), "polygons": [[float(v) for v in parts[1:]]]})
    return out


def tail(path: Path, n: int) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])
```

- [ ] **Step 4: Implement `worker.py`**

```python
"""Training subprocess: `python -m vsp.train.worker <run_dir> [--onnx]` (spec: worker.py).

Started by the scheduler with CUDA_VISIBLE_DEVICES set to one GPU (so Ultralytics always uses device=0) and with
cwd = data/train_weights/, where Ultralytics downloads and caches the pretrained checkpoints. It never writes
run.json; it reports through progress.json, result.json and its exit code (0 done, 2 cancelled, else error).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path

import yaml

from .results import read_metrics, summarize
from .runs import atomic_write_json, best_weights, last_weights, onnx_weights, read_json

CANCELLED_EXIT = 2
PREDICT_CONF = 0.25
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def write_data_yaml(yolo_dir: Path, out: Path) -> Path:
    """The export's data.yaml says `path: .`, which Ultralytics resolves against the CWD; pin it absolute."""
    data = yaml.safe_load((yolo_dir / "data.yaml").read_text())
    data["path"] = str(Path(yolo_dir).resolve())
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return out


class ProgressWriter:
    """progress.json, rewritten atomically; `force=False` writes at most once per `min_interval` seconds."""

    def __init__(self, path: Path, min_interval: float = 1.0, clock=time.monotonic) -> None:
        self.path = path
        self.min_interval = min_interval
        self.clock = clock
        self.state = {"phase": "setup", "epoch": 0, "epochs": 0, "batch": 0, "batches": 0}
        self._last: float | None = None

    def update(self, force: bool = False, **fields) -> None:
        self.state.update(fields)
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.min_interval:
            return
        self._last = now
        atomic_write_json(self.path, self.state)


def instances_from_result(result) -> list[dict]:
    """An Ultralytics Results object as preview instances (normalised polygons, rounded to 5 places)."""
    if result.masks is None or result.boxes is None:
        return []
    out = []
    for poly, cls, conf in zip(result.masks.xyn, result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        if len(poly) < 3:
            continue
        flat = [round(float(v), 5) for v in poly.reshape(-1)]
        out.append({"class_id": int(cls), "score": round(float(conf), 4), "polygons": [flat]})
    return out


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def _quiet_ultralytics():
    from ultralytics import settings

    if settings.get("sync", True):  # write the settings file once, not on every run
        settings.update({"sync": False})


def _attach_progress(model, progress: ProgressWriter) -> None:
    def epoch_start(trainer) -> None:
        progress.update(force=True, phase="train", epoch=trainer.epoch + 1, epochs=trainer.epochs, batch=0,
                        batches=len(trainer.train_loader))

    def batch_end(trainer) -> None:
        progress.update(batch=progress.state["batch"] + 1)

    def val_start(validator) -> None:
        progress.update(force=True, phase="val")

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_val_start", val_start)


def _predict(run_dir: Path, run: dict, progress: ProgressWriter) -> None:
    from ultralytics import YOLO

    images_dir = Path(run["export_dir"]) / "yolo_seg" / "images" / "val"
    names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    model = YOLO(str(best_weights(run_dir)))
    progress.update(force=True, phase="predict", batch=0, batches=len(names))
    items = []
    for i, name in enumerate(names):
        r = model.predict(str(images_dir / name), imgsz=run["params"]["imgsz"], conf=PREDICT_CONF,
                          retina_masks=True, device=0, verbose=False)[0]
        h, w = r.orig_shape
        items.append({"name": name, "width": int(w), "height": int(h), "instances": instances_from_result(r)})
        progress.update(batch=i + 1)
    atomic_write_json(run_dir / "predictions.json", {"conf": PREDICT_CONF, "images": items})


def train(run_dir: Path) -> int:
    run = read_json(run_dir / "run.json")
    progress = ProgressWriter(run_dir / "progress.json")
    signal.signal(signal.SIGTERM, _raise_interrupt)
    (run_dir / "result.json").unlink(missing_ok=True)
    try:
        from ultralytics import YOLO

        _quiet_ultralytics()
        if run.get("resume"):
            model = YOLO(str(last_weights(run_dir)))
            _attach_progress(model, progress)
            model.train(resume=True)
        else:
            data = write_data_yaml(Path(run["export_dir"]) / "yolo_seg", run_dir / "data.yaml")
            p = run["params"]
            model = YOLO(f"{run['model']}.pt")
            _attach_progress(model, progress)
            model.train(
                data=str(data), epochs=p["epochs"], imgsz=p["imgsz"], batch=p["batch"], patience=p["patience"],
                workers=p["workers"], seed=p["seed"], project=str(run_dir), name="train", exist_ok=True,
                device=0, plots=True,
            )
        _predict(run_dir, run, progress)
    except KeyboardInterrupt:
        atomic_write_json(run_dir / "result.json", {"cancelled": True})
        return CANCELLED_EXIT
    except Exception as e:
        traceback.print_exc()
        atomic_write_json(run_dir / "result.json", {"error": f"{type(e).__name__}: {e}"})
        return 1
    atomic_write_json(run_dir / "result.json", {"best": summarize(read_metrics(run_dir))})
    return 0


def export_onnx(run_dir: Path) -> int:
    run = read_json(run_dir / "run.json")
    try:
        from ultralytics import YOLO

        _quiet_ultralytics()
        YOLO(str(best_weights(run_dir))).export(
            format="onnx", imgsz=run["params"]["imgsz"], dynamic=False, simplify=True, device=0,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0 if onnx_weights(run_dir).is_file() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vsp.train.worker")
    parser.add_argument("run_dir")
    parser.add_argument("--onnx", action="store_true", help="export best.pt to ONNX instead of training")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    return export_onnx(run_dir) if args.onnx else train(run_dir)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest server/tests/test_train_results.py server/tests/test_train_worker.py -q && uv run ruff check server`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add server/vsp/train/results.py server/vsp/train/worker.py server/tests/test_train_results.py server/tests/test_train_worker.py
git commit -m "Training: results.csv parsing and the training worker subprocess"
```

---

### Task 3: `TrainScheduler`

**Files:**
- Create: `server/vsp/train/scheduler.py`, `server/tests/fake_train_worker.py`
- Test: `server/tests/test_train_scheduler.py`

**Interfaces:**
- Consumes: Task 1 (`RunStore`, `TERMINAL`, `now_iso`, `read_json`, `last_weights`, `check_model`, `ExportInfo`, `TrainParams`), Task 2 (`read_metrics`, `summarize`, `tail`, `CANCELLED_EXIT`).
- Produces: `Gpu(index: int, name: str, memory_total: int)`, `detect_gpus() -> list[Gpu]`, `default_gpu(gpus, sam3_gpu: int | None) -> int | None`, `WORKER_CMD`, `TrainScheduler(data_root, gpus, *, worker_cmd=WORKER_CMD, kill_after=30.0)` with attributes `runs: RunStore`, `gpus: list[Gpu]` and methods `start(*, recover=True)`, `shutdown(timeout=60.0)`, `recover()`, `submit(export, specs: list[tuple[str, int]], params) -> list[str]`, `cancel(run_id)`, `resume(run_id)`, `resumable(run: dict) -> bool`, `export_onnx(run_id)`, `delete(run_id)`, `snapshot() -> list[dict]`, `idle() -> bool`.

Fake worker behaviour is selected by `params.epochs` (documented in its docstring): 1–12 → that many rows, `best.pt`, `last.pt`, `predictions.json`, exit 0; 13 → `result.json {error}` and exit 1; 14 → no `result.json`, prints a line, exit 3; 99 → one row + `last.pt`, then waits for SIGTERM and exits 2.

- [ ] **Step 1: Write the fake worker**

`server/tests/fake_train_worker.py`:

```python
"""Stands in for `python -m vsp.train.worker` in the scheduler, API and CLI tests.

Behaviour is chosen by run.json's params.epochs:
  1..12  write that many results.csv rows, last.pt, best.pt and predictions.json; exit 0
  13     write result.json {"error": ...}; exit 1
  14     print a line and exit 3 without result.json (a crash)
  99     write one row and last.pt, then wait for SIGTERM and exit 2 (a cancellable long run)
`--onnx` writes train/weights/best.onnx and exits 0.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

HEADER = (
    "                  epoch,                 time,      train/box_loss,      train/seg_loss,      train/cls_loss,"
    "metrics/mAP50(B),metrics/mAP50-95(B),metrics/mAP50(M),metrics/mAP50-95(M),val/box_loss,val/seg_loss,val/cls_loss"
)


def row(e: int) -> str:
    m = e / 10
    return f"{e},{e * 2.5},{1 - m},{1 - m},{1 - m},{m + 0.1},{m},{m + 0.05},{m - 0.05},{1 - m},{1 - m},{1 - m}"


def main() -> int:
    run_dir = Path(sys.argv[1])
    run = json.loads((run_dir / "run.json").read_text())
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    if "--onnx" in sys.argv:
        (weights / "best.onnx").write_bytes(b"onnx")
        return 0
    epochs = run["params"]["epochs"]
    print(f"fake worker gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} cwd={os.getcwd()} "
          f"autoinstall={os.environ.get('YOLO_AUTOINSTALL')} resume={run['resume']}", flush=True)
    (run_dir / "progress.json").write_text(
        json.dumps({"phase": "train", "epoch": 1, "epochs": epochs, "batch": 3, "batches": 10}))
    if epochs == 13:
        (run_dir / "result.json").write_text(json.dumps({"error": "OutOfMemoryError: CUDA out of memory"}))
        return 1
    if epochs == 14:
        print("Segmentation fault in a dataloader worker", flush=True)
        return 3
    n = 1 if epochs == 99 else epochs
    (run_dir / "train" / "results.csv").write_text(HEADER + "\n" + "".join(row(e) + "\n" for e in range(1, n + 1)))
    (weights / "last.pt").write_bytes(b"last")
    if epochs == 99:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(2))
        while True:
            time.sleep(0.05)
    (weights / "best.pt").write_bytes(b"best")
    val = Path(run["export_dir"]) / "yolo_seg" / "images" / "val"
    first = sorted(p.name for p in val.iterdir())[0]
    instance = {"class_id": 0, "score": 0.9, "polygons": [[0.2, 0.2, 0.8, 0.2, 0.8, 0.8]]}
    (run_dir / "predictions.json").write_text(json.dumps(
        {"conf": 0.25, "images": [{"name": first, "width": 96, "height": 64, "instances": [instance]}]}))
    (run_dir / "result.json").write_text(json.dumps({"best": None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the failing tests**

`server/tests/test_train_scheduler.py`:

```python
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from train_fixtures import FAKE_WORKER, make_export, wait_for
from vsp.labeling.errors import BadRequest, Conflict
from vsp.train.models import TrainParams, find_export
from vsp.train.scheduler import Gpu, TrainScheduler, default_gpu

GPUS = [Gpu(0, "Fast", 12 * 2**30), Gpu(1, "Slow", 11 * 2**30)]


@pytest.fixture
def env(tmp_path):
    data_root = tmp_path / "data"
    export = find_export(data_root, make_export(data_root).name)
    sched = TrainScheduler(data_root, GPUS, worker_cmd=FAKE_WORKER, kill_after=5.0)
    yield sched, export
    sched.shutdown(timeout=10)


def status(sched, rid):
    return sched.runs.read(rid)["status"]


def until(sched, rid, *wanted):
    return wait_for(lambda: status(sched, rid) in wanted)


def test_default_gpu():
    assert default_gpu(GPUS, 0) == 1
    assert default_gpu(GPUS, None) == 0
    assert default_gpu(GPUS[:1], 0) == 0
    assert default_gpu([], 0) is None


def test_submit_validates(env):
    sched, export = env
    with pytest.raises(BadRequest):
        sched.submit(export, [], TrainParams())
    with pytest.raises(BadRequest):
        sched.submit(export, [("yolov5s-seg", 0)], TrainParams())
    with pytest.raises(BadRequest):
        sched.submit(export, [("yolo26s-seg", 7)], TrainParams())
    assert sched.runs.list() == []


def test_one_run_per_gpu_and_gpus_in_parallel(env):
    sched, export = env
    sched.start()
    a, b = sched.submit(export, [("yolo26s-seg", 0), ("yolo26m-seg", 0)], TrainParams(epochs=99))
    (c,) = sched.submit(export, [("yolo11s-seg", 1)], TrainParams(epochs=3))
    until(sched, a, "running")
    until(sched, c, "done")  # GPU 1 does not wait for GPU 0
    snap = {g["index"]: g for g in sched.snapshot()}
    assert snap[0]["running"] == a and snap[0]["running_kind"] == "train" and snap[0]["queue"] == [b]
    assert status(sched, b) == "queued"
    sched.cancel(a)
    until(sched, a, "cancelled")
    until(sched, b, "running")
    sched.cancel(b)
    until(sched, b, "cancelled")
    done = sched.runs.read(c)
    assert done["best"]["epoch"] == 3 and done["best"]["mask_map50_95"] == pytest.approx(0.25)
    assert done["finished_at"] and done["pid"] is None
    log = (sched.runs.dir(c) / "worker.log").read_text()
    assert "gpu=1" in log and f"cwd={sched.weights_dir}" in log and "autoinstall=False" in log


def test_cancel_queued_run_never_starts(env):
    sched, export = env
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    sched.cancel(rid)  # not started: the scheduler threads are not running yet
    assert status(sched, rid) == "cancelled"
    sched.start()
    assert sched.idle()
    with pytest.raises(Conflict):
        sched.cancel(rid)


def test_error_message_from_result_json_or_log_tail(env):
    sched, export = env
    sched.start()
    (oom,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=13))
    (crash,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=14))
    until(sched, oom, "error")
    until(sched, crash, "error")
    assert sched.runs.read(oom)["error"] == "OutOfMemoryError: CUDA out of memory"
    assert "Segmentation fault" in sched.runs.read(crash)["error"]
    assert not sched.resumable(sched.runs.read(oom))  # no last.pt


def test_resume_after_cancel(env):
    sched, export = env
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=99))
    until(sched, rid, "running")
    wait_for(lambda: (sched.runs.dir(rid) / "train" / "weights" / "last.pt").is_file())
    sched.cancel(rid)
    until(sched, rid, "cancelled")
    assert sched.resumable(sched.runs.read(rid))
    sched.resume(rid)
    until(sched, rid, "running")
    wait_for(lambda: "resume=True" in (sched.runs.dir(rid) / "worker.log").read_text())
    sched.cancel(rid)
    until(sched, rid, "cancelled")
    assert sched.runs.read(rid)["resume"] is False
    (done,) = sched.submit(export, [("yolo26s-seg", 1)], TrainParams(epochs=1))
    until(sched, done, "done")
    with pytest.raises(Conflict):
        sched.resume(done)


def test_onnx_export(env):
    sched, export = env
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    with pytest.raises(Conflict):
        sched.export_onnx(rid)  # still queued or running
    until(sched, rid, "done")
    sched.export_onnx(rid)
    wait_for(lambda: sched.runs.read(rid)["onnx"] == "done")
    assert (sched.runs.dir(rid) / "train" / "weights" / "best.onnx").is_file()
    assert status(sched, rid) == "done"


def test_delete_only_terminal(env):
    sched, export = env
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=99))
    until(sched, rid, "running")
    with pytest.raises(Conflict):
        sched.delete(rid)
    sched.cancel(rid)
    until(sched, rid, "cancelled")
    sched.delete(rid)
    assert not sched.runs.dir(rid).exists()


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_recover_after_restart(env):
    sched, export = env
    dead = _dead_pid()
    (running, queued, foreign) = [
        sched.runs.create(export, "yolo26s-seg", 1, TrainParams(epochs=2), owner_pid=dead)["id"] for _ in range(3)
    ]
    sched.runs.update(running, status="running", pid=dead, onnx="running")
    sched.runs.update(foreign, status="running", owner_pid=os.getppid())  # a live CLI owns this one
    sched.start()
    until(sched, queued, "done")
    r = sched.runs.read(running)
    assert r["status"] == "interrupted" and r["onnx"] is None and r["pid"] is None
    assert status(sched, foreign) == "running"


def test_shutdown_interrupts_running_runs(tmp_path):
    data_root = tmp_path / "data"
    export = find_export(data_root, make_export(data_root).name)
    sched = TrainScheduler(data_root, GPUS, worker_cmd=FAKE_WORKER, kill_after=5.0)
    sched.start()
    (rid,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=99))
    (waiting,) = sched.submit(export, [("yolo26s-seg", 0)], TrainParams(epochs=2))
    until(sched, rid, "running")
    sched.shutdown(timeout=10)
    assert status(sched, rid) == "interrupted"
    assert status(sched, waiting) == "queued"  # picked up again by the next start()
```

- [ ] **Step 3: Run to see them fail**

Run: `uv run pytest server/tests/test_train_scheduler.py -q`
Expected: `ModuleNotFoundError: No module named 'vsp.train.scheduler'`.

- [ ] **Step 4: Implement `scheduler.py`**

```python
"""Per-GPU training queues (spec: scheduler.py). One worker subprocess per GPU at a time; GPUs run in parallel.

Locking: one Condition guards the queues, the running table and every run.json write made here. A run is popped,
marked running and launched inside one critical section, so a concurrent cancel always finds it either queued
(and drops it) or running with a process to signal.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ..labeling.errors import BadRequest, Conflict
from .models import ExportInfo, TrainParams, check_model
from .results import read_metrics, summarize, tail
from .runs import TERMINAL, RunStore, last_weights, now_iso, read_json
from .worker import CANCELLED_EXIT

log = logging.getLogger(__name__)

WORKER_CMD = (sys.executable, "-m", "vsp.train.worker")


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory_total: int  # bytes


def detect_gpus() -> list[Gpu]:
    """CUDA devices in PyTorch's order, which is also the order CUDA_VISIBLE_DEVICES indexes."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(Gpu(i, props.name, int(props.total_memory)))
    return gpus


def default_gpu(gpus: list[Gpu], sam3_gpu: int | None) -> int | None:
    """The first GPU SAM3 is not on; SAM3's own GPU only when it is the only one."""
    for g in gpus:
        if g.index != sam3_gpu:
            return g.index
    return gpus[0].index if gpus else None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


class TrainScheduler:
    def __init__(self, data_root: Path, gpus: list[Gpu], *, worker_cmd=WORKER_CMD, kill_after: float = 30.0) -> None:
        self.data_root = Path(data_root)
        self.runs = RunStore(self.data_root / "runs")
        self.weights_dir = self.data_root / "train_weights"
        self.gpus = list(gpus)
        self._worker_cmd = tuple(worker_cmd)
        self._kill_after = kill_after
        self._cv = threading.Condition()
        self._queues: dict[int, deque[tuple[str, str]]] = {g.index: deque() for g in self.gpus}
        self._running: dict[int, tuple[str, str, subprocess.Popen]] = {}
        self._cancelled: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._stopping = False

    # ---- lifecycle ----

    def start(self, *, recover: bool = True) -> None:
        """`recover=False` is for the CLI: it must not touch runs a server (or another CLI) owns."""
        if recover:
            self.recover()
        for g in self.gpus:
            t = threading.Thread(target=self._loop, args=(g.index,), name=f"train-gpu{g.index}", daemon=True)
            t.start()
            self._threads.append(t)

    def shutdown(self, timeout: float = 60.0) -> None:
        """Stop taking work and SIGTERM running workers; their runs end `interrupted` (resumable)."""
        with self._cv:
            self._stopping = True
            running = list(self._running.values())
            self._cv.notify_all()
        for _, _, proc in running:
            _signal_group(proc.pid, signal.SIGTERM)
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    def recover(self) -> None:
        """Adopt what a previous process left on disk: running → interrupted, queued → queued here again."""
        me = os.getpid()
        with self._cv:
            for run in sorted(self.runs.list(), key=lambda r: (r["created_at"], r["id"])):
                owner = run.get("owner_pid")
                if owner not in (None, me) and _alive(owner):
                    continue  # another live scheduler (a `vsp train` in a terminal) owns this run
                changes: dict = {}
                if run["status"] == "running":
                    if _alive(run.get("pid")):
                        _signal_group(run["pid"], signal.SIGKILL)  # an orphan of a killed server
                    changes.update(status="interrupted", pid=None, finished_at=now_iso())
                if run.get("onnx") in ("queued", "running"):
                    changes["onnx"] = None
                if run["status"] == "queued":
                    if run["gpu"] in self._queues:
                        self._queues[run["gpu"]].append((run["id"], "train"))
                        changes["owner_pid"] = me
                    else:
                        changes.update(status="error", error=f"GPU {run['gpu']} is not available")
                if changes:
                    self.runs.update(run["id"], **changes)
            self._cv.notify_all()

    # ---- requests ----

    def submit(self, export: ExportInfo, specs: list[tuple[str, int]], params: TrainParams) -> list[str]:
        if not specs:
            raise BadRequest("choose at least one model")
        for model, gpu in specs:
            check_model(model)
            if gpu not in self._queues:
                raise BadRequest(f"GPU {gpu} does not exist")
        ids = []
        with self._cv:
            for model, gpu in specs:
                run = self.runs.create(export, model, gpu, params, owner_pid=os.getpid())
                self._queues[gpu].append((run["id"], "train"))
                ids.append(run["id"])
            self._cv.notify_all()
        return ids

    def cancel(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if run["status"] == "queued":
                self._drop(run_id)
                self.runs.update(run_id, status="cancelled", finished_at=now_iso())
                return
            if run["status"] != "running":
                raise Conflict(f"run {run_id} is {run['status']}")
            proc = self._proc_of(run_id)
            if proc is None:
                raise Conflict(f"run {run_id} is running in another process")
            self._cancelled.add(run_id)
        _signal_group(proc.pid, signal.SIGTERM)
        timer = threading.Timer(self._kill_after, self._kill_if_alive, args=(proc,))
        timer.daemon = True
        timer.start()

    def resumable(self, run: dict) -> bool:
        return run["status"] in ("cancelled", "interrupted", "error") and last_weights(self.runs.dir(run["id"])).is_file()

    def resume(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if not self.resumable(run):
                raise Conflict(f"run {run_id} cannot be resumed")
            if run["gpu"] not in self._queues:
                raise Conflict(f"GPU {run['gpu']} is not available")
            self.runs.update(run_id, status="queued", resume=True, error=None, finished_at=None, owner_pid=os.getpid())
            self._queues[run["gpu"]].append((run_id, "train"))
            self._cv.notify_all()

    def export_onnx(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if run["status"] != "done":
                raise Conflict(f"run {run_id} is {run['status']}; only a finished run can be exported")
            if run.get("onnx") in ("queued", "running"):
                raise Conflict(f"an ONNX export of run {run_id} is already {run['onnx']}")
            if run["gpu"] not in self._queues:
                raise Conflict(f"GPU {run['gpu']} is not available")
            self.runs.update(run_id, onnx="queued")
            self._queues[run["gpu"]].append((run_id, "onnx"))
            self._cv.notify_all()

    def delete(self, run_id: str) -> None:
        with self._cv:
            run = self.runs.read(run_id)
            if run["status"] not in TERMINAL or run.get("onnx") in ("queued", "running"):
                raise Conflict(f"run {run_id} is {run['status']}; cancel it first")
            self.runs.delete(run_id)

    def snapshot(self) -> list[dict]:
        with self._cv:
            out = []
            for g in self.gpus:
                running = self._running.get(g.index)
                out.append({
                    "index": g.index, "name": g.name, "memory_total": g.memory_total,
                    "running": running[0] if running else None, "running_kind": running[1] if running else None,
                    "queue": [rid for rid, _ in self._queues[g.index]],
                })
            return out

    def idle(self) -> bool:
        with self._cv:
            return not self._running and not any(self._queues.values())

    # ---- execution ----

    def _drop(self, run_id: str) -> None:
        for q in self._queues.values():
            for item in [i for i in q if i[0] == run_id]:
                q.remove(item)

    def _proc_of(self, run_id: str) -> subprocess.Popen | None:
        for rid, _, proc in self._running.values():
            if rid == run_id:
                return proc
        return None

    def _kill_if_alive(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            _signal_group(proc.pid, signal.SIGKILL)

    def _loop(self, gpu: int) -> None:
        while True:
            with self._cv:
                while not self._stopping and not self._queues[gpu]:
                    self._cv.wait()
                if self._stopping:
                    return
                run_id, kind = self._queues[gpu].popleft()
                try:
                    proc = self._launch(run_id, kind, gpu)
                except Exception as e:
                    log.exception("could not start %s for run %s", kind, run_id)
                    self._launch_failed(run_id, kind, e)
                    continue
                self._running[gpu] = (run_id, kind, proc)
            code = proc.wait()
            with self._cv:
                del self._running[gpu]
                cancelled = run_id in self._cancelled
                self._cancelled.discard(run_id)
                try:
                    if kind == "train":
                        self._finish_train(run_id, code, cancelled)
                    else:
                        self._finish_onnx(run_id, code)
                except Exception:
                    log.exception("could not record the end of %s for run %s", kind, run_id)
                self._cv.notify_all()

    def _launch(self, run_id: str, kind: str, gpu: int) -> subprocess.Popen:
        run_dir = self.runs.dir(run_id)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        if kind == "train":
            (run_dir / "result.json").unlink(missing_ok=True)
            (run_dir / "progress.json").unlink(missing_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "YOLO_AUTOINSTALL": "False", "PYTHONUNBUFFERED": "1"}
        args = [*self._worker_cmd, str(run_dir), *(["--onnx"] if kind == "onnx" else [])]
        with open(run_dir / "worker.log", "ab") as log_file:
            proc = subprocess.Popen(
                args, cwd=self.weights_dir, env=env, stdin=subprocess.DEVNULL, stdout=log_file,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        if kind == "train":
            self.runs.update(run_id, status="running", started_at=now_iso(), finished_at=None, error=None,
                             pid=proc.pid, owner_pid=os.getpid())
        else:
            self.runs.update(run_id, onnx="running")
        return proc

    def _launch_failed(self, run_id: str, kind: str, e: Exception) -> None:
        with contextlib.suppress(Exception):
            if kind == "train":
                self.runs.update(run_id, status="error", error=f"could not start the worker: {e}", finished_at=now_iso())
            else:
                self.runs.update(run_id, onnx=f"error: could not start the export: {e}")

    def _finish_train(self, run_id: str, code: int, cancelled: bool) -> None:
        run_dir = self.runs.dir(run_id)
        result = read_json(run_dir / "result.json", {}) or {}
        fields: dict = {"pid": None, "resume": False, "finished_at": now_iso()}
        if self._stopping and not cancelled:
            fields["status"] = "interrupted"
        elif cancelled or code == CANCELLED_EXIT:
            fields["status"] = "cancelled"
        elif code == 0:
            fields.update(status="done", best=result.get("best") or summarize(read_metrics(run_dir)))
        else:
            message = result.get("error") or tail(run_dir / "worker.log", 20) or f"worker exited with code {code}"
            fields.update(status="error", error=message)
        self.runs.update(run_id, **fields)

    def _finish_onnx(self, run_id: str, code: int) -> None:
        if code == 0:
            self.runs.update(run_id, onnx="done")
        elif self._stopping:
            self.runs.update(run_id, onnx=None)
        else:
            detail = tail(self.runs.dir(run_id) / "worker.log", 5) or f"exit code {code}"
            self.runs.update(run_id, onnx=f"error: {detail}")
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest server/tests/test_train_scheduler.py -q && uv run ruff check server`
Expected: all pass (about 10 s). If a test hangs, run it with `-x -s` and read `worker.log` under the tmp run dir.

- [ ] **Step 6: Commit**

```bash
git add server/vsp/train/scheduler.py server/tests/fake_train_worker.py server/tests/test_train_scheduler.py
git commit -m "Training: per-GPU scheduler with cancel, resume, ONNX export and restart recovery"
```

---

### Task 4: HTTP routes and app wiring

**Files:**
- Create: `server/vsp/api/train.py`
- Modify: `server/vsp/api/deps.py` (AppState gains `train`), `server/vsp/api/app.py` (build/start/stop the scheduler, include router, 503 mapping)
- Test: `server/tests/test_api_train.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `create_app(settings, *, engine=None, load_engine=True, train=None)`; `AppState.train: TrainScheduler`; `api.train.TrainUnavailable`; routes listed in the spec's API table. Run views returned by the API = `run.json` minus `pid`, `owner_pid`, `export_dir`, plus `epochs_done: int`, `progress: dict | null`, `resumable: bool`, `has_best: bool`, `has_onnx: bool`, and (detail only) `metrics: list[dict]`.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_api_train.py`:

```python
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export, wait_for
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine
from vsp.train.scheduler import Gpu, TrainScheduler


@pytest.fixture
def client(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    sched = TrainScheduler(s.project.data_root, [Gpu(0, "Fast", 12 * 2**30), Gpu(1, "Slow", 11 * 2**30)],
                           worker_cmd=FAKE_WORKER, kill_after=5.0)
    app = create_app(s, engine=FakeEngine(), load_engine=False, train=sched)
    with TestClient(app) as c:
        yield c


def start(client, *, epochs=2, model="yolo26s-seg", gpu=1):
    r = client.post("/api/train/runs", json={
        "export_id": EXPORT_ID, "runs": [{"model": model, "gpu": gpu}], "params": {"epochs": epochs, "imgsz": 640},
    })
    assert r.status_code == 201, r.text
    return r.json()["run_ids"][0]


def until(client, rid, *statuses):
    return wait_for(lambda: (v := client.get(f"/api/train/runs/{rid}").json())["status"] in statuses and v)


def test_status_and_exports(client):
    st = client.get("/api/train/status").json()
    assert st["available"] is True and st["error"] is None
    assert [g["index"] for g in st["gpus"]] == [0, 1] and st["gpus"][0]["queue"] == []
    assert "yolo11m-seg" in st["models"] and st["default_models"][0] == "yolo26s-seg"
    assert st["defaults"]["imgsz"] == 960
    exports = client.get("/api/exports").json()
    assert exports == [{"id": EXPORT_ID, "project_name": "Bench", "created_at": "2026-01-01T00:00:00+00:00",
                        "classes": ["Module"], "train_images": 2, "val_images": 2}]


def test_run_lifecycle_metrics_predictions_weights(client):
    rid = start(client)
    view = until(client, rid, "done")
    assert view["epochs_done"] == 2 and len(view["metrics"]) == 2 and view["metrics"][1]["epoch"] == 2
    assert view["best"]["epoch"] == 2 and view["has_best"] and not view["has_onnx"]
    assert "pid" not in view and "export_dir" not in view and "owner_pid" not in view
    listed = client.get("/api/train/runs").json()
    assert [r["id"] for r in listed] == [rid] and "metrics" not in listed[0]

    page = client.get(f"/api/train/runs/{rid}/predictions?offset=0&limit=5").json()
    assert page["total"] == 1
    item = page["items"][0]
    assert item["name"] == "clip_000000.jpg" and item["predictions"][0]["score"] == 0.9
    assert item["ground_truth"][0]["polygons"][0][:2] == [0.25, 0.25]
    img = client.get(f"/api/exports/{EXPORT_ID}/images/val/clip_000000.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/exports/{EXPORT_ID}/images/val/..%2Fexport.json").status_code == 404

    w = client.get(f"/api/train/runs/{rid}/weights/best.pt")
    assert w.status_code == 200 and w.content == b"best"
    assert client.get(f"/api/train/runs/{rid}/weights/best.onnx").status_code == 404
    assert client.get(f"/api/train/runs/{rid}/weights/args.yaml").status_code == 404
    assert client.post(f"/api/train/runs/{rid}/onnx").status_code == 200
    wait_for(lambda: client.get(f"/api/train/runs/{rid}").json()["has_onnx"])
    assert client.get(f"/api/train/runs/{rid}/weights/best.onnx").content == b"onnx"
    assert "fake worker" in client.get(f"/api/train/runs/{rid}/log").text

    assert client.delete(f"/api/train/runs/{rid}").status_code == 204
    assert client.get(f"/api/train/runs/{rid}").status_code == 404


def test_cancel_resume_and_conflicts(client):
    rid = start(client, epochs=99)
    running = until(client, rid, "running")
    assert running["progress"] is None or running["progress"]["epoch"] == 1
    gpu = next(g for g in client.get("/api/train/status").json()["gpus"] if g["index"] == 1)
    assert gpu["running"] == rid
    assert client.delete(f"/api/train/runs/{rid}").status_code == 409
    assert client.post(f"/api/train/runs/{rid}/onnx").status_code == 409
    assert client.get(f"/api/train/runs/{rid}/predictions").status_code == 404
    assert client.post(f"/api/train/runs/{rid}/cancel").status_code == 200
    cancelled = until(client, rid, "cancelled")
    assert cancelled["resumable"] is True
    assert client.post(f"/api/train/runs/{rid}/cancel").status_code == 409
    assert client.post(f"/api/train/runs/{rid}/resume").json()["status"] in ("queued", "running")
    until(client, rid, "running")
    client.post(f"/api/train/runs/{rid}/cancel")
    until(client, rid, "cancelled")


def test_bad_requests(client):
    body = {"export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 0}]}
    assert client.post("/api/train/runs", json={**body, "runs": [{"model": "yolov8s-seg", "gpu": 0}]}).status_code == 400
    assert client.post("/api/train/runs", json={**body, "runs": [{"model": "yolo26s-seg", "gpu": 5}]}).status_code == 400
    assert client.post("/api/train/runs", json={**body, "runs": []}).status_code == 400
    assert client.post("/api/train/runs", json={**body, "export_id": "missing"}).status_code == 404
    assert client.post("/api/train/runs", json={**body, "params": {"imgsz": 1000}}).status_code == 422
    assert client.get("/api/train/runs/not-a-run").status_code == 404
    assert client.get("/api/exports/missing/images/val/a.jpg").status_code == 404


def test_unavailable_without_ultralytics(client, monkeypatch):
    import vsp.api.train as train_api

    monkeypatch.setattr(train_api, "find_spec", lambda name: None)
    st = client.get("/api/train/status").json()
    assert st["available"] is False and "uv sync" in st["error"]
    body = {"export_id": EXPORT_ID, "runs": [{"model": "yolo26s-seg", "gpu": 0}]}
    assert client.post("/api/train/runs", json=body).status_code == 503
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest server/tests/test_api_train.py -q`
Expected: `TypeError: create_app() got an unexpected keyword argument 'train'`.

- [ ] **Step 3: Implement `server/vsp/api/train.py`**

```python
"""Training routes (spec: API). Runs live on disk; the scheduler queues and starts them."""

from __future__ import annotations

import contextlib
import re
from importlib.util import find_spec
from pathlib import Path

from fastapi import APIRouter, Query, Response
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..store import NotFound
from ..train.models import DEFAULT_MODELS, MODELS, TrainParams, find_export, list_exports
from ..train.results import read_ground_truth, read_metrics, tail
from ..train.runs import best_weights, onnx_weights, read_json
from .deps import State

router = APIRouter(tags=["train"])
_IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(jpg|jpeg|png)$")
_WEIGHT_FILES = {"best.pt": best_weights, "best.onnx": onnx_weights}
_PRIVATE = ("pid", "owner_pid", "export_dir")


class TrainUnavailable(Exception):
    """ultralytics cannot be imported in this environment (HTTP 503)."""


def ultralytics_error() -> str | None:
    if find_spec("ultralytics") is None:
        return "ultralytics is not installed in this environment; run `uv sync`"
    return None


def sam3_gpu(engine) -> int | None:
    device = getattr(engine, "device", None)
    if isinstance(device, str) and device.startswith("cuda:"):
        with contextlib.suppress(ValueError):
            return int(device.removeprefix("cuda:"))
    return None


def _view(state, run: dict, *, detail: bool = False) -> dict:
    run_dir = state.train.runs.dir(run["id"])
    rows = read_metrics(run_dir)
    view = {k: v for k, v in run.items() if k not in _PRIVATE}
    view.update(
        epochs_done=len(rows),
        progress=read_json(run_dir / "progress.json") if run["status"] == "running" else None,
        resumable=state.train.resumable(run),
        has_best=best_weights(run_dir).is_file(),
        has_onnx=onnx_weights(run_dir).is_file(),
    )
    if detail:
        view["metrics"] = rows
    return view


class RunSpec(BaseModel):
    model: str
    gpu: int


class StartBody(BaseModel):
    export_id: str
    runs: list[RunSpec]
    params: TrainParams = Field(default_factory=TrainParams)


@router.get("/api/train/status")
def train_status(state: State) -> dict:
    return {
        "available": ultralytics_error() is None, "error": ultralytics_error(),
        "gpus": state.train.snapshot(), "sam3_gpu": sam3_gpu(state.engine),
        "models": list(MODELS), "default_models": list(DEFAULT_MODELS), "defaults": TrainParams().model_dump(),
    }


@router.get("/api/exports")
def exports(state: State) -> list[dict]:
    return [e.to_json() for e in list_exports(state.settings.project.data_root)]


@router.get("/api/exports/{export_id}/images/val/{name}")
def val_image(export_id: str, name: str, state: State) -> FileResponse:
    export = find_export(state.settings.project.data_root, export_id)
    path = export.yolo_dir / "images" / "val" / name
    if not _IMAGE_NAME_RE.match(name) or not path.is_file():
        raise NotFound(f"image {name} not found")
    return FileResponse(path)


@router.post("/api/train/runs", status_code=201)
def start_runs(body: StartBody, state: State) -> dict:
    error = ultralytics_error()
    if error is not None:
        raise TrainUnavailable(error)
    export = find_export(state.settings.project.data_root, body.export_id)
    return {"run_ids": state.train.submit(export, [(r.model, r.gpu) for r in body.runs], body.params)}


@router.get("/api/train/runs")
def list_runs(state: State) -> list[dict]:
    return [_view(state, r) for r in state.train.runs.list()]


@router.get("/api/train/runs/{run_id}")
def get_run(run_id: str, state: State) -> dict:
    return _view(state, state.train.runs.read(run_id), detail=True)


@router.post("/api/train/runs/{run_id}/cancel")
def cancel_run(run_id: str, state: State) -> dict:
    state.train.cancel(run_id)
    return _view(state, state.train.runs.read(run_id))


@router.post("/api/train/runs/{run_id}/resume")
def resume_run(run_id: str, state: State) -> dict:
    state.train.resume(run_id)
    return _view(state, state.train.runs.read(run_id))


@router.post("/api/train/runs/{run_id}/onnx")
def onnx_run(run_id: str, state: State) -> dict:
    state.train.export_onnx(run_id)
    return _view(state, state.train.runs.read(run_id))


@router.delete("/api/train/runs/{run_id}", status_code=204)
def delete_run(run_id: str, state: State) -> Response:
    state.train.delete(run_id)
    return Response(status_code=204)


@router.get("/api/train/runs/{run_id}/weights/{filename}")
def weights(run_id: str, filename: str, state: State) -> FileResponse:
    run = state.train.runs.read(run_id)
    locate = _WEIGHT_FILES.get(filename)
    path = locate(state.train.runs.dir(run_id)) if locate else None
    if path is None or not path.is_file():
        raise NotFound(f"{filename} not found for run {run_id}")
    return FileResponse(path, filename=f"{run['model']}-{run_id}-{filename}")


@router.get("/api/train/runs/{run_id}/predictions")
def predictions(
    run_id: str, state: State, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50),
) -> dict:
    run = state.train.runs.read(run_id)
    preds = read_json(state.train.runs.dir(run_id) / "predictions.json")
    if preds is None:
        raise NotFound(f"run {run_id} has no predictions yet")
    labels = Path(run["export_dir"]) / "yolo_seg" / "labels" / "val"
    images = preds["images"]
    items = [
        {"name": im["name"], "width": im["width"], "height": im["height"], "predictions": im["instances"],
         "ground_truth": read_ground_truth(labels / f"{Path(im['name']).stem}.txt")}
        for im in images[offset:offset + limit]
    ]
    return {"total": len(images), "conf": preds.get("conf"), "items": items}


@router.get("/api/train/runs/{run_id}/log", response_class=PlainTextResponse)
def run_log(run_id: str, state: State) -> str:
    return tail(state.train.runs.dir(run_id) / "worker.log", 200)
```

- [ ] **Step 4: Wire it in**

`server/vsp/api/deps.py` — add the field (after `jobs`):

```python
@dataclass
class AppState:
    settings: Settings
    store: ProjectStore
    engine: Any  # an Engine; Sam3Engine additionally has load(), device, load_error
    jobs: JobManager
    train: Any = None  # a TrainScheduler
```

`server/vsp/api/app.py`:
- import: `from . import clips, export, jobs, labeling, projects, train as train_api` and `from ..train.scheduler import TrainScheduler, detect_gpus`
- `_STATUS`: add `(train_api.TrainUnavailable, 503)` before `(BadRequest, 400)`.
- signature: `def create_app(settings: Settings, *, engine=None, load_engine: bool = True, train=None) -> FastAPI:`
- `AppState(..., jobs=JobManager(), train=train if train is not None else TrainScheduler(settings.project.data_root, detect_gpus()))`
- lifespan:

```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_engine and hasattr(state.engine, "load"):
            threading.Thread(target=_load, name="engine-load", daemon=True).start()
        state.train.start()
        yield
        state.train.shutdown()
```

- `app.include_router(train_api.router)` after `export.router`.

- [ ] **Step 5: Run the new tests and the whole suite**

Run: `uv run pytest server/tests/test_api_train.py -q && uv run pytest -q && uv run ruff check server`
Expected: all pass (GPU tests skip or pass as before).

- [ ] **Step 6: Commit**

```bash
git add server/vsp/api/train.py server/vsp/api/deps.py server/vsp/api/app.py server/tests/test_api_train.py
git commit -m "Training API: status, exports, runs, cancel/resume/ONNX, weights, predictions and log"
```

---

### Task 5: `vsp train` CLI and the GPU smoke test

**Files:**
- Modify: `server/vsp/cli.py`
- Test: `server/tests/test_train_cli.py`, `server/tests/test_train_gpu.py`

**Interfaces:**
- Consumes: `TrainScheduler`, `detect_gpus`, `default_gpu`, `Gpu`, `find_export`, `list_exports`, `TrainParams`, `DEFAULT_MODELS`, `read_metrics`, `MASK_MAP`, `MASK_MAP50`, `BOX_MAP`, `pick_device`, `cuda_capabilities`.
- Produces: `vsp train [--list] [--export ID] [--model M]… [--gpu N] [--epochs N] [--imgsz N] [--batch N] [--patience N] [--workers N] [--seed N]`; internal `_train(settings, args, *, gpus=None, worker_cmd=None, poll=2.0) -> int` (test seam).

- [ ] **Step 1: Write the failing tests**

`server/tests/test_train_cli.py`:

```python
from __future__ import annotations

import argparse

import vsp.cli
from train_fixtures import EXPORT_ID, FAKE_WORKER, make_export
from vsp.config import Settings
from vsp.train.scheduler import Gpu


def _settings(tmp_path):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    make_export(s.project.data_root)
    return s


def _args(**kw):
    base = dict(list=False, export=EXPORT_ID, model=None, gpu=1, epochs=2, imgsz=640, batch=8, patience=30,
                workers=0, seed=0)
    return argparse.Namespace(**{**base, **kw})


def test_list_exports(tmp_path, capsys):
    rc = vsp.cli._train(_settings(tmp_path), _args(list=True), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER)
    assert rc == 0 and EXPORT_ID in capsys.readouterr().out


def test_trains_models_in_order_and_prints_table(tmp_path, capsys):
    s = _settings(tmp_path)
    rc = vsp.cli._train(s, _args(model=["yolo26s-seg", "yolo11s-seg"]), gpus=[Gpu(0, "A", 1), Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    out = capsys.readouterr().out
    assert rc == 0
    assert "yolo26s-seg" in out and "yolo11s-seg" in out and "epoch   2/2" in out
    assert "mask mAP50-95" in out.lower()
    runs = sorted((s.project.data_root / "runs").iterdir())
    assert len(runs) == 2


def test_failed_run_returns_nonzero(tmp_path, capsys):
    rc = vsp.cli._train(_settings(tmp_path), _args(model=["yolo26s-seg"], epochs=13), gpus=[Gpu(1, "B", 1)],
                        worker_cmd=FAKE_WORKER, poll=0.05)
    assert rc == 1 and "OutOfMemoryError" in capsys.readouterr().out


def test_bad_input(tmp_path, capsys):
    s = _settings(tmp_path)
    assert vsp.cli._train(s, _args(export=None), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER) == 2
    assert vsp.cli._train(s, _args(imgsz=1000), gpus=[Gpu(0, "A", 1)], worker_cmd=FAKE_WORKER) == 2
    assert vsp.cli._train(s, _args(), gpus=[], worker_cmd=FAKE_WORKER) == 1
```

`server/tests/test_train_gpu.py`:

```python
"""Real Ultralytics training on a tiny synthetic export (downloads yolo11n-seg.pt once into the tmp data root)."""

from __future__ import annotations

import pytest
from train_fixtures import make_export, wait_for
from vsp.train.models import TrainParams, find_export
from vsp.train.scheduler import TrainScheduler, detect_gpus

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not detect_gpus(), reason="needs CUDA")
def test_real_worker_trains_one_epoch(tmp_path):
    pytest.importorskip("ultralytics")
    data_root = tmp_path / "data"
    export = find_export(data_root, make_export(data_root, n_train=16, n_val=4).name)
    gpus = detect_gpus()
    sched = TrainScheduler(data_root, gpus)
    sched.start()
    try:
        (rid,) = sched.submit(export, [("yolo11n-seg", gpus[-1].index)],
                              TrainParams(epochs=1, imgsz=320, batch=4, workers=0))
        wait_for(lambda: sched.runs.read(rid)["status"] in ("done", "error", "cancelled"), timeout=600, interval=1)
        run = sched.runs.read(rid)
        assert run["status"] == "done", (sched.runs.dir(rid) / "worker.log").read_text()[-3000:]
        d = sched.runs.dir(rid)
        assert (d / "train" / "weights" / "best.pt").is_file() and (d / "predictions.json").is_file()
        assert run["best"]["epoch"] == 1
    finally:
        sched.shutdown()
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest server/tests/test_train_cli.py -q`
Expected: `AttributeError: module 'vsp.cli' has no attribute '_train'`.

- [ ] **Step 3: Implement in `server/vsp/cli.py`**

Add below `_serve`:

```python
def _fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def _train(settings: Settings, args, *, gpus=None, worker_cmd=None, poll: float = 2.0) -> int:
    import time

    from pydantic import ValidationError

    from .labeling.errors import BadRequest, Conflict
    from .store import NotFound
    from .train.models import DEFAULT_MODELS, TrainParams, find_export, list_exports
    from .train.results import BOX_MAP, MASK_MAP, MASK_MAP50, read_metrics
    from .train.scheduler import WORKER_CMD, TrainScheduler, default_gpu, detect_gpus

    data_root = settings.project.data_root
    if args.list:
        for e in list_exports(data_root):
            print(f"{e.id}  {e.project_name}  train {e.train_images}  val {e.val_images}  classes: {', '.join(e.classes)}")
        return 0
    if not args.export:
        print("error: --export is required (see `vsp train --list`)")
        return 2
    try:
        params = TrainParams(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                             workers=args.workers, seed=args.seed)
    except ValidationError as e:
        print(f"error: {e}")
        return 2
    gpus = detect_gpus() if gpus is None else gpus
    if not gpus:
        print("FAIL: no CUDA GPU is available to train on")
        return 1
    if args.gpu is not None:
        gpu = args.gpu
    else:
        sam3 = pick_device(settings.sam3.device, cuda_capabilities()) if gpus else None
        gpu = default_gpu(gpus, int(sam3.split(":")[1]) if sam3 else None)
    try:
        export = find_export(data_root, args.export)
        sched = TrainScheduler(data_root, gpus, worker_cmd=worker_cmd or WORKER_CMD)
        ids = sched.submit(export, [(m, gpu) for m in (args.model or DEFAULT_MODELS)], params)
    except (NotFound, BadRequest) as e:
        print(f"error: {e}")
        return 2
    print(f"training {len(ids)} model(s) on cuda:{gpu} from export {export.id} ({export.project_name}); "
          f"runs in {sched.runs.root}")
    shown = {rid: 0 for rid in ids}
    states = {rid: "queued" for rid in ids}

    def report() -> None:
        for rid in ids:
            run = sched.runs.read(rid)
            rows = read_metrics(sched.runs.dir(rid))
            for row in rows[shown[rid]:]:
                print(f"{run['model']:<12} epoch {row['epoch']:>3}/{run['params']['epochs']}  "
                      f"mask mAP50-95 {_fmt(row.get(MASK_MAP))}  mask mAP50 {_fmt(row.get(MASK_MAP50))}  "
                      f"box mAP50-95 {_fmt(row.get(BOX_MAP))}", flush=True)
            shown[rid] = len(rows)
            if run["status"] != states[rid]:
                states[rid] = run["status"]
                extra = f": {run['error']}" if run["status"] == "error" else ""
                print(f"{run['model']:<12} {run['status']}{extra}", flush=True)

    sched.start(recover=False)
    try:
        while not sched.idle():
            report()
            time.sleep(poll)
    except KeyboardInterrupt:
        print("stopping: cancelling this command's runs…", flush=True)
        for rid in ids:
            if sched.runs.read(rid)["status"] in ("queued", "running"):
                with contextlib.suppress(Conflict):
                    sched.cancel(rid)
        while not sched.idle():
            time.sleep(0.2)
    finally:
        sched.shutdown()
    report()
    print(f"\n{'model':<12} {'status':<11} {'best epoch':>10} {'mask mAP50':>10} {'mask mAP50-95':>13} "
          f"{'box mAP50-95':>12}  run")
    ok = True
    for rid in ids:
        run = sched.runs.read(rid)
        b = run.get("best") or {}
        ok = ok and run["status"] == "done"
        print(f"{run['model']:<12} {run['status']:<11} {b.get('epoch', '-')!s:>10} {_fmt(b.get('mask_map50')):>10} "
              f"{_fmt(b.get('mask_map50_95')):>13} {_fmt(b.get('box_map50_95')):>12}  {rid}")
    return 0 if ok else 1
```

Add `import contextlib` at the top of `cli.py`. In `main()`, register the subcommand before `args = parser.parse_args(argv)`:

```python
    p_train = sub.add_parser("train", help="train YOLO segmentation models on a yolo_seg export")
    p_train.add_argument("--list", action="store_true", help="list the exports that have a yolo_seg dataset")
    p_train.add_argument("--export", default=None, help="export id (a folder name under data/exports/)")
    p_train.add_argument("--model", action="append", default=None,
                         help="model to train, repeatable (default: yolo26s/m-seg and yolo11s/m-seg)")
    p_train.add_argument("--gpu", type=int, default=None, help="CUDA index (default: the first GPU SAM3 is not on)")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=960)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--patience", type=int, default=30)
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument("--seed", type=int, default=0)
```

and dispatch before `parser.error(...)`:

```python
    if args.cmd == "train":
        return _train(settings, args)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest server/tests/test_train_cli.py -q && uv run pytest server/tests/test_train_gpu.py -q -s && uv run ruff check server`
Expected: CLI tests pass; the GPU test passes in ≤ a few minutes (it downloads `yolo11n-seg.pt`).

- [ ] **Step 5: Commit**

```bash
git add server/vsp/cli.py server/tests/test_train_cli.py server/tests/test_train_gpu.py
git commit -m "vsp train: foreground training from the terminal; GPU smoke test with real Ultralytics"
```

---

### Task 6: Web client types, calls and pure helpers

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/api/rest.ts`
- Create: `web/src/lib/chart.ts`, `web/src/lib/train.ts`, `web/src/hooks/usePoll.ts`
- Test: `web/src/lib/chart.test.ts`, `web/src/lib/train.test.ts`

**Interfaces:**
- Produces types: `TrainParams`, `GpuQueue`, `TrainStatus`, `ExportInfo`, `RunStatus`, `RunBest`, `RunProgress`, `Run`, `MetricRow`, `RunDetail`, `Instance`, `PredictionItem`, `PredictionPage`, `StartRunsRequest`.
- Produces api: `trainStatus()`, `listExports()`, `listRuns()`, `getRun(id)`, `startRuns(body)`, `cancelRun(id)`, `resumeRun(id)`, `exportOnnx(id)`, `deleteRun(id)`, `predictions(id, offset, limit)`; `weightsUrl(runId, file)`, `valImageUrl(exportId, name)`.
- Produces `lib/chart.ts`: `Pt`, `SERIES_COLORS`, `GRID`, `extent(values)`, `scale(domain, range)`, `niceTicks(min, max, count)`, `linePath(points, x, y)`, `nearestIndex(xs, x)`.
- Produces `lib/train.ts`: `MASK_MAP`, `MASK_MAP50`, `BOX_MAP`, `bestRunIds(runs, key)`, `formatDuration(s)`, `defaultGpu(status)`, `polygonPoints(flat, w, h)`, `metricSeries(rows, key)`, `statusText(run)`, `startProblem(...)`, `runLabel(run)`.
- Produces `usePoll<T>(fetcher, intervalMs, deps) -> { data, error, refresh }`.

- [ ] **Step 1: Add the types** (append to `web/src/api/types.ts`)

```ts
export type TrainParams = { epochs: number; imgsz: number; batch: number; patience: number; workers: number; seed: number };
export type GpuQueue = {
  index: number;
  name: string;
  memory_total: number;
  running: string | null;
  running_kind: "train" | "onnx" | null;
  queue: string[];
};
export type TrainStatus = {
  available: boolean;
  error: string | null;
  gpus: GpuQueue[];
  sam3_gpu: number | null;
  models: string[];
  default_models: string[];
  defaults: TrainParams;
};
export type ExportInfo = {
  id: string;
  project_name: string;
  created_at: string;
  classes: string[];
  train_images: number;
  val_images: number;
};
export type RunStatus = "queued" | "running" | "done" | "error" | "cancelled" | "interrupted";
export type RunBest = {
  epoch: number;
  mask_map50: number | null;
  mask_map50_95: number | null;
  box_map50: number | null;
  box_map50_95: number | null;
  train_seconds: number | null;
};
export type RunProgress = { phase: "setup" | "train" | "val" | "predict"; epoch: number; epochs: number; batch: number; batches: number };
export type Run = {
  id: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  export_id: string;
  project_name: string;
  model: string;
  gpu: number;
  params: TrainParams;
  status: RunStatus;
  error: string | null;
  resume: boolean;
  onnx: string | null;
  best: RunBest | null;
  epochs_done: number;
  progress: RunProgress | null;
  resumable: boolean;
  has_best: boolean;
  has_onnx: boolean;
};
export type MetricRow = { epoch: number } & Record<string, number>;
export type RunDetail = Run & { metrics: MetricRow[] };
export type Instance = { class_id: number; score?: number; polygons: number[][] };
export type PredictionItem = { name: string; width: number; height: number; predictions: Instance[]; ground_truth: Instance[] };
export type PredictionPage = { total: number; conf: number | null; items: PredictionItem[] };
export type StartRunsRequest = { export_id: string; runs: { model: string; gpu: number }[]; params: TrainParams };
```

- [ ] **Step 2: Add the calls** (in `web/src/api/rest.ts`: extend the type import with `ExportInfo, PredictionPage, Run, RunDetail, StartRunsRequest, TrainStatus`, add the URL helpers next to `maskUrl`, and add to `api`)

```ts
export const weightsUrl = (runId: string, file: "best.pt" | "best.onnx") => `/api/train/runs/${runId}/weights/${file}`;
export const valImageUrl = (exportId: string, name: string) =>
  `/api/exports/${exportId}/images/val/${encodeURIComponent(name)}`;
```

```ts
  trainStatus: () => request<TrainStatus>("/api/train/status"),
  listExports: () => request<ExportInfo[]>("/api/exports"),
  listRuns: () => request<Run[]>("/api/train/runs"),
  getRun: (id: string) => request<RunDetail>(`/api/train/runs/${id}`),
  startRuns: (body: StartRunsRequest) => request<{ run_ids: string[] }>("/api/train/runs", post(body)),
  cancelRun: (id: string) => request<Run>(`/api/train/runs/${id}/cancel`, post()),
  resumeRun: (id: string) => request<Run>(`/api/train/runs/${id}/resume`, post()),
  exportOnnx: (id: string) => request<Run>(`/api/train/runs/${id}/onnx`, post()),
  deleteRun: (id: string) => request<void>(`/api/train/runs/${id}`, { method: "DELETE" }),
  predictions: (id: string, offset: number, limit: number) =>
    request<PredictionPage>(`/api/train/runs/${id}/predictions?offset=${offset}&limit=${limit}`),
```

- [ ] **Step 3: Write the failing helper tests**

`web/src/lib/chart.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { extent, linePath, nearestIndex, niceTicks, scale } from "./chart";

describe("chart helpers", () => {
  it("extent ignores non-finite values and widens a flat range", () => {
    expect(extent([0.2, NaN, 0.8])).toEqual([0.2, 0.8]);
    expect(extent([])).toEqual([0, 1]);
    expect(extent([3, 3])).toEqual([2.5, 3.5]);
  });
  it("scale maps linearly, including inverted ranges", () => {
    const y = scale([0, 1], [100, 0]);
    expect(y(0)).toBe(100);
    expect(y(0.25)).toBe(75);
    expect(scale([2, 2], [0, 10])(2)).toBe(0);
  });
  it("niceTicks steps by 1, 2 or 5 times a power of ten", () => {
    expect(niceTicks(0, 1, 4)).toEqual([0, 0.5, 1]);
    expect(niceTicks(0, 100, 4)).toEqual([0, 50, 100]);
    expect(niceTicks(0.31, 0.74, 4)).toEqual([0.4, 0.6]);
  });
  it("linePath skips non-finite points", () => {
    expect(linePath([[0, 0], [1, NaN], [2, 1]], (v) => v * 10, (v) => v * 10)).toBe("M0.0,0.0L20.0,10.0");
  });
  it("nearestIndex", () => {
    expect(nearestIndex([1, 2, 3, 10], 6.4)).toBe(2);
    expect(nearestIndex([1, 2, 3, 10], 6.6)).toBe(3);
    expect(nearestIndex([], 1)).toBe(-1);
  });
});
```

`web/src/lib/train.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { Run, TrainStatus } from "../api/types";
import { bestRunIds, defaultGpu, formatDuration, metricSeries, polygonPoints, startProblem, statusText } from "./train";

const run = (id: string, over: Partial<Run> = {}): Run => ({
  id, created_at: "", started_at: null, finished_at: null, export_id: "e", project_name: "P", model: "yolo26s-seg", gpu: 1,
  params: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 }, status: "done", error: null, resume: false,
  onnx: null, best: null, epochs_done: 0, progress: null, resumable: false, has_best: false, has_onnx: false, ...over,
});
const best = (v: number) => ({ epoch: 1, mask_map50: v, mask_map50_95: v, box_map50: v, box_map50_95: v, train_seconds: 60 });

describe("train helpers", () => {
  it("bestRunIds marks every run tied for the maximum", () => {
    const runs = [run("a", { best: best(0.7) }), run("b", { best: best(0.8) }), run("c", { best: best(0.8) }), run("d")];
    expect([...bestRunIds(runs, "mask_map50_95")].sort()).toEqual(["b", "c"]);
    expect(bestRunIds([run("d")], "mask_map50_95").size).toBe(0);
  });
  it("formatDuration", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(45)).toBe("45 s");
    expect(formatDuration(720)).toBe("12 min");
    expect(formatDuration(3900)).toBe("1 h 05 min");
  });
  it("defaultGpu avoids SAM3's GPU", () => {
    const st = { gpus: [{ index: 0 }, { index: 1 }], sam3_gpu: 0 } as unknown as TrainStatus;
    expect(defaultGpu(st)).toBe(1);
    expect(defaultGpu({ ...st, sam3_gpu: null })).toBe(0);
    expect(defaultGpu({ ...st, gpus: [] })).toBeNull();
  });
  it("polygonPoints scales normalised pairs", () => {
    expect(polygonPoints([0.5, 0.25, 1, 1], 200, 100)).toBe("100,25 200,100");
  });
  it("metricSeries keeps rows that have the key", () => {
    expect(metricSeries([{ epoch: 1, a: 0.1 }, { epoch: 2 }, { epoch: 3, a: 0.3 }] as never, "a")).toEqual([[1, 0.1], [3, 0.3]]);
  });
  it("statusText", () => {
    expect(statusText(run("a", { status: "running", progress: { phase: "train", epoch: 3, epochs: 100, batch: 1, batches: 9 } })))
      .toBe("Epoch 3/100");
    expect(statusText(run("a", { status: "running", progress: { phase: "val", epoch: 3, epochs: 100, batch: 9, batches: 9 } })))
      .toBe("Validating epoch 3");
    expect(statusText(run("a", { status: "running", progress: { phase: "predict", epoch: 3, epochs: 3, batch: 1, batches: 9 } })))
      .toBe("Predicting val images");
    expect(statusText(run("a", { status: "running" }))).toBe("Starting");
    expect(statusText(run("a", { status: "error" }))).toBe("Failed");
  });
  it("startProblem", () => {
    expect(startProblem({ available: false, error: "no ultralytics", exportId: "e", picked: 1 })).toBe("no ultralytics");
    expect(startProblem({ available: true, error: null, exportId: null, picked: 1 })).toBe("Choose an export to train on.");
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 0 })).toBe("Tick at least one model.");
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 2 })).toBeNull();
  });
});
```

- [ ] **Step 4: Run to see them fail**

Run: `cd web && npx vitest run src/lib/chart.test.ts src/lib/train.test.ts`
Expected: fail — modules not found.

- [ ] **Step 5: Implement**

`web/src/lib/chart.ts`:

```ts
export type Pt = [number, number];

/** Dark categorical slots 1–4 of the reference palette, validated (CVD and contrast) on the well colour #1E1F21. */
export const SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"] as const;
/** Grid and axis hairlines: the chrome's `line` token. */
export const GRID = "#4A4D52";
export const TEXT_DIM = "#A3A5AA";
export const TEXT_INK = "#E9E7E2";
export const WELL = "#1E1F21";

export function extent(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return [0, 1];
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  return lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi];
}

export function scale(domain: [number, number], range: [number, number]): (v: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v: number) => r0 + (v - d0) * k;
}

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

/** Ticks inside [min, max] at a 1/2/5 × 10^k step; rounded to kill float noise (0.30000000000000004). */
export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min];
  const step = niceStep((max - min) / count);
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

export function linePath(points: Pt[], x: (v: number) => number, y: (v: number) => number): string {
  return points
    .filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b))
    .map(([a, b], i) => `${i === 0 ? "M" : "L"}${x(a).toFixed(1)},${y(b).toFixed(1)}`)
    .join("");
}

export function nearestIndex(xs: number[], x: number): number {
  let best = -1;
  let dist = Infinity;
  xs.forEach((v, i) => {
    const d = Math.abs(v - x);
    if (d < dist) {
      dist = d;
      best = i;
    }
  });
  return best;
}
```

`web/src/lib/train.ts`:

```ts
import type { MetricRow, Run, RunBest, TrainStatus } from "../api/types";
import type { Pt } from "./chart";

export const MASK_MAP = "metrics/mAP50-95(M)";
export const MASK_MAP50 = "metrics/mAP50(M)";
export const BOX_MAP = "metrics/mAP50-95(B)";

/** Ids of the runs whose `best[key]` equals the maximum over all runs that have one. */
export function bestRunIds(runs: Run[], key: keyof RunBest): Set<string> {
  const scored = runs.filter((r) => r.best !== null && typeof r.best[key] === "number");
  if (scored.length === 0) return new Set();
  const top = Math.max(...scored.map((r) => r.best![key] as number));
  return new Set(scored.filter((r) => Math.abs((r.best![key] as number) - top) < 1e-9).map((r) => r.id));
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds - h * 3600) / 60);
  return `${h} h ${String(m).padStart(2, "0")} min`;
}

export function defaultGpu(status: Pick<TrainStatus, "gpus" | "sam3_gpu">): number | null {
  const other = status.gpus.find((g) => g.index !== status.sam3_gpu);
  return other?.index ?? status.gpus[0]?.index ?? null;
}

export function polygonPoints(flat: number[], w: number, h: number): string {
  const out: string[] = [];
  for (let i = 0; i + 1 < flat.length; i += 2) out.push(`${+(flat[i] * w).toFixed(1)},${+(flat[i + 1] * h).toFixed(1)}`);
  return out.join(" ");
}

export function metricSeries(rows: MetricRow[], key: string): Pt[] {
  return rows.filter((r) => typeof r[key] === "number").map((r) => [r.epoch, r[key]]);
}

export function statusText(run: Run): string {
  switch (run.status) {
    case "queued":
      return "Queued";
    case "running": {
      const p = run.progress;
      if (p === null || p.phase === "setup") return "Starting";
      if (p.phase === "val") return `Validating epoch ${p.epoch}`;
      if (p.phase === "predict") return "Predicting val images";
      return `Epoch ${p.epoch}/${p.epochs}`;
    }
    case "done":
      return "Done";
    case "error":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    case "interrupted":
      return "Interrupted";
  }
}

export const runLabel = (run: Pick<Run, "id" | "model">) => `${run.model} · ${run.id.slice(-4)}`;

export function startProblem(o: { available: boolean; error: string | null; exportId: string | null; picked: number }): string | null {
  if (!o.available) return o.error ?? "Training is not available on this server.";
  if (o.exportId === null) return "Choose an export to train on.";
  if (o.picked === 0) return "Tick at least one model.";
  return null;
}
```

`web/src/hooks/usePoll.ts`:

```ts
import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Fetch now, then every `intervalMs` (null = once) until unmount or a deps change. The latest fetcher is kept in
 * a ref, so an inline closure does not restart the loop on every render; `refresh()` fetches immediately.
 */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number | null, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const [nonce, setNonce] = useState(0);
  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const value = await fetcherRef.current();
        if (closed) return;
        setData(value);
        setError(null);
      } catch (e) {
        if (closed) return;
        setError(e instanceof Error ? e.message : String(e));
      }
      if (!closed && intervalMs !== null) timer = setTimeout(() => void tick(), intervalMs);
    };
    void tick();
    return () => {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, nonce, ...deps]);

  return { data, error, refresh };
}
```

- [ ] **Step 6: Run tests and typecheck**

Run: `cd web && npx vitest run src/lib/chart.test.ts src/lib/train.test.ts && npm run typecheck`
Expected: pass, no type errors.

- [ ] **Step 7: Commit**

```bash
git add web/src/api web/src/lib/chart.ts web/src/lib/chart.test.ts web/src/lib/train.ts web/src/lib/train.test.ts web/src/hooks/usePoll.ts
git commit -m "Train page groundwork: client types and calls, chart and run helpers, usePoll"
```

---

### Task 7: Train page components

**Files:**
- Create: `web/src/components/LineChart.tsx`, `web/src/components/MetricCharts.tsx`, `web/src/components/RunsTable.tsx`, `web/src/components/TrainSidebar.tsx`, `web/src/components/PredictionPreview.tsx`
- Test: `web/src/components/LineChart.test.tsx`, `web/src/components/RunsTable.test.tsx`

**Interfaces:**
- Consumes: Task 6 exports.
- Produces:
  - `LineChart({ title, series: { name, color, points: Pt[] }[], yFormat? })`
  - `MetricCharts({ run: RunDetail })`, `CompareChart({ runs: RunDetail[] })`
  - `RunsTable({ runs, selectedId, compareIds, onSelect, onToggleCompare, onCancel, onResume, onOnnx, onDelete })` — each `on*` is `(id: string) => void`
  - `TrainSidebar({ status, exports, runs, onStart, onCancel })` — `onStart(body: StartRunsRequest): Promise<void>`
  - `PredictionPreview({ run: Run })`

- [ ] **Step 1: Write the failing component tests**

`web/src/components/LineChart.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { LineChart } from "./LineChart";

afterEach(cleanup);

describe("LineChart", () => {
  it("draws one path per series, a legend for two, and a tooltip on hover", () => {
    const { container } = render(
      <LineChart
        title="Mask mAP"
        series={[
          { name: "mAP50-95", color: "#3987e5", points: [[1, 0.1], [2, 0.4], [3, 0.5]] },
          { name: "mAP50", color: "#d95926", points: [[1, 0.2], [2, 0.6], [3, 0.7]] },
        ]}
      />,
    );
    expect(container.querySelectorAll("path[data-series]")).toHaveLength(2);
    expect(screen.getByText("mAP50")).toBeTruthy();
    fireEvent.mouseMove(container.querySelector("rect[data-hit]")!, { clientX: 10000, clientY: 10 });
    const tip = screen.getByRole("tooltip");
    expect(tip.textContent).toContain("Epoch 3");
    expect(tip.textContent).toContain("0.70");
  });
  it("has no legend for one series and says so when empty", () => {
    const { container, rerender } = render(<LineChart title="Box" series={[{ name: "Box", color: "#3987e5", points: [[1, 0.3]] }]} />);
    expect(container.querySelectorAll("[data-legend]")).toHaveLength(0);
    rerender(<LineChart title="Box" series={[{ name: "Box", color: "#3987e5", points: [] }]} />);
    expect(screen.getByText("No epochs yet.")).toBeTruthy();
  });
});
```

`web/src/components/RunsTable.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Run } from "../api/types";
import { RunsTable } from "./RunsTable";

afterEach(cleanup);

const base: Run = {
  id: "20260924-120000-yolo26s-seg-aaaa", created_at: "", started_at: null, finished_at: null, export_id: "e", project_name: "Optics",
  model: "yolo26s-seg", gpu: 1, params: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 }, status: "done",
  error: null, resume: false, onnx: null, epochs_done: 100, progress: null, resumable: false, has_best: true, has_onnx: false,
  best: { epoch: 80, mask_map50: 0.9, mask_map50_95: 0.7, box_map50: 0.95, box_map50_95: 0.8, train_seconds: 3700 },
};

describe("RunsTable", () => {
  it("bolds the best value per metric and wires the actions", () => {
    const b: Run = { ...base, id: "20260924-120000-yolo26m-seg-bbbb", model: "yolo26m-seg",
      best: { ...base.best!, mask_map50_95: 0.75 } };
    const q: Run = { ...base, id: "20260924-120000-yolo11s-seg-cccc", model: "yolo11s-seg", status: "queued", best: null,
      has_best: false, epochs_done: 0 };
    const on = { onSelect: vi.fn(), onToggleCompare: vi.fn(), onCancel: vi.fn(), onResume: vi.fn(), onOnnx: vi.fn(), onDelete: vi.fn() };
    render(<RunsTable runs={[base, b, q]} selectedId={null} compareIds={[]} {...on} />);
    const rowB = screen.getByText("yolo26m-seg").closest("tr")!;
    expect(within(rowB).getByText("0.750").tagName).toBe("STRONG");
    const rowA = screen.getByText("yolo26s-seg").closest("tr")!;
    expect(within(rowA).getByText("0.700").tagName).not.toBe("STRONG");
    expect(within(rowA).getByText("1 h 02 min")).toBeTruthy();
    fireEvent.click(within(rowA).getByRole("button", { name: "Export ONNX" }));
    expect(on.onOnnx).toHaveBeenCalledWith(base.id);
    expect(on.onSelect).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByText("yolo11s-seg").closest("tr")!).getByRole("button", { name: "Cancel" }));
    expect(on.onCancel).toHaveBeenCalledWith(q.id);
    fireEvent.click(rowA);
    expect(on.onSelect).toHaveBeenCalledWith(base.id);
  });
});
```

- [ ] **Step 2: Run to see them fail**

Run: `cd web && npx vitest run src/components/LineChart.test.tsx src/components/RunsTable.test.tsx`
Expected: fail — modules not found.

- [ ] **Step 3: Implement `LineChart.tsx`**

```tsx
import { useState } from "react";
import { GRID, TEXT_DIM, TEXT_INK, WELL, extent, linePath, nearestIndex, niceTicks, scale, type Pt } from "../lib/chart";

export type Series = { name: string; color: string; points: Pt[] };

const W = 520;
const H = 200;
const M = { top: 12, right: 44, bottom: 26, left: 40 };

/**
 * A per-epoch line chart: 2 px lines, hairline grid, a legend only for two or more series, the last value written
 * at the end of a single line, and a crosshair + tooltip on hover. Text wears the ink tokens, never a series colour.
 */
export function LineChart({ title, series, yFormat = (v: number) => v.toFixed(2) }: {
  title: string;
  series: Series[];
  yFormat?: (v: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const x0 = xs[0] ?? 0;
  const x1 = Math.max(xs[xs.length - 1] ?? 1, x0 + 1);
  const [lo, hi] = extent(series.flatMap((s) => s.points.map((p) => p[1])));
  const yTicks = niceTicks(lo, hi, 4);
  const yDomain: [number, number] = [Math.min(lo, yTicks[0] ?? lo), Math.max(hi, yTicks[yTicks.length - 1] ?? hi)];
  const x = scale([x0, x1], [M.left, W - M.right]);
  const y = scale(yDomain, [H - M.bottom, M.top]);
  const xTicks = niceTicks(x0, x1, 5).filter(Number.isInteger);

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.ownerSVGElement?.getBoundingClientRect();
    const px = box && box.width > 0 ? ((e.clientX - box.left) * W) / box.width : e.clientX;
    const i = nearestIndex(xs, x0 + ((px - M.left) / (W - M.left - M.right)) * (x1 - x0));
    setHover(i < 0 ? null : xs[i]);
  };

  return (
    <figure className="flex min-w-0 flex-col gap-1 rounded border border-line bg-well p-2">
      <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 font-cond text-sm">
        <span className="font-semibold text-ink">{title}</span>
        {series.length >= 2 &&
          series.map((s) => (
            <span key={s.name} data-legend className="flex items-center gap-1 text-dim">
              <span aria-hidden className="inline-block h-0.5 w-3 rounded" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
      </figcaption>
      {xs.length === 0 ? (
        <p className="note px-2 py-6">No epochs yet.</p>
      ) : (
        <div className="relative">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${title} by epoch`}>
            {yTicks.map((t) => (
              <g key={`y${t}`}>
                <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={1} />
                <text x={M.left - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill={TEXT_DIM}>
                  {yFormat(t)}
                </text>
              </g>
            ))}
            {xTicks.map((t) => (
              <text key={`x${t}`} x={x(t)} y={H - 8} textAnchor="middle" fontSize={11} fill={TEXT_DIM}>
                {t}
              </text>
            ))}
            {series.map((s) =>
              s.points.length === 1 ? (
                <circle key={s.name} data-series={s.name} cx={x(s.points[0][0])} cy={y(s.points[0][1])} r={3} fill={s.color} />
              ) : (
                <path key={s.name} data-series={s.name} d={linePath(s.points, x, y)} fill="none" stroke={s.color}
                  strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              ),
            )}
            {series.length === 1 && series[0].points.length > 0 && (() => {
              const last = series[0].points[series[0].points.length - 1];
              return (
                <text x={x(last[0]) + 6} y={y(last[1]) + 4} fontSize={11} fill={TEXT_INK}>
                  {yFormat(last[1])}
                </text>
              );
            })()}
            {hover !== null && (
              <g>
                <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={H - M.bottom} stroke={TEXT_DIM} strokeWidth={1} />
                {series.map((s) => {
                  const p = s.points.find((q) => q[0] === hover);
                  return p ? <circle key={s.name} cx={x(p[0])} cy={y(p[1])} r={4} fill={s.color} stroke={WELL} strokeWidth={2} /> : null;
                })}
              </g>
            )}
            <rect data-hit x={M.left} y={M.top} width={W - M.left - M.right} height={H - M.top - M.bottom} fill="transparent"
              onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
          </svg>
          {hover !== null && (
            <div role="tooltip" className="pointer-events-none absolute top-1 rounded border border-line bg-panel px-2 py-1 font-cond text-xs"
              style={{ left: `${(x(hover) / W) * 100}%`, transform: x(hover) > W / 2 ? "translateX(calc(-100% - 8px))" : "translateX(8px)" }}>
              <div className="text-dim">Epoch {hover}</div>
              {series.map((s) => {
                const p = s.points.find((q) => q[0] === hover);
                return p ? (
                  <div key={s.name} className="flex items-center gap-1.5 whitespace-nowrap text-ink">
                    <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
                    {s.name} {yFormat(p[1])}
                  </div>
                ) : null;
              })}
            </div>
          )}
        </div>
      )}
    </figure>
  );
}
```

- [ ] **Step 4: Implement `MetricCharts.tsx`**

```tsx
import type { RunDetail } from "../api/types";
import { SERIES_COLORS as C } from "../lib/chart";
import { BOX_MAP, MASK_MAP, MASK_MAP50, metricSeries, runLabel } from "../lib/train";
import { LineChart } from "./LineChart";

const loss = (v: number) => v.toFixed(3);

export function MetricCharts({ run }: { run: RunDetail }) {
  const rows = run.metrics;
  const s = (name: string, key: string, color: string) => ({ name, color, points: metricSeries(rows, key) });
  return (
    <div className="grid gap-3 xl:grid-cols-2">
      <LineChart title="Mask mAP" series={[s("mAP50-95", MASK_MAP, C[0]), s("mAP50", MASK_MAP50, C[1])]} />
      <LineChart title="Box mAP50-95" series={[s("Box mAP50-95", BOX_MAP, C[0])]} />
      <LineChart title="Training loss" yFormat={loss}
        series={[s("box", "train/box_loss", C[0]), s("seg", "train/seg_loss", C[1]), s("cls", "train/cls_loss", C[2])]} />
      <LineChart title="Validation loss" yFormat={loss}
        series={[s("box", "val/box_loss", C[0]), s("seg", "val/seg_loss", C[1]), s("cls", "val/cls_loss", C[2])]} />
    </div>
  );
}

/** Mask mAP50-95 of up to four runs; a run keeps its colour slot for as long as it stays in the comparison. */
export function CompareChart({ runs }: { runs: RunDetail[] }) {
  return (
    <LineChart title="Mask mAP50-95, compared"
      series={runs.slice(0, C.length).map((r, i) => ({ name: runLabel(r), color: C[i], points: metricSeries(r.metrics, MASK_MAP) }))} />
  );
}
```

- [ ] **Step 5: Implement `RunsTable.tsx`**

```tsx
import type { Run, RunBest } from "../api/types";
import { weightsUrl } from "../api/rest";
import { bestRunIds, formatDuration, statusText } from "../lib/train";

const METRICS: { key: keyof RunBest; label: string }[] = [
  { key: "mask_map50", label: "Mask mAP50" },
  { key: "mask_map50_95", label: "Mask mAP50-95" },
  { key: "box_map50_95", label: "Box mAP50-95" },
];
const MAX_COMPARE = 4;
const stop = (e: React.SyntheticEvent) => e.stopPropagation();

type Props = {
  runs: Run[];
  selectedId: string | null;
  compareIds: string[];
  onSelect: (id: string) => void;
  onToggleCompare: (id: string) => void;
  onCancel: (id: string) => void;
  onResume: (id: string) => void;
  onOnnx: (id: string) => void;
  onDelete: (id: string) => void;
};

function OnnxCell({ run, onOnnx }: { run: Run; onOnnx: (id: string) => void }) {
  if (run.has_onnx) return <a className="btn" href={weightsUrl(run.id, "best.onnx")} download>best.onnx</a>;
  if (run.onnx === "queued" || run.onnx === "running") return <span className="note">Exporting…</span>;
  if (run.status !== "done") return null;
  const failed = run.onnx?.startsWith("error") ?? false;
  return (
    <button type="button" className="btn" title={failed ? run.onnx ?? undefined : undefined} onClick={() => onOnnx(run.id)}>
      {failed ? "Retry ONNX" : "Export ONNX"}
    </button>
  );
}

export function RunsTable({ runs, selectedId, compareIds, onSelect, onToggleCompare, onCancel, onResume, onOnnx, onDelete }: Props) {
  const best = Object.fromEntries(METRICS.map((m) => [m.key, bestRunIds(runs, m.key)]));
  if (runs.length === 0) return <p className="note">No training runs yet. Start one from the panel on the right.</p>;
  return (
    <div className="overflow-x-auto rounded border border-line">
      <table className="w-full text-sm">
        <thead className="bg-panel font-cond text-dim">
          <tr className="text-left">
            <th className="px-2 py-1.5 font-medium">Compare</th>
            <th className="px-2 py-1.5 font-medium">Model</th>
            <th className="px-2 py-1.5 font-medium">Export</th>
            <th className="px-2 py-1.5 font-medium">GPU</th>
            <th className="px-2 py-1.5 font-medium">Status</th>
            <th className="px-2 py-1.5 text-right font-medium">Epochs</th>
            {METRICS.map((m) => (
              <th key={m.key} className="px-2 py-1.5 text-right font-medium">{m.label}</th>
            ))}
            <th className="px-2 py-1.5 text-right font-medium">Best epoch</th>
            <th className="px-2 py-1.5 text-right font-medium">Time</th>
            <th className="px-2 py-1.5 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const comparing = compareIds.includes(r.id);
            const active = r.status === "queued" || r.status === "running";
            return (
              <tr key={r.id} aria-selected={r.id === selectedId} onClick={() => onSelect(r.id)}
                className={`cursor-pointer border-t border-line ${r.id === selectedId ? "bg-raise" : "hover:bg-panel"}`}>
                <td className="px-2 py-1.5" onClick={stop}>
                  <input type="checkbox" aria-label={`Compare ${r.model}`} checked={comparing}
                    disabled={!comparing && compareIds.length >= MAX_COMPARE} onChange={() => onToggleCompare(r.id)} />
                </td>
                <td className="px-2 py-1.5 font-medium">{r.model}</td>
                <td className="px-2 py-1.5 text-dim">{r.project_name}</td>
                <td className="px-2 py-1.5 font-cond text-dim">cuda:{r.gpu}</td>
                <td className="px-2 py-1.5">
                  <div className="flex flex-col gap-1">
                    <span className={r.status === "error" ? "text-danger" : ""}>{statusText(r)}</span>
                    {r.status === "running" && r.progress !== null && r.progress.batches > 0 && (
                      <progress value={r.progress.batch} max={r.progress.batches} />
                    )}
                  </div>
                </td>
                <td className="px-2 py-1.5 text-right font-cond">{r.epochs_done}/{r.params.epochs}</td>
                {METRICS.map((m) => {
                  const v = r.best?.[m.key];
                  const text = typeof v === "number" ? v.toFixed(3) : "—";
                  return (
                    <td key={m.key} className="px-2 py-1.5 text-right font-cond">
                      {best[m.key].has(r.id) ? <strong className="font-semibold text-ink">{text}</strong> : <span className="text-dim">{text}</span>}
                    </td>
                  );
                })}
                <td className="px-2 py-1.5 text-right font-cond text-dim">{r.best?.epoch ?? "—"}</td>
                <td className="px-2 py-1.5 text-right font-cond text-dim">{formatDuration(r.best?.train_seconds)}</td>
                <td className="px-2 py-1.5" onClick={stop}>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {r.has_best && <a className="btn" href={weightsUrl(r.id, "best.pt")} download>best.pt</a>}
                    <OnnxCell run={r} onOnnx={onOnnx} />
                    {active && <button type="button" className="btn" onClick={() => onCancel(r.id)}>Cancel</button>}
                    {r.resumable && <button type="button" className="btn" onClick={() => onResume(r.id)}>Resume</button>}
                    {!active && r.onnx !== "queued" && r.onnx !== "running" && (
                      <button type="button" className="btn btn-danger" onClick={() => onDelete(r.id)}>Delete</button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 6: Implement `TrainSidebar.tsx`**

```tsx
import { useEffect, useState } from "react";
import type { ExportInfo, Run, StartRunsRequest, TrainParams, TrainStatus } from "../api/types";
import { defaultGpu, startProblem, statusText } from "../lib/train";

type ModelPick = { checked: boolean; gpu: number };
const PARAM_FIELDS: { key: keyof TrainParams; label: string; min: number; max: number; step: number }[] = [
  { key: "epochs", label: "Epochs", min: 1, max: 1000, step: 1 },
  { key: "imgsz", label: "Image size (px)", min: 320, max: 1920, step: 32 },
  { key: "batch", label: "Batch", min: 1, max: 64, step: 1 },
  { key: "patience", label: "Early-stop patience", min: 0, max: 1000, step: 1 },
];
const gib = (bytes: number) => `${Math.round(bytes / 2 ** 30)} GiB`;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

type Props = {
  status: TrainStatus | null;
  exports: ExportInfo[];
  runs: Run[];
  onStart: (body: StartRunsRequest) => Promise<void>;
  onCancel: (id: string) => void;
};

export function TrainSidebar({ status, exports, runs, onStart, onCancel }: Props) {
  const [exportId, setExportId] = useState<string | null>(null);
  const [picks, setPicks] = useState<Record<string, ModelPick>>({});
  const [showAll, setShowAll] = useState(false);
  const [params, setParams] = useState<TrainParams | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  useEffect(() => {
    if (exportId === null && exports.length > 0) setExportId(exports[0].id);
  }, [exports, exportId]);
  useEffect(() => {
    if (status === null || Object.keys(picks).length > 0) return;
    const gpu = defaultGpu(status) ?? 0;
    setPicks(Object.fromEntries(status.models.map((m) => [m, { checked: status.default_models.includes(m), gpu }])));
    setParams(status.defaults);
  }, [status, picks]);

  if (status === null || params === null) return <aside className="note p-3">Loading…</aside>;

  const visible = showAll ? status.models : status.models.filter((m) => status.default_models.includes(m) || picks[m]?.checked);
  const chosen = status.models.filter((m) => picks[m]?.checked);
  const onSam3 = status.sam3_gpu !== null && chosen.some((m) => picks[m].gpu === status.sam3_gpu);
  const problem = startProblem({ available: status.available, error: status.error, exportId, picked: chosen.length });
  const byId = new Map(runs.map((r) => [r.id, r]));

  const start = async () => {
    if (problem !== null || starting || exportId === null) return;
    setStarting(true);
    setStartError(null);
    try {
      await onStart({ export_id: exportId, runs: chosen.map((m) => ({ model: m, gpu: picks[m].gpu })), params });
    } catch (e) {
      setStartError(message(e));
    } finally {
      setStarting(false);
    }
  };

  return (
    <aside className="flex shrink-0 flex-col gap-5 border-t border-line bg-panel p-3 md:w-96 md:overflow-y-auto md:border-l md:border-t-0">
      <section className="flex flex-col gap-2">
        <h2 className="heading">New runs</h2>
        <label className="flex flex-col gap-1 text-sm text-dim">
          Export
          <select aria-label="Export" className="field" value={exportId ?? ""} onChange={(e) => setExportId(e.target.value || null)}>
            {exports.length === 0 && <option value="">No yolo_seg export yet</option>}
            {exports.map((e) => (
              <option key={e.id} value={e.id}>
                {e.project_name} · {e.id} · {e.train_images} train / {e.val_images} val
              </option>
            ))}
          </select>
        </label>
        <fieldset className="flex flex-col gap-1.5">
          <legend className="mb-1 text-sm text-dim">Models</legend>
          {visible.map((m) => (
            <div key={m} className="flex items-center justify-between gap-2 text-sm">
              <label className="flex items-center gap-2">
                <input type="checkbox" aria-label={m} checked={picks[m]?.checked ?? false}
                  onChange={(e) => setPicks((p) => ({ ...p, [m]: { ...p[m], checked: e.target.checked } }))} />
                {m}
              </label>
              <select aria-label={`GPU for ${m}`} className="field py-0.5" value={picks[m]?.gpu ?? 0}
                onChange={(e) => setPicks((p) => ({ ...p, [m]: { ...p[m], gpu: Number(e.target.value) } }))}>
                {status.gpus.map((g) => (
                  <option key={g.index} value={g.index}>
                    cuda:{g.index} {g.name}{g.index === status.sam3_gpu ? " (shared with SAM3)" : ""}
                  </option>
                ))}
              </select>
            </div>
          ))}
          <button type="button" className="btn self-start" onClick={() => setShowAll((v) => !v)}>
            {showAll ? "Fewer models" : "More models"}
          </button>
        </fieldset>
        {onSam3 && (
          <p className="note text-ink">
            SAM3 runs on cuda:{status.sam3_gpu}. Training there shares its memory; labeling or training may run out of VRAM.
          </p>
        )}
        {PARAM_FIELDS.map((f) => (
          <label key={f.key} className="flex items-center justify-between gap-2 text-sm text-dim">
            {f.label}
            <input aria-label={f.label} type="number" className="field w-24" min={f.min} max={f.max} step={f.step}
              value={params[f.key]} onChange={(e) => setParams({ ...params, [f.key]: Number(e.target.value) || f.min })} />
          </label>
        ))}
        {problem !== null && <p className="note text-ink">{problem}</p>}
        {startError !== null && <p role="alert" className="alert">{startError}</p>}
        <button type="button" className="btn btn-on" disabled={problem !== null || starting} onClick={() => void start()}>
          Start {chosen.length} run{chosen.length === 1 ? "" : "s"}
        </button>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="heading">GPU queues</h2>
        {status.gpus.length === 0 && <p className="note">No CUDA GPU is visible to the server.</p>}
        {status.gpus.map((g) => {
          const current = g.running ? byId.get(g.running) : undefined;
          return (
            <div key={g.index} className="flex flex-col gap-1.5 rounded border border-line p-2 text-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">cuda:{g.index} {g.name}</span>
                <span className="font-cond text-dim">{gib(g.memory_total)}{g.index === status.sam3_gpu ? " · SAM3" : ""}</span>
              </div>
              {current === undefined && g.queue.length === 0 && <span className="note">Idle</span>}
              {current !== undefined && (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center justify-between gap-2">
                    <span>
                      {current.model} · {g.running_kind === "onnx" ? "exporting ONNX" : statusText(current)}
                    </span>
                    {g.running_kind === "train" && (
                      <button type="button" className="btn" onClick={() => onCancel(current.id)}>Cancel</button>
                    )}
                  </div>
                  {g.running_kind === "train" && current.progress !== null && current.progress.batches > 0 && (
                    <progress value={current.progress.batch} max={current.progress.batches} />
                  )}
                </div>
              )}
              {g.queue.map((id, i) => (
                <div key={id} className="flex items-center justify-between gap-2 text-dim">
                  <span>{i + 1}. {byId.get(id)?.model ?? id}</span>
                  <button type="button" className="btn" aria-label={`Cancel queued ${byId.get(id)?.model ?? id}`} onClick={() => onCancel(id)}>
                    Cancel
                  </button>
                </div>
              ))}
            </div>
          );
        })}
      </section>
    </aside>
  );
}
```

- [ ] **Step 7: Implement `PredictionPreview.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api, valImageUrl } from "../api/rest";
import type { PredictionPage, Run } from "../api/types";
import { SERIES_COLORS, TEXT_INK, WELL } from "../lib/chart";
import { polygonPoints } from "../lib/train";

const PRED = SERIES_COLORS[0];

/** A finished run's predictions on the val images: ground truth outlined in ink, predictions filled in blue. */
export function PredictionPreview({ run }: { run: Run }) {
  const [index, setIndex] = useState(0);
  const [page, setPage] = useState<PredictionPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showGt, setShowGt] = useState(true);
  const [showPred, setShowPred] = useState(true);

  useEffect(() => setIndex(0), [run.id]);
  useEffect(() => {
    let closed = false;
    api.predictions(run.id, index, 1).then(
      (p) => !closed && (setPage(p), setError(null)),
      (e: unknown) => !closed && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      closed = true;
    };
  }, [run.id, index]);

  const total = page?.total ?? 0;
  const step = (d: number) => setIndex((i) => (total === 0 ? 0 : (i + d + total) % total));
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (error !== null) return <p role="alert" className="alert">{error}</p>;
  const item = page?.items[0];
  if (!item) return <p className="note">Loading predictions…</p>;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <button type="button" className="btn" aria-label="Previous image" onClick={() => step(-1)}>‹</button>
        <span className="font-cond text-dim">{index + 1} / {total} · {item.name}</span>
        <button type="button" className="btn" aria-label="Next image" onClick={() => step(1)}>›</button>
        <label className="flex items-center gap-1.5 text-dim">
          <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm border-2" style={{ borderColor: TEXT_INK }} />
          Ground truth
        </label>
        <label className="flex items-center gap-1.5 text-dim">
          <input type="checkbox" checked={showPred} onChange={(e) => setShowPred(e.target.checked)} />
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: PRED }} />
          Predictions (conf ≥ {page?.conf ?? 0.25})
        </label>
        <span className="note">{item.predictions.length} predicted, {item.ground_truth.length} labeled</span>
      </div>
      <div className="relative w-full overflow-hidden rounded border border-line bg-well">
        <img src={valImageUrl(run.export_id, item.name)} alt={item.name} className="block w-full" />
        <svg viewBox={`0 0 ${item.width} ${item.height}`} className="absolute inset-0 h-full w-full" aria-hidden>
          {showPred && item.predictions.flatMap((inst, i) =>
            inst.polygons.map((poly, j) => (
              <polygon key={`p${i}-${j}`} points={polygonPoints(poly, item.width, item.height)} fill={PRED} fillOpacity={0.35}
                stroke={PRED} strokeWidth={2} vectorEffect="non-scaling-stroke" />
            )))}
          {showGt && item.ground_truth.flatMap((inst, i) =>
            inst.polygons.map((poly, j) => (
              <polygon key={`g${i}-${j}`} points={polygonPoints(poly, item.width, item.height)} fill="none" stroke={TEXT_INK}
                strokeWidth={2} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
            )))}
          {showPred && item.predictions.map((inst, i) =>
            inst.score !== undefined && inst.polygons[0]?.length >= 2 ? (
              <text key={`s${i}`} x={inst.polygons[0][0] * item.width} y={inst.polygons[0][1] * item.height - 6}
                fontSize={Math.max(12, item.width / 60)} fill={TEXT_INK} stroke={WELL} strokeWidth={3} paintOrder="stroke">
                {inst.score.toFixed(2)}
              </text>
            ) : null)}
        </svg>
      </div>
    </div>
  );
}
```

- [ ] **Step 8: Run tests and typecheck**

Run: `cd web && npx vitest run src/components/LineChart.test.tsx src/components/RunsTable.test.tsx && npm run typecheck`
Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add web/src/components/LineChart.tsx web/src/components/LineChart.test.tsx web/src/components/MetricCharts.tsx \
  web/src/components/RunsTable.tsx web/src/components/RunsTable.test.tsx web/src/components/TrainSidebar.tsx \
  web/src/components/PredictionPreview.tsx
git commit -m "Train page parts: line charts, runs table, sidebar with GPU queues, prediction preview"
```

---

### Task 8: Train page, route and header link

**Files:**
- Create: `web/src/pages/Train.tsx`
- Modify: `web/src/App.tsx` (route `/train`), `web/src/components/AppHeader.tsx` (a "Train" link)
- Test: `web/src/pages/Train.test.tsx`

**Interfaces:**
- Consumes: Tasks 6–7.
- Produces: route `/train`.

- [ ] **Step 1: Write the failing page test**

`web/src/pages/Train.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { ExportInfo, Run, RunDetail, TrainStatus } from "../api/types";
import { Train } from "./Train";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: {
      health: vi.fn(), trainStatus: vi.fn(), listExports: vi.fn(), listRuns: vi.fn(), getRun: vi.fn(), startRuns: vi.fn(),
      cancelRun: vi.fn(), resumeRun: vi.fn(), exportOnnx: vi.fn(), deleteRun: vi.fn(), predictions: vi.fn(),
    },
  };
});

const status: TrainStatus = {
  available: true, error: null, sam3_gpu: 0,
  gpus: [
    { index: 0, name: "RTX 3080 Ti", memory_total: 12 * 2 ** 30, running: null, running_kind: null, queue: [] },
    { index: 1, name: "RTX 2080 Ti", memory_total: 11 * 2 ** 30, running: "r1", running_kind: "train", queue: ["r2"] },
  ],
  models: ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg"],
  default_models: ["yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg"],
  defaults: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 },
};
const exp: ExportInfo = { id: "20260924-040124-8336a8", project_name: "OpticsConverter1", created_at: "", classes: ["Module"], train_images: 798, val_images: 252 };
const run = (id: string, model: string, over: Partial<Run> = {}): Run => ({
  id, created_at: "", started_at: null, finished_at: null, export_id: exp.id, project_name: "OpticsConverter1", model, gpu: 1,
  params: status.defaults, status: "queued", error: null, resume: false, onnx: null, best: null, epochs_done: 0, progress: null,
  resumable: false, has_best: false, has_onnx: false, ...over,
});
const r1 = run("r1", "yolo26s-seg", { status: "running", epochs_done: 2, progress: { phase: "train", epoch: 3, epochs: 100, batch: 5, batches: 100 } });
const r2 = run("r2", "yolo26m-seg");
const m = vi.mocked(api);

const renderPage = () => render(<MemoryRouter><Train /></MemoryRouter>);

beforeEach(() => {
  Object.values(m).forEach((fn) => fn.mockReset());
  m.trainStatus.mockResolvedValue(status);
  m.listExports.mockResolvedValue([exp]);
  m.listRuns.mockResolvedValue([r1, r2]);
  m.getRun.mockImplementation(async (id: string) => ({ ...(id === "r1" ? r1 : r2), metrics: [
    { epoch: 1, "metrics/mAP50-95(M)": 0.2, "metrics/mAP50(M)": 0.4 },
    { epoch: 2, "metrics/mAP50-95(M)": 0.3, "metrics/mAP50(M)": 0.5 },
  ] }) as RunDetail);
  m.startRuns.mockResolvedValue({ run_ids: ["r3"] });
  m.cancelRun.mockResolvedValue(r2);
});
afterEach(cleanup);

describe("Train page", () => {
  it("starts the default models on the non-SAM3 GPU with the export and params", async () => {
    renderPage();
    const start = await screen.findByRole("button", { name: "Start 4 runs" });
    fireEvent.change(screen.getByLabelText("Epochs"), { target: { value: "50" } });
    fireEvent.click(start);
    await waitFor(() => expect(m.startRuns).toHaveBeenCalled());
    const body = m.startRuns.mock.calls[0][0];
    expect(body.export_id).toBe(exp.id);
    expect(body.runs).toEqual([
      { model: "yolo26s-seg", gpu: 1 }, { model: "yolo26m-seg", gpu: 1 }, { model: "yolo11s-seg", gpu: 1 }, { model: "yolo11m-seg", gpu: 1 },
    ]);
    expect(body.params.epochs).toBe(50);
  });

  it("warns when a model is put on SAM3's GPU", async () => {
    renderPage();
    const select = await screen.findByLabelText("GPU for yolo26m-seg");
    expect(screen.queryByText(/SAM3 runs on cuda:0/)).toBeNull();
    fireEvent.change(select, { target: { value: "0" } });
    expect(screen.getByText(/SAM3 runs on cuda:0/)).toBeTruthy();
  });

  it("shows the queues, cancels a queued run, and charts the selected run", async () => {
    renderPage();
    const queue = await screen.findByText("GPU queues");
    const panel = queue.closest("section")!;
    expect(within(panel).getByText(/yolo26s-seg · Epoch 3\/100/)).toBeTruthy();
    fireEvent.click(within(panel).getByRole("button", { name: "Cancel queued yolo26m-seg" }));
    await waitFor(() => expect(m.cancelRun).toHaveBeenCalledWith("r2"));
    expect(await screen.findByRole("img", { name: "Mask mAP by epoch" })).toBeTruthy();
    expect(m.getRun).toHaveBeenCalledWith("r1");
  });

  it("explains when training is unavailable", async () => {
    m.trainStatus.mockResolvedValue({ ...status, available: false, error: "ultralytics is not installed in this environment; run `uv sync`" });
    renderPage();
    expect(await screen.findByText(/ultralytics is not installed/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Start 4 runs" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
```

- [ ] **Step 2: Run to see it fail**

Run: `cd web && npx vitest run src/pages/Train.test.tsx`
Expected: fail — `./Train` not found.

- [ ] **Step 3: Implement `web/src/pages/Train.tsx`**

```tsx
import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/rest";
import type { RunDetail, StartRunsRequest } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { CompareChart, MetricCharts } from "../components/MetricCharts";
import { PredictionPreview } from "../components/PredictionPreview";
import { RunsTable } from "../components/RunsTable";
import { TrainSidebar } from "../components/TrainSidebar";
import { usePoll } from "../hooks/usePoll";
import { statusText } from "../lib/train";

const POLL_MS = 2000;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

const back = (
  <Link to="/" aria-label="Back to clips" className="flex items-center gap-1 font-cond text-dim hover:text-ink">
    <span aria-hidden>‹</span> Clips
  </Link>
);

export function Train() {
  const status = usePoll(() => api.trainStatus(), POLL_MS, []);
  const exports = usePoll(() => api.listExports(), 15000, []);
  const runs = usePoll(() => api.listRuns(), POLL_MS, []);
  const [pickedId, setPickedId] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [actionError, setActionError] = useState<string | null>(null);

  const list = runs.data ?? [];
  const selectedId = pickedId ?? list.find((r) => r.status === "running")?.id ?? list[0]?.id ?? null;
  const selected = list.find((r) => r.id === selectedId) ?? null;
  const detail = usePoll<RunDetail | null>(
    () => (selectedId ? api.getRun(selectedId) : Promise.resolve(null)),
    selected?.status === "running" ? POLL_MS : null,
    [selectedId, selected?.status],
  );
  const anyCompareRunning = list.some((r) => compareIds.includes(r.id) && r.status === "running");
  const compared = usePoll<RunDetail[]>(
    () => Promise.all(compareIds.map((id) => api.getRun(id))),
    anyCompareRunning ? 5000 : null,
    [compareIds.join(",")],
  );

  const act = useCallback(
    (fn: () => Promise<unknown>) => {
      setActionError(null);
      fn().then(
        () => {
          runs.refresh();
          status.refresh();
        },
        (e: unknown) => setActionError(message(e)),
      );
    },
    [runs, status],
  );

  const onStart = async (body: StartRunsRequest) => {
    const { run_ids } = await api.startRuns(body);
    if (run_ids.length > 0) setPickedId(run_ids[0]);
    runs.refresh();
    status.refresh();
  };
  const onDelete = (id: string) => {
    if (!window.confirm("Delete this run and its weights?")) return;
    act(() => api.deleteRun(id));
    setCompareIds((ids) => ids.filter((x) => x !== id));
    if (pickedId === id) setPickedId(null);
  };
  const toggleCompare = (id: string) =>
    setCompareIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]));

  const shown = detail.data !== null && detail.data.id === selectedId ? detail.data : null;

  return (
    <div className="flex h-full flex-col">
      <AppHeader>
        {back}
        <h1 className="truncate font-medium">Train</h1>
      </AppHeader>
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto md:flex-row md:overflow-hidden">
        <div className="flex min-w-0 flex-1 flex-col gap-5 p-4 md:overflow-y-auto">
          {(runs.error ?? status.error ?? actionError) !== null && (
            <p role="alert" className="alert">{runs.error ?? status.error ?? actionError}</p>
          )}
          <section className="flex flex-col gap-2">
            <h2 className="heading">Runs</h2>
            <RunsTable runs={list} selectedId={selectedId} compareIds={compareIds} onSelect={setPickedId}
              onToggleCompare={toggleCompare} onCancel={(id) => act(() => api.cancelRun(id))}
              onResume={(id) => act(() => api.resumeRun(id))} onOnnx={(id) => act(() => api.exportOnnx(id))} onDelete={onDelete} />
          </section>

          {compareIds.length >= 2 && compared.data !== null && (
            <section className="flex flex-col gap-2">
              <h2 className="heading">Comparison</h2>
              <CompareChart runs={compared.data} />
            </section>
          )}

          {selected !== null && (
            <section className="flex flex-col gap-2">
              <h2 className="heading">
                {selected.model} on {selected.project_name} · {statusText(selected)}
              </h2>
              {selected.error !== null && (
                <pre className="alert max-h-40 overflow-auto whitespace-pre-wrap font-cond text-xs">{selected.error}</pre>
              )}
              {shown !== null ? <MetricCharts run={shown} /> : <p className="note">Loading metrics…</p>}
            </section>
          )}

          {selected !== null && (
            <section className="flex flex-col gap-2">
              <h2 className="heading">Validation predictions</h2>
              {selected.status === "done" ? (
                <PredictionPreview run={selected} />
              ) : (
                <p className="note">Predictions appear when the run finishes.</p>
              )}
            </section>
          )}
        </div>
        <TrainSidebar status={status.data} exports={exports.data ?? []} runs={list} onStart={onStart}
          onCancel={(id) => act(() => api.cancelRun(id))} />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Route and header link**

`web/src/App.tsx`: `import { Train } from "./pages/Train";` and add `<Route path="/train" element={<Train />} />` after the export route.

`web/src/components/AppHeader.tsx`: between the `children` div and the engine indicator, add

```tsx
        <Link to="/train" className="shrink-0 font-cond text-dim hover:text-ink">
          Train
        </Link>
```

- [ ] **Step 5: Run the whole web suite, typecheck and build**

Run: `cd web && npm test && npm run build`
Expected: all vitest files pass (existing ones too), build writes `web/dist`.

- [ ] **Step 6: Commit**

```bash
git add web/src/pages/Train.tsx web/src/pages/Train.test.tsx web/src/App.tsx web/src/components/AppHeader.tsx
git commit -m "Train page: runs, live curves, comparison, predictions and the GPU queue sidebar"
```

---

### Task 9: Docs

**Files:**
- Modify: `README.md` (new "Training" section after "Exporting"), `docs/superpowers/plans/2026-09-20-vsp-roadmap.md` (a row for this plan)

- [ ] **Step 1: README section**

Add after the "Exporting" section:

```markdown
### Training

`uv sync` installs Ultralytics (YOLO26 and YOLO11 segmentation). Train from the Train page (header → Train) or from
the terminal:

    uv run vsp train --list                                  # exports that have a yolo_seg dataset
    uv run vsp train --export <id> --gpu 1                   # yolo26s/m-seg and yolo11s/m-seg, one after another
    uv run vsp train --export <id> --model yolo26s-seg --epochs 150 --imgsz 960 --batch 8

Each run lives in `data/runs/<run_id>/`: `run.json` (request and status), `worker.log`, Ultralytics' own output under
`train/` (`weights/best.pt`, `last.pt`, `results.csv`, plots) and `predictions.json` (best.pt on every val image, conf
≥ 0.25). Pretrained checkpoints are cached in `data/train_weights/`. A run is a separate process
(`python -m vsp.train.worker`) with `CUDA_VISIBLE_DEVICES` set to its GPU, so a crash or out-of-memory error ends
that run only.

The server keeps one queue per GPU: one run at a time per GPU, GPUs in parallel. The default GPU is the one SAM3 is not
on; choosing SAM3's GPU is allowed but shares its VRAM. Cancel stops a run after SIGTERM (SIGKILL after 30 s) and keeps
`last.pt`, so Resume continues from the last finished epoch. A run that was training when the server stopped is shown
as "interrupted" and can be resumed; queued runs are queued again. `vsp train` runs in the foreground and owns its
runs; Ctrl-C cancels them. Do not start it on a GPU the server is also training on.

Defaults: 100 epochs, early stop after 30 epochs without improvement, 960 px, batch 8, COCO-pretrained weights,
Ultralytics' default augmentation. Metrics come from `results.csv`; the comparison table ranks runs by the best mask
mAP50-95 epoch. ONNX export (`best.onnx`, static 960 × 960) runs on the run's GPU queue.
```

- [ ] **Step 2: Roadmap row**

In `docs/superpowers/plans/2026-09-20-vsp-roadmap.md`, add a row to the plans table following the existing rows' format: `| 6 | Training | vsp.train (worker, per-GPU scheduler), /api/train, vsp train, Train page | Yes | **Done** (branch yolo-seg-training; plan 2026-09-24-yolo-seg-training.md) |` — adapt the column count to the table's header.

- [ ] **Step 3: Full verification and commit**

Run: `uv run pytest -q && uv run ruff check server && (cd web && npm test && npm run build)`
Expected: everything passes.

```bash
git add README.md docs/superpowers/plans/2026-09-20-vsp-roadmap.md
git commit -m "Document training (CLI, Train page, run folders, GPU queues)"
```

---

### Task 10: Train the four models on export `20260924-040124-8336a8`

Operational, no code. Uses the CLI so the running labeling server (on `cuda:0`) is not restarted.

- [ ] **Step 1: Check free VRAM**

Run: `nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv` (note nvidia-smi orders by PCI bus; map by name: torch `cuda:0` = RTX 3080 Ti, `cuda:1` = RTX 2080 Ti).

- [ ] **Step 2: Start both GPUs' runs in the background**

```bash
uv run vsp train --export 20260924-040124-8336a8 --gpu 1 --model yolo26s-seg --model yolo26m-seg > data/runs/cli-gpu1.log 2>&1 &
uv run vsp train --export 20260924-040124-8336a8 --gpu 0 --model yolo11s-seg --model yolo11m-seg > data/runs/cli-gpu0.log 2>&1 &
```

- [ ] **Step 3: Watch the first epochs**

Confirm each first run reaches epoch 2 and that VRAM on `cuda:0` stays below its total with SAM3 resident. If a medium model runs out of memory on `cuda:0`, resume it later with `--batch 4` on the next free GPU (a new run) and note it in the report.

- [ ] **Step 4: Report**

When both commands exit, collect the final tables from the two logs and report per model: status, best epoch, mask mAP50, mask mAP50-95, box mAP50-95, train time, and the `best.pt` paths. Restarting `vsp serve` (the user's call) makes the runs visible on the Train page.
