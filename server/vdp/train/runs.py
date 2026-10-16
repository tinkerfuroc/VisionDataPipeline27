"""Run folders under data/runs/ and their run.json (spec: Storage). The scheduler is run.json's only
writer."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path

from ..store import NotFound
from ..store.project import now_iso
from .models import ExportInfo, TrainParams
from .sampling import epoch_plan

TERMINAL = frozenset({"done", "error", "cancelled", "interrupted"})
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9-]+-[0-9a-f]{4}$")
_REQUIRED_KEYS = frozenset({"id", "created_at", "status", "gpu", "model", "params"})

# Re-export for callers (later tasks import now_iso from vdp.train.runs)
__all__ = ["now_iso"]


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def new_run_id(model: str, now: datetime | None = None) -> str:
    now = now or datetime.now()  # noqa: DTZ005  # local time, like export ids
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

    def create(
        self, export: ExportInfo, model: str, gpu: int, params: TrainParams, *, owner_pid: int
    ) -> dict:
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
            # sampled epochs for this export (None: whole epochs); a resume keeps it, a new run works it out anew
            "sampling": epoch_plan(params.model_dump(), export.train_images, export.val_images),
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
        """Every run.json under the root, newest first. A run.json that is missing a required key
        or whose `id` does not match `_RUN_ID_RE` (half-written, or corrupted) is skipped rather
        than breaking the whole listing."""
        if not self.root.is_dir():
            return []
        runs = []
        for p in self.root.glob("*/run.json"):
            r = read_json(p)
            if r is None or not _REQUIRED_KEYS.issubset(r) or not _RUN_ID_RE.match(str(r["id"])):
                continue
            runs.append(r)
        return sorted(runs, key=lambda r: (r["created_at"], r["id"]), reverse=True)

    def delete(self, run_id: str) -> None:
        shutil.rmtree(self.dir(run_id))
