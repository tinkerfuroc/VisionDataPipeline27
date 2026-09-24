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
