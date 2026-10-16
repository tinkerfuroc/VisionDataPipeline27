"""Training routes (spec: API). Runs live on disk; the scheduler queues and starts them."""

from __future__ import annotations

import re
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Response
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..store import NotFound
from ..train.gallery import FILTERS, filter_items, load_gallery
from ..train.models import (
    AUG_PRESETS,
    DEFAULT_MODELS,
    MODELS,
    TrainParams,
    find_export,
    list_exports,
    rename_export,
    run_classes,
)
from ..train.results import read_metrics, tail
from ..train.runs import best_weights, onnx_weights, read_json
from ..train.scheduler import (
    Gpu,
    default_gpu,
    estimated_tflops,
    gpu_free_memory,
    gpu_max_clock_ghz,
    rank_gpus,
)
from .deps import State
from .engine import sam3_gpu

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


def _view(state, run: dict, *, detail: bool = False) -> dict:
    run_dir = state.train.runs.dir(run["id"])
    rows = read_metrics(run_dir)
    resumable = state.train.resumable(run)
    has_best = best_weights(run_dir).is_file()
    classes = run_classes(run_dir, run, state.settings.project.data_root)
    view = {k: v for k, v in run.items() if k not in _PRIVATE}
    view.update(
        epochs_done=len(rows),
        progress=read_json(run_dir / "progress.json") if run["status"] == "running" else None,
        resumable=resumable,
        completable=state.train.completable(run),
        has_best=has_best,
        has_onnx=onnx_weights(run_dir).is_file(),
        classes=list(classes) if classes is not None else None,
        source_run=run.get("source_run"),
        start_mode=run.get("start_mode"),
        continued_from_epoch=run.get("continued_from_epoch"),
        # "Train from a previous run" eligibility, mirroring TrainScheduler.submit_from's own checks.
        continuable=resumable,
        finetunable=has_best,
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
    pause_sam3: bool = False  # free SAM3's VRAM until training on its GPU is over (see engine.EngineControl)


def _gpu_view(g: Gpu, snapshot: dict, *, tflops: float, memory_free: int | None) -> dict:
    return {
        **snapshot, "capability": f"{g.capability[0]}.{g.capability[1]}", "sm_count": g.sm_count,
        "memory_free": memory_free, "tflops": round(tflops, 1),
    }


@router.get("/api/train/status")
def train_status(state: State) -> dict:
    gpus = state.train.gpus
    clocks = gpu_max_clock_ghz(gpus)
    free = gpu_free_memory(gpus)
    ranked = rank_gpus(gpus, clocks)
    snapshot_by_index = {s["index"]: s for s in state.train.snapshot()}
    gpu_views = [
        _gpu_view(
            g, snapshot_by_index[g.index],
            tflops=estimated_tflops(g.sm_count, g.capability, clocks.get(g.index, 1.5)),
            memory_free=free.get(g.index),
        )
        for g in ranked
    ]
    return {
        "available": ultralytics_error() is None, "error": ultralytics_error(),
        "gpus": gpu_views, "default_gpu": default_gpu(ranked, state.train.busy_gpus()),
        "sam3_gpu": sam3_gpu(state.engine),  # still reported while SAM3 is paused: where it reloads to
        "sam3_vram_bytes": getattr(state.engine, "vram_bytes", None),
        "models": list(MODELS), "default_models": list(DEFAULT_MODELS),
        "defaults": TrainParams().model_dump(),
        "aug_presets": {k: v.model_dump() for k, v in AUG_PRESETS.items()},
    }


@router.get("/api/exports")
def exports(state: State) -> list[dict]:
    return [e.to_json() for e in list_exports(state.settings.project.data_root)]


class RenameBody(BaseModel):
    name: str | None  # required key (may be null): a PATCH with no `name` at all is a 422, not a no-op clear


@router.patch("/api/exports/{export_id}")
def rename(export_id: str, body: RenameBody, state: State) -> dict:
    return rename_export(state.settings.project.data_root, export_id, body.name).to_json()


@router.get("/api/exports/{export_id}/images/{split}/{name}")
def export_image(export_id: str, split: str, name: str, state: State) -> FileResponse:
    if split not in ("train", "val"):
        raise NotFound(f"image {name} not found")
    export = find_export(state.settings.project.data_root, export_id)
    path = export.yolo_dir / "images" / split / name
    if not _IMAGE_NAME_RE.match(name) or not path.is_file():
        raise NotFound(f"image {name} not found")
    return FileResponse(path)


@router.post("/api/train/runs", status_code=201)
def start_runs(body: StartBody, state: State) -> dict:
    error = ultralytics_error()
    if error is not None:
        raise TrainUnavailable(error)
    export = find_export(state.settings.project.data_root, body.export_id)
    specs = [(r.model, r.gpu) for r in body.runs]
    ids = state.engine_ctl.start_training(
        body.pause_sam3, {r.gpu for r in body.runs},
        lambda before_queue: state.train.submit(export, specs, body.params, before_queue=before_queue),
    )
    return {"run_ids": ids}


class FromCheckpointBody(BaseModel):
    export_id: str
    mode: Literal["continue", "finetune"]
    gpu: int
    params: TrainParams | None = None
    pause_sam3: bool = False


@router.post("/api/train/runs/{run_id}/from-checkpoint", status_code=201)
def start_from_checkpoint(run_id: str, body: FromCheckpointBody, state: State) -> dict:
    """A new run that continues (mode "continue") or fine-tunes (mode "finetune") `run_id`'s
    checkpoint on a different export (same classes required); the source run is untouched."""
    error = ultralytics_error()
    if error is not None:
        raise TrainUnavailable(error)
    export = find_export(state.settings.project.data_root, body.export_id)
    new_id = state.engine_ctl.start_training(
        body.pause_sam3, {body.gpu},
        lambda before_queue: state.train.submit_from(
            run_id, export, body.mode, body.gpu, body.params, before_queue=before_queue,
        ),
    )
    return {"run_id": new_id}


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


class ResumeBody(BaseModel):
    gpu: int | None = None
    pause_sam3: bool = False


def _requeue(state, run_id: str, body: ResumeBody | None, action) -> dict:
    """Resume or Mark complete: `action` is the scheduler method; the run stays on its own GPU without `gpu`."""
    body = body or ResumeBody()
    gpu = body.gpu if body.gpu is not None else state.train.runs.read(run_id)["gpu"]
    state.engine_ctl.start_training(
        body.pause_sam3, {gpu}, lambda before_queue: action(run_id, body.gpu, before_queue=before_queue),
    )
    return _view(state, state.train.runs.read(run_id))


@router.post("/api/train/runs/{run_id}/resume")
def resume_run(run_id: str, state: State, body: ResumeBody | None = None) -> dict:
    return _requeue(state, run_id, body, state.train.resume)


class CompleteBody(ResumeBody):
    """The same fields as ResumeBody."""


@router.post("/api/train/runs/{run_id}/complete")
def complete_run(run_id: str, state: State, body: CompleteBody | None = None) -> dict:
    return _requeue(state, run_id, body, state.train.complete)


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
    filter: Literal["all", "missed", "worst"] = Query("all"),
) -> dict:
    run = state.train.runs.read(run_id)
    labels = Path(run["export_dir"]) / "yolo_seg" / "labels" / "val"
    gallery = load_gallery(state.train.runs.dir(run_id), labels)
    lists = {name: filter_items(gallery["items"], name) for name in FILTERS}
    chosen = lists[filter]
    return {"total": len(chosen), "conf": gallery["conf"], "counts": {k: len(v) for k, v in lists.items()},
            "items": chosen[offset:offset + limit]}


@router.get("/api/train/runs/{run_id}/log", response_class=PlainTextResponse)
def run_log(run_id: str, state: State) -> str:
    state.train.runs.read(run_id)  # 404 for an unknown run, instead of 200 with empty text
    return tail(state.train.runs.dir(run_id) / "worker.log", 200)
