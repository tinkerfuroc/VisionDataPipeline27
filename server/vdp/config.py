"""Typed settings loaded from configs/default.yaml. Unknown keys are errors."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _resolve_repo_path(value: str | Path) -> Path:
    """Relative paths in config are relative to REPO_ROOT, not the process CWD."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerCfg(_Cfg):
    host: str = "127.0.0.1"
    port: int = 28000
    https: bool = False
    cert: Path | None = None  # with `key`: your own certificate; neither = a self-signed one in <data_root>/tls/
    key: Path | None = None


class ProjectCfg(_Cfg):
    data_root: Path = Path("./data")


class IngestCfg(_Cfg):
    allowed_roots: list[str] = Field(default_factory=lambda: ["~"])
    default_stride: int = 1
    jpeg_quality: int = 95
    thumb_width: int = 160
    max_upload_bytes: int = 8 * 2**30  # refuse a larger body before/while it arrives
    max_zip_entries: int = 100_000
    max_zip_bytes: int = 64 * 2**30  # uncompressed total inside one zip


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
    workers: int | None = Field(default=None, ge=1)  # None: cpu_count - 4 (min 1); 1 = in-process


class AugmentCfg(_Cfg):
    workers: int | None = Field(default=None, ge=1)  # None: cpu_count - 4 (min 1); 1 = in-process


class TrainCfg(_Cfg):
    warm_label_cache: bool = True  # scan a new export's YOLO labels in the background so training starts sooner


class Settings(_Cfg):
    server: ServerCfg = Field(default_factory=ServerCfg)
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    sam3: Sam3Cfg = Field(default_factory=Sam3Cfg)
    propagate: PropagateCfg = Field(default_factory=PropagateCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    augment: AugmentCfg = Field(default_factory=AugmentCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        if path is None:
            settings = cls()
        else:
            data = yaml.safe_load(Path(path).read_text()) or {}
            settings = cls.model_validate(data)
        settings.project.data_root = _resolve_repo_path(settings.project.data_root)
        settings.sam3.model_dir = _resolve_repo_path(settings.sam3.model_dir)
        if settings.server.cert is not None:
            settings.server.cert = _resolve_repo_path(settings.server.cert)
        if settings.server.key is not None:
            settings.server.key = _resolve_repo_path(settings.server.key)
        return settings
