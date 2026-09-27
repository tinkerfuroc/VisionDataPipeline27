"""The only module that reads or writes data/projects/. All writes are atomic (temp file + rename)."""

from __future__ import annotations

import io
import os
import re
import shutil
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .models import ClassDef, Clip, LabelObject, Project

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class StoreError(Exception):
    pass


class NotFound(StoreError):
    pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


def encode_mask_png(mask: np.ndarray) -> bytes:
    """1-bit PNG bytes for a boolean mask. An empty mask is an error: absent file = not present."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise StoreError("refusing to write an empty mask; an absent file means 'not present'")
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).convert("1", dither=Image.Dither.NONE).save(buf, format="PNG")
    return buf.getvalue()


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
        self.read_project(project_id)  # raises NotFound before the lock table grows an entry for a bogus id
        name = name.strip()
        if not name:
            raise StoreError("class name must not be empty")
        with self.lock(project_id):
            project = self.read_project(project_id)
            if any(c.name == name for c in project.classes):
                raise StoreError(f"class {name!r} already exists")
            new_id = max((c.id for c in project.classes), default=-1) + 1
            cls = ClassDef(id=new_id, name=name, color=_PALETTE[new_id % len(_PALETTE)])
            project.classes.append(cls)
            self.write_project(project)
            return cls

    def set_class_color(self, project_id: str, class_id: int, color: str) -> ClassDef:
        self.read_project(project_id)  # raises NotFound before the lock table grows an entry for a bogus id
        if not _COLOR_RE.match(color):
            raise StoreError(f"invalid colour: {color!r}; expected #rrggbb")
        color = color.lower()
        with self.lock(project_id):
            project = self.read_project(project_id)
            for cls in project.classes:
                if cls.id == class_id:
                    cls.color = color
                    self.write_project(project)
                    return cls
            raise NotFound(f"class {class_id} not found in project {project_id!r}")

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
        atomic_write_bytes(self.mask_path(clip_id, object_id, idx), encode_mask_png(mask))

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

    def _states(self, clip: Clip, objects: list) -> list[str]:
        """Priority: deleted > keyframe > masked > empty, over `objects`."""
        states = ["empty"] * clip.frame_count
        for obj in objects:
            for i in self.mask_indices(clip.id, obj.id):
                if 0 <= i < clip.frame_count:
                    states[i] = "masked"
        for obj in objects:
            for kf in obj.keyframes:
                if 0 <= kf.frame_idx < clip.frame_count:
                    states[kf.frame_idx] = "keyframe"
        for i in clip.deleted_frames:
            if 0 <= i < clip.frame_count:
                states[i] = "deleted"
        return states

    def frame_states(self, clip: Clip) -> list[str]:
        """Per-frame state of the whole clip (any object), for the Clips page and the timeline summary."""
        return self._states(clip, clip.objects)

    def object_states(self, clip: Clip, obj: LabelObject) -> list[str]:
        """Per-frame state of one object; a deleted frame is `deleted` in every object's lane."""
        return self._states(clip, [obj])

    def delete_object_masks(self, clip_id: str, object_id: int) -> None:
        shutil.rmtree(self.clip_dir(clip_id) / "masks" / str(int(object_id)), ignore_errors=True)
