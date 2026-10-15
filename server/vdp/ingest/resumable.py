"""Resumable uploads (spec: uploads from phones). A file arrives in chunks over many requests, survives a lost
connection, a locked phone or a server restart, and is handed to the ordinary import job once complete.

State lives in `<root>/<id>.part` (the bytes so far) and `<root>/<id>.json` (the record). `received` only ever grows
by whole accepted chunks, so a chunk lost mid-request is simply sent again; a retried chunk that already arrived is
accepted and ignored.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .source import IngestError
from .upload import UploadTooLarge

CHUNK_SIZE = 8 * 2**20
UPLOAD_TTL = 24 * 3600
_HEX = frozenset("0123456789abcdef")


class UploadNotFound(Exception):
    """No such upload, or it expired."""


class OffsetConflict(Exception):
    """A chunk did not start where the server's copy ends; `received` says where it does."""

    def __init__(self, received: int):
        super().__init__(f"the server has {received} bytes of this upload; continue from there")
        self.received = received


@dataclass
class UploadRecord:
    id: str
    project_id: str
    name: str
    kind: str  # "video" | "zip"
    size: int
    received: int
    stride: int
    fps: float
    created_at: float
    updated_at: float
    job_id: str | None = None

    @property
    def complete(self) -> bool:
        return self.received == self.size

    def public(self) -> dict:
        return {"upload_id": self.id, "project_id": self.project_id, "name": self.name, "size": self.size,
                "received": self.received, "job_id": self.job_id}


class ResumableStore:
    def __init__(self, root: Path, max_bytes: int, *, ttl: float = UPLOAD_TTL,
                 clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.ttl = ttl
        self.clock = clock
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _check_id(self, upload_id: str) -> None:
        """Validate that upload_id is exactly 32 lowercase hex chars; raise UploadNotFound otherwise."""
        if len(upload_id) != 32 or not set(upload_id) <= _HEX:
            raise UploadNotFound(f"no upload {upload_id!r}")

    def _lock(self, upload_id: str) -> threading.Lock:
        self._check_id(upload_id)
        with self._locks_guard:
            return self._locks.setdefault(upload_id, threading.Lock())

    def part_path(self, upload_id: str) -> Path:
        self._check_id(upload_id)
        return self.root / f"{upload_id}.part"

    def _record_path(self, upload_id: str) -> Path:
        self._check_id(upload_id)
        return self.root / f"{upload_id}.json"

    def _save(self, rec: UploadRecord) -> None:
        tmp = self.root / f".{rec.id}.json.tmp"
        tmp.write_text(json.dumps(asdict(rec)))
        os.replace(tmp, self._record_path(rec.id))

    def _remove(self, upload_id: str) -> None:
        self.part_path(upload_id).unlink(missing_ok=True)
        self._record_path(upload_id).unlink(missing_ok=True)

    def _load(self, upload_id: str) -> UploadRecord:
        """The record, repaired to match its part file. Call with the upload's lock held."""
        self._check_id(upload_id)
        try:
            rec = UploadRecord(**json.loads(self._record_path(upload_id).read_text()))
        except FileNotFoundError as e:
            raise UploadNotFound(f"no upload {upload_id}") from e
        except (ValueError, TypeError) as e:
            self._remove(upload_id)
            raise UploadNotFound(f"upload {upload_id} was unreadable and has been removed") from e
        if rec.job_id is None and self.clock() - rec.updated_at > self.ttl:
            self._remove(upload_id)
            raise UploadNotFound(f"upload {upload_id} expired")
        part = self.part_path(upload_id)
        if not part.exists():
            part.touch()
        actual = part.stat().st_size
        if actual > rec.received:  # a crash after a chunk was written, before the record was saved
            with part.open("r+b") as f:
                f.truncate(rec.received)
        elif actual < rec.received:  # the record got ahead of the bytes
            rec.received = actual
            self._save(rec)
        return rec

    def create(self, project_id: str, name: str, kind: str, size: int, stride: int, fps: float) -> UploadRecord:
        if size > self.max_bytes:
            raise UploadTooLarge(f"the upload is {size} bytes; the limit is {self.max_bytes}")
        self.sweep()
        self.root.mkdir(parents=True, exist_ok=True)
        now = self.clock()
        rec = UploadRecord(secrets.token_hex(16), project_id, name, kind, size, 0, stride, fps, now, now)
        self.part_path(rec.id).touch()
        self._save(rec)
        return rec

    def get(self, upload_id: str) -> UploadRecord:
        with self._lock(upload_id):
            return self._load(upload_id)

    def list(self, project_id: str) -> list[UploadRecord]:
        """Unfinished uploads of a project, newest first."""
        found = []
        for path in sorted(self.root.glob("*.json")) if self.root.is_dir() else []:
            try:
                rec = self.get(path.stem)
            except UploadNotFound:
                continue
            if rec.project_id == project_id and not rec.complete:
                found.append(rec)
        return sorted(found, key=lambda r: r.created_at, reverse=True)

    def write_chunk(self, upload_id: str, offset: int, data: bytes) -> UploadRecord:
        if not data:
            raise IngestError("the chunk was empty")
        if len(data) > CHUNK_SIZE:
            raise UploadTooLarge(f"a chunk is at most {CHUNK_SIZE} bytes")
        with self._lock(upload_id):
            rec = self._load(upload_id)
            if offset != rec.received:
                if offset < rec.received and offset + len(data) <= rec.received:
                    return rec  # a retry of a chunk that already arrived
                raise OffsetConflict(rec.received)
            if rec.received + len(data) > rec.size:
                raise UploadTooLarge(f"the chunk goes past the declared size of {rec.size} bytes")
            part = self.part_path(upload_id)
            try:
                with part.open("r+b") as f:
                    f.seek(rec.received)
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                with contextlib.suppress(OSError), part.open("r+b") as f:
                    f.truncate(rec.received)  # never keep part of a chunk
                raise
            rec.received += len(data)
            rec.updated_at = self.clock()
            self._save(rec)
            return rec

    def complete(self, upload_id: str, start: Callable[[UploadRecord], str]) -> UploadRecord:
        """Once every byte arrived, call `start` (which starts the import job and returns its id) exactly once,
        however many requests ask at the same time."""
        with self._lock(upload_id):
            rec = self._load(upload_id)
            if rec.complete and rec.job_id is None:
                rec.job_id = start(rec)
                self._save(rec)
            return rec

    def delete(self, upload_id: str) -> None:
        with self._lock(upload_id):
            self._load(upload_id)  # UploadNotFound for an unknown id
            self._remove(upload_id)

    def discard(self, upload_id: str) -> None:
        """Remove an upload's files if present (the import job's cleanup)."""
        with self._lock(upload_id):
            self._remove(upload_id)

    def sweep(self) -> int:
        """Delete unfinished uploads untouched for longer than the TTL; returns how many went."""
        if not self.root.is_dir():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                self.get(path.stem)
            except UploadNotFound:
                removed += 1
        return removed

    def drop_orphaned_completions(self) -> int:
        """At startup: a record with a job id belongs to an import that died with the previous process."""
        if not self.root.is_dir():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                rec = self.get(path.stem)
            except UploadNotFound:
                continue
            if rec.job_id is not None:
                self.discard(rec.id)
                removed += 1
        return removed
