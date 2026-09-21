"""Thread-backed background jobs with an append-only event list (spec: start / poll / cancel / stream)."""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

log = logging.getLogger(__name__)

TERMINAL = frozenset({"done", "error", "cancelled"})
INTERACTIVE = "interactive"  # synthetic job_id reported in the 409 body while an interactive lease is held


class JobBusy(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id} is still running")
        self.job_id = job_id


class JobNotFound(KeyError):
    pass


class Job:
    def __init__(self, kind: str, exclusive: str | None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.exclusive = exclusive
        self.status = "pending"
        self.result: dict | None = None
        self.error: str | None = None
        self.created_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def emit(self, event: dict) -> None:
        with self._lock:
            self._events.append({**event, "seq": len(self._events)})

    def events_since(self, n: int) -> list[dict]:
        with self._lock:
            return list(self._events[n:])

    def request_cancel(self) -> None:
        self._cancel.set()

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            n_events = len(self._events)
        return {
            "id": self.id, "kind": self.kind, "status": self.status, "result": self.result,
            "error": self.error, "created_at": self.created_at, "n_events": n_events,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._leases: dict[str, int] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Job], dict | None], *, exclusive: str | None = None) -> Job:
        with self._lock:
            if exclusive is not None:
                for other in self._jobs.values():
                    if other.exclusive == exclusive and other.status not in TERMINAL:
                        raise JobBusy(other.id)
                if self._leases.get(exclusive, 0) > 0:
                    raise JobBusy(INTERACTIVE)
            job = Job(kind, exclusive)
            self._jobs[job.id] = job
        job._thread = threading.Thread(target=self._run, args=(job, fn), name=f"job-{kind}-{job.id}", daemon=True)
        job._thread.start()
        return job

    @contextlib.contextmanager
    def lease(self, exclusive: str) -> Iterator[None]:
        """An interactive hold on `exclusive` that excludes jobs but not other leases on the same key.

        Used by prompt/accept/clear-range so a `POST /propagate` cannot start (and later overwrite a
        keyframe, or clear ranges out from under it) while one of those calls is mid-flight; two
        overlapping interactive calls queue on the engine's own lock instead of bouncing with 409.
        """
        with self._lock:
            for other in self._jobs.values():
                if other.exclusive == exclusive and other.status not in TERMINAL:
                    raise JobBusy(other.id)
            self._leases[exclusive] = self._leases.get(exclusive, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._leases[exclusive] -= 1

    @staticmethod
    def _run(job: Job, fn: Callable[[Job], dict | None]) -> None:
        job.status = "running"
        try:
            result = fn(job) or {}
        except Exception as e:  # a job must never take the server down
            log.exception("job %s (%s) failed", job.id, job.kind)
            job.error = f"{type(e).__name__}: {e}"
            job.emit({"type": "error", "message": job.error})
            job.status = "error"
            return
        job.result = result
        # A job is "cancelled" only when its work function REPORTS it stopped early -- never merely
        # because a cancel was requested. Work that ignores the request and finishes ends "done".
        final = "cancelled" if result.get("cancelled") else "done"
        job.emit({"type": final, **result})  # event first, status second: streamers rely on this order
        job.status = final

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(f"job {job_id} not found")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.request_cancel()
        return job

    def active(self, kind: str) -> str | None:
        """The id of an unfinished job of this kind, if any."""
        with self._lock:
            for job in self._jobs.values():
                if job.kind == kind and job.status not in TERMINAL:
                    return job.id
        return None

    def running(self, exclusive: str) -> str | None:
        with self._lock:
            for job in self._jobs.values():
                if job.exclusive == exclusive and job.status not in TERMINAL:
                    return job.id
        return None
