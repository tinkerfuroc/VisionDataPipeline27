# Vision Sampler Plan 5 — Web uploads (videos and image-folder ZIPs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator whose browser reaches the GUI through an SSH port forward upload a video file or a `.zip` of images from the Clips page, watch the transfer progress, cancel it mid-flight, and have the upload turn into a normal import job.

**Architecture:** One streaming upload route replaces the multipart `import-video`. The browser sends the file as the **raw request body** (`POST /api/projects/{p}/clips/upload?name=&stride=&fps=`) through `XMLHttpRequest` — the only browser API that reports request progress and can abort. The server streams `request.stream()` straight into `data/uploads/<uuid>_<name>` under a hard byte cap, then hands that temp file to a `FrameSource` (`VideoFileSource`, or the new `ZipFrameSource`, which reads image entries out of the archive without extracting it) inside the existing `import` job, which deletes the temp file when it ends. The Clips page gets one "Upload footage" section: file picker plus drop zone, kind-aware options, progress, cancel.

**Why not multipart (the current `import-video`):** FastAPI parses a whole multipart body (spooling anything over 1 MiB to a temp file, usually `/tmp`) **before** the route function runs. So the handler cannot report progress, cannot refuse an over-size file before it is transferred, and the bytes are written twice (spool, then copy). Over an SSH tunnel a 2 GB video makes all three hurt, and a small `/tmp` (often tmpfs) fills up first.

**Tech stack:** no new Python dependency (`zipfile`, `cv2`, `numpy` are already there) and no new npm package (`XMLHttpRequest` is built in). No GPU involvement anywhere in this plan.

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md` §4 (API table), §6 (Ingest), §9 (errors). GUI rules: `docs/superpowers/specs/2026-09-21-gui-design-pass.md` and the "GUI design pass" section of the roadmap.

## Global Constraints

- Python is managed with uv only: `uv run pytest`, never pip, never conda. Run the suite as `HF_HUB_OFFLINE=1 uv run pytest -q` from the repository root; lint with `uv run ruff check .` and keep it clean (line length and rules come from `pyproject.toml`).
- Web commands run as `bash -c 'source ~/.nvm/nvm.sh && cd web && <command>'`. Web checks: `npx tsc --noEmit`, `npx vitest run`, `npm run build`. Do not add npm packages.
- Never stage or commit anything under `vision_train/`, `data/`, `web/node_modules/`, `web/dist/`, or any `*.pt` / `*.safetensors`. Stage by explicit path only (`git add <path> ...`), never `git add -A` / `git add .`.
- Do not push, do not create a PR, do not touch the global git config.
- **Every task must leave the tree working.** Task 3 adds the new route but keeps `import-video`; Task 5 removes it in the same commit that switches the GUI over. Never leave a commit where the page calls an endpoint that no longer exists.
- The upload endpoint is **unauthenticated, like the rest of the API**. The SSH tunnel is the trust boundary; `vsp serve` keeps binding `127.0.0.1`. Do not add auth, and do not suggest binding `0.0.0.0`.
- Errors: `IngestError` → 400, `UploadTooLarge` → 413, `NotFound` → 404; the app's `_STATUS` table maps them (Starlette picks the most specific class in the MRO, but keep the list ordered most-specific-first like its comment says).
- Uploads never touch the engine and take no GPU lease: two imports may run at once, exactly as today (a job started without `exclusive` does not collide).
- Config values live in `Settings` with a default and in `configs/default.yaml`; `_Cfg` is `extra="forbid"`, so an unknown YAML key is an error.
- GUI rules: every page keeps its controls in a RIGHT-hand sidebar; chrome has no hue — use the `btn`, `btn-on`, `btn-danger`, `field`, `heading`, `note`, `alert` classes and the `surround/well/panel/line/ink/dim/key/danger` tokens; sentence-case copy, no all-caps labels, no middle-dot meta strings, no "→"; a button names exactly what it will do.
- Web test conventions: component tests start with `// @vitest-environment jsdom`, call `cleanup()` in `afterEach`, mock the API with `vi.mock("../api/rest", async (orig) => ({ ...(await orig<typeof import("../api/rest")>()), api: { ... } }))`, and use `vi.resetAllMocks()` (not per-mock `.mockReset()`) in `beforeEach`.
- Commit after every task with a conventional message. Task 6 is a controller task, not a subagent.

## Access model: the browser talks through an SSH port forward

Nothing in this plan is tunnel-specific — that is the point of doing it this way — but every choice above is made for this deployment:

```
laptop: Firefox  ──HTTP──▶  ssh -N -L 28000:127.0.0.1:28000 user@gpu-host  ──▶  vsp serve on the GPU box
        (localhost:28000)                          one TCP stream
```

- The server binds `127.0.0.1` (the default). The forward is the only way in; the CLI already warns when someone binds wider.
- Bandwidth is whatever the SSH link gives (a WAN link makes 1-2 GB take minutes), so the GUI must show progress and offer Cancel; a reload or a dropped link loses that transfer (Phase 2 below is about fixing exactly that).
- The upload is written once, by the server, under `data_root/uploads/`. Peak disk during an import is roughly `upload + the JPEGs/thumbs it produces`; the upload is deleted as soon as the job ends.
- A browser reload during the transfer aborts it; the server sees the disconnect and removes the partial file.
- With compression on, `ssh` wastes CPU on already-compressed video; `-o Compression=no` (or `-C` off) is usually faster here. `-o ServerAliveInterval=30` keeps a NAT'd tunnel from being reaped.

---

### Task 1: `ingest/upload.py` — stream a request body to disk, safely

**Files:**
- Create: `server/vsp/ingest/upload.py`
- Modify: `server/vsp/ingest/__init__.py`
- Test: `server/tests/test_upload_stream.py`

**Interfaces:**
- Consumes: `vsp.ingest.source.IngestError`.
- Produces (Tasks 3 and 6 rely on these exact names):
  - `save_stream(chunks: AsyncIterator[bytes], dest: Path, limit: int) -> int` (async; returns the byte count; removes `dest` on **any** failure)
  - `UploadTooLarge(IngestError)`
  - `clear_uploads(data_root: Path) -> int` (deletes every file in `<data_root>/uploads`, returns how many)

- [ ] **Step 1: Write the failing tests** — `server/tests/test_upload_stream.py`:

```python
from __future__ import annotations

import asyncio

import pytest
from starlette.requests import ClientDisconnect
from vsp.ingest import IngestError, UploadTooLarge, clear_uploads, save_stream


def _chunks(*parts: bytes):
    async def gen():
        for part in parts:
            yield part

    return gen()


def test_writes_every_chunk_and_returns_the_byte_count(tmp_path):
    dest = tmp_path / "up"
    assert asyncio.run(save_stream(_chunks(b"abc", b"", b"defg"), dest, 100)) == 7
    assert dest.read_bytes() == b"abcdefg"


def test_a_body_over_the_limit_removes_the_partial_file(tmp_path):
    dest = tmp_path / "up"
    with pytest.raises(UploadTooLarge, match="larger than 5 bytes"):
        asyncio.run(save_stream(_chunks(b"abc", b"def"), dest, 5))
    assert not dest.exists()


def test_an_empty_body_is_refused_and_leaves_nothing(tmp_path):
    dest = tmp_path / "up"
    with pytest.raises(IngestError, match="empty"):
        asyncio.run(save_stream(_chunks(b"", b""), dest, 100))
    assert not dest.exists()


def test_a_client_disconnect_propagates_and_leaves_nothing(tmp_path):
    dest = tmp_path / "up"

    async def gen():
        yield b"abc"
        raise ClientDisconnect()

    with pytest.raises(ClientDisconnect):
        asyncio.run(save_stream(gen(), dest, 100))
    assert not dest.exists()


def test_clear_uploads_removes_only_files(tmp_path):
    uploads = tmp_path / "uploads"
    (uploads / "nested").mkdir(parents=True)
    (uploads / "a.mp4").write_bytes(b"x")
    (uploads / "nested" / "b.zip").write_bytes(b"x")
    assert clear_uploads(tmp_path) == 1
    assert [p.name for p in uploads.iterdir()] == ["nested"]
    assert clear_uploads(tmp_path / "absent") == 0
```

- [ ] **Step 2: Run them** — `HF_HUB_OFFLINE=1 uv run pytest server/tests/test_upload_stream.py -q`. Expected: FAIL with `ImportError: cannot import name 'save_stream'`.

- [ ] **Step 3: Implement** — `server/vsp/ingest/upload.py`:

```python
"""Streaming an HTTP request body to disk: one write, a hard byte cap, no partial-file litter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from .source import IngestError


class UploadTooLarge(IngestError):
    """The body exceeded `ingest.max_upload_bytes`; the partial file is already removed."""


async def save_stream(chunks: AsyncIterator[bytes], dest: Path, limit: int) -> int:
    """Write `chunks` to `dest` and return the byte count.

    Any failure -- the cap, a client disconnect, a disk error -- removes the partial file, so `dest` exists
    only when the upload completed. An empty body is an error too: the caller would otherwise hand an empty
    file to a decoder and report a confusing "cannot open" instead of "the upload was empty".
    """
    written = 0
    try:
        with dest.open("wb") as out:
            async for chunk in chunks:
                written += len(chunk)
                if written > limit:
                    raise UploadTooLarge(f"the upload is larger than {limit} bytes")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if written == 0:
        dest.unlink(missing_ok=True)
        raise IngestError("the upload was empty")
    return written


def clear_uploads(data_root: Path) -> int:
    """Delete every file in `<data_root>/uploads`. Called once when the app is built: that directory holds
    only in-flight uploads, and a fresh process has none, so anything in it is litter from a killed server."""
    uploads = Path(data_root) / "uploads"
    if not uploads.is_dir():
        return 0
    removed = 0
    for path in uploads.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)
            removed += 1
    return removed
```

In `server/vsp/ingest/__init__.py` add `from .upload import UploadTooLarge, clear_uploads, save_stream` (keep the imports alphabetically sorted: `frame_dir`, `importer`, `source`, `upload`, `video_file` — ruff checks import order) and add `"UploadTooLarge"`, `"clear_uploads"`, `"save_stream"` to `__all__`.

- [ ] **Step 4: Run the tests, the whole suite and ruff** — 5 new tests pass; `uv run ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add server/vsp/ingest/upload.py server/vsp/ingest/__init__.py server/tests/test_upload_stream.py
git commit -m "feat(ingest): stream an upload body to disk under a byte cap, and sweep stale uploads"
```

---

### Task 2: `ingest/zip_file.py` — a `.zip` of images as a frame source

**Files:**
- Create: `server/vsp/ingest/zip_file.py`
- Modify: `server/vsp/ingest/__init__.py`
- Test: `server/tests/test_zip_source.py`

**Interfaces:**
- Consumes: `vsp.ingest.frame_dir.IMAGE_SUFFIXES`, `vsp.ingest.frame_dir.natural_key`, `vsp.ingest.source.{IngestError, SourceFrame}`, `vsp.ingest.importer.import_clip`, `vsp.store.ProjectStore`.
- Produces: `ZipFrameSource(path: Path, fps: float = 30.0, *, max_entries: int = 100_000, max_bytes: int = 64 * 2**30)` with `type = "frame_dir"`, `name = path.stem`, `fps`, and `frames()`.

**Rulings recorded by the plan author:**
- **Nothing is extracted.** Entries are read straight out of the archive and decoded in memory, so a 5 GB zip does not become 5 GB of scratch files. The cost is that JPEG entries are re-encoded rather than copied (`SourceFrame.src_path` is `None`); that is what a PNG frame directory does today too.
- **`type = "frame_dir"`.** `ClipSource.type` is `Literal["video", "frame_dir"]` and a zip *is* a directory of frames, so the stored clip, the exporters and the Label page need no change.
- **`name = the zip's stem`** (`run.zip` → `run`), matching how a frame directory is named. The full relative entry name is kept per frame in `original_names`.
- **Ordering is `natural_key` over the whole relative path**, so folders group and `frame_2` precedes `frame_10` inside each. A flat zip (the common case) behaves exactly like `FrameDirSource`.

- [ ] **Step 1: Write the failing tests** — `server/tests/test_zip_source.py`:

```python
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from vsp.ingest import IngestError, ZipFrameSource, import_clip


def _png(value: int = 90) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((60, 120, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_frames_come_out_in_natural_order_across_folders(tmp_path):
    path = _zip(tmp_path / "run.zip", {
        "b/frame_10.png": _png(2), "b/frame_2.png": _png(3), "a/frame_1.png": _png(1),
    })
    source = ZipFrameSource(path, fps=12)
    assert (source.type, source.name, source.fps) == ("frame_dir", "run", 12.0)
    frames = list(source.frames())
    assert [f.original_name for f in frames] == ["a/frame_1.png", "b/frame_2.png", "b/frame_10.png"]
    assert frames[0].image.shape == (60, 120, 3)
    assert frames[0].image[0, 0, 0] == 1 and frames[0].src_path is None


def test_archive_junk_is_skipped(tmp_path):
    path = _zip(tmp_path / "run.zip", {
        "run/frame_1.png": _png(), "run/notes.txt": b"x", "run/": b"",
        "__MACOSX/run/._frame_1.png": b"x", "run/.hidden.png": _png(),
    })
    assert [f.original_name for f in ZipFrameSource(path).frames()] == ["run/frame_1.png"]


def test_names_that_escape_the_archive_are_ignored(tmp_path):
    path = _zip(tmp_path / "evil.zip", {"../evil.png": _png(), "/abs.png": _png(), "ok.png": _png()})
    assert [f.original_name for f in ZipFrameSource(path).frames()] == ["ok.png"]


def test_a_zip_with_no_images_is_refused(tmp_path):
    with pytest.raises(IngestError, match="no .jpg"):
        ZipFrameSource(_zip(tmp_path / "docs.zip", {"notes.txt": b"x"}))
    with pytest.raises(IngestError, match="no .jpg"):
        ZipFrameSource(_zip(tmp_path / "empty.zip", {}))


def test_a_file_that_is_not_a_zip_is_refused(tmp_path):
    path = tmp_path / "x.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(IngestError, match="cannot read"):
        ZipFrameSource(path)


def test_an_unreadable_image_names_the_entry(tmp_path):
    path = _zip(tmp_path / "run.zip", {"run/frame_1.png": b"not an image"})
    with pytest.raises(IngestError, match="unreadable image: run/frame_1.png"):
        list(ZipFrameSource(path).frames())


def test_entry_and_size_limits_are_enforced(tmp_path):
    path = _zip(tmp_path / "run.zip", {f"f{i}.png": _png() for i in range(3)})
    with pytest.raises(IngestError, match="limit is 2"):
        ZipFrameSource(path, max_entries=2)
    with pytest.raises(IngestError, match="limit is"):
        ZipFrameSource(path, max_bytes=10)


def test_a_zip_imports_into_a_clip_with_the_entry_names_recorded(store, tmp_path):
    project = store.create_project("t")
    path = _zip(tmp_path / "run.zip", {f"run/frame_{i}.png": _png() for i in (1, 2, 3, 4)})
    clip = import_clip(store, project.id, ZipFrameSource(path, fps=12), stride=2)
    assert (clip.frame_count, clip.fps, clip.stride) == (2, 12.0, 2)
    assert clip.source.type == "frame_dir" and clip.source.name == "run"
    assert clip.original_names == {0: "run/frame_1.png", 1: "run/frame_3.png"}
```

- [ ] **Step 2: Run them** — `HF_HUB_OFFLINE=1 uv run pytest server/tests/test_zip_source.py -q`. Expected: FAIL with `ImportError: cannot import name 'ZipFrameSource'`.

- [ ] **Step 3: Implement** — `server/vsp/ingest/zip_file.py`:

```python
"""A .zip of images as a frame source. Entries are read straight out of the archive; nothing is extracted."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .frame_dir import IMAGE_SUFFIXES, natural_key
from .source import IngestError, SourceFrame

MAX_ENTRIES = 100_000
MAX_BYTES = 64 * 2**30


def _select(infos: list[zipfile.ZipInfo], max_entries: int, max_bytes: int) -> list[zipfile.ZipInfo]:
    """Image entries only, in natural order; refuses an archive that holds more than the configured limits.

    Names are checked even though nothing is ever extracted: an archive carrying `../` or an absolute path is
    malformed, and refusing its entries outright keeps that true if extraction is ever added.
    """
    keep = []
    for info in infos:
        name = info.filename
        parts = name.split("/")
        if info.is_dir() or any(p in ("", ".", "..") for p in parts):
            continue
        if parts[0].lower() == "__macosx" or any(p.startswith(".") for p in parts):
            continue  # resource forks, __MACOSX, and dotfiles such as .DS_Store
        if not name.lower().endswith(tuple(IMAGE_SUFFIXES)):
            continue
        keep.append(info)
    if len(keep) > max_entries:
        raise IngestError(f"the zip holds {len(keep)} images; the limit is {max_entries}")
    total = sum(info.file_size for info in keep)
    if total > max_bytes:
        raise IngestError(f"the zip holds {total / 2**30:.1f} GiB of images; the limit is {max_bytes / 2**30:.1f} GiB")
    keep.sort(key=lambda info: natural_key(info.filename))
    return keep


class ZipFrameSource:
    type = "frame_dir"  # a zip is a directory of frames: same clip layout and semantics

    def __init__(
        self, path: Path, fps: float = 30.0, *, max_entries: int = MAX_ENTRIES, max_bytes: int = MAX_BYTES
    ) -> None:
        self.path = Path(path)
        self.name = self.path.stem or self.path.name
        self.fps = float(fps)
        try:
            with zipfile.ZipFile(self.path) as archive:
                self.infos = _select(archive.infolist(), max_entries, max_bytes)
        except zipfile.BadZipFile as e:
            raise IngestError(f"cannot read {self.path.name} as a zip: {e}") from e
        if not self.infos:
            raise IngestError(f"no .jpg/.jpeg/.png files in {self.path.name}")

    def frames(self) -> Iterator[SourceFrame]:
        with zipfile.ZipFile(self.path) as archive:
            for info in self.infos:
                image = cv2.imdecode(np.frombuffer(archive.read(info), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise IngestError(f"unreadable image: {info.filename}")
                yield SourceFrame(image=image, original_name=info.filename, src_path=None)
```

Export it from `server/vsp/ingest/__init__.py`: `from .zip_file import ZipFrameSource` and add `"ZipFrameSource"` to `__all__`.

- [ ] **Step 4: Run the tests, the whole suite and ruff.** 8 new tests pass, everything else stays green, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add server/vsp/ingest/zip_file.py server/vsp/ingest/__init__.py server/tests/test_zip_source.py
git commit -m "feat(ingest): a zip of images as a frame source, streamed without extraction"
```

---

### Task 3: the unified upload route, the size cap, and the startup sweep

**Files:**
- Modify: `server/vsp/api/clips.py`, `server/vsp/api/app.py`, `server/vsp/config.py`, `configs/default.yaml`
- Test: `server/tests/test_api_upload.py`, `server/tests/test_api_core.py` (health assertion)

**Interfaces:**
- Consumes: Task 1 (`save_stream`, `UploadTooLarge`, `clear_uploads`) and Task 2 (`ZipFrameSource`).
- Produces:
  - `POST /api/projects/{project_id}/clips/upload?name=<file>&stride=<int>=1&fps=<float>=30` with the file as the **raw request body** → `202 {"job_id"}`; 400 for an unsupported extension or interrupted upload; 404 for an unknown project; 413 when the body exceeds `ingest.max_upload_bytes`.
  - `GET /api/health` gains `"upload": {"max_bytes": <int>}` (additive; the web type makes it optional).
  - `IngestCfg` gains `max_upload_bytes` (8 GiB), `max_zip_entries` (100 000), `max_zip_bytes` (64 GiB).
  - `_start_import(state, project_id, build_source, stride, cleanup=None)` — `build_source` is now a zero-argument callable so a decoder failure is reported as a **job** error, not a 400.

- [ ] **Step 1: Write the failing API tests** — `server/tests/test_api_upload.py`. The helpers are imported from `test_api_core`; do **not** import its `client` fixture (that trips ruff F811) — copy the fixture, as Plan 4's export tests do.

```python
from __future__ import annotations

import io
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api_core import _project, wait_job  # noqa: F401
from vsp.api.app import create_app
from vsp.config import Settings
from vsp.engine.fake import FakeEngine


def _settings(tmp_path, **ingest):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    s.ingest.allowed_roots = [str(tmp_path)]
    for key, value in ingest.items():
        setattr(s.ingest, key, value)
    return s


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        yield c


def _png(value: int = 90) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((60, 120, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _video_bytes(path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (120, 60))
    for _ in range(10):
        writer.write(np.full((60, 120, 3), 90, np.uint8))
    writer.release()
    return path.read_bytes()


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_a_video_upload_becomes_a_clip(client, tmp_path):
    pid = _project(client)
    body = _video_bytes(tmp_path / "v.avi")
    r = client.post(f"/api/projects/{pid}/clips/upload?name=bench.avi&stride=2", content=body)
    assert r.status_code == 202, r.text
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done", snap
    detail = client.get(f"/api/clips/{snap['result']['clip_id']}").json()
    assert detail["clip"]["frame_count"] == 5
    assert detail["clip"]["source"] == {"type": "video", "name": "bench.avi"}
    assert list((tmp_path / "data" / "uploads").iterdir()) == []  # removed once the job ended


def test_a_zip_upload_becomes_a_frame_directory_clip(client, tmp_path):
    pid = _project(client)
    body = _zip_bytes({
        "run/frame_2.png": _png(2), "run/frame_10.png": _png(3), "run/frame_1.png": _png(1), "run/notes.txt": b"x",
    })
    r = client.post(f"/api/projects/{pid}/clips/upload?name=run.zip&fps=12&stride=2", content=body)
    assert r.status_code == 202, r.text
    snap = wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done", snap
    detail = client.get(f"/api/clips/{snap['result']['clip_id']}").json()
    assert detail["clip"]["source"] == {"type": "frame_dir", "name": "run"}
    assert detail["clip"]["fps"] == 12 and detail["clip"]["frame_count"] == 2
    assert detail["clip"]["original_names"] == {"0": "run/frame_1.png", "1": "run/frame_10.png"}


def test_an_unsupported_name_is_refused_before_anything_is_written(client, tmp_path):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/upload?name=notes.txt", content=b"hello")
    assert r.status_code == 400 and "not a video" in r.json()["detail"]
    assert not (tmp_path / "data" / "uploads").exists()
    assert client.post("/api/projects/nope-000000/clips/upload?name=x.mp4", content=b"x").status_code == 404


def test_a_declared_body_over_the_limit_is_refused_with_413(tmp_path):
    app = create_app(_settings(tmp_path, max_upload_bytes=10), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        pid = _project(c)
        r = c.post(f"/api/projects/{pid}/clips/upload?name=x.mp4", content=b"x" * 100)  # Content-Length: 100
        assert r.status_code == 413 and "limit is 10" in r.json()["detail"]
        assert not (tmp_path / "data" / "uploads").exists()


def test_a_chunked_body_over_the_limit_is_refused_while_it_streams(tmp_path):
    """No Content-Length: the cap has to fire from `save_stream`. TestClient sends a generator body chunked."""
    app = create_app(_settings(tmp_path, max_upload_bytes=10), engine=FakeEngine(), load_engine=False)
    with TestClient(app) as c:
        pid = _project(c)

        def body():
            yield b"x" * 8
            yield b"x" * 8

        r = c.post(f"/api/projects/{pid}/clips/upload?name=x.mp4", content=body())
        assert r.status_code == 413
        assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_an_unreadable_upload_becomes_a_job_error_and_cleans_up(client, tmp_path):
    pid = _project(client)
    bad_video = client.post(f"/api/projects/{pid}/clips/upload?name=x.mp4", content=b"not a video")
    snap = wait_job(client, bad_video.json()["job_id"])
    assert snap["status"] == "error" and "cannot open" in snap["error"]
    bad_zip = client.post(f"/api/projects/{pid}/clips/upload?name=x.zip", content=b"not a zip")
    snap = wait_job(client, bad_zip.json()["job_id"])
    assert snap["status"] == "error" and "cannot read" in snap["error"]
    assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_a_cancelled_upload_import_removes_the_temp_file(client, tmp_path):
    pid = _project(client)
    body = _zip_bytes({f"f{i}.png": _png() for i in range(40)})
    started = client.post(f"/api/projects/{pid}/clips/upload?name=run.zip", content=body).json()["job_id"]
    client.delete(f"/api/jobs/{started}")  # may land before or after the job starts; both are fine
    snap = wait_job(client, started)
    assert snap["status"] in ("done", "cancelled")
    assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_stale_uploads_are_swept_when_the_app_is_built(tmp_path):
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    litter = uploads / "old_upload.mp4"
    litter.write_bytes(b"x")
    create_app(_settings(tmp_path), engine=FakeEngine(), load_engine=False)
    assert not litter.exists()  # a killed server's partial upload never survives the next start
```

In `server/tests/test_api_core.py`, extend `test_health_with_a_loaded_engine`:

```python
    assert body["upload"] == {"max_bytes": 8 * 2**30}
```

- [ ] **Step 2: Run them** — expected: FAIL (404 on the new path; the health key is missing). The two old multipart tests still pass, because Task 3 keeps that route.

- [ ] **Step 3: Implement the config.** `server/vsp/config.py`, in `IngestCfg`:

```python
class IngestCfg(_Cfg):
    allowed_roots: list[str] = Field(default_factory=lambda: ["~"])
    default_stride: int = 1
    jpeg_quality: int = 95
    thumb_width: int = 160
    max_upload_bytes: int = 8 * 2**30  # refuse a larger body before/while it arrives
    max_zip_entries: int = 100_000
    max_zip_bytes: int = 64 * 2**30  # uncompressed total inside one zip
```

`configs/default.yaml` — replace the one-line `ingest` entry with a block:

```yaml
ingest:
  allowed_roots: ["~"]
  default_stride: 1
  jpeg_quality: 95
  thumb_width: 160
  max_upload_bytes: 8589934592   # 8 GiB; a larger upload is refused (413)
  max_zip_entries: 100000
  max_zip_bytes: 68719476736     # 64 GiB uncompressed total inside one zip
```

- [ ] **Step 4: Implement the route.** In `server/vsp/api/clips.py`:

Add to the imports (ruff sorts them): `from collections.abc import Callable`; `Query`, `Request` from fastapi; `from starlette.requests import ClientDisconnect`; and extend the `..ingest` import with `FrameSource, UploadTooLarge, ZipFrameSource, save_stream`.

Add near the top:

```python
VIDEO_SUFFIXES = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".mts", ".m2ts",
})


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:80] or "upload"


def _upload_kind(name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix == ".zip":
        return "zip"
    return "video" if suffix in VIDEO_SUFFIXES else None
```

Change `_start_import` to take a source factory:

```python
def _start_import(
    state: AppState, project_id: str, build_source: Callable[[], FrameSource], stride: int,
    cleanup: Path | None = None,
) -> dict:
    cfg = state.settings.ingest

    def work(job: Job) -> dict:
        try:
            source = build_source()  # a decoder/zip failure here is the job's error, reported like any other
            clip = import_clip(
                state.store, project_id, source, stride=stride, jpeg_quality=cfg.jpeg_quality,
                thumb_width=cfg.thumb_width, progress=lambda n: job.emit({"type": "progress", "done": n}),
                cancel=job.cancel_requested,
            )
            return {"clip_id": clip.id, "frame_count": clip.frame_count}
        except ImportCancelled:
            return {"cancelled": True}
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

    return {"job_id": state.jobs.start("import", work).id}
```

`import_frames` keeps its eager construction (`resolve_allowed` and a missing directory must stay a 400) and passes a factory for the already-built source:

```python
@router.post("/api/projects/{project_id}/clips/import-frames", status_code=202)
def import_frames(project_id: str, body: ImportFramesBody, state: State) -> dict:
    state.store.read_project(project_id)
    path = resolve_allowed(body.path, state.settings.ingest.allowed_roots)
    source = FrameDirSource(path, fps=body.fps)  # eager: a bad directory is a 400, not a job error
    return _start_import(state, project_id, lambda: source, body.stride)
```

`import_video` (the route Task 5 removes) only needs its call updated to `_start_import(state, project_id, lambda: source, stride, cleanup=tmp)`, and may use `_safe_name(file.filename or "upload")`.

Add the new route:

```python
@router.post("/api/projects/{project_id}/clips/upload", status_code=202)
async def upload_clip(
    project_id: str,
    request: Request,
    state: State,
    name: Annotated[str, Query(min_length=1, max_length=1000)],
    stride: Annotated[int, Query(ge=1)] = 1,
    fps: Annotated[float, Query(gt=0)] = 30.0,
) -> dict:
    """Start an import from a file uploaded as the raw request body (not multipart: see the plan's header).

    `fps` is used by a zip only; a video's frame rate comes from its own container.
    """
    state.store.read_project(project_id)
    clean = _safe_name(name)
    kind = _upload_kind(clean)
    if kind is None:
        raise IngestError(f"{clean} is not a video or a .zip of images")
    limit = state.settings.ingest.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        # Refuse before reading a byte. A browser may see this as a broken connection rather than the 413,
        # because it is still sending; the GUI checks the same limit from /api/health before it starts.
        raise UploadTooLarge(f"the upload is {int(declared)} bytes; the limit is {limit}")
    uploads = state.settings.project.data_root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    tmp = uploads / f"{uuid.uuid4().hex}_{clean}"
    try:
        await save_stream(request.stream(), tmp, limit)
    except ClientDisconnect as e:  # the tab closed, or the SSH tunnel dropped
        raise IngestError("the upload was interrupted") from e
    cfg = state.settings.ingest

    def build_source() -> FrameSource:
        if kind == "video":
            return VideoFileSource(tmp, name=clean)
        return ZipFrameSource(tmp, fps=fps, max_entries=cfg.max_zip_entries, max_bytes=cfg.max_zip_bytes)

    return _start_import(state, project_id, build_source, stride, cleanup=tmp)
```

- [ ] **Step 5: Implement the app wiring.** In `server/vsp/api/app.py`:
  - import `UploadTooLarge` from `..ingest` and add `clear_uploads` to the same import;
  - put `(UploadTooLarge, 413)` **before** `(IngestError, 400)` in `_STATUS`;
  - `health` returns one more key:

```python
        return {
            "status": "ok",
            "version": __version__,
            "engine": {
                "loaded": bool(eng.is_loaded()),
                "device": getattr(eng, "device", None),
                "error": getattr(eng, "load_error", None),
            },
            "upload": {"max_bytes": st.settings.ingest.max_upload_bytes},
        }
```

  - as the last thing in `create_app`, before `return app`, sweep the litter from a previous process:

```python
    removed = clear_uploads(settings.project.data_root)
    if removed:
        log.info("removed %d stale upload(s) from a previous run", removed)
```

- [ ] **Step 6: Run the tests, the whole suite and ruff.** All green: the 8 new upload tests, the extended health test, and the untouched multipart tests. Check `uv run ruff check .` is clean (unused imports are the likely slip: `shutil` is still used by `import_video` until Task 5).

- [ ] **Step 7: Commit**

```bash
git add server/vsp/api/clips.py server/vsp/api/app.py server/vsp/config.py configs/default.yaml server/tests/test_api_upload.py server/tests/test_api_core.py
git commit -m "feat(api): a streaming upload route for videos and image zips, with a size cap and a startup sweep"
```

---

### Task 4: the web upload client (XHR, progress, cancel)

**Files:**
- Create: `web/src/lib/upload.ts`, `web/src/lib/upload.test.ts`
- Modify: `web/src/api/rest.ts`, `web/src/api/types.ts`, `web/src/api/rest.test.ts`

**Interfaces:**
- Consumes: `ApiError`, `detailText` from `rest.ts`.
- Produces:
  - `api.uploadClip(projectId, file, options: {stride: number; fps: number}, onProgress?) -> UploadHandle` where `UploadHandle = { promise: Promise<{job_id: string}>; abort: () => void }`
  - `UploadCancelled` (an `Error` subclass) — rejected when `abort()` is called
  - `importKind(name: string): "video" | "zip" | null`, `uploadProblem(draft): string | null`, `formatBytes(n: number): string`, `uploadPercent(sent: number, total: number): number | null`
  - `Health` gains an optional `upload?: { max_bytes: number }` (optional so existing fixtures and an older server keep type-checking).

- [ ] **Step 1: Write the failing helper tests** — `web/src/lib/upload.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { formatBytes, importKind, uploadPercent, uploadProblem } from "./upload";

describe("importKind", () => {
  it("recognises video files and zips by extension, ignoring case", () => {
    expect(importKind("bench.mp4")).toBe("video");
    expect(importKind("RUN.MOV")).toBe("video");
    expect(importKind("frames.zip")).toBe("zip");
    expect(importKind("notes.txt")).toBeNull();
    expect(importKind("noextension")).toBeNull();
  });
});

describe("uploadProblem", () => {
  const ok = { name: "bench.mp4", size: 10, stride: 1, fps: 30, maxBytes: 1000 };
  it("says what stops the upload, or null", () => {
    expect(uploadProblem(ok)).toBeNull();
    expect(uploadProblem({ ...ok, name: null })).toBe("Choose a video or a .zip of images.");
    expect(uploadProblem({ ...ok, name: "notes.txt" })).toBe("That file is not a video or a .zip of images.");
    expect(uploadProblem({ ...ok, stride: 0 })).toBe("Keep every Nth frame needs a whole number of 1 or more.");
    expect(uploadProblem({ ...ok, name: "run.zip", fps: 0 })).toBe("Frames per second must be above 0.");
    expect(uploadProblem({ ...ok, size: 2000 })).toBe("That file is 2.0 kB; the limit is 1000 B.");
    expect(uploadProblem({ ...ok, size: 2000, maxBytes: null })).toBeNull(); // no limit known: let the server decide
  });
});

describe("formatBytes and uploadPercent", () => {
  it("scales units and reports whole percents", () => {
    expect(formatBytes(3)).toBe("3 B");
    expect(formatBytes(1536)).toBe("1.5 kB");
    expect(formatBytes(8 * 2 ** 30)).toBe("8.0 GB");
    expect(uploadPercent(50, 200)).toBe(25);
    expect(uploadPercent(0, 0)).toBeNull();
  });
});
```

- [ ] **Step 2: Implement** — `web/src/lib/upload.ts`:

```ts
export type ImportKind = "video" | "zip";

/** The same extension rule the server applies to the `name` query parameter. */
const VIDEO_SUFFIXES = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".mts", ".m2ts"];

export function importKind(name: string): ImportKind | null {
  const lower = name.toLowerCase();
  if (lower.endsWith(".zip")) return "zip";
  return VIDEO_SUFFIXES.some((suffix) => lower.endsWith(suffix)) ? "video" : null;
}

export type UploadDraft = { name: string | null; size: number; stride: number; fps: number; maxBytes: number | null };

export function uploadProblem(draft: UploadDraft): string | null {
  if (draft.name === null) return "Choose a video or a .zip of images.";
  const kind = importKind(draft.name);
  if (kind === null) return "That file is not a video or a .zip of images.";
  if (!Number.isInteger(draft.stride) || draft.stride < 1) return "Keep every Nth frame needs a whole number of 1 or more.";
  if (kind === "zip" && !(draft.fps > 0)) return "Frames per second must be above 0.";
  // Checked before the transfer starts: a server that refuses mid-body may reset the connection, and the
  // browser would show a network error instead of "too large". The server enforces the same limit anyway.
  if (draft.maxBytes !== null && draft.size > draft.maxBytes)
    return `That file is ${formatBytes(draft.size)}; the limit is ${formatBytes(draft.maxBytes)}.`;
  return null;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[i]}`;
}

export function uploadPercent(sent: number, total: number): number | null {
  return total > 0 ? Math.floor((sent / total) * 100) : null;
}
```

- [ ] **Step 3: Write the failing client tests** — add to `web/src/api/rest.test.ts` a `describe("uploadClip")` with a fake XHR (Node has no `XMLHttpRequest`, so stub it; `File` exists in Node 22 and the file already builds one):

```ts
class FakeXhr {
  static last: FakeXhr | null = null;
  upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = { onprogress: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  status = 0;
  statusText = "";
  responseText = "";
  method = "";
  url = "";
  headers: Record<string, string> = {};
  sent: unknown = null;
  aborted = false;
  constructor() { FakeXhr.last = this; }
  open(method: string, url: string) { this.method = method; this.url = url; }
  setRequestHeader(k: string, v: string) { this.headers[k] = v; }
  send(body: unknown) { this.sent = body; }
  abort() { this.aborted = true; this.onabort?.(); }
}

describe("uploadClip", () => {
  beforeEach(() => vi.stubGlobal("XMLHttpRequest", FakeXhr));
  afterEach(() => vi.unstubAllGlobals());

  it("posts the raw file with the name, stride and fps in the query, and reports progress", async () => {
    const progress: number[] = [];
    const file = new File(["abc"], "bench footage.mp4");
    const handle = api.uploadClip("p1", file, { stride: 2, fps: 30 }, (sent) => progress.push(sent));
    const xhr = FakeXhr.last!;
    expect(xhr.method).toBe("POST");
    expect(xhr.url).toBe("/api/projects/p1/clips/upload?name=bench%20footage.mp4&stride=2&fps=30");
    expect(xhr.headers["Content-Type"]).toBe("application/octet-stream");
    expect(xhr.sent).toBe(file);
    xhr.upload.onprogress!({ lengthComputable: true, loaded: 5, total: 10 });
    xhr.status = 202;
    xhr.responseText = JSON.stringify({ job_id: "j1" });
    xhr.onload!();
    expect(await handle.promise).toEqual({ job_id: "j1" });
    expect(progress).toEqual([5]);
  });

  it("rejects with the server's detail and status", async () => {
    const handle = api.uploadClip("p1", new File(["abc"], "x.mp4"), { stride: 1, fps: 30 });
    const xhr = FakeXhr.last!;
    xhr.status = 413;
    xhr.statusText = "Payload Too Large";
    xhr.responseText = JSON.stringify({ detail: "the upload is 100 bytes; the limit is 10" });
    xhr.onload!();
    await expect(handle.promise).rejects.toMatchObject({ status: 413, message: "the upload is 100 bytes; the limit is 10" });
  });

  it("rejects an aborted upload with UploadCancelled", async () => {
    const handle = api.uploadClip("p1", new File(["abc"], "x.mp4"), { stride: 1, fps: 30 });
    const xhr = FakeXhr.last!;
    handle.abort();
    expect(xhr.aborted).toBe(true);
    await expect(handle.promise).rejects.toBeInstanceOf(UploadCancelled);
  });
});
```

`beforeEach` must come from vitest (add it to the existing import line) and the new names (`UploadCancelled`, `api.uploadClip`) to the `rest` import.

- [ ] **Step 4: Implement the client.** In `web/src/api/types.ts`:

```ts
export type Health = {
  status: string;
  version: string;
  engine: { loaded: boolean; device: string | null; error: string | null };
  /** Absent on an older server: the GUI then skips its own size check and lets the server refuse. */
  upload?: { max_bytes: number };
};
```

In `web/src/api/rest.ts`:
  - extract the error construction that `request` already does, so the XHR path shares it:

```ts
/** The one place a failed response becomes an ApiError; `request` and `uploadClip` both use it. */
export function apiError(status: number, body: unknown, statusText: string, jobId: string | null = null): ApiError {
  return new ApiError(status, detailText(body, `${status} ${statusText}`), jobId);
}
```

    and rewrite `request`'s `if (!r.ok)` branch to use it (behaviour unchanged);
  - add the cancellation error and the upload itself:

```ts
export class UploadCancelled extends Error {
  constructor() {
    super("upload cancelled");
    this.name = "UploadCancelled";
  }
}

export type UploadHandle = { promise: Promise<{ job_id: string }>; abort: () => void };

/**
 * Upload one file as the raw request body. XHR, not fetch: only XHR reports upload progress and can abort a
 * request that is still being sent, which is the whole point of this path over an SSH tunnel.
 */
export function uploadClip(
  projectId: string,
  file: File,
  options: { stride: number; fps: number },
  onProgress?: (sent: number, total: number) => void,
): UploadHandle {
  const url =
    `/api/projects/${projectId}/clips/upload` +
    `?name=${encodeURIComponent(file.name)}&stride=${options.stride}&fps=${options.fps}`;
  const xhr = new XMLHttpRequest();
  const promise = new Promise<{ job_id: string }>((resolve, reject) => {
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded, e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON: keep null */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as { job_id: string });
        return;
      }
      const jobId =
        body !== null && typeof body === "object" && typeof (body as { job_id?: unknown }).job_id === "string"
          ? (body as { job_id: string }).job_id
          : null;
      reject(apiError(xhr.status, body, xhr.statusText, jobId));
    };
    xhr.onerror = () => reject(new Error("The upload failed. Check that the connection is still up."));
    xhr.onabort = () => reject(new UploadCancelled());
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}
```

  - add it to the `api` object: `uploadClip,` (so the page tests can mock it with the existing `api` mock pattern). Keep `importVideo` for now — Task 5 removes it.

- [ ] **Step 5: Run the web checks** — `npx tsc --noEmit` clean, `npx vitest run` green (the previous count plus the new helper and client tests).

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/upload.ts web/src/lib/upload.test.ts web/src/api/rest.ts web/src/api/rest.test.ts web/src/api/types.ts
git commit -m "feat(web): an XHR upload client with progress and cancel, and the upload option helpers"
```

---

### Task 5: the Clips page upload section, and the removal of the multipart route

**Files:**
- Modify: `web/src/pages/Clips.tsx`, `web/src/pages/Clips.test.tsx`, `web/src/api/rest.ts`, `web/src/api/rest.test.ts`
- Modify: `server/vsp/api/clips.py`, `server/tests/test_api_core.py`
- Test: `web/src/pages/Clips.test.tsx`, `server/tests/test_api_core.py`

**Interfaces:**
- Consumes: Task 3's route, Task 4's `api.uploadClip` / `UploadCancelled` / helpers, `HealthContext` (`health?.upload?.max_bytes`).
- Produces: no new exported names. `api.importVideo` and `POST .../clips/import-video` cease to exist.

- [ ] **Step 1: Switch the page.** In `web/src/pages/Clips.tsx`:

  - imports: `useContext` from react; `HealthContext` from `../hooks/useHealth`; `UploadCancelled` and the `UploadHandle` type from `../api/rest`; `formatBytes, importKind, uploadPercent, uploadProblem` from `../lib/upload`.
  - state: replace the `video` / `videoStride` pair with

```tsx
  const { health } = useContext(HealthContext);
  const maxBytes = health?.upload?.max_bytes ?? null;

  const [file, setFile] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [stride, setStride] = useState(1);
  const [fps, setFps] = useState(30);
  const [dragging, setDragging] = useState(false);
  const [upload, setUpload] = useState<{ sent: number; total: number } | null>(null);
  const uploadRef = useRef<UploadHandle | null>(null);
  const uploadIssue = uploadProblem({ name: file?.name ?? null, size: file?.size ?? 0, stride, fps, maxBytes });
```

  - delete `importVideo`; add (note: **not** `guarded` — an upload lasts minutes and must not disable the rest of the page while it runs):

```tsx
  const pickFile = (next: File | null) => {
    setFile(next);  // a new choice replaces the old one; only one upload runs at a time
    setUpload(null);
  };

  const startUpload = () => {
    if (file === null || project === null || uploadRef.current !== null || uploadIssue !== null) return;
    setError(null);
    const handle = api.uploadClip(project.id, file, { stride, fps }, (sent, total) => {
      if (uploadRef.current === handle) setUpload({ sent, total });
    });
    uploadRef.current = handle;
    setUpload({ sent: 0, total: file.size });
    handle.promise.then(
      ({ job_id }) => {
        if (uploadRef.current !== handle) return;
        uploadRef.current = null;
        setUpload(null);
        setImports((list) => [...list, { id: job_id, label: file.name }]);
        setFile(null);
        setFileInputKey((k) => k + 1);
      },
      (e: unknown) => {
        if (uploadRef.current !== handle) return;
        uploadRef.current = null;
        setUpload(null);
        if (e instanceof UploadCancelled) return; // the user asked for it
        fail(e);
      },
    );
  };

  const cancelUpload = () => {
    const handle = uploadRef.current;
    if (handle === null) return;
    uploadRef.current = null; // the abort's rejection is ignored by the identity guard above
    handle.abort();
    setUpload(null);
  };
```

  - replace the "Import a video" `<form>` with (keep the same section order in the sidebar: Export link, Classes, Upload footage, Import a folder of frames):

```tsx
            <section
              className={`flex flex-col gap-2 rounded border p-2 ${dragging ? "border-ink" : "border-transparent"}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                pickFile(e.dataTransfer.files?.[0] ?? null);
              }}
            >
              <h2 className="heading">Upload footage</h2>
              <input
                key={fileInputKey}
                aria-label="Footage file"
                type="file"
                accept="video/*,.zip,application/zip"
                className="text-sm text-dim file:btn file:mr-2 file:border-0"
                onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
              />
              {file === null ? (
                <p className="note">Drop a video or a zip of images here, or choose one above.</p>
              ) : (
                <p className="note text-ink">
                  {importKind(file.name) === "zip" ? "Image folder" : "Video"}, {formatBytes(file.size)}
                </p>
              )}
              <label className="flex items-center justify-between gap-2 text-sm text-dim">
                Keep every Nth frame
                <input
                  aria-label="Upload stride"
                  type="number"
                  min={1}
                  className="field w-16"
                  value={stride}
                  onChange={(e) => setStride(Math.max(1, Number(e.target.value) || 1))}
                />
              </label>
              {file !== null && importKind(file.name) === "zip" && (
                <label className="flex items-center justify-between gap-2 text-sm text-dim">
                  Frames per second
                  <input
                    aria-label="Zip frames per second"
                    type="number"
                    min={1}
                    step="any"
                    className="field w-16"
                    value={fps}
                    onChange={(e) => setFps(Math.max(1, Number(e.target.value) || 30))}
                  />
                </label>
              )}
              {file !== null && uploadIssue !== null && <p className="note text-ink">{uploadIssue}</p>}
              {upload === null ? (
                <button
                  type="button"
                  className="btn btn-on"
                  disabled={file === null || pending || uploadIssue !== null}
                  onClick={startUpload}
                >
                  Upload and import
                </button>
              ) : (
                <>
                  <progress value={upload.sent} max={upload.total || undefined} />
                  <p className="note">
                    {uploadPercent(upload.sent, upload.total) ?? 0}% of {formatBytes(upload.total)}
                  </p>
                  <button type="button" className="btn" onClick={cancelUpload}>
                    Cancel upload
                  </button>
                </>
              )}
            </section>
```

  - update the empty-state sentence to `This project has no clips yet. Upload a video or a zip of images, or import a folder of frames.`

- [ ] **Step 2: Update and extend the page tests** — `web/src/pages/Clips.test.tsx`:
  - in the `vi.mock` api object replace `importVideo: vi.fn()` with `uploadClip: vi.fn()`;
  - add a helper and the new cases:

```tsx
const aFile = (name: string, bytes = "abc") => new File([bytes], name);
const choose = (name: string) =>
  fireEvent.change(screen.getByLabelText("Footage file"), { target: { files: [aFile(name)] } });
const uploadHandle = (jobId = "j10") => ({ promise: Promise.resolve({ job_id: jobId }), abort: vi.fn() });

it("uploads the chosen video, shows progress, then follows the import job", async () => {
  let progress!: (sent: number, total: number) => void;
  m.uploadClip.mockImplementation((_pid, _file, _opts, onProgress) => {
    progress = onProgress!;
    return { promise: Promise.resolve({ job_id: "j10" }), abort: vi.fn() };
  });
  renderPage();
  await screen.findByText("bench.mp4");
  choose("bench.mp4");
  fireEvent.click(screen.getByRole("button", { name: "Upload and import" }));
  await waitFor(() => expect(m.uploadClip).toHaveBeenCalledWith(project.id, expect.any(File), { stride: 1, fps: 30 }, expect.any(Function)));
  act(() => progress(50, 100));
  expect(screen.getByText("50% of 100 B")).toBeTruthy();
  await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j10$/));
  expect((screen.getByLabelText("Footage file") as HTMLInputElement).value).toBe("");
});

it("only offers frames per second for a zip and sends it", async () => {
  m.uploadClip.mockReturnValue(uploadHandle("j11"));
  renderPage();
  await screen.findByText("bench.mp4");
  expect(screen.queryByLabelText("Zip frames per second")).toBeNull();
  choose("run.zip");
  fireEvent.change(screen.getByLabelText("Zip frames per second"), { target: { value: "12" } });
  fireEvent.change(screen.getByLabelText("Upload stride"), { target: { value: "3" } });
  fireEvent.click(screen.getByRole("button", { name: "Upload and import" }));
  await waitFor(() => expect(m.uploadClip).toHaveBeenCalledWith(project.id, expect.any(File), { stride: 3, fps: 12 }, expect.any(Function)));
});

it("cancels an upload in flight without reporting an error", async () => {
  const abort = vi.fn();
  let reject!: (e: unknown) => void;
  m.uploadClip.mockReturnValue({ promise: new Promise((_r, rj) => { reject = rj; }), abort });
  renderPage();
  await screen.findByText("bench.mp4");
  choose("bench.mp4");
  fireEvent.click(screen.getByRole("button", { name: "Upload and import" }));
  fireEvent.click(await screen.findByRole("button", { name: "Cancel upload" }));
  expect(abort).toHaveBeenCalledTimes(1);
  await act(async () => reject(new UploadCancelled()));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByRole("button", { name: "Upload and import" })).toBeTruthy();
});

it("refuses a file that is neither a video nor a zip", async () => {
  renderPage();
  await screen.findByText("bench.mp4");
  choose("notes.txt");
  expect(await screen.findByText("That file is not a video or a .zip of images.")).toBeTruthy();
  expect((screen.getByRole("button", { name: "Upload and import" }) as HTMLButtonElement).disabled).toBe(true);
});

it("shows why the server refused an upload", async () => {
  m.uploadClip.mockReturnValue({ promise: Promise.reject(new ApiError(413, "the upload is 100 bytes; the limit is 10")), abort: vi.fn() });
  renderPage();
  await screen.findByText("bench.mp4");
  choose("bench.mp4");
  fireEvent.click(screen.getByRole("button", { name: "Upload and import" }));
  expect(await screen.findByText("the upload is 100 bytes; the limit is 10")).toBeTruthy();
});
```

    Add `UploadCancelled` to the `../api/rest` import in the test file (`import { ApiError, UploadCancelled, api } from "../api/rest"`). The "disables the other submit buttons while a request is in flight" test keeps working: uploads no longer set `pending` — deliberately, an upload runs for minutes and must not disable Delete or Add class — so it only asserts `Add class` / `Import frames` / `Delete clip`.
    `uploadIssue` is computed once per render from `file`, `stride`, `fps` and `maxBytes`, and drives both the note and the button's `disabled`; with no file chosen there is no issue text, the placeholder note above already says what to do.

- [ ] **Step 3: Remove the multipart endpoint.** In `server/vsp/api/clips.py` delete the whole `import_video` route; drop `File`, `Form`, `UploadFile` and `shutil` from the imports (keep `uuid`, `re`, `Path`); keep `_safe_name` and `_upload_kind`.
In `server/tests/test_api_core.py` delete `test_import_video_upload` and `test_a_bad_upload_becomes_a_job_error` (both behaviours now live in `test_api_upload.py`) and assert the route is gone, so a future edit cannot quietly resurrect two import paths:

```python
def test_the_multipart_import_video_route_is_gone(client, tmp_path):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/clips/import-video", files={"file": ("x.mp4", b"x", "video/mp4")})
    assert r.status_code in (404, 405)
```

- [ ] **Step 4: Remove the multipart client.** In `web/src/api/rest.ts` delete `importVideo`. In `web/src/api/rest.test.ts` split the test named "uses `from` as the prune field name and uploads videos as multipart" down to the prune assertion only (the upload has its own describe block now).

- [ ] **Step 5: Run everything** — `HF_HUB_OFFLINE=1 uv run pytest -q` green; `uv run ruff check .` clean; `npx tsc --noEmit` clean; `npx vitest run` green; `npm run build` succeeds.

- [ ] **Step 6: Commit**

```bash
git add web/src/pages/Clips.tsx web/src/pages/Clips.test.tsx web/src/api/rest.ts web/src/api/rest.test.ts server/vsp/api/clips.py server/tests/test_api_core.py
git commit -m "feat(web): upload videos and image zips from the Clips page; retire the multipart import route"
```

---

### Task 6 (controller, not a subagent): real-server check, documentation, roadmap

- [ ] Start `HF_HUB_OFFLINE=1 uv run vsp serve`. With `curl`, upload a real video and a real image zip (build one with `zip -r run.zip data/demo_frames`), confirm `202 → job done`, the clip appears with the right `source.name`/`frame_count`/`fps`, and `data/uploads/` is empty afterwards.
- [ ] From a second terminal, forward the port as the README will say (`ssh -N -L 28000:127.0.0.1:28000 localhost`) and open `http://127.0.0.1:28000/`: upload a ~100 MB video and watch the percentage move; press Cancel halfway and confirm the browser reports no error, the server log shows the disconnect, and `data/uploads/` is empty.
- [ ] Upload a file larger than a temporarily lowered `ingest.max_upload_bytes` and check the GUI refuses it before the transfer starts (the pre-check from `/api/health`) and that `curl` gets a clean `413`.
- [ ] Confirm the engine banner and the Label page still work while an upload runs (an upload is not a GPU job and must not block prompting/propagation).
- [ ] `README.md`:
  - "Run the server": add the SSH-forward paragraph (the `ssh -N -L 28000:127.0.0.1:28000 user@gpu-host` recipe, that the server keeps binding `127.0.0.1`, and that `-o Compression=no` is usually faster for video).
  - a short "Uploading footage" paragraph: accepted types (video extensions, `.zip` of `.jpg/.jpeg/.png`), that the body is streamed once to `data/uploads/` and deleted when the job ends, that the browser must stay on the page (a reload cancels), the 8 GiB default cap (`ingest.max_upload_bytes`), that `stride` applies to both and `fps` only to a zip, and that resumable uploads are not implemented yet.
  - the README has no API table (the export route is described as a `POST ...` sentence in "Run the server"), so document the upload route the same way, next to the existing import notes: `POST /api/projects/{p}/clips/upload?name=&stride=&fps=` with the file as the raw body → `202 {job_id}`, 400 for an unsupported file, 413 over the cap.
  - manual QA checklist: add an item for uploading a video and a zip through the tunnel (progress, cancel halfway, then a clean import), and one for a file over the cap.
  - the roadmap's Plan 2 "Deferred minors" list says "no upload size limit": note that it is now addressed.
- [ ] `docs/superpowers/plans/2026-09-20-vsp-roadmap.md`: add a Plan 5 row ("Done", branch, test counts) and a short section; note that this plan supersedes the spec's §4 `import-video` row and the §6 "multipart upload" clause.
- [ ] Append an "As executed" section to this plan: what changed against the text, every review fix, and the measured tunnel throughput seen above.

---

## Phase 2 (designed here, NOT built in this plan): resumable chunked uploads

Phase 1 covers a *slow* link; it does not cover a *dropped* one, and it makes the browser tab the owner of the transfer. If that becomes the operator's real pain, the next step is a tus-style chunk session, and the phase-1 shape is already compatible with it (`data/uploads/` exists, the route is one POST that can become create/chunk/complete, and the GUI already has a progress-and-cancel state machine):

1. `POST /api/uploads` `{name, size}` → `{upload_id, offset: 0}`; creates `data/uploads/<id>.part` plus `<id>.json` (name, size, created_at).
2. `PATCH /api/uploads/{id}` with an `Upload-Offset` header and a chunk body (8-16 MiB): 409 when the offset does not match the file's current length, else `{offset}`. Append-only, one writer per id.
3. `POST /api/uploads/{id}/complete?project_id=&stride=&fps=` → validates `size`, renames the part file, starts the import job exactly as Task 3 does.
4. `DELETE /api/uploads/{id}`; sessions older than 24 h are swept by the same startup sweep (extend `clear_uploads`).
5. Client: slice the `File` into chunks, keep `{upload_id, name, size, lastModified}` in `localStorage`, retry a failed chunk with backoff, and resume on reload by asking `GET /api/uploads/{id}` for the offset.
6. Tests: offset mismatch, resume after a gap, completion of an exact-size file, sweeping, and a client test that a mid-chunk failure retries and resumes.

Why it is not in this plan: it adds server-side session state and GC plus a second client state machine, and it should be designed against a measured failure (how often links actually drop) rather than speculatively.

## Out of scope

- Authentication (the SSH tunnel is the boundary) and any change to the bind address.
- `.tar` / `.7z` / `.rar` archives (the same `ZipFrameSource` shape would take a `tarfile` sibling later).
- Multiple files in one request, or a client-side upload queue; one file at a time.
- Extracting an archive to disk to reuse the JPEG copy path (`src_path`) rather than re-encoding.
- Server-side progress for the import itself beyond the existing `progress` events, and progress for the transfer after it finishes.

## As executed

*(filled in when the plan is executed: what changed against the text above, every review fix, and the measured tunnel throughput.)*
