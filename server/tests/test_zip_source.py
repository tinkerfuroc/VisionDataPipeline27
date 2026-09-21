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
