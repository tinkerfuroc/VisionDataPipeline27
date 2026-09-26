"""Helpers shared by the training tests: a minimal yolo_seg export, the fake worker command, polling."""

from __future__ import annotations

import io
import json
import sys
import tarfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

FAKE_WORKER = [sys.executable, str(Path(__file__).with_name("fake_train_worker.py"))]
EXPORT_ID = "20260101-000000-abcdef"
SQUARE = "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"
JPEG = b"\xff\xd8\xff\xe0fakejpeg"


def make_export(
    data_root: Path, export_id: str = EXPORT_ID, n_train: int = 2, n_val: int = 2, *,
    created_at: str = "2026-01-01T00:00:00+00:00", name: str | None = None,
) -> Path:
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
        "export_id": export_id, "created_at": created_at,
        "project": {"id": "p1", "name": "Bench"},
        "classes": [
            {"id": 0, "name": "Module", "color": "#e6194b", "yolo_id": 0, "coco_id": 1, "mask_value": 1}
        ],
        "formats": [
            {
                "name": "yolo_seg", "images": {"train": n_train, "val": n_val},
                "annotations": {"train": n_train, "val": n_val}, "linked": True,
            }
        ],
    }
    if name is not None:
        manifest["name"] = name
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


def _tarball(path: Path) -> None:
    """A DTD-shaped archive: two good images, a labels file, a path-escape entry, an absolute-path
    entry, a doubled-slash entry, a backslash entry and a symlink -- so extraction tests can prove only
    the two safe images make it out."""
    with tarfile.open(path, "w:gz") as tar:

        def add(name, data=JPEG, kind=tarfile.REGTYPE, link=""):
            info = tarfile.TarInfo(name)
            info.type, info.linkname, info.size = kind, link, (len(data) if kind == tarfile.REGTYPE else 0)
            tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)

        add("dtd/images/banded/banded_0001.jpg")
        add("dtd/images/woven/woven_0002.png")
        add("dtd/labels/labels_joint_anno.txt", b"text")
        add("dtd/images/../../evil.jpg")
        add("/abs/dtd/images/x.jpg")
        add("dtd/images//abs.jpg")
        add("dtd/images/evil\\..\\..\\evil.jpg")
        add("dtd/images/link.jpg", kind=tarfile.SYMTYPE, link="/etc/passwd")


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A local HTTP server serving a fake DTD tarball; proxies are bypassed so tests never hit the network."""
    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "www"
    root.mkdir()
    _tarball(root / "dtd.tar.gz")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/dtd.tar.gz"
    httpd.shutdown()
