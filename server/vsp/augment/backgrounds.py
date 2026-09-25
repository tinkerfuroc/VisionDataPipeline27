"""Background image pools for composites (spec B1). One pool: DTD textures, downloaded on request."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from ..store import NotFound

POOLS: dict[str, dict] = {
    "dtd": {
        "label": "DTD textures",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        "prefix": "dtd/images/",
        "size": 625_239_812,
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_CHUNK = 1 << 20

Progress = Callable[[str, int, int], None]


class FetchCancelled(Exception):
    pass


def backgrounds_root(data_root: Path) -> Path:
    return Path(data_root) / "backgrounds"


def pool_dir(data_root: Path, name: str) -> Path:
    if name not in POOLS:
        raise NotFound(f"background pool {name} not found")
    return backgrounds_root(data_root) / name


def list_images(data_root: Path, name: str) -> list[Path]:
    d = pool_dir(data_root, name)
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def safe_member_name(member: tarfile.TarInfo, prefix: str) -> str | None:
    """The path below `prefix` for a regular image file with a clean relative name; None for anything else.

    Computed with `PurePosixPath.relative_to` (not string slicing): a doubled slash in the member name, e.g.
    `dtd/images//x/evil.jpg`, normalises to `x/evil.jpg` here instead of slicing to the leading-slash
    `/x/evil.jpg` that `tmp / rel` would resolve as an absolute path, escaping the pool directory.
    """
    if not member.isfile():
        return None
    name = member.name
    if "\\" in name:
        return None
    try:
        rel = PurePosixPath(name).relative_to(prefix)
    except ValueError:
        return None
    parts = rel.parts
    if not parts or rel.is_absolute() or any(p in (".", "..") for p in parts):
        return None
    if rel.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return rel.as_posix()


def fetch_pool(
    data_root: Path, name: str, *, url: str | None = None, force: bool = False,
    progress: Progress | None = None, cancel: Callable[[], bool] | None = None,
) -> dict:
    pool = POOLS.get(name)
    dest = pool_dir(data_root, name)
    existing = list_images(data_root, name)
    if existing and not force:
        return {"pool": name, "images": len(existing), "skipped": True}
    root = backgrounds_root(data_root)
    root.mkdir(parents=True, exist_ok=True)
    part = root / f".{name}.tar.gz.part"
    tmp = root / f".{name}.tmp"

    def check() -> None:
        if cancel is not None and cancel():
            raise FetchCancelled

    try:
        check()
        with urllib.request.urlopen(url or pool["url"], timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or pool["size"])
            done = 0
            while chunk := resp.read(_CHUNK):
                check()
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress("download", done, total)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        count = 0
        with tarfile.open(part, "r:gz") as tar:
            members = tar.getmembers()
            for i, member in enumerate(members):
                check()
                rel = safe_member_name(member, pool["prefix"])
                if rel is not None:
                    target = tmp / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Belt and suspenders alongside `safe_member_name`: never write outside the pool dir.
                    if target.resolve().is_relative_to(tmp.resolve()):
                        src = tar.extractfile(member)
                        if src is not None:
                            with src, open(target, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            count += 1
                if progress:
                    progress("extract", i + 1, len(members))
        if count == 0:
            raise ValueError(f"the archive holds no images under {pool['prefix']}")
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return {"pool": name, "images": count, "skipped": False}
