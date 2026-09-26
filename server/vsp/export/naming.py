"""Export names (multi-object spec §11): a free-text display name, and a folder id slugged from it."""

from __future__ import annotations

import itertools
import re
import time
import unicodedata
import uuid
from pathlib import Path

from ..labeling.errors import BadRequest

MAX_NAME = 80
MAX_SLUG = 48
_BAD = re.compile(r"[^a-z0-9._-]+")
_DASHES = re.compile(r"-{2,}")


def clean_name(name: str | None) -> str | None:
    """The name as stored: stripped, or None when blank."""
    if name is None:
        return None
    name = name.strip()
    if not name:
        return None
    if len(name) > MAX_NAME:
        raise BadRequest(f"the name must be at most {MAX_NAME} characters")
    if any(unicodedata.category(c) == "Cc" for c in name):
        raise BadRequest("the name must not contain control characters (such as line breaks)")
    return name


def slug(name: str | None) -> str:
    """A folder id from a name: lowercase `[a-z0-9._-]`, starting with a letter or digit; empty when none is left."""
    if not name:
        return ""
    s = _DASHES.sub("-", _BAD.sub("-", name.strip().lower()))
    s = s.lstrip("-._")
    return s[:MAX_SLUG].rstrip("-._")


def timestamp_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def reserve_export_dir(exports_root: Path, name: str | None) -> tuple[str, Path]:
    """Create and return a fresh export folder. `mkdir` without `exist_ok` is the reservation, so two exports
    started at once can never share a folder: the loser of a race moves on to the next number."""
    exports_root = Path(exports_root)
    exports_root.mkdir(parents=True, exist_ok=True)
    base = slug(name)
    if not base:
        export_id = timestamp_id()
        (exports_root / export_id).mkdir()  # a collision here is astronomically unlikely; fail rather than share
        return export_id, exports_root / export_id
    for n in itertools.count(1):
        export_id = base if n == 1 else f"{base}-{n}"
        try:
            (exports_root / export_id).mkdir()
        except FileExistsError:
            continue
        return export_id, exports_root / export_id
    raise AssertionError("unreachable: itertools.count never ends")
