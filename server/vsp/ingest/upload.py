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
