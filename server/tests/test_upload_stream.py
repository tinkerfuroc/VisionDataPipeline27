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
