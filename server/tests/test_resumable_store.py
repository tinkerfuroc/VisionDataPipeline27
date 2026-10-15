from __future__ import annotations

import json
import threading

import pytest
from vdp.ingest import IngestError, UploadTooLarge
from vdp.ingest.resumable import CHUNK_SIZE, OffsetConflict, ResumableStore, UploadNotFound


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return ResumableStore(tmp_path / "resumable", max_bytes=100, clock=clock)


def _new(store, size=10, project="p1", name="v.mp4"):
    return store.create(project, name, "video", size, 1, 30.0)


def test_create_writes_an_empty_part_and_a_record(store):
    rec = _new(store)
    assert len(rec.id) == 32 and rec.received == 0 and not rec.complete
    assert store.part_path(rec.id).read_bytes() == b""
    assert store.get(rec.id).public() == {
        "upload_id": rec.id, "project_id": "p1", "name": "v.mp4", "size": 10, "received": 0, "job_id": None,
    }


def test_create_refuses_a_file_over_the_cap(store):
    with pytest.raises(UploadTooLarge):
        _new(store, size=101)


def test_chunks_append_in_order_and_complete(store):
    rec = _new(store)
    assert store.write_chunk(rec.id, 0, b"abcd").received == 4
    assert store.write_chunk(rec.id, 4, b"efgh").received == 8
    done = store.write_chunk(rec.id, 8, b"ij")
    assert done.received == 10 and done.complete
    assert store.part_path(rec.id).read_bytes() == b"abcdefghij"


def test_a_retried_chunk_that_already_arrived_is_a_no_op(store):
    rec = _new(store)
    store.write_chunk(rec.id, 0, b"abcd")
    assert store.write_chunk(rec.id, 0, b"abcd").received == 4
    assert store.part_path(rec.id).read_bytes() == b"abcd"


def test_any_other_offset_mismatch_is_a_conflict_carrying_received(store):
    rec = _new(store)
    store.write_chunk(rec.id, 0, b"abcd")
    with pytest.raises(OffsetConflict) as gap:
        store.write_chunk(rec.id, 6, b"gh")
    assert gap.value.received == 4
    with pytest.raises(OffsetConflict):
        store.write_chunk(rec.id, 2, b"cdef")  # overlaps the end of what arrived


def test_empty_oversized_and_past_size_chunks_are_refused(store):
    rec = _new(store)
    with pytest.raises(IngestError):
        store.write_chunk(rec.id, 0, b"")
    with pytest.raises(UploadTooLarge):
        store.write_chunk(rec.id, 0, b"x" * (CHUNK_SIZE + 1))
    with pytest.raises(UploadTooLarge):
        store.write_chunk(rec.id, 0, b"x" * 11)
    assert store.get(rec.id).received == 0


@pytest.mark.parametrize("bad", ["", "abc", "../../auth", "G" * 32, "a" * 31, "a" * 33])
def test_ids_that_are_not_32_hex_chars_are_not_found(store, bad):
    with pytest.raises(UploadNotFound):
        store.get(bad)


def test_complete_starts_exactly_one_job_even_when_called_concurrently(store):
    rec = _new(store, size=2)
    store.write_chunk(rec.id, 0, b"ab")
    started: list[str] = []
    lock = threading.Lock()

    def start(r):
        with lock:
            started.append(r.id)
        return f"job{len(started)}"

    results = []
    threads = [threading.Thread(target=lambda: results.append(store.complete(rec.id, start))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert started == [rec.id]
    assert {r.job_id for r in results} == {"job1"}


def test_complete_does_nothing_before_the_last_byte(store):
    rec = _new(store)
    store.write_chunk(rec.id, 0, b"abcd")
    assert store.complete(rec.id, lambda r: "job").job_id is None


def test_list_shows_unfinished_uploads_of_a_project_newest_first(store, clock):
    a = _new(store)
    clock.t += 1
    b = _new(store)
    _new(store, project="other")
    done = _new(store, size=1)
    store.write_chunk(done.id, 0, b"x")
    assert [r.id for r in store.list("p1")] == [b.id, a.id]


def test_delete_and_discard(store):
    rec = _new(store)
    store.delete(rec.id)
    with pytest.raises(UploadNotFound):
        store.get(rec.id)
    with pytest.raises(UploadNotFound):
        store.delete(rec.id)
    store.discard(rec.id)  # already gone: no error


def test_untouched_uploads_expire_after_the_ttl(store, clock):
    rec = _new(store)
    clock.t += 24 * 3600 - 1
    store.write_chunk(rec.id, 0, b"ab")  # activity resets the clock
    clock.t += 24 * 3600 + 1
    with pytest.raises(UploadNotFound):
        store.get(rec.id)
    assert not store.part_path(rec.id).exists()


def test_sweep_removes_expired_uploads(store, clock):
    _new(store)
    clock.t += 24 * 3600 + 1
    assert store.sweep() == 1
    assert list(store.root.iterdir()) == []


def test_drop_orphaned_completions_removes_records_with_a_job(store):
    rec = _new(store, size=1)
    store.write_chunk(rec.id, 0, b"x")
    store.complete(rec.id, lambda r: "job1")
    keep = _new(store)
    assert store.drop_orphaned_completions() == 1
    assert [r.id for r in store.list("p1")] == [keep.id]
    assert not store.part_path(rec.id).exists()


def test_a_part_longer_than_received_is_truncated_on_load(store):
    rec = _new(store)
    store.write_chunk(rec.id, 0, b"abcd")
    with store.part_path(rec.id).open("ab") as f:
        f.write(b"zz")  # a crash after the write, before the record was saved
    assert store.get(rec.id).received == 4
    assert store.part_path(rec.id).read_bytes() == b"abcd"


def test_a_part_shorter_than_received_lowers_received(store):
    rec = _new(store)
    store.write_chunk(rec.id, 0, b"abcd")
    store.part_path(rec.id).write_bytes(b"ab")
    assert store.get(rec.id).received == 2


def test_a_corrupt_record_is_not_found_and_removed(store):
    rec = _new(store)
    (store.root / f"{rec.id}.json").write_text("{not json")
    with pytest.raises(UploadNotFound):
        store.get(rec.id)
    assert not store.part_path(rec.id).exists()


def test_state_survives_a_new_store_on_the_same_root(tmp_path, clock):
    first = ResumableStore(tmp_path / "r", max_bytes=100, clock=clock)
    rec = first.create("p1", "v.mp4", "video", 10, 2, 25.0)
    first.write_chunk(rec.id, 0, b"abc")
    again = ResumableStore(tmp_path / "r", max_bytes=100, clock=clock).get(rec.id)
    assert (again.received, again.stride, again.fps) == (3, 2, 25.0)
    assert json.loads((tmp_path / "r" / f"{rec.id}.json").read_text())["received"] == 3


def test_discard_with_path_traversal_raises_and_touches_nothing_outside_root(store, tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.touch()
    with pytest.raises(UploadNotFound):
        store.discard("../../x")
    assert sentinel.exists()  # sentinel still exists; nothing was touched outside root


def test_part_path_with_path_traversal_raises(store):
    with pytest.raises(UploadNotFound):
        store.part_path("../x")


def test_write_chunk_with_path_traversal_raises_and_touches_nothing(store, tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.touch()
    with pytest.raises(UploadNotFound):
        store.write_chunk("../../x", 0, b"a")
    assert sentinel.exists()


def test_record_file_with_invalid_utf8_bytes_is_not_found_and_removed(store):
    rec = _new(store)
    record_path = store.root / f"{rec.id}.json"
    record_path.write_bytes(b"\x80\x81\x82")  # invalid UTF-8
    with pytest.raises(UploadNotFound):
        store.get(rec.id)
    assert not store.part_path(rec.id).exists()
