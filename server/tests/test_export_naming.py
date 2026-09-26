import threading

import pytest
from vsp.export.naming import clean_name, reserve_export_dir, slug
from vsp.labeling.errors import BadRequest


@pytest.mark.parametrize("name, expected", [
    ("Cups v2", "cups-v2"),
    ("  Tassen / Schalen (final)  ", "tassen-schalen-final"),
    ("--weird__name..", "weird__name"),
    ("a" * 60, "a" * 48),
    ("x" * 47 + "-yz", "x" * 47),
    ("杯子", ""),
    ("", ""),
    (None, ""),
    ("v1.2_test", "v1.2_test"),
])
def test_slug(name, expected):
    assert slug(name) == expected


def test_clean_name():
    assert clean_name("  Cups v2 ") == "Cups v2"
    assert clean_name("   ") is None and clean_name(None) is None
    with pytest.raises(BadRequest, match="at most 80"):
        clean_name("a" * 81)
    with pytest.raises(BadRequest, match="control"):
        clean_name("two\nlines")
    with pytest.raises(BadRequest, match="control"):
        clean_name("two lines")  # U+2028 LINE SEPARATOR (category Zl)


def test_reserve_uses_the_slug_then_numbers(tmp_path):
    assert reserve_export_dir(tmp_path, "Cups v2")[0] == "cups-v2"
    assert reserve_export_dir(tmp_path, "cups V2")[0] == "cups-v2-2"
    assert reserve_export_dir(tmp_path, "Cups-v2")[0] == "cups-v2-3"
    assert all((tmp_path / d).is_dir() for d in ("cups-v2", "cups-v2-2", "cups-v2-3"))


def test_reserve_falls_back_to_a_timestamp_id(tmp_path):
    export_id, path = reserve_export_dir(tmp_path, "杯子")
    assert len(export_id) == len("20260926-104433-a1b2c3") and export_id[8] == "-" and path.is_dir()


def test_concurrent_reservations_never_share_a_folder(tmp_path):
    got: list[str] = []
    lock = threading.Lock()

    def one():
        export_id, _ = reserve_export_dir(tmp_path, "same")
        with lock:
            got.append(export_id)

    threads = [threading.Thread(target=one) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(got)) == 12
