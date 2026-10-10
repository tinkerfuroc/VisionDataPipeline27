from __future__ import annotations

import tarfile

import pytest
from vdp.augment.backgrounds import FetchCancelled, fetch_pool, list_images, safe_member_name


def test_only_safe_images_are_extracted(tmp_path, server):
    data_root = tmp_path / "data"
    phases = []
    result = fetch_pool(data_root, "dtd", url=server, progress=lambda ph, d, t: phases.append(ph))
    # "dtd/images//abs.jpg" (a doubled slash) normalises to the contained "abs.jpg" and is extracted safely;
    # the path-escape, absolute-path, backslash and symlink entries are all rejected.
    assert result == {"pool": "dtd", "images": 3, "skipped": False}
    pool_root = data_root / "backgrounds" / "dtd"
    names = sorted(p.relative_to(pool_root).as_posix() for p in list_images(data_root, "dtd"))
    assert names == ["abs.jpg", "banded/banded_0001.jpg", "woven/woven_0002.png"]
    assert not (tmp_path / "evil.jpg").exists() and "download" in phases and "extract" in phases
    assert not list((data_root / "backgrounds").glob(".*"))  # no partial archive or temp dir left


def test_existing_pool_is_skipped_unless_forced(tmp_path, server):
    data_root = tmp_path / "data"
    fetch_pool(data_root, "dtd", url=server)
    assert fetch_pool(data_root, "dtd", url=server)["skipped"] is True
    assert fetch_pool(data_root, "dtd", url=server, force=True)["images"] == 3


def test_cancel_leaves_nothing(tmp_path, server):
    data_root = tmp_path / "data"
    with pytest.raises(FetchCancelled):
        fetch_pool(data_root, "dtd", url=server, cancel=lambda: True)
    assert list_images(data_root, "dtd") == []
    assert not list((data_root / "backgrounds").glob(".*"))


def test_safe_member_name():
    info = tarfile.TarInfo("dtd/images/a/b.jpg")
    assert safe_member_name(info, "dtd/images/") == "a/b.jpg"
    # A doubled slash normalises to a name safely contained under `prefix`, instead of the naive
    # string-sliced "/abs.jpg" that a caller could resolve as an absolute (escaping) path.
    assert safe_member_name(tarfile.TarInfo("dtd/images//abs.jpg"), "dtd/images/") == "abs.jpg"
    for bad in (
        "dtd/images/../x.jpg", "/dtd/images/x.jpg", "dtd/images/x.txt", "other/x.jpg",
        "dtd/images/evil\\..\\..\\evil.jpg", "dtd\\images\\evil.jpg",
    ):
        assert safe_member_name(tarfile.TarInfo(bad), "dtd/images/") is None
