import cv2
import numpy as np
import pytest
from vsp.ingest import (
    FrameDirSource,
    IngestError,
    VideoFileSource,
    import_clip,
    natural_key,
    resolve_allowed,
)
from vsp.store import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "data")


def _img(value, w=64, h=48):
    img = np.full((h, w, 3), value, dtype=np.uint8)
    img[:, : w // 2] = 255 - value  # some structure so JPEG has content
    return img


def _write(path, value, **kw):
    assert cv2.imwrite(str(path), _img(value, **kw))


def test_natural_key_orders_numbers_numerically():
    names = ["frame_10.png", "frame_2.png", "frame_1.png"]
    assert sorted(names, key=natural_key) == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_frame_dir_natural_order_gaps_and_original_names(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for n, v in [(1, 10), (2, 60), (10, 200)]:  # gap between 2 and 10 is fine
        _write(src / f"frame_{n}.png", v)
    (src / "notes.txt").write_text("ignored")
    p = store.create_project("p")
    clip = import_clip(store, p.id, FrameDirSource(src, fps=12.0))
    assert clip.frame_count == 3
    assert (clip.width, clip.height, clip.fps) == (64, 48, 12.0)
    assert clip.source.type == "frame_dir"
    assert clip.original_names == {0: "frame_1.png", 1: "frame_2.png", 2: "frame_10.png"}
    # index 2 must be the image written as frame_10 (value 200 on the right half)
    got = cv2.imread(str(store.frame_path(clip.id, 2)))
    assert abs(int(got[0, -1, 0]) - 200) <= 3
    assert store.thumb_path(clip.id, 0).is_file()
    assert store.read_clip(clip.id).frame_count == 3


def test_jpeg_is_copied_byte_identical_png_is_reencoded(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _write(src / "0.jpg", 50)
    _write(src / "1.png", 90)
    p = store.create_project("p")
    clip = import_clip(store, p.id, FrameDirSource(src))
    assert store.frame_path(clip.id, 0).read_bytes() == (src / "0.jpg").read_bytes()
    assert store.frame_path(clip.id, 1).read_bytes()[:2] == b"\xff\xd8"  # JPEG magic


def test_stride_keeps_every_nth(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for n in range(6):
        _write(src / f"{n}.png", n * 40)
    p = store.create_project("p")
    clip = import_clip(store, p.id, FrameDirSource(src), stride=2)
    assert clip.frame_count == 3
    assert clip.stride == 2
    assert clip.original_names == {0: "0.png", 1: "2.png", 2: "4.png"}


def test_mixed_sizes_rejected_and_partial_clip_removed(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _write(src / "0.png", 10)
    _write(src / "1.png", 10, w=32, h=32)
    p = store.create_project("p")
    with pytest.raises(IngestError, match="mixed frame sizes"):
        import_clip(store, p.id, FrameDirSource(src))
    assert list((store.projects_dir / p.id / "clips").iterdir()) == []


def test_unreadable_image_rejected(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    (src / "0.png").write_bytes(b"not a png")
    p = store.create_project("p")
    with pytest.raises(IngestError, match="unreadable"):
        import_clip(store, p.id, FrameDirSource(src))


def test_empty_dir_and_missing_dir_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IngestError, match="no .jpg"):
        FrameDirSource(empty)
    with pytest.raises(IngestError, match="not a directory"):
        FrameDirSource(tmp_path / "nope")


def test_resolve_allowed(tmp_path):
    root = tmp_path / "ok"
    (root / "sub").mkdir(parents=True)
    assert resolve_allowed(root / "sub", [str(root)]) == (root / "sub").resolve()
    with pytest.raises(IngestError, match="outside"):
        resolve_allowed(tmp_path / "other", [str(root)])
    with pytest.raises(IngestError, match="outside"):
        resolve_allowed(root / "sub" / ".." / ".." / "other", [str(root)])


def test_video_import_counts_decoded_frames(store, tmp_path):
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (64, 48))
    assert writer.isOpened()
    for n in range(30):
        writer.write(_img(n * 8))
    writer.release()
    p = store.create_project("p")
    seen = []
    clip = import_clip(store, p.id, VideoFileSource(path), progress=seen.append)
    assert clip.frame_count == 30
    assert clip.source.type == "video" and clip.source.name == "v.avi"
    assert clip.fps == pytest.approx(15.0, abs=0.01)
    assert clip.original_names == {}
    assert seen[-1] == 30


def test_bad_video_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(IngestError, match="cannot open"):
        VideoFileSource(bad)


def test_invalid_stride(store, tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    _write(src / "0.png", 1)
    p = store.create_project("p")
    with pytest.raises(IngestError, match="stride"):
        import_clip(store, p.id, FrameDirSource(src), stride=0)


def test_thumbnail_write_failure_raises_and_cleans_up(store, tmp_path, monkeypatch):
    import vsp.ingest.importer as importer_mod

    src = tmp_path / "frames"
    src.mkdir()
    _write(src / "0.png", 10)
    p = store.create_project("p")

    real_imwrite = cv2.imwrite

    def fake_imwrite(path, *a, **kw):
        if "thumbs" in path:
            return False
        return real_imwrite(path, *a, **kw)

    monkeypatch.setattr(importer_mod.cv2, "imwrite", fake_imwrite)
    with pytest.raises(IngestError, match="thumbnail"):
        import_clip(store, p.id, FrameDirSource(src))
    assert list((store.projects_dir / p.id / "clips").iterdir()) == []
