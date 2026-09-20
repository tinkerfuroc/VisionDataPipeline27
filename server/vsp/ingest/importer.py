"""Any FrameSource -> frames/NNNNNN.jpg + thumbs/ + clip.json. Failure removes the partial clip."""

from __future__ import annotations

import shutil
from collections.abc import Callable

import cv2

from ..store import Clip, ClipSource, ProjectStore, now_iso
from .source import FrameSource, IngestError

_JPEG_SUFFIXES = {".jpg", ".jpeg"}


class ImportCancelled(IngestError):
    """The import was cancelled cooperatively via the `cancel` callback."""


def import_clip(
    store: ProjectStore,
    project_id: str,
    source: FrameSource,
    *,
    stride: int = 1,
    jpeg_quality: int = 95,
    thumb_width: int = 160,
    progress: Callable[[int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Clip:
    if stride < 1:
        raise IngestError("stride must be >= 1")
    clip_id, cdir = store.new_clip_dir(project_id)
    try:
        size: tuple[int, int] | None = None
        kept = 0
        names: dict[int, str] = {}
        for src_idx, fr in enumerate(source.frames()):
            if cancel is not None and cancel():
                raise ImportCancelled("import cancelled")
            if src_idx % stride:
                continue
            h, w = fr.image.shape[:2]
            if size is None:
                size = (w, h)
            elif (w, h) != size:
                label = fr.original_name or f"frame {src_idx}"
                raise IngestError(f"mixed frame sizes: {label} is {w}x{h}, expected {size[0]}x{size[1]}")
            dst = cdir / "frames" / f"{kept:06d}.jpg"
            if fr.src_path is not None and fr.src_path.suffix.lower() in _JPEG_SUFFIXES:
                shutil.copyfile(fr.src_path, dst)  # copy, never hardlink: store frames are immutable
            elif not cv2.imwrite(str(dst), fr.image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
                raise IngestError(f"failed to write {dst.name}")
            th = max(1, round(h * thumb_width / w))
            thumb = cv2.resize(fr.image, (thumb_width, th), interpolation=cv2.INTER_AREA)
            thumb_name = f"{kept:06d}.jpg"
            if not cv2.imwrite(str(cdir / "thumbs" / thumb_name), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80]):
                raise IngestError(f"failed to write thumbnail {thumb_name}")
            if fr.original_name:
                names[kept] = fr.original_name
            kept += 1
            if progress is not None:
                progress(kept)
        if kept == 0 or size is None:
            raise IngestError("source produced no frames")
        clip = Clip(
            id=clip_id, project_id=project_id, source=ClipSource(type=source.type, name=source.name),
            fps=float(source.fps), stride=stride, width=size[0], height=size[1], frame_count=kept,
            created_at=now_iso(), original_names=names,
        )
        store.write_clip(clip)
        return clip
    except BaseException:
        shutil.rmtree(cdir, ignore_errors=True)
        raise
