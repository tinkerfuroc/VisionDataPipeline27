"""Derived export: real frames plus synthetic composites on background textures (spec B2).

Real frames are never re-labelled: their images are hard-linked in (as for any export) and their per-format
label artifacts (`yolo_seg`/`yolo_det` label files, `masks` PNGs, COCO images+annotations) are copied
verbatim from the source, for both splits. Synthetic (composite) frames are made on a process pool
(`vdp.parallel`); each worker writes its composites' images and per-format label files itself, with the
same per-frame label functions every exporter uses (`yolo_seg_lines`, `write_semantic_png`, ...), and the main
process appends their COCO images/annotations to the train JSON (everything else lives in its own per-stem
file, so "merge" is just placing more files side by side).

Lighting variants (`lighting` per real train frame, named `lit_<seed>_<stem>_<k>`) are the source frame relit as
low light or mild over-exposure (`vdp.augment.lighting`); since only pixels change, each one's labels are its
source frame's, copied under the new stem like a real frame's (COCO: the image entry and its annotations are
duplicated into the train JSON).

Cutouts for composites are cut per object from the source's COCO train annotations when the source has the
`coco` format (two touching objects of one class stay two cutouts), and per class region of the semantic `masks`
PNGs otherwise; the derived manifest records which (`augment.cutouts`). Either way they feed composites only,
never a real frame's own labels.

Composites and lighting copies are made in one pooled phase, their units interleaved in proportion. An image
limit (`max_train_images`) trims the plan up front (`vdp.augment.limits.plan_generation`) to an exact subset of
an unlimited run; a time limit (`time_limit_s` after the job starts) drops the units still queued at the deadline
and keeps what was made. The manifest's `augment.limits` records what was asked for, planned and produced.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple
from uuid import uuid4

import numpy as np
from PIL import Image

from ..export import ExportCancelled, manifest_for
from ..export.base import ExportSummary, place_image
from ..export.coco import coco_annotations
from ..export.mask_png import MaskSummary, paint, palette_for, write_semantic_png
from ..export.naming import MAX_NAME
from ..export.plan import ExportPlan, classes_from_manifest, options_from_manifest
from ..export.polygons import clean_mask, polygons_to_mask
from ..export.yolo import yolo_det_lines, yolo_seg_lines
from ..labeling.errors import BadRequest
from ..parallel import run_units
from ..train.models import find_export
from .backgrounds import POOLS, list_images
from .composite import Cutout, cutout_from, make_composite
from .cutout_pack import CutoutPackWriter, load_cutouts
from .lighting import LOW_SHARE, relight
from .limits import GenerationPlan, plan_generation

MAX_LIGHTING = 3
MAX_TRAIN_IMAGES = 10_000_000  # upper bound on `max_train_images` (keeps the export id and name short)
UNIT_STEMS = 64  # mask files per cutout work unit
UNIT_COMPOSITES = 16  # composites per work unit
UNIT_LIGHTING = 16  # source frames per lighting work unit
LIGHTING_STREAM = 1  # seeds lighting rngs as [seed, LIGHTING_STREAM, frame, k]; composites use [seed, n]

_now = time.monotonic  # the job's start time comes from here (tests swap in a fake clock)


@dataclass(frozen=True)
class AugmentOptions:
    per_image: int = 3
    max_objects: int = 2
    seed: int = 0
    pool: str = "dtd"
    lighting: int = 0  # relit copies per real train frame
    max_train_images: int | None = None  # cap on the whole train split (real + generated)
    time_limit_s: int | None = None  # stop generating this many seconds after the job starts


def _derived_name(
    source_name: str, per_image: int, lighting: int = 0, max_train_images: int | None = None,
    time_limit_s: int | None = None,
) -> str:
    """The source's display name (name, else id) plus ` ×<N> augmented` (with composites) and/or
    ` +<K> lighting` (with lighting variants), then ` ≤<N> images` (image limit) and/or ` ≤<M> min` (time limit,
    minutes rounded up), truncated so the result never exceeds `MAX_NAME`: the suffix is always kept whole, and
    the source part is shortened (trailing whitespace stripped off the cut) to make room for it."""
    suffix = (f" ×{per_image} augmented" if per_image else "") + (f" +{lighting} lighting" if lighting else "")
    if max_train_images is not None:
        suffix += f" ≤{max_train_images} images"
    if time_limit_s is not None:
        suffix += f" ≤{math.ceil(time_limit_s / 60)} min"
    base = source_name
    max_base = max(0, MAX_NAME - len(suffix))
    if len(base) > max_base:
        base = base[:max_base].rstrip()
    return f"{base}{suffix}"


def check_augment(data_root: Path, export_id: str, opts: AugmentOptions) -> dict:
    """The source manifest, once it and `opts` check out. Raises `NotFound` / `BadRequest`."""
    info = find_export(data_root, export_id)  # id pattern, existence, and the yolo_seg format
    if info.derived_from is not None:
        raise BadRequest(f"export {export_id} is already an augmented copy; augment the original export")
    if "masks" not in info.formats:
        raise BadRequest(f"export {export_id} has no masks format; re-export with Mask images ticked")
    if not 0 <= opts.per_image <= 10:
        raise BadRequest("per_image must be between 0 and 10")
    if not 0 <= opts.lighting <= MAX_LIGHTING:
        raise BadRequest(f"lighting must be between 0 and {MAX_LIGHTING}")
    if opts.per_image == 0 and opts.lighting == 0:
        raise BadRequest("ask for at least one composite or lighting variant per frame")
    if opts.per_image and opts.pool not in POOLS:
        raise BadRequest(f"background pool {opts.pool} not found")
    if opts.per_image and not list_images(data_root, opts.pool):
        raise BadRequest(f"background pool {opts.pool} is empty; download it first")
    if not 1 <= opts.max_objects <= 3:
        raise BadRequest("max_objects must be between 1 and 3")
    if opts.seed < 0:
        raise BadRequest("seed must be 0 or more")
    if opts.time_limit_s is not None and not 10 <= opts.time_limit_s <= 86400:
        raise BadRequest("time_limit_s must be between 10 and 86400")
    manifest = json.loads((info.dir / "export.json").read_text())
    real = manifest["counts"]["train"]["images"]
    if opts.max_train_images is not None and opts.max_train_images > MAX_TRAIN_IMAGES:
        raise BadRequest(f"max_train_images must be at most {MAX_TRAIN_IMAGES}")
    if opts.max_train_images is not None and opts.max_train_images <= real:
        raise BadRequest(
            f"this export already has {real} training frames; set a limit above that, or leave it empty"
        )
    if opts.per_image and manifest["counts"]["train"]["annotated"] == 0:
        raise BadRequest(f"export {export_id} has no train-split objects to build composites from")
    return manifest


CutoutMode = Literal["per_object", "per_class_region"]


def _coco_train_objects(source_dir: Path) -> dict[str, list[tuple[int, list[np.ndarray]]]]:
    """Each train image's objects, by image stem: `(export class id, polygons)`, in annotation order. The COCO
    exporter writes one annotation per object, so objects that touch or overlap stay apart. Each polygon is an
    (n, 2) float64 array (the same values as Python floats, at an eighth of their memory: a large export's
    train JSON is hundreds of MB, several GB as lists of floats)."""
    data = json.loads((source_dir / "coco" / "annotations" / "instances_train.json").read_text())
    stems = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
    out: dict[str, list[tuple[int, list[np.ndarray]]]] = {}
    for ann in sorted(data["annotations"], key=lambda a: a["id"]):
        polygons = [np.asarray(p[:len(p) // 2 * 2], dtype=np.float64).reshape(-1, 2) for p in ann["segmentation"]]
        out.setdefault(stems[ann["image_id"]], []).append((int(ann["category_id"]) - 1, polygons))
    del data
    return out


# `paint`'s canvas is uint8 (0..255): a per-object index needs one value per annotation, so a class with more
# than this many objects in one frame falls back to per-class-region for the WHOLE frame instead (not just that
# class): tracking a partial, per-class fallback inside an otherwise per-object frame would need its own code
# path, where falling back for the whole frame is simply the same path a `coco`-less export already always takes
# (so it is exercised and understood), and a frame this crowded, in even one class, is far outside normal use.
MAX_PAINT_OBJECTS = 255



def _cutout_unit(
    args: tuple[Path, list[str], bool, dict[str, list], int],
) -> tuple[list[Cutout], list[tuple[int, int]]]:
    """(source_dir, mask stems in order, per_object, COCO objects for those stems, min_area_px) -> the unit's
    cutouts and sizes, in order (the per-frame logic of `_cutouts_and_sizes`). Module-level so the process
    pool can run it."""
    source_dir, stems, per_object, by_stem, min_area_px = args
    masks_dir = source_dir / "masks" / "masks" / "train"
    cutouts: list[Cutout] = []
    sizes: list[tuple[int, int]] = []
    for stem in stems:
        with Image.open(masks_dir / f"{stem}.png") as png:
            width, height = png.size
            semantic = np.array(png)
        sizes.append((width, height))
        objects = by_stem.get(stem, [])
        counts: dict[int, int] = {}
        for cls, _polygons in objects:
            counts[cls] = counts.get(cls, 0) + 1
        if per_object and all(n <= MAX_PAINT_OBJECTS for n in counts.values()):
            found: list[tuple[int, np.ndarray] | None] = [None] * len(objects)
            for cls in counts:
                idxs = [i for i, (c, _polygons) in enumerate(objects) if c == cls]
                class_masks = [polygons_to_mask(objects[i][1], width, height) for i in idxs]
                canvas, _overlap = paint(list(enumerate(class_masks, start=1)), height, width)
                for k, i in enumerate(idxs, start=1):
                    found[i] = (cls, (canvas == k) & (semantic == cls + 1))
        else:
            found = [(int(v) - 1, semantic == v) for v in sorted(int(v) for v in np.unique(semantic) if v > 0)]
        found = [(cls, clean_mask(mask, min_area_px)) for cls, mask in found]
        found = [(cls, mask) for cls, mask in found if mask.any()]
        if not found:
            continue
        image = np.array(Image.open(source_dir / "images" / f"{stem}.jpg").convert("RGB"))
        for cls, mask in found:
            cut = cutout_from(image, mask, cls)
            if cut is not None:
                cutouts.append(cut)
    return cutouts, sizes


def _cutouts_and_sizes(
    source_dir: Path, formats: list[str], min_area_px: int, tick: Callable[[], None],
    sink: Callable[[int, list[Cutout]], None], workers: int | None = None, cancel: Callable[[], bool] = lambda: False,
) -> tuple[list[tuple[int, int]], CutoutMode]:
    """Cutouts and `(width, height)` of every real train frame (the frames are listed from the `masks` format).
    The cutouts go to `sink(unit_index, cutouts)` as each unit of frames finishes, in completion order (a large
    export's cutouts are tens of GB: they are never all held here); the cutout order is units by index, each
    unit's cutouts in order. Returns the sizes, in frame order, and the cutout mode.
    Per object from the COCO train annotations when the source has `coco`, else per class region of the indexed
    mask PNGs (touching objects of one class then come out as one). Feeds composites only.

    The per-object case resolves shared pixels one class at a time, then restores holes from the semantic mask
    PNG (which is authoritative on which class owns a pixel, since that is what the export already decided):
    COCO polygons are outer contours only (`mask_to_polygons` uses `RETR_EXTERNAL`), so `polygons_to_mask` fills
    a holed object's hole in and gives an object entirely inside another one the same footprint as its host. So,
    per class, that class's annotations' polygons are rasterised and painted onto their own canvas with `paint`
    (the same "largest first" ownership `masks` uses, so a smaller object on top of a larger one of the SAME
    class wins the shared pixels) using per-object indices, and each of that class's objects' cutout is
    `(canvas_c == its index) & (semantic_png == its class value)`: the per-class canvas settles who owns a pixel
    among that class's own objects, and the true (unfilled, unmerged) `masks` PNG says whether the pixel really
    belongs to that class at all, restoring holes and dropping the leftover pixels a filled-in hole steals from
    another class entirely (a washer's hole shows the board underneath, not a filled-in disc of washer).

    Resolving per class does NOT help a hole onto a larger object of the SAME class: the export's semantic PNG
    only records a class per pixel, so it cannot tell two same-class objects apart there, and the canvas (built
    from the same outer-contour, hole-filled polygons for both) can't either. A washer on a board of its own
    class still comes out with the board's pixels showing through its hole; this is a known limit of the COCO
    export, not something this function can fix from here.
    """
    per_object = "coco" in formats
    by_stem = _coco_train_objects(source_dir) if per_object else {}
    # sorted paths, not stems: "a-b.png" < "a.png" but "a" < "a-b" (composites index this order)
    stems = [p.stem for p in sorted((source_dir / "masks" / "masks" / "train").glob("*.png"))]
    chunks = [stems[i:i + UNIT_STEMS] for i in range(0, len(stems), UNIT_STEMS)]
    units = [(source_dir, chunk, per_object, {s: by_stem[s] for s in chunk if s in by_stem}, min_area_px)
             for chunk in chunks]

    def on_done(n: int) -> None:
        for _ in range(n):
            tick()

    del by_stem  # each unit holds its own frames' objects

    unit_sizes: dict[int, list[tuple[int, int]]] = {}

    def on_result(i: int, result: tuple[list[Cutout], list[tuple[int, int]]]) -> None:
        sink(i, result[0])
        unit_sizes[i] = result[1]

    run_units(_cutout_unit, units, workers=workers, weight=lambda u: len(u[1]), on_done=on_done,
              cancel=cancel, on_result=on_result)
    sizes = [s for i in range(len(units)) for s in unit_sizes[i]]
    return sizes, "per_object" if per_object else "per_class_region"



class SynthResult(NamedTuple):
    """What a worker reports for one composite with objects (its files are already written)."""
    n: int
    stem: str
    width: int
    height: int
    coco: list[dict]  # per-object COCO annotations, without id / image_id ([] unless the coco format is chosen)
    yolo_seg_lines: int
    yolo_det_lines: int
    objects: int
    overlap_px: int  # the masks format's overlapping pixels
    linked: bool  # every format's image was hard-linked (not copied)


# Per-process state; relies on augment jobs being exclusive (api/augment.py starts them with exclusive=EXCLUSIVE),
# so two in-process runs never overlap.
_worker: dict = {}  # a composite worker's state, set by _init_composite_worker


def _init_composite_worker(
    work_dir: Path, pool_images: list[Path], sizes: list[tuple[int, int]], opts_dict: dict, formats: list[str],
    min_area_px: int, palette: list[int], out: Path,
) -> None:
    """Pool initializer (also called in-process for one worker): memory-maps the packed cutouts and sets up a
    per-worker background cache."""

    @lru_cache(maxsize=8)  # per worker: gone with the worker (or cleared by run_augment when run in-process)
    def load_background(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    _worker.clear()
    _worker.update(
        cutouts=load_cutouts(work_dir), pool_images=pool_images, sizes=sizes, opts=AugmentOptions(**opts_dict),
        formats=formats, min_area_px=min_area_px, palette=palette, out=out, load_background=load_background,
    )


def _composite_unit(ns: list[int]) -> list[SynthResult | None]:
    """Make composites `ns` (each from its own `default_rng([seed, n])`) and write each one's image into
    `out/images/`, place it into every format's train images, and write its per-format label files. `None` for
    a composite that came out without objects (nothing is written for it)."""
    w = _worker
    opts, formats, out = w["opts"], w["formats"], w["out"]
    results: list[SynthResult | None] = []
    for n in ns:
        rng = np.random.default_rng([opts.seed, n])
        width, height = w["sizes"][int(rng.integers(len(w["sizes"])))]
        background = w["load_background"](w["pool_images"][int(rng.integers(len(w["pool_images"])))])
        composite = make_composite(
            rng, width, height, background, w["cutouts"], opts.max_objects, w["min_area_px"],
        )
        if not composite.objects:
            results.append(None)
            continue
        objects = composite.objects  # (export class id, mask), already cleaned by make_composite
        stem = f"syn_{opts.seed}_{n:06d}"
        image_path = out / "images" / f"{stem}.jpg"
        Image.fromarray(composite.image).save(image_path, quality=90)
        linked = True
        for fmt in formats:
            linked = place_image(image_path, out / fmt / "images" / "train" / f"{stem}.jpg") and linked
        seg = yolo_seg_lines(objects, width, height) if "yolo_seg" in formats else []
        det = yolo_det_lines(objects, width, height) if "yolo_det" in formats else []
        for fmt, lines in (("yolo_seg", seg), ("yolo_det", det)):
            if fmt in formats:  # an empty file would be YOLO's "background image"; never the case here
                (out / fmt / "labels" / "train" / f"{stem}.txt").write_text("".join(f"{ln}\n" for ln in lines))
        overlap = 0
        if "masks" in formats:
            overlap = write_semantic_png(
                objects, height, width, w["palette"], out / "masks" / "masks" / "train" / f"{stem}.png",
            )
        coco = coco_annotations(objects) if "coco" in formats else []
        results.append(SynthResult(n, stem, width, height, coco, len(seg), len(det), len(objects), overlap, linked))
    return results


def _make_format_dirs(out: Path, formats: list[str]) -> None:
    """The directories every format's exporter creates, so a split without frames still has its (empty) dirs."""
    (out / "images").mkdir()
    sub = {"yolo_seg": ("images", "labels"), "yolo_det": ("images", "labels"), "masks": ("images", "masks"),
           "coco": ("images",)}
    for fmt in formats:
        for kind in sub[fmt]:
            for split in ("train", "val"):
                (out / fmt / kind / split).mkdir(parents=True, exist_ok=True)
        if fmt == "coco":
            (out / fmt / "annotations").mkdir(parents=True, exist_ok=True)


def _synth_summaries(formats: list[str], synthetic: list[SynthResult]) -> list[ExportSummary]:
    """The composites' per-format summaries, counted as each exporter counts its frames."""
    linked = all(s.linked for s in synthetic)
    images = {"train": len(synthetic), "val": 0}
    counted = {
        "yolo_seg": lambda s: s.yolo_seg_lines, "yolo_det": lambda s: s.yolo_det_lines,
        "masks": lambda s: s.objects, "coco": lambda s: len(s.coco),
    }
    summaries: list[ExportSummary] = []
    for fmt in formats:
        annotations = {"train": sum(counted[fmt](s) for s in synthetic), "val": 0}
        if fmt == "masks":
            summaries.append(MaskSummary(fmt, dict(images), annotations, linked,
                                         sum(s.overlap_px for s in synthetic)))
        else:
            summaries.append(ExportSummary(fmt, dict(images), annotations, linked))
    return summaries


def _copy_label(
    source_dir: Path, out: Path, fmt: str, split: str, stem: str, new_stem: str | None = None,
) -> None:
    """Copy one frame's `fmt` label artifact verbatim, under `new_stem` when given (a lighting variant)."""
    if fmt in ("yolo_seg", "yolo_det"):
        rel, suffix = Path(fmt) / "labels" / split, ".txt"
    elif fmt == "masks":
        rel, suffix = Path("masks") / "masks" / split, ".png"
    else:
        return  # coco: images only here; its annotations are merged as one JSON in _merge_coco
    dst = out / rel / f"{new_stem or stem}{suffix}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / rel / f"{stem}{suffix}", dst)


def _merge_real_frames(source_dir: Path, out: Path, formats: list[str], tick: Callable[[], None]) -> bool:
    """Hard-link every real frame's image (both splits, into `out/images/` and each format's own images
    dir) and copy its per-format label artifacts verbatim from the source."""
    linked = True
    for split in ("train", "val"):
        stems = sorted(p.stem for p in (source_dir / "masks" / "masks" / split).glob("*.png"))
        for stem in stems:
            tick()
            src_image = source_dir / "images" / f"{stem}.jpg"
            linked = place_image(src_image, out / "images" / f"{stem}.jpg") and linked
            for fmt in formats:
                linked = place_image(src_image, out / fmt / "images" / split / f"{stem}.jpg") and linked
                _copy_label(source_dir, out, fmt, split, stem)
    return linked



def _lighting_unit(items: list[tuple[int, str, int]]) -> list[tuple[str, str]]:
    """Relit copies `(frame index i in the full sorted list, stem, copy k)`, in order (a frame's copies are
    adjacent, so its image is read once) -> the unit's `(stem, new_stem)` pairs, in order. Each copy is written
    into `out/images/`, placed into each format's train images, and gets its source frame's per-format labels
    under the new stem (COCO's are added in `_merge_coco`). Reads `source_dir`, `out`, `formats` and `seed`
    from the worker state set by `_init_generation_worker`."""
    w = _worker
    source_dir, out, formats, seed = w["source_dir"], w["out"], w["formats"], w["seed"]
    pairs: list[tuple[str, str]] = []
    loaded: tuple[int, np.ndarray] | None = None
    for i, stem, k in items:
        if loaded is None or loaded[0] != i:
            loaded = (i, np.array(Image.open(source_dir / "images" / f"{stem}.jpg").convert("RGB")))
        rng = np.random.default_rng([seed, LIGHTING_STREAM, i, k])
        kind = "low" if rng.random() < LOW_SHARE else "over"
        new_stem = f"lit_{seed}_{stem}_{k}"
        image_path = out / "images" / f"{new_stem}.jpg"
        Image.fromarray(relight(loaded[1], rng, kind)).save(image_path, quality=95)
        for fmt in formats:
            place_image(image_path, out / fmt / "images" / "train" / f"{new_stem}.jpg")
            _copy_label(source_dir, out, fmt, "train", stem, new_stem)
        pairs.append((stem, new_stem))
    return pairs


def _init_generation_worker(
    composite_initargs: tuple | None, source_dir: Path, out: Path, formats: list[str], seed: int,
) -> None:
    """Pool initializer for the generation phase (also called in-process for one worker): the composite worker
    state when composites are planned, plus what lighting units read."""
    _worker.clear()
    if composite_initargs is not None:
        _init_composite_worker(*composite_initargs)
    _worker.update(source_dir=source_dir, out=out, formats=formats, seed=seed)


GenerationUnit = tuple[Literal["composite", "lighting"], list]


def _generation_unit(unit: GenerationUnit) -> list:
    """One composite unit (`("composite", [n, ...])`) or lighting unit (`("lighting", [(i, stem, k), ...])`)."""
    kind, items = unit
    return _composite_unit(items) if kind == "composite" else _lighting_unit(items)


def _interleave(a: list, b: list) -> list:
    """`a` and `b` merged in proportion to their lengths (Bresenham style), each keeping its own order: item
    `j` of a list lands where `(j + 1/2) / len` of it would, so `a` = 3 units and `b` = 2 give a b a b a."""
    merged, i, j = [], 0, 0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and (2 * i + 1) * len(b) <= (2 * j + 1) * len(a)):
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    return merged


def _generation_units(plan: GenerationPlan, stems: list[str]) -> list[GenerationUnit]:
    """The plan's composites (chunks of UNIT_COMPOSITES, in `n` order) and lighting copies (UNIT_LIGHTING source
    frames per unit) interleaved in proportion. The planner lists lighting copies in selection order (round by
    round); they are written in `(i, k)` order -- each frame's copies together, frames ascending -- which is the
    order an unlimited run has always used, so a limited run is an exact subset and an unlimited one unchanged."""
    ns = plan.composites
    composite = [("composite", ns[j:j + UNIT_COMPOSITES]) for j in range(0, len(ns), UNIT_COMPOSITES)]
    by_frame: dict[int, list[tuple[int, str, int]]] = {}
    for i, k in sorted(plan.lighting):
        by_frame.setdefault(i, []).append((i, stems[i], k))
    frames = list(by_frame.values())
    lighting = [("lighting", [item for frame in frames[j:j + UNIT_LIGHTING] for item in frame])
                for j in range(0, len(frames), UNIT_LIGHTING)]
    return _interleave(composite, lighting)


def _lighting_annotations(source_dir: Path, fmt: str, pairs: list[tuple[str, str]]) -> int:
    """How many annotations the lighting copies add to `fmt`'s train split, counted as its exporter counts
    them: label lines for YOLO, objects for `masks` (taken from the YOLO-det boxes, one per object, else the
    YOLO-seg lines, which count an object once per polygon). COCO's count comes from `_merge_coco`."""
    if fmt == "masks":
        fmt = "yolo_det" if (source_dir / "yolo_det").is_dir() else "yolo_seg"
    labels = source_dir / fmt / "labels" / "train"
    return sum(len((labels / f"{stem}.txt").read_text().splitlines()) for stem, _new in pairs)


def _merge_coco(
    source_dir: Path, out: Path, lighting: list[tuple[str, str]], synthetic: list[SynthResult],
) -> int:
    """`instances_train.json` = the source's train JSON, then a copy of each lighting variant's source image
    entry and annotations, then the composites' images/annotations in `n` order (ids continuing past the
    source's own); `instances_val.json` = the source's, copied verbatim. Returns the lighting copies'
    annotation count."""
    annotations_dir = Path("coco") / "annotations"
    src_train = json.loads((source_dir / annotations_dir / "instances_train.json").read_text())

    next_image_id = max((im["id"] for im in src_train["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in src_train["annotations"]), default=0) + 1
    images = list(src_train["images"])
    annotations = list(src_train["annotations"])

    by_stem = {Path(im["file_name"]).stem: im for im in src_train["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in sorted(src_train["annotations"], key=lambda a: a["id"]):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    lighting_annotations = 0
    for stem, new_stem in lighting:
        im = by_stem[stem]
        file_name = str(Path(im["file_name"]).with_name(f"{new_stem}.jpg"))
        images.append({**im, "id": next_image_id, "file_name": file_name})
        for ann in anns_by_image.get(im["id"], []):
            annotations.append({**ann, "id": next_ann_id, "image_id": next_image_id})
            next_ann_id += 1
            lighting_annotations += 1
        next_image_id += 1

    for s in synthetic:
        images.append({"id": next_image_id, "file_name": f"{s.stem}.jpg", "width": s.width, "height": s.height})
        for ann in s.coco:
            annotations.append({"id": next_ann_id, "image_id": next_image_id, **ann})
            next_ann_id += 1
        next_image_id += 1

    train_json = {**src_train, "images": images, "annotations": annotations}
    (out / annotations_dir / "instances_train.json").write_text(json.dumps(train_json))
    val_name = annotations_dir / "instances_val.json"
    shutil.copy2(source_dir / val_name, out / val_name)
    return lighting_annotations


def _merged_format_summaries(
    src_formats: list[dict], synth_summaries: list[ExportSummary], real_linked: bool,
    lighting: dict[str, tuple[int, int]] | None = None,
) -> list[dict]:
    """Per format: the source's counts plus the composites' plus the lighting copies' (`lighting`: format ->
    (train images, train annotations)). `overlap_px` sums the source's and the composites' only: a frame's own
    overlap is not recorded, so the lighting copies' (a repeat of their source frames') is not added."""
    lighting = lighting or {}
    synth_by_name = {s.name: asdict(s) for s in synth_summaries}
    empty = {"images": {"train": 0, "val": 0}, "annotations": {"train": 0, "val": 0}, "linked": True}
    merged = []
    for src in src_formats:
        synth = synth_by_name.get(src["name"], empty)
        lit_images, lit_annotations = lighting.get(src["name"], (0, 0))
        extra_images = {"train": lit_images, "val": 0}
        extra_annotations = {"train": lit_annotations, "val": 0}
        entry = {
            "name": src["name"],
            "images": {sp: src["images"][sp] + synth["images"][sp] + extra_images[sp] for sp in ("train", "val")},
            "annotations": {
                sp: src["annotations"][sp] + synth["annotations"][sp] + extra_annotations[sp]
                for sp in ("train", "val")
            },
            "linked": bool(src["linked"]) and bool(synth["linked"]) and real_linked,
        }
        if "overlap_px" in src or "overlap_px" in synth:  # the masks format; an older source counts as 0
            entry["overlap_px"] = int(src.get("overlap_px", 0)) + int(synth.get("overlap_px", 0))
        merged.append(entry)
    return merged


def run_augment(
    data_root: Path, export_id: str, opts: AugmentOptions,
    progress: Callable[[int, int], None] | None = None, cancel: Callable[[], bool] | None = None,
    workers: int | None = None,
) -> dict:
    """Derive an augmented copy of export `export_id`. Cutouts are extracted first (always to the end), then
    one pooled generation phase makes the planned composites and lighting copies, interleaved, until done or
    `opts.time_limit_s` after the job started (whatever was made by then is kept); real frames, COCO and the
    manifest follow. `opts.max_train_images` caps the plan up front (see `plan_generation`)."""
    start = _now()
    deadline = start + opts.time_limit_s if opts.time_limit_s is not None else None
    data_root = Path(data_root)
    manifest = check_augment(data_root, export_id, opts)
    source_dir = data_root / "exports" / export_id
    tags = ([f"aug{opts.per_image}x"] if opts.per_image else []) + ([f"lit{opts.lighting}"] if opts.lighting else [])
    tags += [f"cap{opts.max_train_images}"] if opts.max_train_images is not None else []
    tags += [f"t{opts.time_limit_s}"] if opts.time_limit_s is not None else []
    new_id = f"{export_id}-{'-'.join(tags)}-{uuid4().hex[:4]}"
    out = data_root / "exports" / new_id
    out.mkdir()  # outside the try: as run_export, fail without touching (or later deleting) another export
    try:
        formats = [f["name"] for f in manifest["formats"]]
        min_area_px = int(manifest["options"]["min_area_px"])
        real_train_n = manifest["counts"]["train"]["images"]
        real_val_n = manifest["counts"]["val"]["images"]
        gen = plan_generation(real_train_n, opts.per_image, opts.lighting, opts.max_train_images)
        # computed once, up front, from the source manifest and the plan; never shrunk or grown as the job runs
        merge_ticks = real_train_n + real_val_n
        cutout_ticks = real_train_n if opts.per_image else 0
        total = cutout_ticks + len(gen.composites) + merge_ticks + len(gen.lighting)
        state = {"done": 0, "total": max(1, total)}

        def tick() -> None:
            if cancel is not None and cancel():
                raise ExportCancelled
            state["done"] += 1
            if progress is not None:
                progress(min(state["done"], state["total"]), state["total"])

        def on_done(k: int) -> None:
            for _ in range(k):
                tick()

        classes, class_map = classes_from_manifest(manifest)
        _make_format_dirs(out, formats)
        work = out / ".work"
        work.mkdir()
        cutout_mode: CutoutMode | None = None
        composite_initargs = None
        if opts.per_image:  # no deadline: composites need every cutout, and the phase is short
            # packed straight to disk for the composite workers to memory-map; an image limit can leave no
            # composites, and then no worker needs the cutouts (or backgrounds)
            with contextlib.ExitStack() as stack:
                sink = stack.enter_context(CutoutPackWriter(work)).add if gen.composites else lambda i, c: None
                sizes, cutout_mode = _cutouts_and_sizes(
                    source_dir, formats, min_area_px, tick, sink, workers, cancel or (lambda: False),
                )
            if gen.composites:
                composite_initargs = (work, list_images(data_root, opts.pool), sizes, asdict(opts), formats,
                                      min_area_px, palette_for(classes), out)

        train_stems = sorted(p.stem for p in (source_dir / "masks" / "masks" / "train").glob("*.png"))
        units = _generation_units(gen, train_stems)
        try:
            results = run_units(
                _generation_unit, units, workers=workers, weight=lambda u: len(u[1]), on_done=on_done,
                cancel=cancel or (lambda: False), initializer=_init_generation_worker,
                initargs=(composite_initargs, source_dir, out, formats, opts.seed), deadline=deadline,
            )
        finally:
            _worker.clear()  # a one-worker run initialised this process: drop its memmap and cache
        timed_out = any(r is None for r in results)
        # units, and the items within them, are in n order (composites) and (i, k) order (lighting)
        done = [(unit[0], r) for unit, r in zip(units, results, strict=True) if r is not None]
        synthetic = [s for kind, r in done if kind == "composite" for s in r if s is not None]
        lit = [pair for kind, r in done if kind == "lighting" for pair in r]

        options = options_from_manifest(manifest["options"])
        plan = ExportPlan(  # for manifest_for only: no frames (real ones are copied, composites written above)
            project_id=manifest["project"]["id"], project_name=manifest["project"]["name"], classes=classes,
            class_map=class_map, options=options, clip_ids=tuple(c["id"] for c in manifest["clips"]),
            clip_names={c["id"]: c["name"] for c in manifest["clips"]}, val_clip_ids=options.val_clip_ids or (),
            frames=(),
        )
        synth_summaries = _synth_summaries(formats, synthetic)
        synth_linked = all(s.linked for s in synth_summaries)

        real_linked = _merge_real_frames(source_dir, out, formats, tick)
        lit_counts = {
            fmt: (len(lit), _lighting_annotations(source_dir, fmt, lit)) for fmt in formats if fmt != "coco"
        }
        if "coco" in formats:
            lit_counts["coco"] = (len(lit), _merge_coco(source_dir, out, lit, synthetic))
        yolo_labels = source_dir / "yolo_seg" / "labels" / "train"
        lit_annotated = sum(1 for stem, _new in lit if (yolo_labels / f"{stem}.txt").read_text().strip())
        for fmt in ("yolo_seg", "yolo_det"):
            if fmt in formats:
                shutil.copy2(source_dir / fmt / "data.yaml", out / fmt / "data.yaml")

        counts = {
            "train": {
                "images": real_train_n + len(synthetic) + len(lit),
                "annotated": manifest["counts"]["train"]["annotated"] + len(synthetic) + lit_annotated,
                "empty": manifest["counts"]["train"]["empty"] + len(lit) - lit_annotated,
            },
            "val": manifest["counts"]["val"],
        }
        limits = {
            "max_train_images": opts.max_train_images, "time_limit_s": opts.time_limit_s,
            "requested": {"composites": gen.requested_composites, "lighting": gen.requested_lighting},
            "planned": {"composites": len(gen.composites), "lighting": len(gen.lighting)},
            "produced": {"composites": len(synthetic), "lighting": len(lit)},
            "stopped_by": "time" if timed_out else "images" if gen.capped else None,
        }
        linked = synth_linked and real_linked
        result_manifest = manifest_for(plan, new_id, synth_summaries, counts, linked)
        result_manifest["clips"] = manifest["clips"]
        result_manifest["regions_applied"] = manifest.get("regions_applied", 0)
        result_manifest["formats"] = _merged_format_summaries(
            manifest["formats"], synth_summaries, real_linked, lit_counts,
        )
        result_manifest["counts"] = counts
        result_manifest["derived_from"] = export_id
        result_manifest["name"] = _derived_name(
            manifest.get("name") or export_id, opts.per_image, opts.lighting, opts.max_train_images,
            opts.time_limit_s,
        )
        result_manifest["augment"] = {
            "per_image": opts.per_image, "max_objects": opts.max_objects, "seed": opts.seed,
            "pool": opts.pool if opts.per_image else None, "synthetic_images": len(synthetic),
            "cutouts": cutout_mode, "lighting_variants": opts.lighting, "lighting_images": len(lit),
            "limits": limits,
        }
        (out / "export.json").write_text(json.dumps(result_manifest, indent=2))
        shutil.rmtree(work)
        if timed_out and progress is not None:
            progress(state["total"], state["total"])  # the dropped units' ticks never came: complete the bar
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {
        "export_id": new_id, "path": str(out), "counts": counts, "synthetic": len(synthetic),
        "name": result_manifest["name"], "cutouts": cutout_mode, "limits": limits,
    }
