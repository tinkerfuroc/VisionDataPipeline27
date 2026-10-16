"""Sampled epochs: train on a slice of the train set per epoch, validate on a fixed subset of the val set.

Shorter epochs make validation, best-checkpoint selection and early stopping happen more often. Pure helpers only
(numpy, files): the server works out a run's plan when it creates the run (`epoch_plan`, stored as run.json's
`sampling`), and the training worker applies it (`vdp.train.sampled_trainer`).

- Train: cycle c shuffles the whole train set (seed, c) and hands it out in consecutive slices of N, one per epoch, so
  every image is seen once per cycle before any repeats. Epoch e's slice is images [e*N, (e+1)*N) of the endless
  stream of cycle permutations: a pure function of (seed, epoch, dataset size, N).
- The epoch-denominated settings that measure "how much data" (warmup, close_mosaic) are scaled by (train size / N)
  so they cover as many images as with full epochs; patience stays in (short) epochs, which is the point.
- Val: one fixed, seeded subset, proportional per clip and with every class of the val split in at least one image,
  so per-epoch scores stay comparable. The final evaluation of best.pt still uses the whole val split.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

# Ultralytics' own defaults (cfg/default.yaml), which vdp never overrides; kept here so the server never imports it.
ULTRALYTICS_WARMUP_EPOCHS = 3.0
ULTRALYTICS_CLOSE_MOSAIC = 10
# A scaled warmup never takes more than this share of a run's epochs (but always at least Ultralytics' own 3): short
# sampled runs would otherwise warm up for almost all of their epochs.
MAX_WARMUP_SHARE = 0.3
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
VAL_SUBSET_DIR = "val_subset"  # in the run dir: images/val and labels/val, linked from the export


def cycle_permutation(seed: int, cycle: int, size: int) -> np.ndarray:
    return np.random.default_rng([seed, cycle]).permutation(size)


def epoch_slice(seed: int, epoch: int, size: int, per_epoch: int) -> np.ndarray:
    """Dataset indices of epoch `epoch`'s slice: `per_epoch` of them (one slice may run over into the next cycle)."""
    parts = []
    pos, end = epoch * per_epoch, (epoch + 1) * per_epoch
    while pos < end:
        cycle, offset = divmod(pos, size)
        take = min(end - pos, size - offset)
        parts.append(cycle_permutation(seed, cycle, size)[offset:offset + take])
        pos += take
    return np.concatenate(parts)


def scaled_schedule(epochs: int, train_images: int, per_epoch: int) -> dict:
    """Ultralytics' warmup_epochs and close_mosaic, scaled to cover the same number of images with epochs of
    `per_epoch`. Warmup is capped at MAX_WARMUP_SHARE of the epochs (at least Ultralytics' 3) and at epochs - 1
    (Ultralytics' own cap, so this is what it will do), close_mosaic at the epochs (as with full epochs and a run
    shorter than 10 epochs, mosaic is then off from the start)."""
    factor = train_images / per_epoch
    cap = max(ULTRALYTICS_WARMUP_EPOCHS, round(MAX_WARMUP_SHARE * epochs, 2))
    warmup = min(round(ULTRALYTICS_WARMUP_EPOCHS * factor, 2), cap, max(epochs - 1, 0))
    close_mosaic = min(max(1, round(ULTRALYTICS_CLOSE_MOSAIC * factor)), epochs)
    return {"warmup_epochs": float(warmup), "close_mosaic": int(close_mosaic)}


def epoch_plan(params: dict, train_images: int, val_images: int) -> dict | None:
    """What a run's `train_images_per_epoch` / `val_images_per_epoch` params mean for its export: None when neither
    applies (unset, or at least the split's size: all of it, as today), else the effective sizes (None = all) plus,
    for sampled training, the scaled warmup_epochs and close_mosaic."""
    n = params.get("train_images_per_epoch")
    m = params.get("val_images_per_epoch")
    n = n if n is not None and n < train_images else None
    m = m if m is not None and m < val_images else None
    if n is None and m is None:
        return None
    plan = {"train_images": train_images, "train_images_per_epoch": n, "val_images": val_images,
            "val_images_per_epoch": m}
    if n is not None:
        plan.update(scaled_schedule(params["epochs"], train_images, n))
    return plan


def clip_of(name: str) -> str:
    """An export image's clip id: its stem is `<clip_id>_<frame>` (vdp.export.plan._stems)."""
    return Path(name).stem.split("_", 1)[0]


def _largest_remainder(total: int, sizes: dict[str, int]) -> dict[str, int]:
    count = sum(sizes.values())
    exact = {k: total * v / count for k, v in sizes.items()}
    quota = {k: int(x) for k, x in exact.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(exact[k] - quota[k]), -sizes[k], k))
    for k in by_remainder[: total - sum(quota.values())]:
        quota[k] += 1
    return quota


def choose_val_subset(images: dict[str, set[int]], size: int, seed: int) -> list[str]:
    """`size` image names out of `images` ({name: class ids in its labels}), sorted: per clip in proportion to the
    clip's share of the val split, and with every class that occurs at all in at least one chosen image (that may
    take more than `size` images only when `size` is smaller than the number of classes). All of them when `size`
    is at least their number. Deterministic for a seed."""
    names = sorted(images)
    if size >= len(names):
        return names
    rng = np.random.default_rng(seed)
    by_clip: dict[str, list[str]] = {}
    for name in names:
        by_clip.setdefault(clip_of(name), []).append(name)
    # 1. one image per class, rarest class first, unless an image picked for another class already has it
    forced: set[str] = set()
    covered: set[int] = set()
    classes = sorted({c for cs in images.values() for c in cs})
    holders = {c: [n for n in names if c in images[n]] for c in classes}
    for c in sorted(classes, key=lambda c: (len(holders[c]), c)):
        if c not in covered:
            pick = holders[c][int(rng.integers(len(holders[c])))]
            forced.add(pick)
            covered |= images[pick]
    # 2. per-clip quotas; a clip's forced images count towards its quota, taking room from the clips with most slack
    quota = _largest_remainder(size, {k: len(v) for k, v in by_clip.items()})
    forced_in = {k: sum(n in forced for n in v) for k, v in by_clip.items()}
    quota = {k: max(q, forced_in[k]) for k, q in quota.items()}
    for _ in range(sum(quota.values()) - size):
        slack = max(by_clip, key=lambda k: (quota[k] - forced_in[k], quota[k], k))
        if quota[slack] == forced_in[slack]:
            break  # only forced images left: more classes than `size`
        quota[slack] -= 1
    # 3. fill each clip's quota with its other images in a seeded random order
    chosen = set(forced)
    for clip in sorted(by_clip):
        rest = [n for n in by_clip[clip] if n not in forced]
        order = rng.permutation(len(rest))
        chosen.update(rest[i] for i in order[: quota[clip] - forced_in[clip]])
    return sorted(chosen)


def label_path(image: Path) -> Path:
    """Where Ultralytics looks for an image's label (`img2label_paths`): the last /images/ becomes /labels/."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(str(image).rsplit(sa, 1))).with_suffix(".txt")


def read_image_classes(images_dir: Path) -> dict[str, set[int]]:
    """{image name: the class ids in its YOLO label file} for every image in `images_dir` (no label = no class)."""
    out = {}
    for image in sorted(Path(images_dir).iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            lines = label_path(image).read_text().splitlines()
        except FileNotFoundError:
            lines = []
        out[image.name] = {int(float(parts[0])) for parts in (line.split() for line in lines) if parts}
    return out


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:  # another filesystem, or links not allowed: a symlink (Ultralytics reads through it)
        os.symlink(src.resolve(), dst)


def write_val_subset(images_dir: Path, names: list[str], out: Path) -> Path:
    """Link the chosen val images (and their labels) into `out/images/val` and `out/labels/val`, and return the
    images dir, for the run's data.yaml `val:`. Ultralytics keeps a split's label cache next to its labels dir, so the
    subset's cache lands in `out/labels/val.cache` -- in the run dir, never over the export's own (shared) val.cache.
    An `out` that already holds exactly these images is kept (with its cache), otherwise it is rebuilt."""
    images_dir, out = Path(images_dir), Path(out)
    subset_images = out / "images" / images_dir.name
    if subset_images.is_dir() and sorted(p.name for p in subset_images.iterdir()) == sorted(names):
        return subset_images
    tmp = out.with_name(f".{out.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp_images = tmp / "images" / images_dir.name
    tmp_labels = tmp / "labels" / images_dir.name
    tmp_images.mkdir(parents=True)
    tmp_labels.mkdir(parents=True)
    for name in names:
        _link(images_dir / name, tmp_images / name)
        label = label_path(images_dir / name)
        if label.is_file():
            _link(label, tmp_labels / label.name)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return subset_images
