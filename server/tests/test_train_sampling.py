from __future__ import annotations

import os

import numpy as np
import pytest
from vdp.train import sampling
from vdp.train.sampling import (
    choose_val_subset,
    epoch_plan,
    epoch_slice,
    read_image_classes,
    scaled_schedule,
    write_val_subset,
)

# ---- train slices ----


def test_slices_partition_each_shuffled_cycle():
    size, n = 12, 4
    for cycle in range(3):
        epochs = range(cycle * 3, cycle * 3 + 3)
        seen = np.concatenate([epoch_slice(7, e, size, n) for e in epochs])
        assert sorted(seen.tolist()) == list(range(size))  # every image once per cycle


def test_slices_cover_the_whole_set_before_repeating_when_n_does_not_divide_it():
    size, n = 10, 4
    stream = np.concatenate([epoch_slice(3, e, size, n) for e in range(5)])  # 20 = two whole cycles
    assert sorted(stream[:10].tolist()) == list(range(size))
    assert sorted(stream[10:].tolist()) == list(range(size))
    assert all(len(epoch_slice(3, e, size, n)) == n for e in range(7))


def test_consecutive_epochs_move_to_the_next_slice():
    a, b = epoch_slice(0, 0, 100, 10), epoch_slice(0, 1, 100, 10)
    assert len(a) == len(b) == 10
    assert not set(a.tolist()) & set(b.tolist())


def test_slice_is_a_pure_function_of_seed_epoch_size_and_n():
    assert epoch_slice(5, 3, 50, 7).tolist() == epoch_slice(5, 3, 50, 7).tolist()
    assert epoch_slice(5, 3, 50, 7).tolist() != epoch_slice(6, 3, 50, 7).tolist()


def test_a_new_cycle_uses_a_different_permutation():
    first = np.concatenate([epoch_slice(1, e, 20, 5) for e in range(4)])
    second = np.concatenate([epoch_slice(1, e, 20, 5) for e in range(4, 8)])
    assert sorted(first.tolist()) == sorted(second.tolist())
    assert first.tolist() != second.tolist()


# ---- scaled epoch-denominated settings ----


def test_scaled_schedule_keeps_the_same_amount_of_data():
    s = scaled_schedule(epochs=100, train_images=28000, per_epoch=7000)
    assert s == {"warmup_epochs": 12.0, "close_mosaic": 40}


def test_scaled_schedule_never_exceeds_the_epochs():
    s = scaled_schedule(epochs=20, train_images=28000, per_epoch=2000)  # x14: warmup 42, close_mosaic 140
    assert s["close_mosaic"] == 20
    assert s["warmup_epochs"] == 6  # at most 30 % of the run's epochs
    assert scaled_schedule(epochs=5, train_images=100, per_epoch=10)["warmup_epochs"] == 3  # but at least Ultralytics' 3
    assert scaled_schedule(epochs=3, train_images=100, per_epoch=10)["warmup_epochs"] == 2  # and never the last epoch
    assert scaled_schedule(epochs=1, train_images=100, per_epoch=50)["warmup_epochs"] == 0


def test_scaled_schedule_rounds_close_mosaic_to_at_least_one():
    s = scaled_schedule(epochs=100, train_images=1000, per_epoch=999)
    assert s["close_mosaic"] == 10 and s["warmup_epochs"] == 3.0


def test_ultralytics_defaults_match_the_constants():
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import DEFAULT_CFG

    assert DEFAULT_CFG.warmup_epochs == sampling.ULTRALYTICS_WARMUP_EPOCHS
    assert DEFAULT_CFG.close_mosaic == sampling.ULTRALYTICS_CLOSE_MOSAIC


def test_epoch_plan_is_none_without_sampling_and_clamps_to_the_dataset():
    p = {"epochs": 100}
    assert epoch_plan(p, 28000, 2000) is None
    assert epoch_plan({**p, "train_images_per_epoch": 30000, "val_images_per_epoch": 2000}, 28000, 2000) is None
    plan = epoch_plan({**p, "train_images_per_epoch": 7000, "val_images_per_epoch": None}, 28000, 2000)
    assert plan == {
        "train_images": 28000, "train_images_per_epoch": 7000, "val_images": 2000, "val_images_per_epoch": None,
        "warmup_epochs": 12.0, "close_mosaic": 40,
    }
    plan = epoch_plan({**p, "val_images_per_epoch": 400}, 28000, 2000)
    assert plan == {"train_images": 28000, "train_images_per_epoch": None, "val_images": 2000,
                    "val_images_per_epoch": 400}


# ---- fixed validation subset ----


def _val_images(clips: dict[str, int], rare: dict[str, int] | None = None) -> dict[str, set[int]]:
    """`clips` = {clip id: image count}; every image has class 0, except `rare` = {name: class id} adds a class."""
    out = {f"{clip}_{i:06d}.jpg": {0} for clip, n in clips.items() for i in range(n)}
    for name, cls in (rare or {}).items():
        out[name] = out[name] | {cls}
    return out


def test_val_subset_is_fixed_seed_stable_and_of_the_asked_size():
    images = _val_images({"aaaaaaaaaaaa": 60, "bbbbbbbbbbbb": 40})
    a = choose_val_subset(images, 20, seed=3)
    assert a == choose_val_subset(images, 20, seed=3)
    assert len(a) == 20 and len(set(a)) == 20 and set(a) <= set(images)
    assert a == sorted(a)
    assert a != choose_val_subset(images, 20, seed=4)


def test_val_subset_is_proportional_per_clip():
    images = _val_images({"aaaaaaaaaaaa": 600, "bbbbbbbbbbbb": 300, "cccccccccccc": 100})
    chosen = choose_val_subset(images, 100, seed=0)
    per_clip = {c: sum(n.startswith(c) for n in chosen) for c in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc")}
    assert per_clip == {"aaaaaaaaaaaa": 60, "bbbbbbbbbbbb": 30, "cccccccccccc": 10}


def test_val_subset_covers_every_class_present():
    images = _val_images({"aaaaaaaaaaaa": 200, "bbbbbbbbbbbb": 200},
                         rare={"aaaaaaaaaaaa_000123.jpg": 1, "bbbbbbbbbbbb_000007.jpg": 2})
    images["bbbbbbbbbbbb_000050.jpg"] = set()  # a background image: no class at all
    for seed in range(5):
        chosen = choose_val_subset(images, 10, seed=seed)
        assert len(chosen) == 10
        assert set().union(*(images[n] for n in chosen)) == {0, 1, 2}


def test_val_subset_of_at_least_the_whole_set_is_all_of_it():
    images = _val_images({"aaaaaaaaaaaa": 5})
    assert choose_val_subset(images, 5, seed=0) == sorted(images)
    assert choose_val_subset(images, 50, seed=0) == sorted(images)


def test_read_image_classes_reads_the_yolo_labels(tmp_path):
    images = tmp_path / "yolo_seg" / "images" / "val"
    labels = tmp_path / "yolo_seg" / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name in ("c1_000000.jpg", "c1_000001.png", "c2_000000.jpg", "notes.txt"):
        (images / name).write_bytes(b"x")
    (labels / "c1_000000.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n2 0.1 0.1 0.2 0.1 0.2 0.2\n")
    (labels / "c1_000001.txt").write_text("")
    assert read_image_classes(images) == {"c1_000000.jpg": {0, 2}, "c1_000001.png": set(), "c2_000000.jpg": set()}


def test_write_val_subset_links_images_and_labels_into_the_run(tmp_path):
    images = tmp_path / "export" / "yolo_seg" / "images" / "val"
    labels = tmp_path / "export" / "yolo_seg" / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(4):
        (images / f"c_{i:06d}.jpg").write_bytes(b"img%d" % i)
        (labels / f"c_{i:06d}.txt").write_text(f"0 {i}\n")
    out = tmp_path / "run" / "val_subset"

    subset_images = write_val_subset(images, ["c_000001.jpg", "c_000003.jpg"], out)

    assert subset_images == out / "images" / "val"
    assert sorted(p.name for p in subset_images.iterdir()) == ["c_000001.jpg", "c_000003.jpg"]
    # Ultralytics finds a label by swapping /images/ for /labels/ in the image path: inside the run dir too
    assert (out / "labels" / "val" / "c_000003.txt").read_text() == "0 3\n"
    assert os.path.samefile(subset_images / "c_000001.jpg", images / "c_000001.jpg")

    (out / "labels" / "val.cache").write_bytes(b"cache")  # Ultralytics' label cache for the subset
    write_val_subset(images, ["c_000001.jpg", "c_000003.jpg"], out)  # the same subset again (a resume): kept
    assert (out / "labels" / "val.cache").is_file()
    write_val_subset(images, ["c_000000.jpg"], out)  # a different one: rebuilt
    assert sorted(p.name for p in subset_images.iterdir()) == ["c_000000.jpg"]
    assert not (out / "labels" / "val.cache").exists()
    assert sorted(p.name for p in out.parent.iterdir()) == ["val_subset"]  # no temp dir left behind
