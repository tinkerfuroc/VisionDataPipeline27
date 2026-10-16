from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.utils.data
from vdp.train.sampled_trainer import EpochSliceSampler, SliceFeed, sampled_trainer, use_sampler
from vdp.train.sampling import epoch_slice


def test_sampler_serves_consecutive_slices_from_the_epoch_it_was_set_to():
    s = EpochSliceSampler(size=30, per_epoch=8, seed=2)
    assert len(s) == 8
    assert list(s) == epoch_slice(2, 0, 30, 8).tolist()
    assert list(s) == epoch_slice(2, 1, 30, 8).tolist()
    s.set_epoch(5)  # e.g. resuming at epoch 5
    assert list(s) == epoch_slice(2, 5, 30, 8).tolist()
    assert s.served == [0, 1, 5]


class _Indices(torch.utils.data.Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def _ultralytics_loader(size: int, batch: int, workers: int):
    pytest.importorskip("ultralytics")
    from ultralytics.data.build import build_dataloader

    return build_dataloader(_Indices(size), batch=batch, workers=workers, shuffle=True, rank=-1, device="cpu")


def _run_epochs(loader, feed: SliceFeed, epochs: range, *, close_at: int | None = None, repeat: int | None = None):
    """What Ultralytics' `_do_train` does with the train loader, epoch by epoch: on_train_epoch_start, the
    close-mosaic loader reset (in its last epochs), one pass of len(loader) batches, on_train_epoch_end; `repeat`
    replays that epoch once, as its NaN recovery does. Returns the indices each epoch trained on."""
    seen: dict[int, list[int]] = {}
    queue = list(epochs)
    repeated = False
    while queue:
        epoch = queue[0]
        feed.start_epoch(epoch)
        if epoch == close_at:
            feed.restarted(epoch)  # SampledTrainer._close_dataloader_mosaic, then Ultralytics' loader.reset()
            loader.reset()
        seen[epoch] = [int(i) for b in loader for i in b]
        feed.end_epoch()
        if epoch == repeat and not repeated:
            repeated = True  # the epoch runs again with the same number
            continue
        queue.pop(0)
    return seen


@pytest.mark.parametrize("workers", [0, 2])
def test_each_epoch_of_the_real_loader_trains_on_its_own_slice(workers):
    """Ultralytics' InfiniteDataLoader keeps one iterator and (with workers) prefetches several epochs ahead when the
    epochs are this short; each epoch must still train on exactly its own slice, also across the close-mosaic reset,
    a NaN-recovery replay and a resume at a later epoch."""
    size, n, batch = 40, 8, 4
    loader = _ultralytics_loader(size, batch, workers)
    sampler = EpochSliceSampler(size, n, seed=3)
    use_sampler(loader, sampler)
    assert len(loader) == math.ceil(n / batch) and len(loader.sampler) == n
    feed = SliceFeed(loader, sampler, 0)
    try:
        seen = _run_epochs(loader, feed, range(9), close_at=6, repeat=3)
        assert seen == {e: epoch_slice(3, e, size, n).tolist() for e in range(9)}
        assert seen[0] != seen[1]
        resumed = _run_epochs(loader, feed, range(12, 15))  # e.g. a fresh loader resumed at epoch 12
        assert resumed == {e: epoch_slice(3, e, size, n).tolist() for e in range(12, 15)}
    finally:
        loader.close()


def test_a_partial_last_batch_still_ends_the_epoch_on_its_slice():
    size, n, batch = 25, 10, 4  # 3 batches of 4, 4, 2
    loader = _ultralytics_loader(size, batch, workers=2)
    sampler = EpochSliceSampler(size, n, seed=0)
    use_sampler(loader, sampler)
    feed = SliceFeed(loader, sampler, 0)
    try:
        assert len(loader) == 3
        seen = _run_epochs(loader, feed, range(6))
        assert seen == {e: epoch_slice(0, e, size, n).tolist() for e in range(6)}
    finally:
        loader.close()


def test_final_eval_validates_best_on_the_full_val_split(tmp_path, monkeypatch):
    """The per-epoch validator reads the run's val subset; the final evaluation of best.pt swaps in a loader over the
    whole val split, and its metrics are written to final_val.json."""
    pytest.importorskip("ultralytics")
    from ultralytics.engine.trainer import BaseTrainer

    seen = {}

    def fake_final_eval(self):  # Ultralytics' own: validates best.pt with self.validator's loader
        seen["dataloader"] = self.validator.dataloader
        self.metrics = {"metrics/mAP50-95(M)": 0.5, "metrics/mAP50(M)": 0.75}

    monkeypatch.setattr(BaseTrainer, "final_eval", fake_final_eval)
    cls = sampled_trainer(train_images_per_epoch=None, seed=0, full_val="/export/yolo_seg/images/val",
                          final_val_path=tmp_path / "final_val.json")
    trainer = cls.__new__(cls)  # no Ultralytics setup: only what final_eval touches
    (tmp_path / "best.pt").write_bytes(b"best")
    full = SimpleNamespace(dataset=list(range(7)), close=lambda: seen.setdefault("closed", True))
    built = []

    def get_dataloader(path, batch_size=16, rank=0, mode="train"):
        built.append((path, batch_size, rank, mode))
        return full

    subset_loader = object()
    trainer.__dict__.update(best=Path(tmp_path / "best.pt"), batch_size=4, world_size=1,
                            validator=SimpleNamespace(dataloader=subset_loader), get_dataloader=get_dataloader)

    trainer.final_eval()

    assert built == [("/export/yolo_seg/images/val", 8, -1, "val")]
    assert seen["dataloader"] is full and seen["closed"]
    assert json.loads((tmp_path / "final_val.json").read_text()) == {
        "images": 7, "metrics": {"metrics/mAP50-95(M)": 0.5, "metrics/mAP50(M)": 0.75},
    }


def test_real_cpu_training_with_sampled_epochs(tmp_path, monkeypatch):
    """A real (tiny, CPU) Ultralytics run with the worker's data prep and SampledTrainer: each epoch is
    ceil(N / batch) iterations over its own slice (consecutive epochs see different images, also across the
    close-mosaic loader reset), every per-epoch validation reads the fixed val subset (whose label cache stays in the
    run dir), and the final evaluation of best.pt reads the whole val split."""
    pytest.importorskip("ultralytics")
    import numpy as np
    from train_fixtures import make_export
    from ultralytics import YOLO
    from ultralytics.data import base as ul_base
    from ultralytics.data import utils as ul_utils
    from vdp.train import worker
    from vdp.train.sampling import epoch_plan

    monkeypatch.setattr(ul_utils, "check_font", lambda *a, **k: None)  # no font download
    monkeypatch.chdir(tmp_path)
    export = make_export(tmp_path / "data", n_train=12, n_val=6)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    params = {"epochs": 3, "imgsz": 64, "batch": 2, "seed": 4, "train_images_per_epoch": 4, "val_images_per_epoch": 3}
    run = {"export_dir": str(export), "params": params, "sampling": epoch_plan(params, 12, 6)}
    data, full_val = worker.prepare_data(run_dir, run, rewrite=True)

    trained: dict[int, list[str]] = {}
    current = {"epoch": None}
    original = ul_base.BaseDataset.__getitem__

    def recording_getitem(self, index):
        if self.augment:  # the train dataset (validation does not augment)
            trained.setdefault(current["epoch"], []).append(Path(self.im_files[index]).name)
        return original(self, index)

    monkeypatch.setattr(ul_base.BaseDataset, "__getitem__", recording_getitem)
    batches: dict[int, int] = {}
    loader_len: dict[int, int] = {}
    val_sizes: list[int] = []
    model = YOLO("yolo26n-seg.yaml")  # built from its config: nothing to download

    def epoch_start(t):
        current["epoch"] = t.epoch
        loader_len[t.epoch] = len(t.train_loader)

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", lambda t: batches.__setitem__(t.epoch, batches.get(t.epoch, 0) + 1))
    model.add_callback("on_val_start", lambda v: val_sizes.append(len(v.dataloader.dataset)))
    trainer_cls = sampled_trainer(train_images_per_epoch=4, seed=4, full_val=full_val,
                                  final_val_path=run_dir / "final_val.json")

    model.train(trainer=trainer_cls, data=str(data), epochs=3, imgsz=64, batch=2, workers=0, device="cpu",
                seed=4, plots=False, amp=False, project=str(run_dir), name="train", exist_ok=True,
                warmup_epochs=run["sampling"]["warmup_epochs"], close_mosaic=2, verbose=False)

    im_files = sorted(p.name for p in (export / "yolo_seg" / "images" / "train").iterdir())
    assert loader_len == {0: 2, 1: 2, 2: 2} and batches == {0: 2, 1: 2, 2: 2}  # ceil(4 / 2) iterations per epoch
    assert trained == {e: [im_files[i] for i in epoch_slice(4, e, 12, 4)] for e in range(3)}
    assert set(trained[0]).isdisjoint(trained[1]) and set(trained[1]).isdisjoint(trained[2])
    assert val_sizes == [3, 3, 3, 6]  # per epoch on the subset, then best.pt on the whole val split
    assert json.loads((run_dir / "final_val.json").read_text())["images"] == 6
    assert (run_dir / "val_subset" / "labels" / "val.cache").is_file()
    export_cache = export / "yolo_seg" / "labels" / "val.cache"  # the export's own: the whole split, never the subset
    assert len(np.load(str(export_cache), allow_pickle=True).item()["labels"]) == 6


def test_real_cpu_resume_picks_up_the_slice_of_the_epoch_it_resumes_at(tmp_path, monkeypatch):
    """Stopped after its first epoch (last.pt saved), a resumed run trains epoch 1 on slice 1 and epoch 2 on slice 2:
    the slice follows Ultralytics' own epoch counter, which a resume continues."""
    pytest.importorskip("ultralytics")
    from train_fixtures import make_export
    from ultralytics import YOLO
    from ultralytics.data import base as ul_base
    from ultralytics.data import utils as ul_utils
    from vdp.train import worker
    from vdp.train.sampling import epoch_plan

    monkeypatch.setattr(ul_utils, "check_font", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    export = make_export(tmp_path / "data", n_train=12, n_val=4)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    params = {"epochs": 3, "imgsz": 64, "batch": 2, "seed": 9, "train_images_per_epoch": 4}
    run = {"export_dir": str(export), "params": params, "sampling": epoch_plan(params, 12, 4)}
    data, _ = worker.prepare_data(run_dir, run, rewrite=True)
    trained: dict[int, list[str]] = {}
    current = {"epoch": None}
    original = ul_base.BaseDataset.__getitem__

    def recording_getitem(self, index):
        if self.augment:
            trained.setdefault(current["epoch"], []).append(Path(self.im_files[index]).name)
        return original(self, index)

    monkeypatch.setattr(ul_base.BaseDataset, "__getitem__", recording_getitem)

    def make(path):
        model = YOLO(path)
        model.add_callback("on_train_epoch_start", lambda t: current.__setitem__("epoch", t.epoch))
        return model

    def stop_after_first_epoch(t):
        raise KeyboardInterrupt  # like the worker's SIGTERM, after last.pt of epoch 0 was saved

    first = make("yolo26n-seg.yaml")
    first.add_callback("on_fit_epoch_end", stop_after_first_epoch)
    trainer_cls = sampled_trainer(train_images_per_epoch=4, seed=9, full_val=None, final_val_path=None)
    with pytest.raises(KeyboardInterrupt):
        first.train(trainer=trainer_cls, data=str(data), epochs=3, imgsz=64, batch=2, workers=0, device="cpu",
                    seed=9, plots=False, amp=False, project=str(run_dir), name="train", exist_ok=True, verbose=False)
    make(str(run_dir / "train" / "weights" / "last.pt")).train(
        resume=True, data=str(data), save_dir=str(run_dir / "train"), trainer=trainer_cls, close_mosaic=1)

    im_files = sorted(p.name for p in (export / "yolo_seg" / "images" / "train").iterdir())
    assert trained == {e: [im_files[i] for i in epoch_slice(9, e, 12, 4)] for e in range(3)}


def test_auto_optimizer_counts_the_sampled_iterations(monkeypatch):
    """Ultralytics' optimizer='auto' picks MuSGD above 10 000 optimizer steps, else AdamW, counting steps from the whole
    train set; a sampled run's steps are counted from its images per epoch."""
    pytest.importorskip("ultralytics")
    from ultralytics.engine.trainer import BaseTrainer

    seen = []
    monkeypatch.setattr(BaseTrainer, "build_optimizer",
                        lambda self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5:
                        seen.append(iterations))
    cls = sampled_trainer(train_images_per_epoch=7000, seed=0, full_val=None, final_val_path=None)
    trainer = cls.__new__(cls)
    sampler = EpochSliceSampler(28000, 7000, 0)
    trainer.__dict__.update(slice_feed=object(), train_loader=SimpleNamespace(sampler=sampler), batch_size=8,
                            args=SimpleNamespace(nbs=64), epochs=50)
    trainer.build_optimizer(None, iterations=math.ceil(28000 / 64) * 50)  # what Ultralytics passes: 21 900
    assert seen == [math.ceil(7000 / 64) * 50]  # 5 500: AdamW, not MuSGD
    whole = sampled_trainer(train_images_per_epoch=None, seed=0, full_val="/v", final_val_path=None)
    trainer = whole.__new__(whole)
    trainer.__dict__.update(slice_feed=None)
    trainer.build_optimizer(None, iterations=21900)
    assert seen[-1] == 21900  # whole epochs: Ultralytics' own count


def test_the_trainer_class_follows_the_task():
    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.segment import SegmentationTrainer

    det = sampled_trainer(train_images_per_epoch=4, seed=0, full_val=None, final_val_path=None, task="detect")
    seg = sampled_trainer(train_images_per_epoch=4, seed=0, full_val=None, final_val_path=None)
    assert issubclass(det, DetectionTrainer) and not issubclass(det, SegmentationTrainer)
    assert issubclass(seg, SegmentationTrainer)
