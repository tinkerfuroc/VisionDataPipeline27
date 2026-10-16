"""Sampled epochs in the training worker (vdp.train.sampling): an Ultralytics SegmentationTrainer subclass, passed as
`model.train(trainer=...)`, whose train loader hands out one slice of the train set per epoch and whose final
evaluation of best.pt runs on the whole val split while the per-epoch validation uses the run's fixed subset.

How Ultralytics (8.4.x) feeds the train loader, which the slice bookkeeping below is built around:
- `InfiniteDataLoader` keeps ONE iterator for the whole run over `_RepeatSampler(BatchSampler(sampler))`, which calls
  `iter(sampler)` again whenever the previous pass is used up; an epoch is `len(loader)` = ceil(len(sampler) / batch)
  batches of it. So each `iter(sampler)` call is one epoch, provided every epoch takes exactly `len(loader)` batches.
- With worker processes the iterator prefetches (4 batches per worker), so the next epoch's `iter(sampler)` is
  called while the current epoch is still running -- before `on_train_epoch_start` of the epoch it serves. The slice
  can therefore not be picked at epoch start; the sampler instead serves consecutive epochs from a starting point.
- That starting point is re-set whenever the iterator is (re)made: by `_close_dataloader_mosaic` (it is followed by
  `train_loader.reset()`), and, as a safety net, at the start of any epoch that would not get its own slice next
  (the first epoch of a resumed run, or an epoch Ultralytics repeats after a NaN recovery), by resetting the loader.
"""

from __future__ import annotations

from pathlib import Path

from .runs import atomic_write_json
from .sampling import epoch_slice


class EpochSliceSampler:
    """Each `iter()` yields the next epoch's slice of `size` images (`per_epoch` indices), starting at the epoch last
    given to `set_epoch`. A plain iterable (all a torch BatchSampler needs), so this module imports no torch: the
    server imports the worker module for its constants."""

    def __init__(self, size: int, per_epoch: int, seed: int) -> None:
        self.size, self.per_epoch, self.seed = size, per_epoch, seed
        self.next_epoch = 0
        self.served: list[int] = []  # the epoch of every iter() call, for tests and the smoke run

    def set_epoch(self, epoch: int) -> None:
        self.next_epoch = epoch

    def __iter__(self):
        epoch, self.next_epoch = self.next_epoch, self.next_epoch + 1
        self.served.append(epoch)
        return iter(epoch_slice(self.seed, epoch, self.size, self.per_epoch).tolist())

    def __len__(self) -> int:
        return self.per_epoch


def use_sampler(loader, sampler) -> None:
    """Swap the sampler of a loader built by Ultralytics' `build_dataloader` (single GPU: shuffled, no sampler of its
    own), keeping everything else it set up (workers, seeds, pinning). The batches come from
    `loader.batch_sampler` (`_RepeatSampler`) -> `BatchSampler` -> sampler; `loader.sampler` is what Ultralytics
    reads the epoch size from (`len(train_loader.sampler)`)."""
    loader.batch_sampler.sampler.sampler = sampler
    object.__setattr__(loader, "sampler", sampler)  # DataLoader forbids setting it; Ultralytics does the same


class SliceFeed:
    """Keeps a loader's EpochSliceSampler on the trainer's epoch. `consuming` is the epoch whose slice the loader's
    next batch belongs to: it advances by one per finished epoch."""

    def __init__(self, loader, sampler: EpochSliceSampler, epoch: int) -> None:
        self.loader, self.sampler = loader, sampler
        self.restarted(epoch)

    def restarted(self, epoch: int) -> None:
        """The loader's iterator is new (or about to be): its first slice is `epoch`'s."""
        self.sampler.set_epoch(epoch)
        self.consuming = epoch

    def start_epoch(self, epoch: int) -> None:
        if self.consuming != epoch:  # e.g. a resumed run's first epoch, or an epoch repeated after a NaN recovery
            self.loader.reset()
            self.restarted(epoch)

    def end_epoch(self) -> None:
        self.consuming += 1


def sampled_trainer(*, train_images_per_epoch: int | None, seed: int, full_val: str | None,
                    final_val_path: Path | None):
    """The trainer class for `model.train(trainer=...)`: slices of `train_images_per_epoch` (None: whole epochs) and,
    with `full_val` (the export's val images dir, while data.yaml's `val` is the run's subset), a final evaluation of
    best.pt on the whole val split whose metrics are written to `final_val_path`."""
    from ultralytics.models.yolo.segment import SegmentationTrainer

    class SampledTrainer(SegmentationTrainer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.slice_feed: SliceFeed | None = None
            if train_images_per_epoch:
                self.add_callback("on_train_epoch_start", lambda t: t.slice_feed.start_epoch(t.epoch))
                self.add_callback("on_train_epoch_end", lambda t: t.slice_feed.end_epoch())

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if mode == "train" and train_images_per_epoch:
                if rank != -1:
                    raise RuntimeError("sampled epochs train on one GPU only")
                size = len(loader.dataset)  # the images Ultralytics kept (it drops unreadable ones)
                sampler = EpochSliceSampler(size, min(train_images_per_epoch, size), seed)
                use_sampler(loader, sampler)
                # epoch 0 at setup (a resumed run's start epoch is known only later); an out-of-memory retry
                # rebuilds the loader mid-run, where `self.epoch` is the epoch it restarts
                self.slice_feed = SliceFeed(loader, sampler, getattr(self, "epoch", 0))
            return loader

        def _close_dataloader_mosaic(self) -> None:
            super()._close_dataloader_mosaic()
            if self.slice_feed is not None:
                self.slice_feed.restarted(self.epoch)  # Ultralytics resets the loader right after this

        def final_eval(self) -> None:
            loader = None
            if full_val is not None and self.best.exists():
                batch = max(self.batch_size // max(self.world_size, 1), 1) * 2  # as for the per-epoch val loader
                loader = self.get_dataloader(full_val, batch_size=batch, rank=-1, mode="val")
                self.validator.dataloader = loader  # final_eval validates best.pt with the validator's loader
            try:
                super().final_eval()
            finally:
                if loader is not None:
                    loader.close()
            if loader is not None and self.metrics and final_val_path is not None:
                atomic_write_json(final_val_path, {
                    "images": len(loader.dataset),
                    "metrics": {k: float(v) for k, v in self.metrics.items()},
                })

    return SampledTrainer
