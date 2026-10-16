"""VRAM probe: `python -m vdp.train.vram_probe <model> <imgsz> [--objects N]` (see vram.py for the server side).

Started like the train worker -- CUDA_VISIBLE_DEVICES set to one GPU (so it always uses cuda:0) and cwd
data/train_weights/, where Ultralytics downloads and caches the pretrained checkpoints -- but at low priority. For
each batch size in BATCHES it runs one real training step on random images and records the peak memory PyTorch
reserved (`torch.cuda.max_memory_reserved`, the same counter Ultralytics' `GPU_mem` column shows). Its PyTorch memory
is capped at the GPU's free memory at start minus HEADROOM_BYTES (reported as `limit_bytes`), and it stops at the
first out-of-memory error: that batch size and every larger one are reported as `null` ("needs more than the limit").

The training step is the trainer's own, minus the data loader: the model is built the way the segment trainer
builds it (a fresh SegmentationModel from the checkpoint's yaml with the dataset's class count, then the pretrained
weights loaded into it, the trainer's default hyper-parameters attached), and each step is a forward pass and the
model's real segmentation loss under AMP autocast (fp16, as training uses), a GradScaler-scaled backward, gradient
clipping, an AdamW step (the optimizer Ultralytics' "auto" picks for runs under 10k iterations; it keeps two state
tensors per weight, more than SGD/MuSGD's one, so it is the conservative choice; its state is allocated before the
first measurement) and an EMA update. The labels are synthetic (`--objects` per image) but shaped exactly as the trainer's collate function produces them (overlapping instance masks at
1/mask_ratio resolution, semantic masks for YOLO26's semantic branch).

Output goes to stdout as lines starting with PREFIX followed by JSON (Ultralytics' own logging is free-form text on
the same streams): {"type": "start", gpu_name, total_bytes, limit_bytes}, one {"type": "batch", batch, bytes} per
size, then {"type": "done", estimates} or {"type": "error", message}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Callable

BATCHES = (1, 2, 4, 8, 12)
NICE = 10
PREFIX = "VRAM_PROBE "
# The class count the probe's model head is built with. A real run uses its export's (this tool's projects have
# one to a few classes); it changes memory very little, except through YOLO26's semantic-segmentation branch,
# whose output and one-hot targets have a channel per class, so a few more than usual errs on the safe side.
PROBE_CLASSES = 4
PROBE_INSTANCES = 8  # synthetic objects per image unless told (--objects): the mask loss runs per matched instance
# Memory the probe leaves to everyone else on the GPU (SAM3 allocating for a prompt, say): it caps its own PyTorch
# allocations at the GPU's free memory at start minus this, so a batch that needs more counts as not fitting.
HEADROOM_BYTES = 2**30
_OOM_MARKERS = ("CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_INTERNAL_ERROR", "unable to find an engine")


def is_memory_error(e: BaseException) -> bool:
    """The same test Ultralytics' trainer (and the train worker) uses for a GPU memory error."""
    text = str(e)
    return "out of memory" in text.lower() or any(m in text for m in _OOM_MARKERS)


def measure_batches(
    step: Callable[[int], int], batches: tuple[int, ...] = BATCHES,
    on_result: Callable[[int, int | None], None] | None = None,
) -> list[dict]:
    """`step(batch)` runs one training step and returns its peak bytes; the first memory error ends the loop and
    marks that batch and every larger one None. Any other error propagates."""
    out: list[dict] = []
    fits = True
    for batch in batches:
        peak = None
        if fits:
            try:
                peak = int(step(batch))
            except Exception as e:  # re-raised unless it is a memory error
                if not is_memory_error(e):
                    raise
                fits = False
        out.append({"batch": batch, "bytes": peak})
        if on_result is not None:
            on_result(batch, peak)
    return out


def memory_cap(free: int, total: int, headroom: int = HEADROOM_BYTES) -> tuple[int, float]:
    """(limit bytes, fraction of `total`) for `torch.cuda.set_per_process_memory_fraction`: free memory minus the
    headroom, never below zero (then nothing fits)."""
    limit = max(0, min(free - headroom, total))
    return limit, (limit / total if total > 0 else 0.0)


def prime_optimizer(optimizer, params) -> None:
    """Make the optimizer allocate its state now, with zero gradients, so every measurement includes it: during a
    measured step GradScaler skips `optimizer.step()` whenever the scaled gradients overflow (common on the first
    steps), which would leave AdamW's state unallocated."""
    import torch

    for p in params:
        p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def synthetic_batch(batch: int, imgsz: int, *, device, classes: int = PROBE_CLASSES,
                    instances: int = PROBE_INSTANCES, mask_ratio: int = 4) -> dict:
    """Random images plus labels shaped like the segment trainer's collated batch (after `preprocess_batch`):
    `instances` boxes per image in a grid, normalised xywh, their overlap masks (instance index + 1 per pixel) and
    class-index semantic masks at 1/mask_ratio resolution."""
    import torch

    side = imgsz // mask_ratio
    grid = int(instances ** 0.5 + 0.999)
    cell = 1.0 / grid
    boxes, masks = [], torch.zeros(batch, side, side)
    for k in range(instances):
        cx, cy = (k % grid + 0.5) * cell, (k // grid + 0.5) * cell
        boxes.append([cx, cy, cell * 0.6, cell * 0.6])
        x0, x1 = int((cx - cell * 0.3) * side), int((cx + cell * 0.3) * side)
        y0, y1 = int((cy - cell * 0.3) * side), int((cy + cell * 0.3) * side)
        masks[:, y0:y1, x0:x1] = k + 1
    cls = torch.arange(instances, dtype=torch.float32) % classes
    sem = torch.zeros(batch, side, side)
    for k in range(instances):
        sem[masks == k + 1] = cls[k]
    return {
        "img": torch.rand(batch, 3, imgsz, imgsz, device=device),
        "batch_idx": torch.arange(batch, dtype=torch.float32).repeat_interleave(instances).to(device),
        "cls": cls.repeat(batch).view(-1, 1).to(device),
        "bboxes": torch.tensor(boxes).repeat(batch, 1).to(device),
        "masks": masks.to(device),
        "sem_masks": sem.to(device),
    }


def build_step(model_name: str, imgsz: int, objects: int = PROBE_INSTANCES) -> Callable[[int], int]:
    """Load the model as the segment trainer does and return `step(batch) -> peak reserved bytes`."""
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.torch_utils import ModelEMA, autocast, init_seeds

    device = torch.device("cuda:0")
    cfg = get_cfg()  # the trainer's defaults: loss gains, overlap_mask, mask_ratio, epochs, deterministic
    # As the trainer starts: deterministic cuDNN/cuBLAS algorithms choose different workspaces than the default ones.
    init_seeds(cfg.seed + 1, deterministic=cfg.deterministic)
    pretrained = YOLO(f"{model_name}.pt").model  # downloaded into the cwd (data/train_weights) when missing
    model = SegmentationModel(pretrained.yaml, nc=PROBE_CLASSES, ch=3, verbose=False)
    model.load(pretrained, verbose=False)
    del pretrained
    model.args = cfg
    model.nc = PROBE_CLASSES
    # The trainer trains channels_last (NHWC) on CUDA; its convolutions use other algorithms and much less memory
    # than NCHW ones (about 1 GiB less for yolo26s-seg at batch 8, 960 px), so leaving this out overestimates.
    model.to(device).to(memory_format=torch.channels_last).train()
    for name, p in model.named_parameters():
        p.requires_grad = ".dfl" not in name  # the trainer always freezes the DFL layer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, fused=True)
    prime_optimizer(optimizer, trainable)
    print(f"optimizer state allocated for {len(optimizer.state)} of {len(trainable)} trainable tensors", flush=True)
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model)

    def step(batch: int) -> int:
        data = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            data = synthetic_batch(batch, imgsz, device=device, instances=objects)
            with autocast(True):
                loss, _ = model(data)
            scaler.scale(loss.sum()).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)
            torch.cuda.synchronize(device)
            return int(torch.cuda.max_memory_reserved(device))
        finally:
            del data
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    return step


def _claim_device() -> tuple[str, int, int]:
    """(GPU name, total bytes, limit bytes): caps this process's PyTorch memory at what is free now minus the
    headroom, so the probe never takes memory another process (SAM3) is about to need."""
    import torch

    free, total = torch.cuda.mem_get_info(0)
    limit, fraction = memory_cap(int(free), int(total))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return torch.cuda.get_device_properties(0).name, int(total), limit


def _say(**event) -> None:
    print(PREFIX + json.dumps(event), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vdp.train.vram_probe")
    parser.add_argument("model")
    parser.add_argument("imgsz", type=int)
    parser.add_argument("--objects", type=int, default=PROBE_INSTANCES, help="synthetic objects per image")
    args = parser.parse_args(argv)
    os.nice(NICE)  # low priority: a training run or the server must not wait for this
    try:
        name, total, limit = _claim_device()
        _say(type="start", gpu_name=name, total_bytes=total, limit_bytes=limit)
        step = build_step(args.model, args.imgsz, args.objects)
        estimates = measure_batches(step, on_result=lambda b, n: _say(type="batch", batch=b, bytes=n))
    except Exception as e:  # noqa: BLE001 -- reported to the server, which turns it into a job error
        traceback.print_exc()
        _say(type="error", message=f"{type(e).__name__}: {e}")
        return 1
    _say(type="done", estimates=estimates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
