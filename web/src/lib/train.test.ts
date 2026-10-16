import { describe, expect, it } from "vitest";
import type { ExportInfo, GpuQueue, Run, TrainStatus } from "../api/types";
import {
  bestRunIds, checkpointSources, classesCompatible, continueModeLabel, defaultGpu, defaultMode, epochEstimate, finalMetricsNote,
  finetuneModeLabel, formatDuration, gpuOptionLabel, measuredSpeed, metricSeries, modeDisabledReason, polygonPoints,
  samplingNote, sendableParams, sourceRunLabel, startModeHeaderNote, startModeNote, startProblem, statusText, valSubsetNote,
  vramBusyReason, vramFit, vramLabel,
} from "./train";

const gpu = (over: Partial<GpuQueue> = {}): GpuQueue => ({
  index: 0, name: "NVIDIA GeForce RTX 3080 Ti", memory_total: 12 * 2 ** 30, memory_free: 9.1 * 2 ** 30,
  capability: "8.6", sm_count: 80, tflops: 34, running: null, running_kind: null, queue: [], ...over,
});

const run = (id: string, over: Partial<Run> = {}): Run => ({
  id, created_at: "", started_at: null, finished_at: null, export_id: "e", project_name: "P", model: "yolo26s-seg", gpu: 1,
  params: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 }, status: "done", error: null, resume: false,
  onnx: null, best: null, epochs_done: 0, progress: null, resumable: false, completable: false, has_best: false,
  has_onnx: false, ...over,
});
const best = (v: number) => ({ epoch: 1, mask_map50: v, mask_map50_95: v, box_map50: v, box_map50_95: v, train_seconds: 60 });

describe("train helpers", () => {
  it("bestRunIds marks every run tied for the maximum", () => {
    const runs = [run("a", { best: best(0.7) }), run("b", { best: best(0.8) }), run("c", { best: best(0.8) }), run("d")];
    expect([...bestRunIds(runs, "mask_map50_95")].sort()).toEqual(["b", "c"]);
    expect(bestRunIds([run("d")], "mask_map50_95").size).toBe(0);
  });
  it("formatDuration", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(45)).toBe("45 s");
    expect(formatDuration(720)).toBe("12 min");
    expect(formatDuration(3900)).toBe("1 h 05 min");
  });
  it("defaultGpu uses the server's default_gpu, even when that is SAM3's GPU", () => {
    const st = { gpus: [{ index: 0 }, { index: 1 }], default_gpu: 0 } as unknown as TrainStatus;
    expect(defaultGpu(st)).toBe(0);
    expect(defaultGpu({ ...st, default_gpu: 1 })).toBe(1);
    // Falls back to the first listed GPU only if the server ever omits default_gpu.
    expect(defaultGpu({ ...st, default_gpu: null })).toBe(0);
    expect(defaultGpu({ ...st, default_gpu: null, gpus: [] })).toBeNull();
  });
  it("gpuOptionLabel: name, tflops, free VRAM, SAM3 and the default marker", () => {
    const st = { default_gpu: 0, sam3_gpu: 0 } as unknown as TrainStatus;
    expect(gpuOptionLabel(gpu(), st)).toBe("cuda:0 · RTX 3080 Ti · default · ~34 TFLOPS · 9.1 GiB free · SAM3");
    expect(gpuOptionLabel(gpu({ index: 1, memory_free: null }), { default_gpu: 0, sam3_gpu: 0 })).toBe(
      "cuda:1 · RTX 3080 Ti · ~34 TFLOPS",
    );
  });
  it("polygonPoints scales normalised pairs", () => {
    expect(polygonPoints([0.5, 0.25, 1, 1], 200, 100)).toBe("100,25 200,100");
  });
  it("metricSeries keeps rows that have the key", () => {
    expect(metricSeries([{ epoch: 1, a: 0.1 }, { epoch: 2 }, { epoch: 3, a: 0.3 }] as never, "a")).toEqual([[1, 0.1], [3, 0.3]]);
  });
  it("statusText", () => {
    expect(statusText(run("a", { status: "running", progress: { phase: "train", epoch: 3, epochs: 100, batch: 1, batches: 9 } })))
      .toBe("Epoch 3/100");
    expect(statusText(run("a", { status: "running", progress: { phase: "val", epoch: 3, epochs: 100, batch: 9, batches: 9 } })))
      .toBe("Validating epoch 3");
    expect(statusText(run("a", { status: "running", progress: { phase: "predict", epoch: 3, epochs: 3, batch: 1, batches: 9 } })))
      .toBe("Predicting val images");
    expect(statusText(run("a", { status: "running" }))).toBe("Starting");
    expect(statusText(run("a", { status: "error" }))).toBe("Failed");
    expect(statusText(run("a", { status: "done" }))).toBe("Done");
    expect(statusText(run("a", { status: "done", completed_early: true, epochs_done: 65, params: { ...run("a").params, epochs: 200 } })))
      .toBe("Done (stopped at epoch 65 of 200)");
  });
  it("startProblem", () => {
    expect(startProblem({ available: false, error: "no ultralytics", exportId: "e", picked: 1 })).toBe("no ultralytics");
    expect(startProblem({ available: true, error: null, exportId: null, picked: 1 })).toBe("Choose an export to train on.");
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 0 })).toBe("Tick at least one model.");
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 2 })).toBeNull();
    const params = { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 };
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 1, params: { ...params, train_images_per_epoch: 7 } }))
      .toBe("Images per epoch must be at least the batch size.");
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 1, params: { ...params, train_images_per_epoch: 8 } }))
      .toBeNull();
  });

  it("checkpointSources keeps only continuable-or-finetunable runs, in their given order", () => {
    const runs = [
      run("a", { continuable: true }), run("b", { finetunable: true }),
      run("c", { continuable: false, finetunable: false }), run("d", { continuable: true, finetunable: true }),
    ];
    expect(checkpointSources(runs).map((r) => r.id)).toEqual(["a", "b", "d"]);
  });

  it("sourceRunLabel names the model, the export (or project) label, epoch and status", () => {
    const exports: ExportInfo[] = [
      { id: "e2", name: "Cups v2", project_name: "Optics", created_at: "", classes: ["Module"], train_images: 1,
        val_images: 1, formats: [], derived_from: null },
    ];
    const r = run("a", { model: "yolo26s-seg", export_id: "e2", epochs_done: 70, status: "cancelled",
      params: { ...run("a").params, epochs: 200 } });
    expect(sourceRunLabel(r, exports)).toBe("yolo26s-seg · Cups v2 · epoch 70/200 · Cancelled");
    const noExport = run("b", { model: "yolo11m-seg", export_id: "gone", project_name: "Bench", epochs_done: 5,
      params: { ...run("b").params, epochs: 10 }, status: "interrupted" });
    expect(sourceRunLabel(noExport, [])).toBe("yolo11m-seg · Bench · epoch 5/10 · Interrupted");
  });

  it("continueModeLabel and finetuneModeLabel name the epoch(s)", () => {
    const r = run("a", { epochs_done: 70, params: { ...run("a").params, epochs: 200 } });
    expect(continueModeLabel(r)).toBe(
      "Continue (from epoch 70 of 200: same optimizer and schedule, remaining epochs on the new data)",
    );
    expect(finetuneModeLabel(r)).toBe("Fine-tune (fresh schedule from best.pt)"); // stopped: best epoch unknown
    expect(finetuneModeLabel(run("b", { best: best(0.5), epochs_done: 40 }))).toBe(
      "Fine-tune (fresh schedule from best.pt, epoch 1)",
    );
  });

  it("modeDisabledReason and defaultMode follow continuable/finetunable", () => {
    const both = run("a", { continuable: true, finetunable: true });
    expect(modeDisabledReason(both, "continue")).toBeNull();
    expect(modeDisabledReason(both, "finetune")).toBeNull();
    expect(defaultMode(both)).toBe("continue");
    const finetuneOnly = run("b", { continuable: false, finetunable: true });
    expect(modeDisabledReason(finetuneOnly, "continue")).toMatch(/stopped run/);
    expect(defaultMode(finetuneOnly)).toBe("finetune");
    const neither = run("c", { continuable: false, finetunable: false });
    expect(modeDisabledReason(neither, "finetune")).toMatch(/best\.pt/);
  });

  it("classesCompatible: order-sensitive, and unknown (null) classes never block", () => {
    const exp = (classes: string[]): ExportInfo => ({
      id: "e", name: null, project_name: "P", created_at: "", classes, train_images: 0, val_images: 0,
      formats: [], derived_from: null,
    });
    expect(classesCompatible(exp(["A", "B"]), run("a", { classes: ["A", "B"] }))).toBe(true);
    expect(classesCompatible(exp(["B", "A"]), run("a", { classes: ["A", "B"] }))).toBe(false);
    expect(classesCompatible(exp(["A"]), run("a", { classes: ["A", "B"] }))).toBe(false);
    expect(classesCompatible(exp(["A"]), run("a", { classes: null }))).toBe(true);
    expect(classesCompatible(exp(["A"]), run("a", {}))).toBe(true); // classes absent (older server)
  });

  it("startModeNote and startModeHeaderNote", () => {
    expect(startModeNote(run("a"))).toBeNull();
    const cont = run("a", { source_run: "20260927-072200-yolo26s-seg-a1b2", start_mode: "continue", continued_from_epoch: 70 });
    expect(startModeNote(cont)).toBe("continued from 20260927-0722…");
    expect(startModeHeaderNote(cont)).toBe("continued from 20260927-0722… at epoch 70");
    const fine = run("b", { source_run: "20260927-072200-yolo26s-seg-a1b2", start_mode: "finetune" });
    expect(startModeNote(fine)).toBe("fine-tuned from 20260927-0722…");
    expect(startModeHeaderNote(fine)).toBe("fine-tuned from 20260927-0722…");
  });
});

describe("VRAM estimate helpers", () => {
  const G = 2 ** 30;
  it("vramFit compares an estimate with free memory, tight within 10 %", () => {
    const est = (bytes: number | null, over_bytes: number | null = null) => ({ bytes, over_bytes });
    expect(vramFit(est(5 * G), 9 * G)).toBe("fits");
    expect(vramFit(est(8.5 * G), 9 * G)).toBe("tight");
    expect(vramFit(est(9 * G), 9 * G)).toBe("tight");
    expect(vramFit(est(9.5 * G), 9 * G)).toBe("no");
    expect(vramFit(est(5 * G), null)).toBeNull(); // free memory unknown
  });
  it("vramFit: a batch that did not fit needs more than over_bytes", () => {
    expect(vramFit({ bytes: null, over_bytes: 8 * G }, 7 * G)).toBe("no");
    expect(vramFit({ bytes: null, over_bytes: 8 * G }, 8 * G)).toBe("no");
    expect(vramFit({ bytes: null, over_bytes: 8 * G }, 11 * G)).toBe("unknown"); // more memory than it was tried with
    expect(vramFit({ bytes: null, over_bytes: null }, 11 * G)).toBe("no");
  });
  it("vramLabel", () => {
    expect(vramLabel(7.84 * G)).toBe("about 7.8 GiB");
    expect(vramLabel(null)).toBe("—");
  });
  it("vramBusyReason names a GPU that is training, queued, exporting or busy in another process", () => {
    const st = { gpus: [gpu(), gpu({ index: 1, running: "r1", running_kind: "train" }), gpu({ index: 2, queue: ["q"] }),
      gpu({ index: 3, running: "r2", running_kind: "onnx" })] };
    expect(vramBusyReason(st, [], 0)).toBeNull();
    expect(vramBusyReason(st, [], 1)).toMatch(/^cuda:1 is training/);
    expect(vramBusyReason(st, [], 2)).toMatch(/^cuda:2 has a run queued/);
    expect(vramBusyReason(st, [], 3)).toMatch(/^cuda:3 is exporting/);
    expect(vramBusyReason(st, [run("x", { status: "running", gpu: 0 })], 0)).toMatch(/^cuda:0 is training/);
  });
});

describe("sampled epochs", () => {
  const params = { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 };
  const speed = { train_images_per_s: 65, val_images_per_s: 80, batch: 8, imgsz: 960 };

  it("sendableParams leaves out the empty (all) sizes", () => {
    expect(sendableParams({ ...params, train_images_per_epoch: null, val_images_per_epoch: null })).toEqual(params);
    expect(sendableParams({ ...params, train_images_per_epoch: 7000, val_images_per_epoch: null }))
      .toEqual({ ...params, train_images_per_epoch: 7000 });
  });

  it("measuredSpeed takes the newest run measured with the same model, image size and GPU", () => {
    const runs = [
      run("new-other-gpu", { gpu: 0, speed }),
      run("new", { gpu: 1, speed: { ...speed, train_images_per_s: 70 } }),
      run("old", { gpu: 1, speed: { ...speed, train_images_per_s: 50 } }),
      run("unmeasured", { gpu: 1 }),
    ];
    expect(measuredSpeed(runs, "yolo26s-seg", 960, 1)?.train_images_per_s).toBe(70);
    expect(measuredSpeed(runs, "yolo26s-seg", 640, 1)).toBeNull();
    expect(measuredSpeed(runs, "yolo11s-seg", 960, 1)).toBeNull();
  });

  it("epochEstimate: images per epoch, minutes from the measured speed, full passes and validation", () => {
    expect(epochEstimate({ trainImages: 28000, valImages: 1643, epochs: 100, trainPerEpoch: 7000, valPerEpoch: 400, speed }))
      .toEqual(["1 epoch = 7,000 images (≈ 2 min) · 100 epochs ≈ 25 full passes", "Validation: 400 of 1,643 images per epoch (≈ 5 s)"]);
    expect(epochEstimate({ trainImages: 28000, valImages: 1643, epochs: 30, trainPerEpoch: null, valPerEpoch: null, speed: null }))
      .toEqual(["1 epoch = 28,000 images · 30 epochs ≈ 30 full passes", "Validation: all 1,643 images per epoch"]);
    // more than the split holds means all of it; partial passes keep one decimal
    expect(epochEstimate({ trainImages: 1000, valImages: 100, epochs: 5, trainPerEpoch: 300, valPerEpoch: 500, speed: null }))
      .toEqual(["1 epoch = 300 images · 5 epochs ≈ 1.5 full passes", "Validation: all 100 images per epoch"]);
  });

  it("samplingNote and valSubsetNote describe a run's sampled epochs", () => {
    expect(samplingNote(run("a"))).toBeNull();
    const sampled = run("b", { sampling: {
      train_images: 28000, train_images_per_epoch: 7000, val_images: 1643, val_images_per_epoch: 400,
      warmup_epochs: 12, close_mosaic: 40,
    } });
    expect(samplingNote(sampled)).toBe(
      "Each epoch trains on the next 7,000 of 28,000 images (warmup 12 epochs, mosaic off for the last 40). " +
      "Per-epoch validation uses a fixed 400 of 1,643 val images; the final metrics are on all 1,643.",
    );
    expect(valSubsetNote(sampled)).toBe("val subset of 400");
    const grown = run("e", { sampling: { ...sampled.sampling!, val_images_per_epoch: 10, val_images_chosen: 13 } });
    expect(samplingNote(grown)).toContain(
      "Per-epoch validation uses a fixed 13 of 1,643 val images (10 asked, more to cover every class); the final metrics are on all 1,643.",
    );
    expect(valSubsetNote(grown)).toBe("val subset of 13");
    const valOnly = run("c", { sampling: { train_images: 28000, train_images_per_epoch: null, val_images: 1643, val_images_per_epoch: 400 } });
    expect(samplingNote(valOnly)).toBe("Per-epoch validation uses a fixed 400 of 1,643 val images; the final metrics are on all 1,643.");
    const trainOnly = run("d", { sampling: { train_images: 28000, train_images_per_epoch: 7000, val_images: 1643, val_images_per_epoch: null, warmup_epochs: 12, close_mosaic: 40 } });
    expect(valSubsetNote(trainOnly)).toBeNull();
  });
  it("finalMetricsNote names the full-val final evaluation", () => {
    expect(finalMetricsNote(run("a", { best: best(0.5) }))).toBeNull();
    expect(finalMetricsNote(run("b", { best: { ...best(0.5), epoch: 12, mask_map50_95: 0.7123, val_images: 1643 } }))).toBe(
      "Final, best.pt (epoch 12) on all 1,643 val images: mask mAP50-95 0.712 · mask mAP50 0.500 · box mAP50-95 0.500",
    );
  });
});
