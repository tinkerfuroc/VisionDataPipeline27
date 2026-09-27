import { describe, expect, it } from "vitest";
import type { GpuQueue, Run, TrainStatus } from "../api/types";
import { bestRunIds, defaultGpu, formatDuration, gpuOptionLabel, metricSeries, polygonPoints, startProblem, statusText } from "./train";

const gpu = (over: Partial<GpuQueue> = {}): GpuQueue => ({
  index: 0, name: "NVIDIA GeForce RTX 3080 Ti", memory_total: 12 * 2 ** 30, memory_free: 9.1 * 2 ** 30,
  capability: "8.6", sm_count: 80, tflops: 34, running: null, running_kind: null, queue: [], ...over,
});

const run = (id: string, over: Partial<Run> = {}): Run => ({
  id, created_at: "", started_at: null, finished_at: null, export_id: "e", project_name: "P", model: "yolo26s-seg", gpu: 1,
  params: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 }, status: "done", error: null, resume: false,
  onnx: null, best: null, epochs_done: 0, progress: null, resumable: false, has_best: false, has_onnx: false, ...over,
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
  });
  it("startProblem", () => {
    expect(startProblem({ available: false, error: "no ultralytics", exportId: "e", picked: 1 })).toBe("no ultralytics");
    expect(startProblem({ available: true, error: null, exportId: null, picked: 1 })).toBe("Choose an export to train on.");
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 0 })).toBe("Tick at least one model.");
    expect(startProblem({ available: true, error: null, exportId: "e", picked: 2 })).toBeNull();
  });
});
