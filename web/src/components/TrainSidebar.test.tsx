// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AugParams, Run, TrainStatus } from "../api/types";
import { TrainSidebar } from "./TrainSidebar";

afterEach(cleanup);

const augDefaults: AugParams = {
  degrees: 0, shear: 0, perspective: 0, flipud: 0, fliplr: 0.5, scale: 0.5, translate: 0.1,
  hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, mosaic: 1.0, mixup: 0, copy_paste: 0, photometric: false,
};

const status: TrainStatus = {
  available: true, error: null, sam3_gpu: null, default_gpu: 0,
  gpus: [
    { index: 0, name: "NVIDIA GeForce RTX 3080 Ti", memory_total: 12 * 2 ** 30, memory_free: 9 * 2 ** 30, capability: "8.6", sm_count: 80, tflops: 34, running: null, running_kind: null, queue: [] },
    { index: 1, name: "NVIDIA GeForce RTX 2080 Ti", memory_total: 11 * 2 ** 30, memory_free: 5 * 2 ** 30, capability: "7.5", sm_count: 68, tflops: 13.5, running: null, running_kind: null, queue: [] },
  ],
  models: ["yolo11s-seg"], default_models: ["yolo11s-seg"],
  defaults: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0, augment: "default", aug: null },
  aug_presets: { default: augDefaults, strong: { ...augDefaults, degrees: 15 } },
};

const run = (over: Partial<Run> = {}): Run => ({
  id: "20260924-120000-yolo11s-seg-aaaa", created_at: "", started_at: null, finished_at: null, export_id: "e",
  project_name: "P", model: "yolo11s-seg", gpu: 1, params: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 },
  status: "cancelled", error: null, resume: false, onnx: null, best: null, epochs_done: 40, progress: null,
  resumable: false, completable: false, has_best: false, has_onnx: false, ...over,
});

const noop = () => Promise.resolve();

function renderSidebar(selectedRun: Run | null, onComplete: (id: string, gpu: number) => Promise<void> = noop) {
  return render(
    <TrainSidebar status={status} exports={[]} runs={selectedRun ? [selectedRun] : []} exportId={null}
      onExportChange={() => {}} onExportsChanged={() => {}} onStart={noop} onCancel={() => {}}
      selectedRun={selectedRun} onResume={noop} onComplete={onComplete} />,
  );
}

describe("TrainSidebar stopped-run section", () => {
  it("shows only Resume for a resumable, not-completable run", () => {
    renderSidebar(run({ resumable: true }));
    expect(screen.getByText("Stopped yolo11s-seg run")).toBeTruthy();
    expect(screen.getByRole("button", { name: /^Resume on cuda:/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /^Mark complete on cuda:/ })).toBeNull();
  });

  it("shows only Mark complete for a completable, not-resumable run", () => {
    renderSidebar(run({ completable: true }));
    expect(screen.getByText("Stopped yolo11s-seg run")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /^Resume on cuda:/ })).toBeNull();
    expect(screen.getByRole("button", { name: /^Mark complete on cuda:/ })).toBeTruthy();
  });

  it("shows both buttons when a run is both resumable and completable", () => {
    renderSidebar(run({ resumable: true, completable: true }));
    expect(screen.getByRole("button", { name: /^Resume on cuda:/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /^Mark complete on cuda:/ })).toBeTruthy();
  });

  it("shows no stopped-run section for a run that is neither resumable nor completable", () => {
    renderSidebar(run());
    expect(screen.queryByText("Stopped yolo11s-seg run")).toBeNull();
  });

  it("calls onComplete with the chosen GPU", async () => {
    const onComplete = vi.fn().mockResolvedValue(undefined);
    renderSidebar(run({ completable: true, gpu: 1 }), onComplete);
    fireEvent.change(screen.getByLabelText("Resume GPU"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Mark complete on cuda:1" }));
    await Promise.resolve();
    expect(onComplete).toHaveBeenCalledWith(run().id, 1);
  });

  it("Mark complete's note includes the best epoch and mask mAP when the run already carries one", () => {
    const withBest = run({
      completable: true,
      best: { epoch: 12, mask_map50: 0.9, mask_map50_95: 0.734, box_map50: 0.9, box_map50_95: 0.8, train_seconds: 100 },
    });
    renderSidebar(withBest);
    expect(screen.getByText(
      "Keeps the best epoch so far (epoch 12, mask mAP50-95 0.734), runs the validation predictions on that GPU, then marks the run done.",
    )).toBeTruthy();
  });

  it("Mark complete's note omits the parenthetical when no best is known yet", () => {
    renderSidebar(run({ completable: true }));
    expect(screen.getByText(
      "Keeps the best epoch so far, runs the validation predictions on that GPU, then marks the run done.",
    )).toBeTruthy();
  });
});
