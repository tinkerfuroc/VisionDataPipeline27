// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AugParams, ExportInfo, Run, TrainStatus } from "../api/types";
import { TrainSidebar } from "./TrainSidebar";

afterEach(cleanup);

const augDefaults: AugParams = {
  degrees: 0, shear: 0, perspective: 0, flipud: 0, fliplr: 0.5, scale: 0.5, translate: 0.1,
  hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, mosaic: 1.0, mixup: 0, copy_paste: 0, photometric: false, lighting: false,
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
      selectedRun={selectedRun} onResume={noop} onComplete={onComplete} sourceRunId={null}
      onSourceRunChange={() => {}} onStartFromCheckpoint={noop} onTrainOnNewData={() => {}} />,
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

  it("Train on new data calls onTrainOnNewData with the stopped run's id", () => {
    const onTrainOnNewData = vi.fn();
    render(
      <TrainSidebar status={status} exports={[]} runs={[run({ resumable: true })]} exportId={null}
        onExportChange={() => {}} onExportsChanged={() => {}} onStart={noop} onCancel={() => {}}
        selectedRun={run({ resumable: true })} onResume={noop} onComplete={noop} sourceRunId={null}
        onSourceRunChange={() => {}} onStartFromCheckpoint={noop} onTrainOnNewData={onTrainOnNewData} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Train on new data" }));
    expect(onTrainOnNewData).toHaveBeenCalledWith(run().id);
  });
});

const exportInfo = (over: Partial<ExportInfo> = {}): ExportInfo => ({
  id: "e1", name: null, project_name: "Optics", created_at: "", classes: ["Module"], train_images: 10,
  val_images: 2, formats: ["yolo_seg"], derived_from: null, ...over,
});

describe("TrainSidebar Start from a checkpoint", () => {
  const continuableRun = run({
    id: "20260924-120000-yolo11s-seg-src1", status: "cancelled", resumable: true, continuable: true,
    epochs_done: 40, params: { ...run().params, epochs: 100 }, classes: ["Module"],
  });
  const doneRun = run({
    id: "20260924-120000-yolo11s-seg-src2", status: "done", finetunable: true, has_best: true, epochs_done: 100,
    best: { epoch: 80, mask_map50: 0.9, mask_map50_95: 0.7, box_map50: 0.9, box_map50_95: 0.8, train_seconds: 60 },
  });

  function renderFromCheckpoint(opts: {
    runs: Run[];
    sourceRunId: string | null;
    exports?: ExportInfo[];
    exportId?: string | null;
    onSourceRunChange?: (id: string | null) => void;
    onStartFromCheckpoint?: (id: string, body: unknown) => Promise<void>;
  }) {
    return render(
      <TrainSidebar status={status} exports={opts.exports ?? [exportInfo()]} runs={opts.runs}
        exportId={opts.exportId ?? "e1"} onExportChange={() => {}} onExportsChanged={() => {}} onStart={noop}
        onCancel={() => {}} selectedRun={null} onResume={noop} onComplete={noop}
        sourceRunId={opts.sourceRunId} onSourceRunChange={opts.onSourceRunChange ?? (() => {})}
        onStartFromCheckpoint={(opts.onStartFromCheckpoint ?? noop) as never} onTrainOnNewData={() => {}} />,
    );
  }

  it("the Start-from select lists only continuable/finetunable runs", () => {
    const ineligible = run({ id: "20260924-120000-yolo11s-seg-nope", status: "queued" });
    renderFromCheckpoint({ runs: [continuableRun, doneRun, ineligible], sourceRunId: null });
    const select = screen.getByLabelText("Start from") as HTMLSelectElement;
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(["", continuableRun.id, doneRun.id]);
  });

  it("Continue is the default mode for a continuable run; a done run defaults to and is limited to Fine-tune", () => {
    renderFromCheckpoint({ runs: [continuableRun], sourceRunId: continuableRun.id });
    expect((screen.getByRole("radio", { name: "Continue" }) as HTMLInputElement).checked).toBe(true);
    expect((screen.getByRole("radio", { name: "Continue" }) as HTMLInputElement).disabled).toBe(false);
    expect(screen.getByText(`Model: ${continuableRun.model}`)).toBeTruthy();
    cleanup();

    renderFromCheckpoint({ runs: [doneRun], sourceRunId: doneRun.id });
    const continueRadio = screen.getByRole("radio", { name: "Continue" }) as HTMLInputElement;
    expect(continueRadio.disabled).toBe(true); // Continue disabled for a done run
    expect(screen.getByText(/only a stopped run/)).toBeTruthy();
    expect((screen.getByRole("radio", { name: "Fine-tune" }) as HTMLInputElement).checked).toBe(true);
  });

  it("marks an incompatible export disabled in the Export select and disables Start with a note", () => {
    const mismatched = exportInfo({ id: "e-mismatch", classes: ["Widget"] });
    renderFromCheckpoint({
      runs: [continuableRun], sourceRunId: continuableRun.id, exports: [exportInfo(), mismatched],
      exportId: "e-mismatch",
    });
    const options = Array.from((screen.getByLabelText("Export") as HTMLSelectElement).options);
    const bad = options.find((o) => o.value === "e-mismatch")!;
    expect(bad.disabled).toBe(true);
    expect(bad.text).toMatch(/ — different classes$/);
    const good = options.find((o) => o.value === "e1")!;
    expect(good.disabled).toBe(false);
    expect(screen.getByText(/do not match the source run's classes/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Continue on new data" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("Continue shows the source run's own params, disabled, with a note; Fine-tune leaves them editable", () => {
    renderFromCheckpoint({ runs: [continuableRun], sourceRunId: continuableRun.id });
    expect(screen.getByText("Continue keeps the run's own settings.")).toBeTruthy();
    const epochs = screen.getByLabelText("Epochs") as HTMLInputElement;
    expect(epochs.disabled).toBe(true);
    expect(epochs.value).toBe("100"); // the source run's own epochs, not the server defaults
    cleanup();

    renderFromCheckpoint({ runs: [continuableRun, doneRun], sourceRunId: continuableRun.id });
    fireEvent.click(screen.getByRole("radio", { name: "Fine-tune" }));
    expect(screen.queryByText("Continue keeps the run's own settings.")).toBeNull();
    expect((screen.getByLabelText("Epochs") as HTMLInputElement).disabled).toBe(false);
  });

  it("Start button reads Continue/Fine-tune on new data and sends the request body", async () => {
    const onStartFromCheckpoint = vi.fn().mockResolvedValue(undefined);
    renderFromCheckpoint({ runs: [continuableRun], sourceRunId: continuableRun.id, onStartFromCheckpoint });
    expect(screen.getByRole("button", { name: "Continue on new data" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Continue on new data" }));
    await Promise.resolve();
    expect(onStartFromCheckpoint).toHaveBeenCalledWith(continuableRun.id, { export_id: "e1", mode: "continue", gpu: 0 });
    // Continue never sends `params` (the source's own are used, ignoring anything client-side).
    expect(onStartFromCheckpoint.mock.calls[0][1]).not.toHaveProperty("params");
  });

  it("Fine-tune sends the edited params in the request body", async () => {
    const onStartFromCheckpoint = vi.fn().mockResolvedValue(undefined);
    renderFromCheckpoint({ runs: [doneRun], sourceRunId: doneRun.id, onStartFromCheckpoint });
    fireEvent.change(screen.getByLabelText("Epochs"), { target: { value: "12" } });
    fireEvent.click(screen.getByRole("button", { name: "Fine-tune on new data" }));
    await Promise.resolve();
    const [id, body] = onStartFromCheckpoint.mock.calls[0];
    expect(id).toBe(doneRun.id);
    expect(body.mode).toBe("finetune");
    expect(body.params.epochs).toBe(12);
  });
});

describe("TrainSidebar Pause SAM3 checkbox", () => {
  const onSam3: TrainStatus = { ...status, sam3_gpu: 0, sam3_vram_bytes: 3.4e9 };
  const pauseBox = () => screen.queryByRole("checkbox", { name: /^Pause SAM3 while training/ });

  function renderNewRuns(st: TrainStatus, extra: {
    onStart?: (body: unknown) => Promise<void>;
    runs?: Run[];
    sourceRunId?: string | null;
    onStartFromCheckpoint?: (id: string, body: unknown) => Promise<void>;
    selectedRun?: Run | null;
    onResume?: (id: string, gpu: number, pauseSam3?: boolean) => Promise<void>;
  } = {}) {
    return render(
      <TrainSidebar status={st} exports={[exportInfo()]} runs={extra.runs ?? []} exportId="e1"
        onExportChange={() => {}} onExportsChanged={() => {}} onStart={(extra.onStart ?? noop) as never}
        onCancel={() => {}} selectedRun={extra.selectedRun ?? null} onResume={extra.onResume ?? noop} onComplete={noop}
        sourceRunId={extra.sourceRunId ?? null} onSourceRunChange={() => {}}
        onStartFromCheckpoint={(extra.onStartFromCheckpoint ?? noop) as never} onTrainOnNewData={() => {}} />,
    );
  }

  it("appears only with the SAM3 warning, unchecked, naming the VRAM it frees", () => {
    renderNewRuns(status); // SAM3's GPU unknown: no warning, no checkbox
    expect(pauseBox()).toBeNull();
    cleanup();
    renderNewRuns(onSam3);
    expect(screen.getByText(/SAM3 runs on cuda:0/)).toBeTruthy();
    const box = pauseBox() as HTMLInputElement;
    expect(box.checked).toBe(false);
    expect(box.closest("label")?.textContent).toBe("Pause SAM3 while training (frees about 3.4 GB of VRAM)");
    fireEvent.change(screen.getByLabelText("GPU"), { target: { value: "1" } }); // off SAM3's GPU
    expect(pauseBox()).toBeNull();
  });

  it("leaves out the amount when the server does not know it", () => {
    renderNewRuns({ ...onSam3, sam3_vram_bytes: null });
    expect(pauseBox()?.closest("label")?.textContent).toBe("Pause SAM3 while training");
  });

  it("sends pause_sam3 only when checked", async () => {
    const onStart = vi.fn().mockResolvedValue(undefined);
    renderNewRuns(onSam3, { onStart });
    fireEvent.click(screen.getByRole("button", { name: "Start 1 run" }));
    await Promise.resolve();
    expect(onStart.mock.calls[0][0]).not.toHaveProperty("pause_sam3");
    fireEvent.click(pauseBox()!);
    expect(screen.getByText(/SAM3 reloads by itself/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Start 1 run" }));
    await Promise.resolve();
    expect(onStart.mock.calls[1][0]).toMatchObject({ runs: [{ model: "yolo11s-seg", gpu: 0 }], pause_sam3: true });
  });

  it("sends pause_sam3 when training from a checkpoint", async () => {
    const source = run({
      id: "20260924-120000-yolo11s-seg-src2", status: "done", finetunable: true, has_best: true, classes: ["Module"],
    });
    const onStartFromCheckpoint = vi.fn().mockResolvedValue(undefined);
    renderNewRuns(onSam3, { runs: [source], sourceRunId: source.id, onStartFromCheckpoint });
    fireEvent.click(pauseBox()!);
    fireEvent.click(screen.getByRole("button", { name: "Fine-tune on new data" }));
    await Promise.resolve();
    expect(onStartFromCheckpoint.mock.calls[0][1]).toMatchObject({ gpu: 0, pause_sam3: true });
  });

  it("passes the choice to Resume in the stopped-run section", async () => {
    const stopped = run({ resumable: true });
    const onResume = vi.fn().mockResolvedValue(undefined);
    renderNewRuns(onSam3, { runs: [stopped], selectedRun: stopped, onResume });
    const section = screen.getByText("Stopped yolo11s-seg run").closest("section")!;
    fireEvent.click(within(section).getByRole("checkbox", { name: /^Pause SAM3 while training/ }));
    fireEvent.click(within(section).getByRole("button", { name: "Resume on cuda:0" }));
    await Promise.resolve();
    expect(onResume).toHaveBeenCalledWith(stopped.id, 0, true);
  });
});
