// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { BackgroundPool, ExportInfo, ExportSample, Run, RunDetail, TrainStatus } from "../api/types";
import { SERIES_COLORS } from "../lib/chart";
import { runLabel } from "../lib/train";
import { Train } from "./Train";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: {
      health: vi.fn(), trainStatus: vi.fn(), listExports: vi.fn(), listRuns: vi.fn(), getRun: vi.fn(), startRuns: vi.fn(),
      cancelRun: vi.fn(), resumeRun: vi.fn(), exportOnnx: vi.fn(), deleteRun: vi.fn(), predictions: vi.fn(),
      listBackgrounds: vi.fn(), fetchBackgrounds: vi.fn(), augmentExport: vi.fn(), exportSamples: vi.fn(), renameExport: vi.fn(),
    },
  };
});

const status: TrainStatus = {
  available: true, error: null, sam3_gpu: 0, default_gpu: 0,
  gpus: [
    // Best (fastest estimated) first: cuda:0 is the fastest GPU here, and idle, so it is the
    // default even though SAM3 shares it. cuda:1 is slower and already training r1 (queue: r2).
    {
      index: 0, name: "NVIDIA GeForce RTX 3080 Ti", memory_total: 12 * 2 ** 30, memory_free: 9.1 * 2 ** 30,
      capability: "8.6", sm_count: 80, tflops: 34, running: null, running_kind: null, queue: [],
    },
    {
      index: 1, name: "NVIDIA GeForce RTX 2080 Ti", memory_total: 11 * 2 ** 30, memory_free: 5.5 * 2 ** 30,
      capability: "7.5", sm_count: 68, tflops: 13.5, running: "r1", running_kind: "train", queue: ["r2"],
    },
  ],
  models: ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg"],
  default_models: ["yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg"],
  defaults: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0, augment: "default", aug: null },
  aug_presets: {
    default: {
      degrees: 0, shear: 0, perspective: 0, flipud: 0, fliplr: 0.5, scale: 0.5, translate: 0.1,
      hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, mosaic: 1.0, mixup: 0, copy_paste: 0, photometric: false,
    },
    strong: {
      degrees: 15, shear: 5, perspective: 0.0005, flipud: 0.5, fliplr: 0.5, scale: 0.6, translate: 0.2,
      hsv_h: 0.02, hsv_s: 0.7, hsv_v: 0.5, mosaic: 1.0, mixup: 0.1, copy_paste: 0.3, photometric: true,
    },
  },
};
const exp: ExportInfo = {
  id: "20260924-040124-8336a8", name: null, project_name: "OpticsConverter1", created_at: "", classes: ["Module"],
  train_images: 798, val_images: 252, formats: ["yolo_seg", "masks"], derived_from: null,
};
const derivedExp: ExportInfo = {
  id: "20260924-040124-8336a6-aug3x-3385", name: null, project_name: "OpticsConverter1", created_at: "", classes: ["Module"],
  train_images: 3192, val_images: 252, formats: ["yolo_seg", "masks"], derived_from: exp.id,
};
const pool: BackgroundPool = { name: "dtd", label: "DTD textures", images: 5640, size_bytes: 625_239_812, fetching: null };
const run = (id: string, model: string, over: Partial<Run> = {}): Run => ({
  id, created_at: "", started_at: null, finished_at: null, export_id: exp.id, project_name: "OpticsConverter1", model, gpu: 1,
  params: status.defaults, status: "queued", error: null, resume: false, onnx: null, best: null, epochs_done: 0, progress: null,
  resumable: false, has_best: false, has_onnx: false, ...over,
});
const r1 = run("r1", "yolo26s-seg", { status: "running", epochs_done: 2, progress: { phase: "train", epoch: 3, epochs: 100, batch: 5, batches: 100 } });
const r2 = run("r2", "yolo26m-seg");
const m = vi.mocked(api);

const renderPage = () => render(<MemoryRouter><Train /></MemoryRouter>);

beforeEach(() => {
  Object.values(m).forEach((fn) => fn.mockReset());
  m.trainStatus.mockResolvedValue(status);
  m.listExports.mockResolvedValue([exp]);
  m.listRuns.mockResolvedValue([r1, r2]);
  m.listBackgrounds.mockResolvedValue([pool]);
  m.exportSamples.mockResolvedValue([]);
  m.getRun.mockImplementation(async (id: string) => ({ ...(id === "r1" ? r1 : r2), metrics: [
    { epoch: 1, "metrics/mAP50-95(M)": 0.2, "metrics/mAP50(M)": 0.4 },
    { epoch: 2, "metrics/mAP50-95(M)": 0.3, "metrics/mAP50(M)": 0.5 },
  ] }) as RunDetail);
  m.startRuns.mockResolvedValue({ run_ids: ["r3"] });
  m.cancelRun.mockResolvedValue(r2);
});
afterEach(cleanup);

describe("Train page", () => {
  it("starts the default models on the fastest idle GPU (SAM3's), with the export and params", async () => {
    renderPage();
    const start = await screen.findByRole("button", { name: "Start 4 runs" });
    expect((screen.getByLabelText("GPU") as HTMLSelectElement).value).toBe("0");
    fireEvent.change(screen.getByLabelText("Epochs"), { target: { value: "50" } });
    fireEvent.click(start);
    await waitFor(() => expect(m.startRuns).toHaveBeenCalled());
    const body = m.startRuns.mock.calls[0][0];
    expect(body.export_id).toBe(exp.id);
    expect(body.runs).toEqual([
      { model: "yolo26s-seg", gpu: 0 }, { model: "yolo26m-seg", gpu: 0 }, { model: "yolo11s-seg", gpu: 0 }, { model: "yolo11m-seg", gpu: 0 },
    ]);
    expect(body.params.epochs).toBe(50);
  });

  it("lists GPUs best first in the batch select, labelled with estimated TFLOPS, free VRAM and SAM3", async () => {
    renderPage();
    const select = (await screen.findByLabelText("GPU")) as HTMLSelectElement;
    const options = Array.from(select.options).map((o) => o.text);
    expect(options).toEqual([
      "cuda:0 · RTX 3080 Ti · est. 34 TFLOPS · 9.1 GiB free · SAM3 · default",
      "cuda:1 · RTX 2080 Ti · est. 14 TFLOPS · 5.5 GiB free",
    ]);
  });

  it("Different GPU per model reveals per-model overrides seeded from the batch choice", async () => {
    renderPage();
    await screen.findByRole("button", { name: "Start 4 runs" });
    expect(screen.queryByLabelText("GPU for yolo26m-seg")).toBeNull();

    fireEvent.click(screen.getByLabelText("Different GPU per model"));
    const perModel = (await screen.findByLabelText("GPU for yolo26m-seg")) as HTMLSelectElement;
    expect(perModel.value).toBe("0"); // seeded from the batch choice (cuda:0)

    fireEvent.change(perModel, { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Start 4 runs" }));
    await waitFor(() => expect(m.startRuns).toHaveBeenCalled());
    const body = m.startRuns.mock.calls[0][0];
    expect(body.runs).toEqual([
      { model: "yolo26s-seg", gpu: 0 }, { model: "yolo26m-seg", gpu: 1 }, { model: "yolo11s-seg", gpu: 0 }, { model: "yolo11m-seg", gpu: 0 },
    ]);
  });

  it("starts with the strong augmentation preset and no custom aug values", async () => {
    renderPage();
    const start = await screen.findByRole("button", { name: "Start 4 runs" });
    fireEvent.change(screen.getByLabelText("Augmentation"), { target: { value: "strong" } });
    fireEvent.click(start);
    await waitFor(() => expect(m.startRuns).toHaveBeenCalled());
    const body = m.startRuns.mock.calls[0][0];
    expect(body.params.augment).toBe("strong");
    expect(body.params.aug).toBeNull();
  });

  it("starts with custom augmentation values prefilled from the strong preset", async () => {
    renderPage();
    const start = await screen.findByRole("button", { name: "Start 4 runs" });
    fireEvent.change(screen.getByLabelText("Augmentation"), { target: { value: "custom" } });
    fireEvent.change(screen.getByLabelText("degrees"), { target: { value: "7" } });
    fireEvent.click(start);
    await waitFor(() => expect(m.startRuns).toHaveBeenCalled());
    const body = m.startRuns.mock.calls[0][0];
    expect(body.params.aug?.degrees).toBe(7);
    expect(body.params.aug?.mixup).toBe(0.1);
  });

  it("warns about SAM3 by default (its GPU is the fastest idle one), clearing when a different GPU is picked", async () => {
    renderPage();
    expect(await screen.findByText(/SAM3 runs on cuda:0/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("GPU"), { target: { value: "1" } });
    expect(screen.queryByText(/SAM3 runs on cuda:0/)).toBeNull();
  });

  it("shows the queues, cancels a queued run, and charts the selected run", async () => {
    renderPage();
    const queue = await screen.findByText("GPU queues");
    const panel = queue.closest("section")!;
    expect(within(panel).getByText(/yolo26s-seg · Epoch 3\/100/)).toBeTruthy();
    fireEvent.click(within(panel).getByRole("button", { name: "Cancel queued yolo26m-seg" }));
    await waitFor(() => expect(m.cancelRun).toHaveBeenCalledWith("r2"));
    expect(await screen.findByRole("img", { name: "Mask mAP by epoch" })).toBeTruthy();
    expect(m.getRun).toHaveBeenCalledWith("r1");
  });

  it("explains when training is unavailable", async () => {
    m.trainStatus.mockResolvedValue({ ...status, available: false, error: "ultralytics is not installed in this environment; run `uv sync`" });
    renderPage();
    expect(await screen.findByText(/ultralytics is not installed/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Start 4 runs" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows GPU-busy notes for New runs and lists a foreign process's run in GPU queues", async () => {
    const r3 = run("r3", "yolo11m-seg", {
      status: "running", gpu: 0,
      progress: { phase: "train", epoch: 5, epochs: 100, batch: 2, batches: 20 },
    });
    m.listRuns.mockResolvedValue([r1, r2, r3]);
    renderPage();

    // The batch default (cuda:0, the fastest idle GPU by this server's own bookkeeping) is busy
    // with a foreign `vsp train` process's run that this server does not own.
    expect(await screen.findByText(/cuda:0 is busy with a vsp train run in another process/)).toBeTruthy();

    // Switching to cuda:1 instead shows this server's own queue-behind note (it is training r1).
    fireEvent.change(screen.getByLabelText("GPU"), { target: { value: "1" } });
    expect(
      await screen.findByText("cuda:1 is already training yolo26s-seg; this run will queue behind it."),
    ).toBeTruthy();

    const queue = await screen.findByText("GPU queues");
    const panel = queue.closest("section")!;
    const gpu0 = within(panel).getByText(/cuda:0 NVIDIA GeForce RTX 3080 Ti/).parentElement!.parentElement!;
    expect(
      within(gpu0).getByText("yolo11m-seg · Epoch 5/100 · running in another process (vsp train)"),
    ).toBeTruthy();
    expect(within(gpu0).queryByText("Idle")).toBeNull();
    expect(within(gpu0).queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("keeps a compared run's chart colour on its slot across removal, and hands a new run the freed slot", async () => {
    const r3 = run("r3", "yolo26n-seg", { status: "done" });
    const r4 = run("r4", "yolo11s-seg", { status: "done" });
    const byId: Record<string, Run> = { r1, r2, r3, r4 };
    m.listRuns.mockResolvedValue([r1, r2, r3, r4]);
    m.getRun.mockImplementation(
      async (id: string) =>
        ({
          ...byId[id],
          metrics: [
            { epoch: 1, "metrics/mAP50-95(M)": 0.1 },
            { epoch: 2, "metrics/mAP50-95(M)": 0.2 },
          ],
        }) as RunDetail,
    );

    renderPage();
    fireEvent.click(await screen.findByLabelText("Compare yolo26s-seg")); // r1 -> slot 0
    fireEvent.click(screen.getByLabelText("Compare yolo26m-seg")); // r2 -> slot 1
    fireEvent.click(screen.getByLabelText("Compare yolo26n-seg")); // r3 -> slot 2

    const chart = await screen.findByRole("img", { name: "Mask mAP50-95, compared by epoch" });
    const strokeOf = (r: Run) => chart.querySelector(`path[data-series="${runLabel(r)}"]`)?.getAttribute("stroke");
    await waitFor(() => {
      expect(strokeOf(r1)).toBe(SERIES_COLORS[0]);
      expect(strokeOf(r2)).toBe(SERIES_COLORS[1]);
      expect(strokeOf(r3)).toBe(SERIES_COLORS[2]);
    });

    fireEvent.click(screen.getByLabelText("Compare yolo26m-seg")); // remove r2, freeing slot 1
    await waitFor(() => {
      // Compare against a boolean, not the raw element: asserting a DOM node directly makes a failed retry's
      // message inspect React's internal props on it, which trips React's dev-only "`key` is not a prop" trap.
      expect(chart.querySelector(`path[data-series="${runLabel(r2)}"]`) === null).toBe(true);
      expect(strokeOf(r1)).toBe(SERIES_COLORS[0]); // unchanged
      expect(strokeOf(r3)).toBe(SERIES_COLORS[2]); // unchanged, did not shift down into slot 1
    });

    fireEvent.click(screen.getByLabelText("Compare yolo11s-seg")); // r4 takes r2's freed slot 1
    await waitFor(() => expect(strokeOf(r4)).toBe(SERIES_COLORS[1]));
  });

  it("shows composite samples when the selected export is a derived (augmented) export", async () => {
    m.listExports.mockResolvedValue([exp, derivedExp]);
    const sample: ExportSample = {
      name: "syn_0_000001.jpg", width: 640, height: 480, ground_truth: [{ class_id: 0, polygons: [[0.1, 0.1, 0.9, 0.9]] }],
    };
    m.exportSamples.mockResolvedValue([sample]);
    renderPage();

    expect(screen.queryByText(/Composite samples/)).toBeNull();
    fireEvent.change(await screen.findByLabelText("Export"), { target: { value: derivedExp.id } });
    expect(await screen.findByText(`Composite samples · ${derivedExp.id}`)).toBeTruthy();
    expect(m.exportSamples).toHaveBeenCalledWith(derivedExp.id, 12, 0);
  });

  it("shows the export's name at the start of its select option, else the project name", async () => {
    const named: ExportInfo = { ...exp, id: "cups-v2", name: "Cups v2" };
    m.listExports.mockResolvedValue([exp, named]);
    renderPage();
    const select = (await screen.findByLabelText("Export")) as HTMLSelectElement;
    const optionTexts = Array.from(select.options).map((o) => o.text);
    expect(optionTexts.some((t) => t.startsWith(`Cups v2 · ${named.id}`))).toBe(true);
    expect(optionTexts.some((t) => t.startsWith(`${exp.project_name} · ${exp.id}`))).toBe(true);
  });

  it("shows the selected run's export name in the run-detail heading, else the project name", async () => {
    const named: ExportInfo = { ...exp, id: "cups-v2", name: "Cups v2" };
    m.listExports.mockResolvedValue([exp, named]);
    m.listRuns.mockResolvedValue([r1, { ...r2, export_id: named.id }]);
    renderPage();

    // r1 is running, so it is selected by default; its export (`exp`) has no name, so the heading
    // falls back to the run's project name.
    expect(await screen.findByText(`yolo26s-seg on ${exp.project_name} · Epoch 3/100`)).toBeTruthy();

    // Selecting r2 (whose export is named) switches the heading to the export's name.
    fireEvent.click(screen.getByRole("checkbox", { name: "Compare yolo26m-seg" }).closest("tr")!);
    expect(await screen.findByText("yolo26m-seg on Cups v2 · Queued")).toBeTruthy();
  });

  it("renames the selected export: prefilled field, Save disabled until changed, then refreshes the exports", async () => {
    const named: ExportInfo = { ...exp, name: "Cups v2" };
    m.listExports.mockResolvedValue([named]);
    m.renameExport.mockResolvedValue({ ...named, name: "Better" });
    renderPage();
    await screen.findByLabelText("Export");

    const input = screen.getByLabelText("Rename export") as HTMLInputElement;
    expect(input.value).toBe("Cups v2");
    const save = screen.getByRole("button", { name: "Save" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);

    fireEvent.change(input, { target: { value: "Better" } });
    expect(save.disabled).toBe(false);
    expect(m.listExports).toHaveBeenCalledTimes(1);

    fireEvent.click(save);
    await waitFor(() => expect(m.renameExport).toHaveBeenCalledWith(named.id, "Better"));
    await waitFor(() => expect(m.listExports).toHaveBeenCalledTimes(2));
  });
});
