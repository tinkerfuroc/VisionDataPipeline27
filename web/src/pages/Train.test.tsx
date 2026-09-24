// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { ExportInfo, Run, RunDetail, TrainStatus } from "../api/types";
import { Train } from "./Train";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: {
      health: vi.fn(), trainStatus: vi.fn(), listExports: vi.fn(), listRuns: vi.fn(), getRun: vi.fn(), startRuns: vi.fn(),
      cancelRun: vi.fn(), resumeRun: vi.fn(), exportOnnx: vi.fn(), deleteRun: vi.fn(), predictions: vi.fn(),
    },
  };
});

const status: TrainStatus = {
  available: true, error: null, sam3_gpu: 0,
  gpus: [
    { index: 0, name: "RTX 3080 Ti", memory_total: 12 * 2 ** 30, running: null, running_kind: null, queue: [] },
    { index: 1, name: "RTX 2080 Ti", memory_total: 11 * 2 ** 30, running: "r1", running_kind: "train", queue: ["r2"] },
  ],
  models: ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg"],
  default_models: ["yolo26s-seg", "yolo26m-seg", "yolo11s-seg", "yolo11m-seg"],
  defaults: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 },
};
const exp: ExportInfo = { id: "20260924-040124-8336a8", project_name: "OpticsConverter1", created_at: "", classes: ["Module"], train_images: 798, val_images: 252 };
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
  m.getRun.mockImplementation(async (id: string) => ({ ...(id === "r1" ? r1 : r2), metrics: [
    { epoch: 1, "metrics/mAP50-95(M)": 0.2, "metrics/mAP50(M)": 0.4 },
    { epoch: 2, "metrics/mAP50-95(M)": 0.3, "metrics/mAP50(M)": 0.5 },
  ] }) as RunDetail);
  m.startRuns.mockResolvedValue({ run_ids: ["r3"] });
  m.cancelRun.mockResolvedValue(r2);
});
afterEach(cleanup);

describe("Train page", () => {
  it("starts the default models on the non-SAM3 GPU with the export and params", async () => {
    renderPage();
    const start = await screen.findByRole("button", { name: "Start 4 runs" });
    fireEvent.change(screen.getByLabelText("Epochs"), { target: { value: "50" } });
    fireEvent.click(start);
    await waitFor(() => expect(m.startRuns).toHaveBeenCalled());
    const body = m.startRuns.mock.calls[0][0];
    expect(body.export_id).toBe(exp.id);
    expect(body.runs).toEqual([
      { model: "yolo26s-seg", gpu: 1 }, { model: "yolo26m-seg", gpu: 1 }, { model: "yolo11s-seg", gpu: 1 }, { model: "yolo11m-seg", gpu: 1 },
    ]);
    expect(body.params.epochs).toBe(50);
  });

  it("warns when a model is put on SAM3's GPU", async () => {
    renderPage();
    const select = await screen.findByLabelText("GPU for yolo26m-seg");
    expect(screen.queryByText(/SAM3 runs on cuda:0/)).toBeNull();
    fireEvent.change(select, { target: { value: "0" } });
    expect(screen.getByText(/SAM3 runs on cuda:0/)).toBeTruthy();
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
});
