// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { GpuQueue, Run, TrainStatus, VramEstimate } from "../api/types";
import type { JobHandle } from "../hooks/useJob";
import { useJob } from "../hooks/useJob";
import { VramEstimatePanel } from "./VramEstimate";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return { ...real, api: { vramEstimate: vi.fn() } };
});
vi.mock("../hooks/useJob", () => ({ useJob: vi.fn() }));

const m = vi.mocked(api);
const mUseJob = vi.mocked(useJob);
const G = 2 ** 30;

const idle: JobHandle = {
  status: "idle", nextSeq: 0, done: 0, total: null, lastFrame: null, logs: [], result: null, error: null, transport: "ws",
};
const jobs: Record<string, JobHandle> = {};

const gpu = (over: Partial<GpuQueue> = {}): GpuQueue => ({
  index: 0, name: "NVIDIA GeForce RTX 3080 Ti", memory_total: 12 * G, memory_free: 9 * G, capability: "8.6", sm_count: 80,
  tflops: 34, running: null, running_kind: null, queue: [], ...over,
});
const status = (over: Partial<TrainStatus> = {}): TrainStatus => ({
  available: true, error: null, sam3_gpu: null, default_gpu: 0, gpus: [gpu(), gpu({ index: 1, memory_free: 5 * G })],
  models: ["yolo26s-seg"], default_models: ["yolo26s-seg"],
  defaults: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0, augment: "default", aug: null },
  aug_presets: {} as TrainStatus["aug_presets"], ...over,
});
const result = (over: Partial<VramEstimate> = {}): VramEstimate => ({
  model: "yolo26s-seg", imgsz: 960, gpu: 0, gpu_name: "NVIDIA GeForce RTX 3080 Ti", total_bytes: 12 * G, free_bytes: 9 * G,
  estimates: [
    { batch: 1, bytes: 1.5 * G }, { batch: 2, bytes: 2.6 * G }, { batch: 4, bytes: 4.6 * G }, { batch: 8, bytes: 8.5 * G },
    { batch: 12, bytes: null },
  ],
  measured_at: "2026-10-16T10:00:00+00:00", cached: false, ...over,
});

type PanelProps = Partial<Parameters<typeof VramEstimatePanel>[0]>;
function renderPanel(props: PanelProps = {}) {
  const all = { status: status(), runs: [] as Run[], model: "yolo26s-seg", imgsz: 960, gpu: 0, onPickBatch: vi.fn(), ...props };
  return { ...render(<VramEstimatePanel {...all} />), props: all };
}

async function measured(res: VramEstimate, props: PanelProps = {}) {
  m.vramEstimate.mockResolvedValue({ job_id: "v1" });
  jobs.v1 = { ...idle, status: "done", result: res as unknown as Record<string, unknown> };
  const out = renderPanel(props);
  fireEvent.click(screen.getByRole("button", { name: "Estimate VRAM" }));
  await screen.findByRole("table");
  return out;
}

const rowOf = (batch: number) => screen.getByRole("row", { name: new RegExp(`^Use batch ${batch}\\b`) });

beforeEach(() => {
  m.vramEstimate.mockReset();
  Object.keys(jobs).forEach((k) => delete jobs[k]);
  mUseJob.mockReset();
  mUseJob.mockImplementation((jobId: string | null) => (jobId === null ? idle : (jobs[jobId] ?? idle)));
});
afterEach(cleanup);

describe("VramEstimatePanel", () => {
  it("starts an estimate for the chosen model, image size and GPU and shows progress", async () => {
    m.vramEstimate.mockResolvedValue({ job_id: "v1" });
    jobs.v1 = { ...idle, status: "running", done: 2, total: 5 };
    renderPanel({ gpu: 1, imgsz: 640 });
    fireEvent.click(screen.getByRole("button", { name: "Estimate VRAM" }));
    await waitFor(() => expect(m.vramEstimate).toHaveBeenCalledWith({ model: "yolo26s-seg", imgsz: 640, gpu: 1 }));
    expect(await screen.findByText("Measuring… 2 of 5 batch sizes")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Estimate VRAM" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("is disabled, with the reason as its tooltip, while the GPU is training", () => {
    renderPanel({ status: status({ gpus: [gpu({ running: "r1", running_kind: "train" })] }) });
    const btn = screen.getByRole("button", { name: "Estimate VRAM" }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.title).toMatch(/^cuda:0 is training/);
  });

  it("is disabled without a model", () => {
    renderPanel({ model: null });
    const btn = screen.getByRole("button", { name: "Estimate VRAM" }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.title).toBe("Pick a model first.");
  });

  it("shows each batch against the GPU's current free memory: fits, tight, doesn't fit", async () => {
    await measured(result());
    expect(within(rowOf(1)).getByText("about 1.5 GiB")).toBeTruthy();
    expect(within(rowOf(1)).getByText("fits")).toBeTruthy();
    expect(within(rowOf(8)).getByText("about 8.5 GiB")).toBeTruthy();
    expect(within(rowOf(8)).getByText("tight")).toBeTruthy(); // within 10 % of the 9 GiB free
    expect(within(rowOf(12)).getByText("doesn't fit")).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "cuda:0 now (9.0 GiB free)" })).toBeTruthy();
    expect(screen.queryByRole("columnheader", { name: /SAM3 paused/ })).toBeNull();
  });

  it("uses the live free memory from the status poll over the one in the answer", async () => {
    await measured(result(), { status: status({ gpus: [gpu({ memory_free: 4 * G })] }) });
    expect(within(rowOf(4)).getByText("doesn't fit")).toBeTruthy();
  });

  it("adds a 'with SAM3 paused' column when SAM3 shares the GPU", async () => {
    await measured(result(), { status: status({ sam3_gpu: 0, sam3_vram_bytes: 2 * G }) });
    expect(screen.getByRole("columnheader", { name: "With SAM3 paused (11.0 GiB)" })).toBeTruthy();
    const cells = within(rowOf(8)).getAllByRole("cell");
    expect(cells.map((c) => c.textContent)).toEqual(["8", "about 8.5 GiB", "tight", "fits"]);
    expect(within(rowOf(12)).getAllByRole("cell").map((c) => c.textContent)).toEqual(["12", "—", "doesn't fit", "doesn't fit"]);
  });

  it("no SAM3 column when SAM3 is on another GPU or its VRAM is unknown", async () => {
    await measured(result(), { status: status({ sam3_gpu: 1, sam3_vram_bytes: 2 * G }) });
    expect(screen.queryByRole("columnheader", { name: /SAM3 paused/ })).toBeNull();
    cleanup();
    await measured(result(), { status: status({ sam3_gpu: 0, sam3_vram_bytes: null }) });
    expect(screen.queryByRole("columnheader", { name: /SAM3 paused/ })).toBeNull();
  });

  it("clicking a row sets the batch", async () => {
    const { props } = await measured(result());
    fireEvent.click(rowOf(4));
    expect(props.onPickBatch).toHaveBeenCalledWith(4);
  });

  it("rows are not clickable when the batch is locked", async () => {
    await measured(result(), { onPickBatch: undefined });
    expect(screen.queryByRole("row", { name: /^Use batch/ })).toBeNull();
    expect(screen.getByText("about 1.5 GiB")).toBeTruthy();
  });

  it("re-measures, skipping the cache", async () => {
    await measured(result({ cached: true }));
    expect(screen.getByText(/measured earlier/)).toBeTruthy();
    m.vramEstimate.mockResolvedValue({ job_id: "v2" });
    fireEvent.click(screen.getByRole("button", { name: "Re-measure" }));
    await waitFor(() =>
      expect(m.vramEstimate).toHaveBeenLastCalledWith({ model: "yolo26s-seg", imgsz: 960, gpu: 0, remeasure: true }));
  });

  it("hides the table when the model, image size or GPU changes", async () => {
    const { props, rerender } = await measured(result());
    rerender(<VramEstimatePanel {...props} imgsz={640} />);
    expect(screen.queryByRole("table")).toBeNull();
    rerender(<VramEstimatePanel {...props} gpu={1} />);
    expect(screen.queryByRole("table")).toBeNull();
    rerender(<VramEstimatePanel {...props} model="yolo26m-seg" />);
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("shows a refused request's message", async () => {
    m.vramEstimate.mockRejectedValue(new ApiError(409, "a training run is queued or running on cuda:0"));
    renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "Estimate VRAM" }));
    expect((await screen.findByRole("alert")).textContent).toBe("a training run is queued or running on cuda:0");
  });

  it("shows a failed job's error", async () => {
    m.vramEstimate.mockResolvedValue({ job_id: "v1" });
    jobs.v1 = { ...idle, status: "error", error: "ProbeFailed: FileNotFoundError: yolo26s-seg.pt" };
    renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "Estimate VRAM" }));
    expect((await screen.findByRole("alert")).textContent).toBe("ProbeFailed: FileNotFoundError: yolo26s-seg.pt");
  });
});
