// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExportInfo, Run } from "../api/types";
import { RunsTable } from "./RunsTable";

afterEach(cleanup);

const base: Run = {
  id: "20260924-120000-yolo26s-seg-aaaa", created_at: "", started_at: null, finished_at: null, export_id: "e", project_name: "Optics",
  model: "yolo26s-seg", gpu: 1, params: { epochs: 100, imgsz: 960, batch: 8, patience: 30, workers: 8, seed: 0 }, status: "done",
  error: null, resume: false, onnx: null, epochs_done: 100, progress: null, resumable: false, has_best: true, has_onnx: false,
  best: { epoch: 80, mask_map50: 0.9, mask_map50_95: 0.7, box_map50: 0.95, box_map50_95: 0.8, train_seconds: 3700 },
};

describe("RunsTable", () => {
  it("bolds the best value per metric and wires the actions", () => {
    const b: Run = { ...base, id: "20260924-120000-yolo26m-seg-bbbb", model: "yolo26m-seg",
      best: { ...base.best!, mask_map50_95: 0.75 } };
    const q: Run = { ...base, id: "20260924-120000-yolo11s-seg-cccc", model: "yolo11s-seg", status: "queued", best: null,
      has_best: false, epochs_done: 0 };
    const on = { onSelect: vi.fn(), onToggleCompare: vi.fn(), onCancel: vi.fn(), onResume: vi.fn(), onOnnx: vi.fn(), onDelete: vi.fn() };
    render(<RunsTable runs={[base, b, q]} selectedId={null} compareIds={[]} {...on} />);
    const rowB = screen.getByText("yolo26m-seg").closest("tr")!;
    expect(within(rowB).getByText("0.750").tagName).toBe("STRONG");
    const rowA = screen.getByText("yolo26s-seg").closest("tr")!;
    expect(within(rowA).getByText("0.700").tagName).not.toBe("STRONG");
    expect(within(rowA).getByText("1 h 02 min")).toBeTruthy();
    fireEvent.click(within(rowA).getByRole("button", { name: "Export ONNX" }));
    expect(on.onOnnx).toHaveBeenCalledWith(base.id);
    expect(on.onSelect).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByText("yolo11s-seg").closest("tr")!).getByRole("button", { name: "Cancel" }));
    expect(on.onCancel).toHaveBeenCalledWith(q.id);
    fireEvent.click(rowA);
    expect(on.onSelect).toHaveBeenCalledWith(base.id);
  });

  it("shows the export id under the project name, so a run on an augmented export is distinguishable", () => {
    const original: Run = { ...base, id: "20260924-120000-yolo26s-seg-eeee" };
    const augmented: Run = { ...base, id: "20260924-120000-yolo26s-seg-ffff", export_id: "e-aug3x-3385" };
    const on = { onSelect: vi.fn(), onToggleCompare: vi.fn(), onCancel: vi.fn(), onResume: vi.fn(), onOnnx: vi.fn(), onDelete: vi.fn() };
    render(<RunsTable runs={[original, augmented]} selectedId={null} compareIds={[]} {...on} />);
    const rows = screen.getAllByText("Optics").map((el) => el.closest("tr")!);
    expect(within(rows[0]).getByText("e")).toBeTruthy();
    expect(within(rows[1]).getByText("e-aug3x-3385")).toBeTruthy();
  });

  it("shows the export's name from `exports`, falling back to the project name when unknown or unnamed", () => {
    const named: Run = { ...base, id: "20260924-120000-yolo26s-seg-gggg", export_id: "cups-v2" };
    const unnamed: Run = { ...base, id: "20260924-120000-yolo26s-seg-hhhh", export_id: "e-plain" };
    const unknown: Run = { ...base, id: "20260924-120000-yolo26s-seg-iiii", export_id: "e-gone" };
    const exports: ExportInfo[] = [
      { id: "cups-v2", name: "Cups v2", project_name: "Optics", created_at: "", classes: [], train_images: 0, val_images: 0, formats: [], derived_from: null },
      { id: "e-plain", name: null, project_name: "Optics", created_at: "", classes: [], train_images: 0, val_images: 0, formats: [], derived_from: null },
    ];
    const on = { onSelect: vi.fn(), onToggleCompare: vi.fn(), onCancel: vi.fn(), onResume: vi.fn(), onOnnx: vi.fn(), onDelete: vi.fn() };
    render(<RunsTable runs={[named, unnamed, unknown]} selectedId={null} compareIds={[]} exports={exports} {...on} />);

    const namedRow = screen.getByText("Cups v2").closest("tr")!;
    expect(within(namedRow).getByText("cups-v2")).toBeTruthy();

    const fallbackRows = screen.getAllByText("Optics").map((el) => el.closest("tr")!);
    expect(fallbackRows).toHaveLength(2);
    expect(within(fallbackRows[0]).getByText("e-plain")).toBeTruthy();
    expect(within(fallbackRows[1]).getByText("e-gone")).toBeTruthy();
  });

  it("shows the Aug column from params.augment, defaulting to default", () => {
    const strong: Run = { ...base, id: "20260924-120000-yolo26m-seg-dddd", model: "yolo26m-seg",
      params: { ...base.params, augment: "strong" } };
    const on = { onSelect: vi.fn(), onToggleCompare: vi.fn(), onCancel: vi.fn(), onResume: vi.fn(), onOnnx: vi.fn(), onDelete: vi.fn() };
    render(<RunsTable runs={[base, strong]} selectedId={null} compareIds={[]} {...on} />);
    const rowDefault = screen.getByText("yolo26s-seg").closest("tr")!;
    const rowStrong = screen.getByText("yolo26m-seg").closest("tr")!;
    expect(within(rowDefault).getByText("default")).toBeTruthy();
    expect(within(rowStrong).getByText("strong")).toBeTruthy();
  });
});
