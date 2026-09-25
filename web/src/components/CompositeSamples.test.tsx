// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { ExportInfo, ExportSample } from "../api/types";
import { CompositeSamples } from "./CompositeSamples";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return { ...real, api: { exportSamples: vi.fn() } };
});
const m = vi.mocked(api);

const exp = (over: Partial<ExportInfo> = {}): ExportInfo => ({
  id: "expA", project_name: "OpticsConverter1", created_at: "", classes: ["Module"],
  train_images: 3192, val_images: 252, formats: ["yolo_seg", "masks"], derived_from: "src", ...over,
});
const sample = (n: number): ExportSample => ({
  name: `syn_0_${String(n).padStart(6, "0")}.jpg`, width: 640, height: 480,
  ground_truth: [{ class_id: 0, polygons: [[0.1, 0.1, 0.9, 0.9]] }],
});

beforeEach(() => {
  m.exportSamples.mockReset();
  m.exportSamples.mockResolvedValue([sample(1), sample(2)]);
});
afterEach(cleanup);

describe("CompositeSamples", () => {
  it("fetches samples for the export and increments the seed on Shuffle", async () => {
    render(<CompositeSamples exportInfo={exp()} />);
    await waitFor(() => expect(m.exportSamples).toHaveBeenLastCalledWith("expA", 12, 0));
    fireEvent.click(await screen.findByRole("button", { name: "Shuffle" }));
    await waitFor(() => expect(m.exportSamples).toHaveBeenLastCalledWith("expA", 12, 1));
  });

  it("resets the shuffle seed to 0 when the export changes", async () => {
    const { rerender } = render(<CompositeSamples exportInfo={exp()} />);
    await waitFor(() => expect(m.exportSamples).toHaveBeenLastCalledWith("expA", 12, 0));
    fireEvent.click(await screen.findByRole("button", { name: "Shuffle" }));
    await waitFor(() => expect(m.exportSamples).toHaveBeenLastCalledWith("expA", 12, 1));

    rerender(<CompositeSamples exportInfo={exp({ id: "expB" })} />);
    await waitFor(() => expect(m.exportSamples).toHaveBeenLastCalledWith("expB", 12, 0));

    // a subsequent Shuffle on the new export starts counting from its own reset seed, not the old one
    fireEvent.click(await screen.findByRole("button", { name: "Shuffle" }));
    await waitFor(() => expect(m.exportSamples).toHaveBeenLastCalledWith("expB", 12, 1));
  });
});
