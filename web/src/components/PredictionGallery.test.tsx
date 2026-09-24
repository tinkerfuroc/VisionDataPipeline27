// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { PredictionItem, PredictionPage, Run } from "../api/types";
import { PredictionGallery } from "./PredictionGallery";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return { ...real, api: { predictions: vi.fn() } };
});
const m = vi.mocked(api);

const item = (i: number, over: Partial<PredictionItem> = {}): PredictionItem => ({
  index: i, name: `f_${String(i).padStart(3, "0")}.jpg`, width: 96, height: 64, iou: 0.9,
  predictions: [{ class_id: 0, score: 0.9, polygons: [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9]] }],
  ground_truth: [{ class_id: 0, polygons: [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9]] }], ...over,
});
const page = (offset: number, limit: number, total: number, filter = "all"): PredictionPage => ({
  total, conf: 0.25, counts: { all: 30, missed: 3, worst: 29 },
  items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, k) =>
    filter === "missed" ? item(offset + k, { predictions: [], iou: 0 }) : item(offset + k)),
});
const run = { id: "r1", export_id: "e1", status: "done" } as Run;

beforeEach(() => {
  m.predictions.mockReset();
  m.predictions.mockImplementation(async (_id: string, offset: number, limit: number, filter = "all") =>
    page(offset, limit, filter === "missed" ? 3 : 30, filter));
});
afterEach(cleanup);

describe("PredictionGallery", () => {
  it("pages by four, loads more near the end, and filters", async () => {
    render(<PredictionGallery run={run} />);
    expect(await screen.findByText("1–4 of 30")).toBeTruthy();
    expect(m.predictions).toHaveBeenLastCalledWith("r1", 0, 20, "all");
    const next = screen.getByRole("button", { name: "Next four" });
    fireEvent.click(next);
    expect(screen.getByText("5–8 of 30")).toBeTruthy();
    fireEvent.click(next);
    fireEvent.click(next);
    await waitFor(() => expect(m.predictions).toHaveBeenLastCalledWith("r1", 20, 20, "all"));
    fireEvent.click(screen.getByRole("button", { name: /Missed 3/ }));
    await waitFor(() => expect(m.predictions).toHaveBeenLastCalledWith("r1", 0, 20, "missed"));
    expect(await screen.findByText("1–3 of 3")).toBeTruthy();
    expect(screen.getAllByText(/missed/).length).toBeGreaterThan(0);
  });

  it("opens a card large, steps, and returns with Escape", async () => {
    render(<PredictionGallery run={run} />);
    fireEvent.click(await screen.findByRole("button", { name: "Open f_002.jpg" }));
    expect(screen.getByText(/3 \/ 30 · f_002.jpg/)).toBeTruthy();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByText(/4 \/ 30 · f_003.jpg/)).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByText("1–4 of 30")).toBeTruthy();
  });
});
