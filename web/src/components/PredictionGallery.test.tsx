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

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

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

  it("scrolls the remounted strip back under the counter after closing a card past the first page", async () => {
    const scrollIntoView = vi.fn();
    const original = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = scrollIntoView;
    try {
      render(<PredictionGallery run={run} />);
      fireEvent.click(await screen.findByRole("button", { name: "Open f_009.jpg" }));
      expect(screen.getByText(/10 \/ 30 · f_009.jpg/)).toBeTruthy();
      scrollIntoView.mockClear();
      fireEvent.keyDown(window, { key: "Escape" });
      expect(screen.getByText("9–12 of 30")).toBeTruthy();
      expect(scrollIntoView).toHaveBeenCalledWith({ inline: "start", block: "nearest", behavior: "auto" });
    } finally {
      Element.prototype.scrollIntoView = original;
    }
  });

  it("does not let the previous filter's total leak into the counter while the next one is loading", async () => {
    render(<PredictionGallery run={run} />);
    await screen.findByText("1–4 of 30");

    const gate = deferred<PredictionPage>();
    m.predictions.mockImplementation(async (_id: string, offset: number, limit: number, filter = "all") =>
      filter === "missed" ? gate.promise : page(offset, limit, 30, filter));

    fireEvent.click(screen.getByRole("button", { name: /Missed 3/ }));
    // the "missed" fetch is still pending: nothing on screen may use the stale total of 30, and there
    // is no Next button that could page `first` past a total that hasn't arrived yet.
    expect(screen.queryByText(/of 30/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Next four" })).toBeNull();

    gate.resolve(page(0, 20, 3, "missed"));
    expect(await screen.findByText("1–3 of 3")).toBeTruthy();
  });

  it("reads the strip's real column gap instead of an assumed 12px", async () => {
    const { container } = render(<PredictionGallery run={run} />);
    await screen.findByText("1–4 of 30");
    const strip = container.querySelector(".snap-x") as HTMLDivElement;
    const card = strip.children[0] as HTMLElement;
    Object.defineProperty(card, "offsetWidth", { configurable: true, value: 20 });
    // card 11 sits exactly at the left edge once a 13.5px gap (gap-3 at this app's 112.5% root font)
    // is accounted for; the old hardcoded-12px math rounds that same scrollLeft up to card 12.
    const gapSpy = vi.spyOn(window, "getComputedStyle").mockReturnValue({ columnGap: "13.5px" } as CSSStyleDeclaration);
    fireEvent.scroll(strip, { target: { scrollLeft: 11 * (20 + 13.5) } });
    gapSpy.mockRestore();
    expect(screen.getByText("12–15 of 30")).toBeTruthy();
  });

  it("ignores a hand-scroll event while a ‹ ›-triggered scroll is still settling", async () => {
    const { container } = render(<PredictionGallery run={run} />);
    await screen.findByText("1–4 of 30");
    const strip = container.querySelector(".snap-x") as HTMLDivElement;
    Object.defineProperty(strip.children[0], "offsetWidth", { configurable: true, value: 100 });

    vi.useFakeTimers();
    try {
      fireEvent.click(screen.getByRole("button", { name: "Next four" }));
      expect(screen.getByText("5–8 of 30")).toBeTruthy();

      // a stray scroll while the click's own scroll is still settling must not fight it
      fireEvent.scroll(strip, { target: { scrollLeft: 0 } });
      expect(screen.getByText("5–8 of 30")).toBeTruthy();

      vi.advanceTimersByTime(500); // the scrollend-less (jsdom) fallback clears the guard
    } finally {
      vi.useRealTimers();
    }

    // once settled, hand-scrolling drives the counter again
    fireEvent.scroll(strip, { target: { scrollLeft: 0 } });
    expect(screen.getByText("1–4 of 30")).toBeTruthy();
  });
});
