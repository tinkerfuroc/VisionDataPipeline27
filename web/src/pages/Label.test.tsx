// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Label } from "./Label";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { getClip: vi.fn() },
}));

const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "bench.mp4" }, fps: 10, stride: 1, width: 640, height: 480,
    frame_count: 40, created_at: "", deleted_frames: [30], original_names: { "3": "img_0007.png" },
    objects: [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states: Array.from({ length: 40 }, (_, i) => (i === 0 ? "keyframe" : i < 20 ? "masked" : i === 30 ? "deleted" : "empty")),
  classes: [{ id: 0, name: "connector", color: "#e6194b" }],
};

const renderLabel = () =>
  render(
    <MemoryRouter initialEntries={["/clips/c1"]}>
      <Routes>
        <Route path="/clips/:clipId" element={<Label />} />
      </Routes>
    </MemoryRouter>,
  );

beforeEach(() => {
  stubCanvasAndObservers();
  vi.resetAllMocks();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Label page (viewer)", () => {
  it("shows the clip, its statistics and the current frame", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    expect(await screen.findByText("bench.mp4")).toBeTruthy();
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    expect(screen.getByText(/1 keyframe/)).toBeTruthy();
    expect(screen.getByText(/51% labeled/)).toBeTruthy();
  });
  it("steps with the transport buttons, clamps at the ends and shows original file names", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Back 10 frames" }));
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    expect(screen.getByText("frame 3 / 39")).toBeTruthy();
    expect(screen.getByText("img_0007.png")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Forward 10 frames" }));
    expect(screen.getByText("frame 13 / 39")).toBeTruthy();
  });
  it("toggles playback and the overlay", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    const overlay = screen.getByRole("checkbox", { name: /overlay/i }) as HTMLInputElement;
    expect(overlay.checked).toBe(true);
    fireEvent.click(overlay);
    expect(overlay.checked).toBe(false);
  });
  it("shows a load error with a way back", async () => {
    vi.mocked(api.getClip).mockRejectedValue(new ApiError(404, "clip c1 not found"));
    renderLabel();
    expect(await screen.findByText(/clip c1 not found/)).toBeTruthy();
    expect(screen.getByRole("link", { name: /back to clips/i })).toBeTruthy();
  });
});
