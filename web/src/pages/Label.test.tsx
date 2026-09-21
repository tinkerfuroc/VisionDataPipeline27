// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { HealthContext, type HealthState } from "../hooks/useHealth";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Label } from "./Label";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { getClip: vi.fn(), prompt: vi.fn(), accept: vi.fn(), addClass: vi.fn() },
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

const ready: HealthState = {
  health: { status: "ok", version: "0", engine: { loaded: true, device: "cuda:0", error: null } },
  unreachable: false,
};

const renderLabel = (health: HealthState = ready) =>
  render(
    <HealthContext.Provider value={health}>
      <MemoryRouter initialEntries={["/clips/c1"]}>
        <Routes>
          <Route path="/clips/:clipId" element={<Label />} />
        </Routes>
      </MemoryRouter>
    </HealthContext.Provider>,
  );

beforeEach(() => {
  stubCanvasAndObservers();
  vi.resetAllMocks();
  // a 656×496 box fits the 640×480 frame at scale 1 with an 8 px margin, so screen (108, 58) is image pixel (100, 50)
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 656 });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => 496 });
  HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ left: 0, top: 0, width: 656, height: 496 }) as DOMRect;
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

describe("Label page (prompting)", () => {
  it("previews a click on the canvas and accepts it", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockResolvedValue({
      candidates: [{ score: 0.9, bbox: [80, 30, 120, 70], area: 1200, phrase: null, mask_png_b64: "AAAA" }],
    });
    const accepted = { ...detail, states: detail.states.map((s, i) => (i === 25 ? "keyframe" : s)) } as ClipDetail;
    vi.mocked(api.accept).mockResolvedValue(accepted);
    renderLabel();
    await screen.findByText("bench.mp4");
    for (let i = 0; i < 25; i++) fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 0 });
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 25, { type: "clicks", points: [[100, 50, 1]] }));
    const accept = screen.getByRole("button", { name: /Accept/ }) as HTMLButtonElement;
    await waitFor(() => expect(accept.disabled).toBe(false));
    fireEvent.click(accept);
    await waitFor(() => expect(api.accept).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/2 keyframes/)).toBeTruthy();
  });

  it("right-click adds a negative point; prompting is off while the engine is not loaded", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [] });
    const { unmount } = renderLabel();
    await screen.findByText("bench.mp4");
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 2 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 2 });
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 0]] }));
    unmount();

    vi.mocked(api.prompt).mockClear();
    renderLabel({ health: { status: "ok", version: "0", engine: { loaded: false, device: null, error: null } }, unreachable: false });
    await screen.findByText("bench.mp4");
    const again = screen.getByLabelText("Frame");
    fireEvent.mouseDown(again, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(again, { clientX: 108, clientY: 58, button: 0 });
    expect(screen.getByText(/engine is not loaded/i)).toBeTruthy();
    expect(api.prompt).not.toHaveBeenCalled();
  });
});
