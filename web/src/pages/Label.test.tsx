// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { HealthContext, type HealthState } from "../hooks/useHealth";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Label } from "./Label";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: {
    getClip: vi.fn(),
    prompt: vi.fn(),
    accept: vi.fn(),
    addClass: vi.fn(),
    propagate: vi.fn(),
    cancelJob: vi.fn(),
    deleteFrame: vi.fn(),
    restoreFrame: vi.fn(),
    prune: vi.fn(),
    clearRange: vi.fn(),
    getJob: vi.fn(),
  },
}));

class FakeSocket {
  static last: FakeSocket | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(readonly url: string) {
    FakeSocket.last = this;
  }
  close() {}
  emit(event: object) {
    act(() => this.onmessage?.({ data: JSON.stringify(event) }));
  }
}

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
  FakeSocket.last = null;
  vi.stubGlobal("WebSocket", FakeSocket);
  sessionStorage.clear();
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
    // both the prompt panel and (now) the propagate panel report the disabled engine
    expect(screen.getAllByText(/engine is not loaded/i).length).toBeGreaterThan(0);
    expect(api.prompt).not.toHaveBeenCalled();
  });
});

describe("Label page (propagation, cleanup, keyboard)", () => {
  it("propagates from a keyframe, follows the streamed frames and reloads when the job ends", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j1" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(api.propagate).toHaveBeenCalledWith("c1", 0, "forward"));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));
    expect(sessionStorage.getItem("vsp.job.c1")).toBe("j1");
    FakeSocket.last!.emit({ seq: 0, type: "progress", done: 0, total: 39 });
    FakeSocket.last!.emit({ seq: 1, type: "frame", frame: 20, has_mask: true, done: 1, total: 39 });
    // F3: frame events are buffered and applied on the next animation frame, not synchronously
    expect(await screen.findByText("frame 20 / 39")).toBeTruthy();
    expect(screen.getByText(/20 masked/)).toBeTruthy();
    expect(screen.getByText(/propagation is running/i)).toBeTruthy();
    const loads = vi.mocked(api.getClip).mock.calls.length;
    FakeSocket.last!.emit({ seq: 2, type: "done", processed: 39, written: 38, lost_at: 33, cancelled: false });
    await waitFor(() => expect(vi.mocked(api.getClip).mock.calls.length).toBe(loads + 1));
    expect(await screen.findByText("frame 33 / 39")).toBeTruthy();
    expect(screen.getByText(/lost the object at frame 33/i)).toBeTruthy();
    expect(sessionStorage.getItem("vsp.job.c1")).toBeNull();
  });

  it("only offers propagation on a keyframe, and Stop cancels the job", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j2" });
    vi.mocked(api.cancelJob).mockResolvedValue({ id: "j2", kind: "propagate", status: "running", result: null, error: null, created_at: "", n_events: 0 });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/Go to a keyframe/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Previous frame" }));
    fireEvent.click(screen.getByRole("button", { name: "Both" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j2$/));
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(api.cancelJob).toHaveBeenCalledWith("j2"));
  });

  it("attaches to the job a 409 names, but a foreign job never drives this clip and Stop still cancels it", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockRejectedValue(new ApiError(409, "job j7 is still running", "j7"));
    vi.mocked(api.cancelJob).mockResolvedValue({
      id: "j7", kind: "propagate", status: "running", result: null, error: null, created_at: "", n_events: 0,
    });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "◀ Backward" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j7$/));
    expect(sessionStorage.getItem("vsp.job.c1")).toBeNull();
    expect(screen.getByText(/GPU is busy with a propagation that was not started from this page/i)).toBeTruthy();
    FakeSocket.last!.emit({ seq: 0, type: "frame", frame: 20, has_mask: true, done: 1, total: 39 });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    expect(screen.getByText(/19 masked/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(api.cancelJob).toHaveBeenCalledWith("j7"));
    // frame events are applied on an animation frame: look again once one has certainly passed
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    expect(screen.getByText(/19 masked/)).toBeTruthy();
  });

  it("does not re-attach to a finished foreign job when a job of its own ends", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockRejectedValue(new ApiError(409, "job j7 is still running", "j7"));
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j8" });
    renderLabel();
    await screen.findByText("bench.mp4");
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 0 });
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j7$/)); // the prompt's 409 attached the foreign job
    FakeSocket.last!.emit({ seq: 0, type: "done", processed: 5, written: 5, lost_at: null, cancelled: false });
    await waitFor(() => expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j8$/));
    FakeSocket.last!.emit({ seq: 0, type: "done", processed: 39, written: 39, lost_at: null, cancelled: false });
    expect(await screen.findByText(/Done: 39 frames processed/)).toBeTruthy();
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    expect(FakeSocket.last?.url).toMatch(/j8$/); // no new socket to j7
    expect(screen.queryByText(/not started from this page/i)).toBeNull();
  });

  it("re-attaches to its own job from sessionStorage and applies its frame events", async () => {
    sessionStorage.setItem("vsp.job.c1", "j5");
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j5$/));
    FakeSocket.last!.emit({ seq: 0, type: "frame", frame: 20, has_mask: true, done: 1, total: 39 });
    expect(await screen.findByText("frame 20 / 39")).toBeTruthy();
    expect(screen.getByText(/20 masked/)).toBeTruthy();
  });

  it("runs a second propagation cleanly: the first job's terminal status never leaks onto the second", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValueOnce({ job_id: "j1" }).mockResolvedValueOnce({ job_id: "j2" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));
    FakeSocket.last!.emit({ seq: 0, type: "done", processed: 39, written: 38, lost_at: 33, cancelled: false });
    expect(await screen.findByText("frame 33 / 39")).toBeTruthy();
    expect(sessionStorage.getItem("vsp.job.c1")).toBeNull();

    fireEvent.keyDown(window, { key: "K", shiftKey: true }); // back to frame 0, a keyframe
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    const loads = vi.mocked(api.getClip).mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(sessionStorage.getItem("vsp.job.c1")).toBe("j2"));
    expect(screen.getByText("frame 0 / 39")).toBeTruthy(); // no jump to the first job's lost_at (33)
    expect(vi.mocked(api.getClip).mock.calls.length).toBe(loads); // no extra reload at the moment j2 started
  });

  it("coalesces a burst of streamed frame events into a single flush", async () => {
    const emptyDetail: ClipDetail = {
      ...detail,
      states: Array.from({ length: 40 }, (_, i) => (i === 0 ? "keyframe" : "empty")),
    };
    vi.mocked(api.getClip).mockResolvedValue(emptyDetail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j1" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));

    const seen = new Set<string>();
    const observer = new MutationObserver(() => {
      const el = screen.queryByText(/^frame \d+ \/ 39$/);
      if (el?.textContent !== undefined && el?.textContent !== null) seen.add(el.textContent);
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    for (let f = 1; f <= 30; f++) FakeSocket.last!.emit({ seq: f, type: "frame", frame: f, has_mask: true, done: f, total: 39 });
    expect(await screen.findByText("frame 30 / 39")).toBeTruthy();
    observer.disconnect();
    expect(screen.getByText(/30 masked/)).toBeTruthy();
    // The observer also wakes on unrelated mutations (each event bumps the job's own progress text), so it can
    // legitimately resample the frame counter's pre-flush value ("frame 0 / 39") more than once; what matters is
    // that the coalesced flush jumped straight from 0 to 30 and never showed any of the 29 frames in between.
    const seenFrames = [...seen].map((t) => Number(/^frame (\d+) \/ 39$/.exec(t)?.[1]));
    expect(seenFrames.every((n) => n === 0 || n === 30)).toBe(true);
    expect(seenFrames).toContain(30);
  });

  it("handles the keyboard map", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const deleted = { ...detail, states: detail.states.map((s, i) => (i === 11 ? "deleted" : s)) } as ClipDetail;
    vi.mocked(api.deleteFrame).mockResolvedValue(deleted);
    vi.mocked(api.restoreFrame).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    expect(screen.getByText("frame 11 / 39")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Delete" });
    await waitFor(() => expect(api.deleteFrame).toHaveBeenCalledWith("c1", 11));
    await screen.findByText(/2 deleted/);
    fireEvent.keyDown(window, { key: "Delete" });
    await waitFor(() => expect(api.restoreFrame).toHaveBeenCalledWith("c1", 11));
    fireEvent.keyDown(window, { key: "K", shiftKey: true });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    fireEvent.keyDown(window, { key: "g" });
    expect(screen.getByText("frame 20 / 39")).toBeTruthy();
    fireEvent.keyDown(window, { key: "m" });
    expect((screen.getByRole("checkbox", { name: /overlay/i }) as HTMLInputElement).checked).toBe(false);
    fireEvent.keyDown(window, { key: " " });
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
  });

  it("ignores shortcuts typed into a field", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(screen.getByLabelText("New class"), { key: "ArrowRight" });
    fireEvent.keyDown(screen.getByLabelText("New class"), { key: "m" });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    expect((screen.getByRole("checkbox", { name: /overlay/i }) as HTMLInputElement).checked).toBe(true);
  });

  it("does not treat a checkbox as a field: shortcuts still work when it is focused", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    const overlay = screen.getByRole("checkbox", { name: /overlay/i });
    fireEvent.click(overlay);
    fireEvent.keyDown(overlay, { key: "ArrowRight" });
    expect(screen.getByText("frame 1 / 39")).toBeTruthy();
  });

  it("prunes after confirmation and clears a range", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prune).mockResolvedValue(detail);
    vi.mocked(api.clearRange).mockResolvedValue(detail);
    const confirm = vi.fn().mockReturnValueOnce(false).mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    fireEvent.click(screen.getByRole("button", { name: /Prune/ }));
    expect(api.prune).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Prune/ }));
    await waitFor(() => expect(api.prune).toHaveBeenCalledWith("c1", 10));
    fireEvent.change(screen.getByLabelText("Range start"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("Range end"), { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: /Clear masks/ }));
    await waitFor(() => expect(api.clearRange).toHaveBeenCalledWith("c1", 5, 9));
  });
});
