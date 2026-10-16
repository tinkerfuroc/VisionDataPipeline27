// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipDetail, FrameState } from "../api/types";
import { HealthContext, type HealthState } from "../hooks/useHealth";
import { FrameStore } from "../lib/frameCache";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Label } from "./Label";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: {
    getClip: vi.fn(),
    prompt: vi.fn(),
    accept: vi.fn(),
    addClass: vi.fn(),
    setClassColor: vi.fn(),
    relabelObject: vi.fn(),
    deleteObject: vi.fn(),
    propagate: vi.fn(),
    cancelJob: vi.fn(),
    deleteFrame: vi.fn(),
    restoreFrame: vi.fn(),
    prune: vi.fn(),
    deleteFrames: vi.fn(),
    restoreFrames: vi.fn(),
    clearRange: vi.fn(),
    getJob: vi.fn(),
    setMask: vi.fn(),
    setRegions: vi.fn(),
  },
}));

vi.mock("../lib/brushCanvas", () => ({
  createLayer: () => ({ canvas: document.createElement("canvas"), ctx: null }),
  loadBase: async () => null,
  encodeLayer: () => ({ pngB64: "UE5H", any: true }),
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

const states: FrameState[] = Array.from({ length: 40 }, (_, i) =>
  i === 0 ? "keyframe" : i < 20 ? "masked" : i === 30 ? "deleted" : "empty",
);
const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "bench.mp4" }, fps: 10, stride: 1, width: 640, height: 480,
    frame_count: 40, created_at: "", deleted_frames: [30], original_names: { "3": "img_0007.png" },
    objects: [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states,
  classes: [{ id: 0, name: "connector", color: "#e6194b" }],
  object_states: { "1": states },
};
/** a one-object detail with these states; its object's lane equals the clip's, as the server reports for one object */
const oneObject = (s: FrameState[]): ClipDetail => ({ ...detail, states: s, object_states: { "1": s } });

/** two objects: #1 connector (as in `detail`) and #2 cup, keyframed on 5 and masked on 6..9 */
const cupLane: FrameState[] = Array.from({ length: 40 }, (_, i) =>
  i === 5 ? "keyframe" : i > 5 && i < 10 ? "masked" : i === 30 ? "deleted" : "empty",
);
const two: ClipDetail = {
  ...detail,
  clip: { ...detail.clip, objects: [...detail.clip.objects, { id: 2, class_id: 1, keyframes: [], propagated_ranges: [] }] },
  states: states.map((s, i) => (i === 5 ? "keyframe" : s)),
  classes: [...detail.classes, { id: 1, name: "cup", color: "#3cb44b" }],
  object_states: { "1": states, "2": cupLane },
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
    expect(screen.getByText("1 keyframe")).toBeTruthy(); // the statistics; the Cleanup panel also mentions it
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
  it("previews a class colour change at once, saves it debounced, and reloads the clip", async () => {
    vi.mocked(api.getClip).mockResolvedValueOnce(detail);
    vi.mocked(api.setClassColor).mockResolvedValue({ ...detail.classes[0], color: "#123456" });
    renderLabel();
    await screen.findByText("bench.mp4");
    const input = screen.getByLabelText("Class colour") as HTMLInputElement;
    expect(input.value).toBe("#e6194b");

    const recolored: ClipDetail = { ...detail, classes: [{ ...detail.classes[0], color: "#123456" }] };
    vi.mocked(api.getClip).mockResolvedValue(recolored); // what reload() fetches once the save lands

    vi.useFakeTimers();
    fireEvent.change(input, { target: { value: "#111111" } });
    fireEvent.change(input, { target: { value: "#123456" } });
    expect(input.value).toBe("#123456"); // shown locally at once, before the debounce fires
    expect(api.setClassColor).not.toHaveBeenCalled();
    await act(async () => vi.advanceTimersByTime(300));
    vi.useRealTimers();
    expect(api.setClassColor).toHaveBeenCalledTimes(1);
    expect(api.setClassColor).toHaveBeenCalledWith("p", 0, "#123456");
    await waitFor(() => expect(api.getClip).toHaveBeenCalledTimes(2)); // reload, so fills/lanes/previews repaint
  });
  it("shows a class colour save error the way the page already shows errors", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.setClassColor).mockRejectedValue(new Error("disk full"));
    renderLabel();
    await screen.findByText("bench.mp4");
    vi.useFakeTimers();
    fireEvent.change(screen.getByLabelText("Class colour"), { target: { value: "#123456" } });
    await act(async () => vi.advanceTimersByTime(300));
    vi.useRealTimers();
    expect(await screen.findByText("disk full")).toBeTruthy();
  });
  it("disables the class colour row when there is no class to colour", async () => {
    const noClasses: ClipDetail = { ...detail, clip: { ...detail.clip, objects: [] }, classes: [], object_states: {} };
    vi.mocked(api.getClip).mockResolvedValue(noClasses);
    renderLabel();
    await screen.findByText("bench.mp4");
    expect((screen.getByLabelText("Class colour") as HTMLInputElement).disabled).toBe(true);
  });
  it("shows a load error with a way back", async () => {
    vi.mocked(api.getClip).mockRejectedValue(new ApiError(404, "clip c1 not found"));
    renderLabel();
    expect(await screen.findByText(/clip c1 not found/)).toBeTruthy();
    expect(screen.getByRole("link", { name: /back to clips/i })).toBeTruthy();
  });
  it("draws every visible object's stored mask on the frame, the active object last and outlined", async () => {
    const composite = vi.spyOn(FrameStore.prototype, "composite").mockResolvedValue(null);
    try {
      vi.mocked(api.getClip).mockResolvedValue(two);
      renderLabel();
      await screen.findByText("bench.mp4");
      for (let i = 0; i < 6; i++) fireEvent.keyDown(window, { key: "ArrowRight" });
      expect(screen.getByText("frame 6 / 39")).toBeTruthy();
      const onFrame6 = composite.mock.calls.map(([parts]) => parts).filter((parts) => parts.some((p) => p.url.includes("/masks/6?")));
      const last = onFrame6[onFrame6.length - 1];
      expect(last.map((p) => p.url.split("?")[0])).toEqual(["/api/clips/c1/objects/2/masks/6", "/api/clips/c1/objects/1/masks/6"]);
      expect(last[0].outline).toBeNull();
      expect(last[1].outline).toEqual({ rgb: [230, 25, 75], edge: 5 });
    } finally {
      composite.mockRestore();
    }
  });
});

describe("Label page (prompting)", () => {
  it("previews a click on the canvas and accepts it", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockResolvedValue({
      candidates: [{ score: 0.9, bbox: [80, 30, 120, 70], area: 1200, phrase: null, mask_png_b64: "AAAA" }],
    });
    const accepted = { ...oneObject(detail.states.map((s, i) => (i === 25 ? "keyframe" : s))), object_ids: [1] };
    vi.mocked(api.accept).mockResolvedValue(accepted);
    renderLabel();
    await screen.findByText("bench.mp4");
    for (let i = 0; i < 25; i++) fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 0 });
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 25, { type: "clicks", points: [[100, 50, 1]] }, 1));
    const accept = screen.getByRole("button", { name: /Accept/ }) as HTMLButtonElement;
    await waitFor(() => expect(accept.disabled).toBe(false));
    fireEvent.click(accept);
    await waitFor(() =>
      expect(api.accept).toHaveBeenCalledWith("c1", 25, { type: "clicks", points: [[100, 50, 1]] }, { objectId: 1 }, { candidate: 0 }),
    );
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
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 0]] }, 1));
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

  it("creates a new object from its first accepted mask and makes it active", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockResolvedValue({
      candidates: [{ score: 0.9, bbox: [80, 30, 120, 70], area: 1200, phrase: null, mask_png_b64: "AAAA" }],
    });
    vi.mocked(api.accept).mockResolvedValue({ ...two, object_ids: [2] });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "n" });
    expect(screen.getByText("Accept creates a new connector object.")).toBeTruthy();
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 0 });
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 1]] }, null));
    const accept = screen.getByRole("button", { name: /^Accept/ }) as HTMLButtonElement;
    await waitFor(() => expect(accept.disabled).toBe(false));
    fireEvent.click(accept);
    await waitFor(() =>
      expect(api.accept).toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 1]] }, { objectId: null, classId: 0 }, { candidate: 0 }),
    );
    const created = await within(screen.getByRole("list", { name: "Objects" })).findByRole("button", { name: /^#2 / });
    expect(created.getAttribute("aria-pressed")).toBe("true");
  });
});

describe("Label page (propagation, cleanup, keyboard)", () => {
  it("propagates from a keyframe, follows the streamed frames and reloads when the job ends", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j1" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(api.propagate).toHaveBeenCalledWith("c1", 0, "forward", "all"));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));
    expect(sessionStorage.getItem("vdp.job.c1")).toBe("j1");
    FakeSocket.last!.emit({ seq: 0, type: "progress", done: 0, total: 39 });
    FakeSocket.last!.emit({ seq: 1, type: "frame", frame: 20, objects: [{ id: 1, has_mask: true }], done: 1, total: 39 });
    // F3: frame events are buffered and applied on the next animation frame, not synchronously
    expect(await screen.findByText("frame 20 / 39")).toBeTruthy();
    expect(screen.getByText(/20 masked/)).toBeTruthy();
    expect(screen.getByText("A propagation is running. Stop it to prompt.")).toBeTruthy();
    const loads = vi.mocked(api.getClip).mock.calls.length;
    FakeSocket.last!.emit({ seq: 2, type: "done", processed: 39, written: 38, lost_at: 33, lost: { "1": 33 }, skipped: [], cancelled: false });
    await waitFor(() => expect(vi.mocked(api.getClip).mock.calls.length).toBe(loads + 1));
    expect(await screen.findByText("frame 33 / 39")).toBeTruthy();
    expect(screen.getByText(/lost #1 connector at frame 33/i)).toBeTruthy();
    expect(sessionStorage.getItem("vdp.job.c1")).toBeNull();
  });

  it("follows the furthest frame of a run, not an earlier frame a lagging object re-emits", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j1" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));
    // one batch: the leading group reaches frame 22, then the lagging group re-emits frame 21 at the same `done`
    FakeSocket.last!.emit({ seq: 0, type: "frame", frame: 22, objects: [{ id: 1, has_mask: true }], done: 3, total: 39 });
    FakeSocket.last!.emit({ seq: 1, type: "frame", frame: 21, objects: [{ id: 1, has_mask: true }], done: 3, total: 39 });
    expect(await screen.findByText("frame 22 / 39")).toBeTruthy();
    // a later batch holding only a re-emitted earlier frame does not pull the playhead back either
    FakeSocket.last!.emit({ seq: 2, type: "frame", frame: 20, objects: [{ id: 1, has_mask: true }], done: 3, total: 39 });
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    expect(screen.getByText("frame 22 / 39")).toBeTruthy();
    FakeSocket.last!.emit({ seq: 3, type: "frame", frame: 23, objects: [{ id: 1, has_mask: true }], done: 4, total: 39 });
    expect(await screen.findByText("frame 23 / 39")).toBeTruthy();
  });

  it("offers propagation on any masked frame but not on an empty one, and Stop cancels the job", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j2" });
    vi.mocked(api.cancelJob).mockResolvedValue({ id: "j2", kind: "propagate", status: "running", result: null, error: null, created_at: "", n_events: 0 });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Next frame" })); // frame 1: masked by a propagation, not a keyframe
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.queryByText(/has no mask/)).toBeNull();
    fireEvent.keyDown(window, { key: "g" }); // next empty frame (20)
    expect(await screen.findByText("frame 20 / 39")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/This frame has no mask/)).toBeTruthy();
    for (let i = 0; i < 10; i++) fireEvent.keyDown(window, { key: "ArrowRight" }); // frame 30 is deleted
    expect(await screen.findByText("frame 30 / 39")).toBeTruthy();
    expect(screen.getByText(/Restore it, or go to a masked frame/)).toBeTruthy();
    fireEvent.keyDown(window, { key: "K", shiftKey: true }); // back to frame 0, the keyframe
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
    expect(sessionStorage.getItem("vdp.job.c1")).toBeNull();
    expect(screen.getByText(/GPU is busy with a propagation that was not started from this page/i)).toBeTruthy();
    FakeSocket.last!.emit({ seq: 0, type: "frame", frame: 20, objects: [{ id: 1, has_mask: true }], done: 1, total: 39 });
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
    sessionStorage.setItem("vdp.job.c1", "j5");
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j5$/));
    FakeSocket.last!.emit({ seq: 0, type: "frame", frame: 20, objects: [{ id: 1, has_mask: true }], done: 1, total: 39 });
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
    expect(sessionStorage.getItem("vdp.job.c1")).toBeNull();

    fireEvent.keyDown(window, { key: "K", shiftKey: true }); // back to frame 0, a keyframe
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    const loads = vi.mocked(api.getClip).mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(sessionStorage.getItem("vdp.job.c1")).toBe("j2"));
    expect(screen.getByText("frame 0 / 39")).toBeTruthy(); // no jump to the first job's lost_at (33)
    expect(vi.mocked(api.getClip).mock.calls.length).toBe(loads); // no extra reload at the moment j2 started
  });

  it("coalesces a burst of streamed frame events into a single flush", async () => {
    const emptyDetail = oneObject(Array.from({ length: 40 }, (_, i) => (i === 0 ? "keyframe" : "empty")));
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
    for (let f = 1; f <= 30; f++) FakeSocket.last!.emit({ seq: f, type: "frame", frame: f, objects: [{ id: 1, has_mask: true }], done: f, total: 39 });
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
    const deleted = oneObject(detail.states.map((s, i) => (i === 11 ? "deleted" : s)));
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

  it("marks a range with [ and ], names it on the button, and unmarks it once cleared", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.clearRange).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    // nothing marked: the button acts on the current frame; frame 0 is a keyframe, and keyframes are cleared too
    expect((screen.getByRole("button", { name: "Clear masks in frame 0" }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByText("Removes 1 keyframe.")).toBeTruthy();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true }); // frame 12 (Shift = 10 frames)
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true }); // frame 22: empty
    expect((screen.getByRole("button", { name: "Clear masks in frame 22" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("There is nothing to clear in frame 22.")).toBeTruthy();
    fireEvent.keyDown(window, { key: "K", shiftKey: true });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "[" });
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    fireEvent.keyDown(window, { key: "]" });
    expect(screen.getByText(/Removes 11 propagated masks/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frames 1 to 11" }));
    await waitFor(() => expect(api.clearRange).toHaveBeenCalledWith("c1", 1, 11, [1]));
    expect(await screen.findByRole("button", { name: "Clear masks in frame 11" })).toBeTruthy(); // unmarked on success
  });

  it("keeps the mark when clearing fails, so it can be retried", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.clearRange).mockRejectedValue(new Error("disk full"));
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "[" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "]" });
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frames 1 to 2" }));
    expect(await screen.findByText(/disk full/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Clear masks in frames 1 to 2" })).toBeTruthy();
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
    await waitFor(() => expect(api.clearRange).toHaveBeenCalledWith("c1", 5, 9, [1]));
  });

  it("deletes and restores a marked range of frames, keeping the mark", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.deleteFrames).mockResolvedValue(detail);
    vi.mocked(api.restoreFrames).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Range start"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("Range end"), { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: "Delete frames 5 to 9" }));
    await waitFor(() => expect(api.deleteFrames).toHaveBeenCalledWith("c1", 5, 9));
    expect((screen.getByLabelText("Range start") as HTMLInputElement).value).toBe("5"); // still marked
  });
});

describe("Label page (brush)", () => {
  const paintOnce = () => {
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { button: 0, clientX: 40, clientY: 40 });
    fireEvent.mouseMove(canvas, { button: 0, buttons: 1, clientX: 60, clientY: 60 });
    fireEvent.mouseUp(canvas, { button: 0, clientX: 60, clientY: 60 });
  };
  // Switching to Brush mode kicks off useBrush's async base-mask load (F1); flush it before painting so start()
  // isn't refused as "still loading" — a real drag could never begin before the browser's next tick either.
  const flushBaseLoad = () => act(() => new Promise((r) => setTimeout(r, 0)));

  it("switches to the Brush panel, locks the page while strokes are unsaved, and saves the mask", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.setMask).mockResolvedValue({ ...detail, object_id: 1 });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Brush" }));
    expect(screen.getByRole("heading", { name: "Brush" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Prompt" })).toBeNull();
    expect(screen.getByText("No strokes yet on frame 0.")).toBeTruthy();
    await flushBaseLoad();

    paintOnce();
    expect(screen.getByText("1 stroke on frame 0, not saved.")).toBeTruthy();
    const lock = /Save or discard the brush strokes on frame 0 first/;
    expect(screen.getAllByText(lock).length).toBeGreaterThan(0);
    expect((screen.getByRole("button", { name: "Next frame" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Clear masks in frame 0" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Click" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy(); // the step key is ignored while locked
    fireEvent.keyDown(window, { key: "=" });
    expect((screen.getByLabelText("Brush size") as HTMLInputElement).value).toBe("29"); // 24 * 1.2

    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(api.setMask).toHaveBeenCalledWith("c1", 0, "UE5H", { objectId: 1 }));
    expect(await screen.findByText("No strokes yet on frame 0.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Next frame" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("discards with Escape and undoes with Z; Brush is unavailable on a deleted frame", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Brush" }));
    await flushBaseLoad();
    paintOnce();
    paintOnce();
    expect(screen.getByText("2 strokes on frame 0, not saved.")).toBeTruthy();
    fireEvent.keyDown(window, { key: "z" });
    expect(screen.getByText("1 stroke on frame 0, not saved.")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByText("No strokes yet on frame 0.")).toBeTruthy();
    for (let i = 0; i < 3; i++) fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    expect(await screen.findByText("frame 30 / 39")).toBeTruthy(); // deleted in the fixture
    expect(screen.getByText("This frame is deleted. Restore it to paint on it.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Brush" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("Label page (objects)", () => {
  const panel = () => within(screen.getByRole("list", { name: "Objects" }));
  const row = (name: RegExp) => panel().getByRole("button", { name });

  it("lists the objects; one is selected by click or with O / Shift+O, and N starts a new object", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    expect(row(/^#1 connector/).getAttribute("aria-pressed")).toBe("true");
    expect(row(/^#2 cup/).textContent).toContain("5 frames");
    fireEvent.click(row(/^#2 cup/));
    expect(row(/^#2 cup/).getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(window, { key: "o" });
    expect(row(/^#1 connector/).getAttribute("aria-pressed")).toBe("true"); // wraps around
    fireEvent.keyDown(window, { key: "O", shiftKey: true });
    expect(row(/^#2 cup/).getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(window, { key: "n" });
    expect(panel().getByText("New object (cup)", { selector: "div" }).getAttribute("aria-current")).toBe("true");
  });

  it("K and G follow the active object's lane", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "k" });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy(); // #1 has no keyframe after 0 (the clip does: 5)
    fireEvent.keyDown(window, { key: "g" });
    expect(screen.getByText("frame 20 / 39")).toBeTruthy();
    fireEvent.click(row(/^#2 cup/));
    fireEvent.keyDown(window, { key: "K", shiftKey: true });
    expect(screen.getByText("frame 5 / 39")).toBeTruthy(); // #2's keyframe
    fireEvent.keyDown(window, { key: "g" });
    expect(screen.getByText("frame 10 / 39")).toBeTruthy(); // #2 is masked on 6..9
  });

  it("relabels the active object via the explicit Change control", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    const relabelled: ClipDetail = {
      ...two, clip: { ...two.clip, objects: [{ ...two.clip.objects[0], class_id: 1 }, two.clip.objects[1]] },
    };
    vi.mocked(api.relabelObject).mockResolvedValue(relabelled);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("New class of #1"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Change" }));
    await waitFor(() => expect(api.relabelObject).toHaveBeenCalledWith("c1", 1, 1));
    expect(await panel().findByRole("button", { name: /^#1 cup/ })).toBeTruthy();
  });
  it("picking a class in the Objects panel starts a new object of it instead of relabelling", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Class"), { target: { value: "1" } });
    expect(panel().getByText("New object (cup)", { selector: "div" }).getAttribute("aria-current")).toBe("true");
    expect(api.relabelObject).not.toHaveBeenCalled();
  });

  it("deletes the active object after a confirmation naming its frame count, then selects the next one", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    const onlyTwo: ClipDetail = {
      ...two, clip: { ...two.clip, objects: [two.clip.objects[1]] }, states: cupLane, object_states: { "2": cupLane },
    };
    vi.mocked(api.deleteObject).mockResolvedValue(onlyTwo);
    const confirm = vi.fn().mockReturnValueOnce(false).mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Delete #1 connector" }));
    expect(confirm).toHaveBeenCalledWith("Delete #1 connector and its masks on 20 frames?");
    expect(api.deleteObject).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete #1 connector" }));
    await waitFor(() => expect(api.deleteObject).toHaveBeenCalledWith("c1", 1));
    expect((await panel().findByRole("button", { name: /^#2 cup/ })).getAttribute("aria-pressed")).toBe("true");
  });

  it("shows one timeline lane per class; a lane's name selects that class's (lowest-id) object", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "cup" })); // the class name in the timeline's gutter
    expect(row(/^#2 cup/).getAttribute("aria-pressed")).toBe("true");
  });

  it("a class with no object in the clip gets no lane until it is picked, and picking it starts a new object rather than relabelling", async () => {
    // one Module object (class 0) and a project class GreenTab (class 1) with none in this clip
    const oneModuleTwoClasses: ClipDetail = {
      ...detail,
      classes: [{ id: 0, name: "Module", color: "#e6194b" }, { id: 1, name: "GreenTab", color: "#3cb44b" }],
    };
    vi.mocked(api.getClip).mockResolvedValue(oneModuleTwoClasses);
    renderLabel();
    await screen.findByText("bench.mp4");
    expect(screen.getByRole("button", { name: "Module" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "GreenTab" })).toBeNull(); // no object in this clip: no lane
    fireEvent.change(screen.getByLabelText("Class"), { target: { value: "1" } });
    expect(panel().getByText("New object (GreenTab)", { selector: "div" }).getAttribute("aria-current")).toBe("true");
    expect(screen.getByRole("button", { name: "GreenTab" })).toBeTruthy(); // the chosen class shows its lane
    expect(api.relabelObject).not.toHaveBeenCalled();
    fireEvent.click(row(/^#1 Module/)); // back to the existing object: the empty class's lane goes again
    expect(screen.queryByRole("button", { name: "GreenTab" })).toBeNull();
    expect(api.relabelObject).not.toHaveBeenCalled();
  });

  it("propagates the selected object alone when chosen", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j3" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(within(screen.getByRole("group", { name: "Objects to propagate" })).getByRole("radio", { name: "Selected: #1 connector" }));
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(api.propagate).toHaveBeenCalledWith("c1", 0, "forward", [1]));
  });

  it("checks the selected object's own mask before propagating it alone, and any object's for All objects", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(row(/^#2 cup/));
    for (let i = 0; i < 10; i++) fireEvent.keyDown(window, { key: "ArrowRight" }); // frame 10: #1 masked, #2 empty
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(within(screen.getByRole("group", { name: "Objects to propagate" })).getByRole("radio", { name: "Selected: #2 cup" }));
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/#2 cup has no mask on this frame/)).toBeTruthy();
  });

  it("clears the active object's masks by default, or every object's when chosen", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    vi.mocked(api.clearRange).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frame 0" }));
    await waitFor(() => expect(api.clearRange).toHaveBeenLastCalledWith("c1", 0, 0, [1]));
    fireEvent.click(within(screen.getByRole("group", { name: "Objects to clear" })).getByRole("radio", { name: "All objects" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frame 0" }));
    await waitFor(() => expect(api.clearRange).toHaveBeenLastCalledWith("c1", 0, 0, "all"));
  });
});

describe("Label page (regions)", () => {
  const openRegions = async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    fireEvent.click(await screen.findByRole("button", { name: "Regions" }));
    fireEvent.click(screen.getByRole("button", { name: "Add region from here" }));
  };

  it("edits, saves and discards regions with the Regions tool", async () => {
    await openRegions();
    expect(screen.getByText("Unsaved changes")).toBeTruthy();
    vi.mocked(api.setRegions).mockResolvedValue({
      ...detail,
      clip: { ...detail.clip, regions: [{ start: 0, end: detail.clip.frame_count - 1, crop: null, blocks: [] }] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save regions" }));
    await waitFor(() => expect(api.setRegions).toHaveBeenCalledWith("c1", [expect.objectContaining({ start: 0 })]));
    await waitFor(() => expect(screen.queryByText("Unsaved changes")).toBeNull());
  });

  it("asks before leaving the Regions tool with unsaved changes", async () => {
    await openRegions();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    fireEvent.click(screen.getByRole("button", { name: "Click" }));
    expect(confirm).toHaveBeenCalled();
    expect(screen.getByText("Unsaved changes")).toBeTruthy(); // stayed
    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "Click" }));
    expect(screen.queryByText("Unsaved changes")).toBeNull();
    confirm.mockRestore();
  });
});
