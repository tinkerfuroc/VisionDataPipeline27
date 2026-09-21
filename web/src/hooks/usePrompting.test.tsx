// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { Candidate, ClipDetail } from "../api/types";
import type { ClipHandle } from "./useClip";
import { usePrompting } from "./usePrompting";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { prompt: vi.fn(), accept: vi.fn() },
}));

const cand = (area: number): Candidate => ({ score: 0.9, bbox: [10, 10, 50, 50], area, phrase: null, mask_png_b64: "AAAA" });
const baseDetail = (withObject: boolean): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 640, height: 480,
    frame_count: 10, created_at: "", deleted_frames: [],
    objects: withObject ? [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }] : [],
  },
  states: Array(10).fill("empty"),
  classes: [{ id: 0, name: "a", color: "#ff0000" }, { id: 1, name: "b", color: "#00ff00" }],
});
const handle = (detail: ClipDetail): ClipHandle => ({
  detail, error: null, refreshError: null, reload: vi.fn(), apply: vi.fn(), patchFrame: vi.fn(), maskVersion: () => "0.0",
});
type Args = { frame: number; classId: number | null; clip: ClipHandle; enabled: boolean };
const setup = (over: Partial<Args> = {}) =>
  renderHook((a: Args) => usePrompting({ clipId: "c1", ...a }), {
    initialProps: { frame: 2, classId: 0, clip: handle(baseDetail(true)), enabled: true, ...over },
  });

beforeEach(() => {
  vi.resetAllMocks();
});
afterEach(cleanup);

describe("usePrompting", () => {
  it("previews after every click", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    expect(api.prompt).toHaveBeenLastCalledWith("c1", 2, { type: "clicks", points: [[5, 6, 1]] });
    expect(result.current.canAccept).toBe(true);
    act(() => result.current.dispatch({ type: "point", x: 7, y: 8, label: 0 }));
    await waitFor(() => expect(api.prompt).toHaveBeenCalledTimes(2));
  });

  it("drops a response that arrives after a newer request", async () => {
    let resolveFirst: (v: { candidates: Candidate[] }) => void = () => {};
    vi.mocked(api.prompt)
      .mockReturnValueOnce(new Promise((r) => (resolveFirst = r)))
      .mockResolvedValueOnce({ candidates: [cand(222)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "point", x: 1, y: 1, label: 1 }));
    act(() => result.current.dispatch({ type: "point", x: 2, y: 2, label: 1 }));
    await waitFor(() => expect(result.current.candidates[0]?.area).toBe(222));
    await act(async () => resolveFirst({ candidates: [cand(111)] }));
    expect(result.current.candidates[0].area).toBe(222);
  });

  it("does not re-request when only the chosen candidate changes", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(1), cand(2)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "tool", tool: "text" }));
    act(() => result.current.dispatch({ type: "phrase", phrase: "bolt" }));
    act(() => result.current.dispatch({ type: "submit" }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(2));
    act(() => result.current.dispatch({ type: "candidate", index: 1 }));
    expect(api.prompt).toHaveBeenCalledTimes(1);
  });

  it("clears the draft when the frame changes or prompting is disabled", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result, rerender } = setup();
    const clip = handle(baseDetail(true));
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    rerender({ frame: 3, classId: 0, clip, enabled: true });
    expect(result.current.draft.points).toEqual([]);
    expect(result.current.candidates).toEqual([]);
    expect(api.prompt).toHaveBeenCalledTimes(1); // the old frame's draft is never sent for the new frame
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    rerender({ frame: 3, classId: 0, clip, enabled: false });
    expect(result.current.draft.points).toEqual([]);
  });

  it("explains an empty result and surfaces server errors with the blocking job id", async () => {
    vi.mocked(api.prompt).mockResolvedValueOnce({ candidates: [] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.error).toMatch(/No usable mask/));
    expect(result.current.canAccept).toBe(false);
    vi.mocked(api.prompt).mockRejectedValueOnce(new ApiError(409, "job j9 is still running", "j9"));
    act(() => result.current.dispatch({ type: "point", x: 7, y: 8, label: 1 }));
    await waitFor(() => expect(result.current.error).toBe("job j9 is still running"));
    expect(result.current.busyJobId).toBe("j9");
  });

  it("accepts with the chosen candidate, sends the class only when needed and adopts the result", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const accepted = baseDetail(true);
    vi.mocked(api.accept).mockResolvedValue(accepted);
    const clip = handle(baseDetail(true));
    const { result, rerender } = setup({ clip });
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.canAccept).toBe(true));
    await act(() => result.current.accept());
    expect(api.accept).toHaveBeenLastCalledWith("c1", 2, { type: "clicks", points: [[5, 6, 1]] }, { classId: undefined, candidate: 0 });
    expect(clip.apply).toHaveBeenCalledWith(accepted, true);
    expect(result.current.draft.points).toEqual([]);

    rerender({ frame: 2, classId: 1, clip, enabled: true }); // the user picked another class
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.canAccept).toBe(true));
    await act(() => result.current.accept());
    expect(vi.mocked(api.accept).mock.calls[1][3]).toEqual({ classId: 1, candidate: 0 });
  });

  it("requires a class for the first accept on a clip", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result } = setup({ clip: handle(baseDetail(false)), classId: null });
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    expect(result.current.canAccept).toBe(false);
    await act(() => result.current.accept());
    expect(api.accept).not.toHaveBeenCalled();
    expect(result.current.error).toMatch(/class/i);
  });
});
