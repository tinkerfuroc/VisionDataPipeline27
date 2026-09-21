// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { useClip } from "./useClip";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { getClip: vi.fn() },
}));

const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 640, height: 480,
    frame_count: 3, created_at: "", deleted_frames: [], objects: [],
  },
  states: ["keyframe", "empty", "masked"],
  classes: [],
};

beforeEach(() => vi.resetAllMocks());
afterEach(cleanup);

describe("useClip", () => {
  it("loads the clip detail", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).toEqual(detail));
    expect(api.getClip).toHaveBeenCalledWith("c1");
  });
  it("reports a load failure", async () => {
    vi.mocked(api.getClip).mockRejectedValue(new ApiError(404, "clip c1 not found"));
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.error).toBe("clip c1 not found"));
  });
  it("patches one frame from a job event and gives that frame's mask a new version", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = [result.current.maskVersion(1), result.current.maskVersion(2)];
    act(() => result.current.patchFrame(1, true));
    expect(result.current.detail?.states).toEqual(["keyframe", "masked", "masked"]);
    expect(result.current.maskVersion(1)).not.toBe(before[0]);
    expect(result.current.maskVersion(2)).toBe(before[1]);
  });
  it("re-versions a frame whose mask was overwritten even if its state did not change", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = result.current.maskVersion(2);
    act(() => result.current.patchFrame(2, true));
    expect(result.current.maskVersion(2)).not.toBe(before);
  });
  it("apply() replaces the detail and re-versions every mask only when masks changed", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = result.current.maskVersion(0);
    const next: ClipDetail = { ...detail, states: ["keyframe", "deleted", "masked"] };
    act(() => result.current.apply(next, false));
    expect(result.current.detail).toBe(next);
    expect(result.current.maskVersion(0)).toBe(before);
    act(() => result.current.apply(next, true));
    expect(result.current.maskVersion(0)).not.toBe(before);
  });
});
