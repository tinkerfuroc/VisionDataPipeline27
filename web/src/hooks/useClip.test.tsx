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
    frame_count: 3, created_at: "", deleted_frames: [], objects: [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states: ["keyframe", "empty", "masked"],
  classes: [],
  object_states: { "1": ["keyframe", "empty", "masked"] },
};

beforeEach(() => vi.resetAllMocks());
afterEach(cleanup);

/** A promise this test can resolve/reject on its own schedule. */
function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

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
  it("patches an object's lane from a job event, recomputes the clip state and re-versions that frame only", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = [result.current.maskVersion(1), result.current.maskVersion(2)];
    act(() => result.current.patchFrames([{ frame: 1, objects: [{ id: 1, hasMask: true }] }]));
    expect(result.current.detail?.object_states["1"]).toEqual(["keyframe", "masked", "masked"]);
    expect(result.current.detail?.states).toEqual(["keyframe", "masked", "masked"]);
    expect(result.current.maskVersion(1)).not.toBe(before[0]);
    expect(result.current.maskVersion(2)).toBe(before[1]);
  });
  it("patchFrames applies a batch in one update and re-versions every touched frame", async () => {
    const lane: ClipDetail["states"] = ["keyframe", "empty", "empty", "empty"];
    vi.mocked(api.getClip).mockResolvedValue({ ...detail, states: lane, object_states: { "1": lane } });
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = [0, 1, 2, 3].map((f) => result.current.maskVersion(f));
    act(() =>
      result.current.patchFrames([
        { frame: 1, objects: [{ id: 1, hasMask: true }] },
        { frame: 2, objects: [{ id: 1, hasMask: true }] },
        { frame: 3, objects: [{ id: 1, hasMask: false }] },
      ]),
    );
    expect(result.current.detail?.states).toEqual(["keyframe", "masked", "masked", "empty"]);
    expect(result.current.maskVersion(0)).toBe(before[0]); // not in the batch: unchanged
    expect(result.current.maskVersion(1)).not.toBe(before[1]);
    expect(result.current.maskVersion(2)).not.toBe(before[2]);
    expect(result.current.maskVersion(3)).not.toBe(before[3]); // stayed "empty", but its mask was still touched
  });
  it("re-versions a frame whose mask was overwritten even if no state changed", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const shown = result.current.detail;
    const before = result.current.maskVersion(2);
    act(() => result.current.patchFrames([{ frame: 2, objects: [{ id: 1, hasMask: true }] }]));
    expect(result.current.maskVersion(2)).not.toBe(before);
    expect(result.current.detail).toBe(shown); // nothing changed: the same detail object
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
  it("update() applies a function against whatever detail is current, not a stale snapshot", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    // a lane patch lands (as it would while some other request is in flight) before update() runs
    act(() => result.current.patchFrames([{ frame: 1, objects: [{ id: 1, hasMask: true }] }]));
    const patchedStates = result.current.detail?.states;
    act(() =>
      result.current.update((d) => ({ ...d, classes: [...d.classes, { id: 9, name: "plate", color: "#fff" }] })),
    );
    expect(result.current.detail?.classes).toEqual([{ id: 9, name: "plate", color: "#fff" }]);
    expect(result.current.detail?.states).toEqual(patchedStates); // the concurrent patch survives, not rolled back
  });
  it("drops a reload that resolves after a newer one", async () => {
    const first = deferred<ClipDetail>();
    vi.mocked(api.getClip).mockReturnValueOnce(first.promise);
    const { result } = renderHook(() => useClip("c1"));

    const second: ClipDetail = { ...detail, states: ["keyframe", "masked", "masked"] };
    vi.mocked(api.getClip).mockResolvedValueOnce(second);
    await act(async () => {
      await result.current.reload();
    });
    expect(result.current.detail).toEqual(second);
    const versionAfterSecond = result.current.maskVersion(0);

    await act(async () => {
      first.resolve(detail);
      await Promise.resolve();
    });
    expect(result.current.detail).toEqual(second);
    expect(result.current.maskVersion(0)).toBe(versionAfterSecond);
  });
  it("drops the response for a previous clip id", async () => {
    const c1 = deferred<ClipDetail>();
    const c2Detail: ClipDetail = { ...detail, clip: { ...detail.clip, id: "c2" } };
    vi.mocked(api.getClip).mockImplementation((id: string) => (id === "c1" ? c1.promise : Promise.resolve(c2Detail)));
    const { result, rerender } = renderHook(({ id }) => useClip(id), { initialProps: { id: "c1" } });

    rerender({ id: "c2" });
    await waitFor(() => expect(result.current.detail?.clip.id).toBe("c2"));

    await act(async () => {
      c1.resolve(detail);
      await Promise.resolve();
    });
    expect(result.current.detail?.clip.id).toBe("c2");
  });
  it("a failed refresh keeps the loaded clip and reports refreshError", async () => {
    vi.mocked(api.getClip).mockResolvedValueOnce(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).toEqual(detail));

    vi.mocked(api.getClip).mockRejectedValueOnce(new ApiError(500, "server exploded"));
    await act(async () => {
      await result.current.reload();
    });
    expect(result.current.detail).toEqual(detail);
    expect(result.current.error).toBeNull();
    expect(result.current.refreshError).toBe("server exploded");

    vi.mocked(api.getClip).mockResolvedValueOnce(detail);
    await act(async () => {
      await result.current.reload();
    });
    expect(result.current.refreshError).toBeNull();
  });
});
