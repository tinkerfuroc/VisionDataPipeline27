// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import { useJob } from "./useJob";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { getJob: vi.fn() },
}));

class FakeSocket {
  static instances: FakeSocket[] = [];
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
  emit(event: object) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
}
const socket = () => FakeSocket.instances[FakeSocket.instances.length - 1];

beforeEach(() => {
  FakeSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.mocked(api.getJob).mockReset();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("useJob", () => {
  it("is idle without a job id and opens no socket", () => {
    const { result } = renderHook(() => useJob(null));
    expect(result.current.status).toBe("idle");
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it("streams events, drops replays and forwards each new event once", () => {
    const seen: number[] = [];
    const { result } = renderHook(() => useJob("j1", (e) => seen.push(e.seq)));
    expect(socket().url).toMatch(/\/ws\/jobs\/j1$/);
    act(() => {
      socket().emit({ seq: 0, type: "progress", done: 0, total: 3 });
      socket().emit({ seq: 1, type: "frame", frame: 4, has_mask: true, done: 1, total: 3 });
      socket().emit({ seq: 0, type: "progress", done: 0, total: 3 });
    });
    expect(seen).toEqual([0, 1]);
    expect(result.current).toMatchObject({ status: "running", done: 1, total: 3, lastFrame: 4, transport: "ws" });
  });

  it("falls back to polling when the socket closes before the job ends", async () => {
    vi.mocked(api.getJob).mockResolvedValue({
      id: "j1", kind: "import", status: "done", result: { clip_id: "c9" }, error: null, created_at: "", n_events: 3,
    });
    const { result } = renderHook(() => useJob("j1"));
    act(() => socket().onclose?.());
    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.transport).toBe("poll");
    expect(result.current.result).toEqual({ clip_id: "c9" });
  });

  it("does not poll when the socket closes after the final event", () => {
    const { result } = renderHook(() => useJob("j1"));
    act(() => {
      socket().emit({ seq: 0, type: "done", clip_id: "c1" });
      socket().onclose?.();
    });
    expect(result.current.status).toBe("done");
    expect(api.getJob).not.toHaveBeenCalled();
  });

  it("closes the socket on unmount and reconnects for a new job id", () => {
    const { rerender, unmount } = renderHook(({ id }) => useJob(id), { initialProps: { id: "j1" } });
    const first = socket();
    rerender({ id: "j2" });
    expect(first.closed).toBe(true);
    expect(socket().url).toMatch(/j2$/);
    unmount();
    expect(socket().closed).toBe(true);
  });
});
