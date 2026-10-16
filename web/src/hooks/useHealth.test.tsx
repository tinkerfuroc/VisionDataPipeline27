// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { Health } from "../api/types";
import { useHealth } from "./useHealth";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { health: vi.fn() },
}));

const paused: Health = { status: "ok", version: "0", engine: { loaded: false, device: "cuda:0", error: null, paused: true } };

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(api.health).mockReset().mockResolvedValue(paused);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("useHealth", () => {
  it("polls a paused engine slowly, and refresh() polls at once", async () => {
    const { result } = renderHook(() => useHealth());
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(result.current.health).toEqual(paused);
    expect(api.health).toHaveBeenCalledTimes(1);
    await act(() => vi.advanceTimersByTimeAsync(5000)); // the 2 s "loading" pace would have polled twice by now
    expect(api.health).toHaveBeenCalledTimes(1);
    act(() => result.current.refresh?.());
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(api.health).toHaveBeenCalledTimes(2);
    await act(() => vi.advanceTimersByTimeAsync(15000)); // the refresh replaced the pending poll: one more, not two
    expect(api.health).toHaveBeenCalledTimes(3);
  });
});
