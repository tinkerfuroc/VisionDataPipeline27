// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePlayback } from "./usePlayback";

function usePlayer(frameCount: number, fps: number) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });
  return { frame, playing };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("usePlayback", () => {
  it("advances one frame per 1/fps seconds", () => {
    const { result } = renderHook(() => usePlayer(100, 10));
    act(() => vi.advanceTimersByTime(350));
    expect(result.current.frame).toBe(3);
  });
  it("stops on the last frame", () => {
    const { result } = renderHook(() => usePlayer(3, 10));
    act(() => vi.advanceTimersByTime(1000));
    expect(result.current.frame).toBe(2);
    expect(result.current.playing).toBe(false);
  });
});
