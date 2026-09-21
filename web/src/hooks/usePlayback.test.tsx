// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePlayback } from "./usePlayback";

function usePlayer(frameCount: number, fps: number, initial: { frame?: number; playing?: boolean } = {}) {
  const [frame, setFrame] = useState(initial.frame ?? 0);
  const [playing, setPlaying] = useState(initial.playing ?? true);
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });
  return { frame, playing, setPlaying };
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
  it("stops at once on a one-frame clip instead of latching on", () => {
    const { result } = renderHook(() => usePlayer(1, 10, { frame: 0, playing: false }));
    act(() => result.current.setPlaying(true)); // pressed Play: a false → true transition, like the restart case
    act(() => vi.advanceTimersByTime(300));
    expect(result.current.frame).toBe(0);
    expect(result.current.playing).toBe(false);
  });

  it("restarts from the beginning when Play is pressed while parked on the last frame", () => {
    const { result } = renderHook(() => usePlayer(3, 10, { frame: 2, playing: false }));
    expect(result.current.frame).toBe(2);
    act(() => result.current.setPlaying(true));
    expect(result.current.frame).toBe(0); // restarted, not stopped immediately
    expect(result.current.playing).toBe(true);
    act(() => vi.advanceTimersByTime(150));
    expect(result.current.frame).toBe(1); // and it keeps advancing from there
  });
});
