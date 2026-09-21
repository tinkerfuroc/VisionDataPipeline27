import { describe, expect, it } from "vitest";
import type { FrameState } from "../api/types";
import { applyFrameEvent, clampFrame, counts, findNext, frameAtX, labeledPercent, runs } from "./timeline";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);

describe("runs", () => {
  it("is empty for no frames", () => expect(runs([])).toEqual([]));
  it("run-length encodes with inclusive ends", () => {
    expect(runs(S("eekmmd"))).toEqual([
      { state: "empty", start: 0, end: 1 },
      { state: "keyframe", start: 2, end: 2 },
      { state: "masked", start: 3, end: 4 },
      { state: "deleted", start: 5, end: 5 },
    ]);
  });
});

describe("navigation", () => {
  it("clamps and rounds frame indices", () => {
    expect(clampFrame(-3, 10)).toBe(0);
    expect(clampFrame(12, 10)).toBe(9);
    expect(clampFrame(4.6, 10)).toBe(5);
    expect(clampFrame(5, 0)).toBe(0);
  });
  it("finds the next frame in a state strictly beyond the start, without wrapping", () => {
    const s = S("kmekde");
    expect(findNext(s, 0, 1, "keyframe")).toBe(3);
    expect(findNext(s, 3, 1, "keyframe")).toBeNull();
    expect(findNext(s, 3, -1, "keyframe")).toBe(0);
    expect(findNext(s, 0, 1, "empty")).toBe(2);
    expect(findNext(s, 2, 1, "empty")).toBe(5);
  });
  it("maps a timeline x position to a frame", () => {
    expect(frameAtX(0, 100, 10)).toBe(0);
    expect(frameAtX(55, 100, 10)).toBe(5);
    expect(frameAtX(99.9, 100, 10)).toBe(9);
    expect(frameAtX(100, 100, 10)).toBe(9);
    expect(frameAtX(-5, 100, 10)).toBe(0);
    expect(frameAtX(5, 0, 10)).toBe(0);
  });
});

describe("statistics", () => {
  it("counts every state", () => {
    expect(counts(S("kmeed"))).toEqual({ empty: 2, masked: 1, keyframe: 1, deleted: 1 });
  });
  it("reports labeled frames as a percentage of non-deleted frames", () => {
    expect(labeledPercent(S("kmeed"))).toBe(50);
    expect(labeledPercent(S("dd"))).toBe(0);
    expect(labeledPercent([])).toBe(0);
  });
});

describe("applyFrameEvent", () => {
  it("marks a frame masked or empty and returns a new array", () => {
    const s = S("kee");
    const masked = applyFrameEvent(s, 1, true);
    expect(masked).toEqual(S("kme"));
    expect(masked).not.toBe(s);
    expect(applyFrameEvent(masked, 1, false)).toEqual(S("kee"));
  });
  it("never touches keyframes or deleted frames and returns the same array when nothing changes", () => {
    const s = S("kdm");
    expect(applyFrameEvent(s, 0, false)).toBe(s);
    expect(applyFrameEvent(s, 1, true)).toBe(s);
    expect(applyFrameEvent(s, 2, true)).toBe(s);
    expect(applyFrameEvent(s, 9, true)).toBe(s);
  });
});
