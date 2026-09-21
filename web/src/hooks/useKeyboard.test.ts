import { describe, expect, it } from "vitest";
import { keyAction } from "./useKeyboard";

const key = (k: string, mods: Partial<{ shiftKey: boolean; ctrlKey: boolean; metaKey: boolean; altKey: boolean }> = {}) =>
  keyAction({ key: k, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false, ...mods });

describe("keyAction", () => {
  it("maps accept, clear and undo", () => {
    expect(key("Enter")).toEqual({ type: "accept" });
    expect(key("Escape")).toEqual({ type: "clear" });
    expect(key("z")).toEqual({ type: "undo" });
    expect(key("Z")).toEqual({ type: "undo" });
  });
  it("steps by 1, or by 10 with Shift", () => {
    expect(key("ArrowRight")).toEqual({ type: "step", delta: 1 });
    expect(key("ArrowLeft")).toEqual({ type: "step", delta: -1 });
    expect(key("ArrowRight", { shiftKey: true })).toEqual({ type: "step", delta: 10 });
    expect(key("ArrowLeft", { shiftKey: true })).toEqual({ type: "step", delta: -10 });
  });
  it("jumps between keyframes and to the next empty frame", () => {
    expect(key("k")).toEqual({ type: "keyframe", dir: 1 });
    expect(key("K", { shiftKey: true })).toEqual({ type: "keyframe", dir: -1 });
    expect(key("g")).toEqual({ type: "nextEmpty" });
  });
  it("maps play, delete and overlay", () => {
    expect(key(" ")).toEqual({ type: "play" });
    expect(key("Delete")).toEqual({ type: "delete" });
    expect(key("m")).toEqual({ type: "overlay" });
  });
  it("ignores unknown keys and browser shortcuts", () => {
    expect(key("q")).toBeNull();
    expect(key("z", { ctrlKey: true })).toBeNull();
    expect(key("k", { metaKey: true })).toBeNull();
    expect(key("ArrowRight", { altKey: true })).toBeNull();
  });
});
