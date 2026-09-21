// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { isField, keyAction } from "./useKeyboard";

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

describe("isField", () => {
  const input = (type?: string) => {
    const el = document.createElement("input");
    if (type !== undefined) el.type = type;
    return el;
  };

  it("treats a text-entry input, a textarea and a select as a field", () => {
    // jsdom does not implement `isContentEditable` (it stays `undefined` even after `contentEditable = "true"`),
    // so the contenteditable branch of `isField` cannot be exercised from a unit test in this environment.
    expect(isField(input())).toBe(true); // default type is "text"
    expect(isField(input("text"))).toBe(true);
    expect(isField(input("number"))).toBe(true);
    expect(isField(document.createElement("textarea"))).toBe(true);
    expect(isField(document.createElement("select"))).toBe(true);
  });
  it("does not treat a non-text input as a field", () => {
    for (const type of ["checkbox", "radio", "button", "submit", "range", "file", "color"]) {
      expect(isField(input(type))).toBe(false);
    }
  });
  it("does not treat a plain element, or null, as a field", () => {
    expect(isField(document.createElement("button"))).toBe(false);
    expect(isField(document.createElement("div"))).toBe(false);
    expect(isField(null)).toBe(false);
  });
});
