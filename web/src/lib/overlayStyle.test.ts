// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_STYLE, MAX_THICKNESS, loadStyle, outlineFor, parseStyle, saveStyle } from "./overlayStyle";

describe("overlay style", () => {
  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });
  it("defaults to the class colour at 5 px", () => {
    expect(parseStyle(null)).toEqual({ color: null, thickness: 5 });
    expect(parseStyle("not json")).toEqual(DEFAULT_STYLE);
  });
  it("keeps valid fields, drops bad ones and clamps the thickness", () => {
    expect(parseStyle('{"color":"#FFAA00","thickness":0}')).toEqual({ color: "#ffaa00", thickness: 0 });
    expect(parseStyle('{"color":"red","thickness":99}')).toEqual({ color: null, thickness: MAX_THICKNESS });
    expect(parseStyle('{"thickness":-3.2}')).toEqual({ color: null, thickness: 0 });
    expect(parseStyle('{"thickness":"7"}')).toEqual(DEFAULT_STYLE);
  });
  it("round-trips through localStorage and survives storage that throws", () => {
    saveStyle({ color: "#000000", thickness: 2 });
    expect(loadStyle()).toEqual({ color: "#000000", thickness: 2 });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(loadStyle()).toEqual(DEFAULT_STYLE);
    expect(() => saveStyle(DEFAULT_STYLE)).not.toThrow();
  });
  it("resolves the outline colour: the class colour unless a custom one is set", () => {
    expect(outlineFor({ color: null, thickness: 4 }, [1, 2, 3])).toEqual({ rgb: [1, 2, 3], edge: 4 });
    expect(outlineFor({ color: "#ff0000", thickness: 0 }, [1, 2, 3])).toEqual({ rgb: [255, 0, 0], edge: 0 });
  });
});
