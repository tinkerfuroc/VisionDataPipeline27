import { describe, expect, it } from "vitest";
import { hexToRgb, tintMask } from "./mask";

describe("mask tinting", () => {
  it("parses class colours and falls back to green", () => {
    expect(hexToRgb("#e6194b")).toEqual([230, 25, 75]);
    expect(hexToRgb("3cb44b")).toEqual([60, 180, 75]);
    expect(hexToRgb("nope")).toEqual([0, 255, 0]);
  });
  it("colours object pixels and makes the background transparent, in place", () => {
    const px = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]);
    const out = tintMask(px, [10, 20, 30], 0.5);
    expect(out).toBe(px);
    expect(Array.from(out)).toEqual([10, 20, 30, 128, 0, 0, 0, 0]);
  });
  it("clamps alpha to 0..1", () => {
    expect(tintMask(new Uint8ClampedArray([255, 255, 255, 255]), [1, 2, 3], 7)[3]).toBe(255);
    expect(tintMask(new Uint8ClampedArray([255, 255, 255, 255]), [1, 2, 3], -1)[3]).toBe(0);
  });
});
