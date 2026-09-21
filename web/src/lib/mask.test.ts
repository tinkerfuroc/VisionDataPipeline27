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
  it("outlines the object when it knows the image size: solid rings inside, a white halo outside", () => {
    // 9×9 image with a 7×7 object in the middle; at this size the edge is 2 px wide
    const w = 9;
    const px = new Uint8ClampedArray(w * w * 4);
    for (let y = 1; y <= 7; y++) for (let x = 1; x <= 7; x++) px.fill(255, (y * w + x) * 4, (y * w + x) * 4 + 4);
    const out = tintMask(px, [10, 20, 30], 0.5, { width: w, height: w });
    const at = (x: number, y: number) => Array.from(out.slice((y * w + x) * 4, (y * w + x) * 4 + 4));
    expect(at(0, 0)).toEqual([255, 255, 255, 230]); // halo, diagonal neighbour of the object's corner
    expect(at(0, 4)).toEqual([255, 255, 255, 230]);
    expect(at(1, 4)).toEqual([10, 20, 30, 255]); // outermost ring
    expect(at(2, 4)).toEqual([10, 20, 30, 255]); // second ring
    expect(at(3, 4)).toEqual([10, 20, 30, 128]); // interior keeps the translucent tint
    expect(at(4, 4)).toEqual([10, 20, 30, 128]);
  });
  it("draws no edge where the frame cuts the object off, and leaves far background transparent", () => {
    const w = 8;
    const px = new Uint8ClampedArray(w * w * 4);
    for (let y = 0; y < w; y++) for (let x = 0; x < 3; x++) px.fill(255, (y * w + x) * 4, (y * w + x) * 4 + 4);
    const out = tintMask(px, [10, 20, 30], 0.5, { width: w, height: w });
    const alpha = (x: number, y: number) => out[(y * w + x) * 4 + 3];
    expect(alpha(0, 4)).toBe(128); // touches the image border only: interior, not edge
    expect(alpha(2, 4)).toBe(255); // the real boundary
    expect(alpha(3, 4)).toBe(230); // halo
    expect(alpha(6, 4)).toBe(0);
  });
  it("clamps alpha to 0..1", () => {
    expect(tintMask(new Uint8ClampedArray([255, 255, 255, 255]), [1, 2, 3], 7)[3]).toBe(255);
    expect(tintMask(new Uint8ClampedArray([255, 255, 255, 255]), [1, 2, 3], -1)[3]).toBe(0);
  });
});
