import { describe, expect, it } from "vitest";
import { clampPoint, fitView, insideImage, normBox, panBy, pickCandidate, toImage, toScreen, zoomAt } from "./geometry";

describe("view transform", () => {
  it("fits and centres the image inside the box with a margin", () => {
    expect(fitView(640, 480, 1296, 736)).toEqual({ scale: 1.5, tx: 168, ty: 8 });
  });
  it("round-trips between screen and image coordinates", () => {
    const v = { scale: 2, tx: 10, ty: -4 };
    expect(toScreen(v, 5, 7)).toEqual([20, 10]);
    expect(toImage(v, 20, 10)).toEqual([5, 7]);
  });
  it("zooms about the cursor so the image point under it stays put", () => {
    const v = zoomAt({ scale: 1, tx: 0, ty: 0 }, 100, 50, 2, 0.1, 10);
    expect(v).toEqual({ scale: 2, tx: -100, ty: -50 });
    expect(toImage(v, 100, 50)).toEqual([100, 50]);
  });
  it("clamps the zoom to its limits", () => {
    expect(zoomAt({ scale: 1, tx: 0, ty: 0 }, 0, 0, 100, 0.1, 10).scale).toBe(10);
    expect(zoomAt({ scale: 1, tx: 0, ty: 0 }, 0, 0, 0.001, 0.1, 10).scale).toBe(0.1);
  });
  it("pans by a screen delta", () => {
    expect(panBy({ scale: 3, tx: 1, ty: 2 }, 10, -5)).toEqual({ scale: 3, tx: 11, ty: -3 });
  });
});

describe("image-space helpers", () => {
  it("tests whether a point is inside the image", () => {
    expect(insideImage(639.5, 479.9, 640, 480)).toBe(true);
    expect(insideImage(640, 10, 640, 480)).toBe(false);
    expect(insideImage(-0.1, 5, 640, 480)).toBe(false);
  });
  it("clamps a point to integer pixel indices", () => {
    expect(clampPoint(-3.2, 500.7, 640, 480)).toEqual([0, 479]);
    expect(clampPoint(12.9, 7.1, 640, 480)).toEqual([12, 7]);
  });
  it("normalises a drag in any direction to an in-frame integer box", () => {
    expect(normBox([250, 290], [150, 190], 640, 480)).toEqual([150, 190, 250, 290]);
    expect(normBox([-20, -5], [700, 100.4], 640, 480)).toEqual([0, 0, 640, 100]);
  });
  it("picks the smallest candidate box containing the point", () => {
    const boxes: [number, number, number, number][] = [[0, 0, 100, 100], [10, 10, 30, 30], [200, 200, 300, 300]];
    expect(pickCandidate(boxes, 20, 20)).toBe(1);
    expect(pickCandidate(boxes, 50, 50)).toBe(0);
    expect(pickCandidate(boxes, 150, 150)).toBeNull();
  });
});
