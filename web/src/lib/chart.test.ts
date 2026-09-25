import { describe, expect, it } from "vitest";
import { extent, linePath, nearestIndex, niceTicks, scale } from "./chart";

describe("chart helpers", () => {
  it("extent ignores non-finite values and widens a flat range", () => {
    expect(extent([0.2, NaN, 0.8])).toEqual([0.2, 0.8]);
    expect(extent([])).toEqual([0, 1]);
    expect(extent([3, 3])).toEqual([2.5, 3.5]);
  });
  it("scale maps linearly, including inverted ranges", () => {
    const y = scale([0, 1], [100, 0]);
    expect(y(0)).toBe(100);
    expect(y(0.25)).toBe(75);
    expect(scale([2, 2], [0, 10])(2)).toBe(0);
  });
  it("niceTicks steps by 1, 2 or 5 times a power of ten", () => {
    expect(niceTicks(0, 1, 4)).toEqual([0, 0.5, 1]);
    expect(niceTicks(0, 100, 4)).toEqual([0, 50, 100]);
    expect(niceTicks(0.31, 0.74, 4)).toEqual([0.4, 0.6]);
  });
  it("linePath skips non-finite points", () => {
    expect(linePath([[0, 0], [1, NaN], [2, 1]], (v) => v * 10, (v) => v * 10)).toBe("M0.0,0.0L20.0,10.0");
  });
  it("nearestIndex", () => {
    expect(nearestIndex([1, 2, 3, 10], 6.4)).toBe(2);
    expect(nearestIndex([1, 2, 3, 10], 6.6)).toBe(3);
    expect(nearestIndex([], 1)).toBe(-1);
  });
});
