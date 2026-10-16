import { describe, expect, it } from "vitest";
import type { RegionRange } from "../api/types";
import {
  activeRegion, addRangeFrom, dragRect, hitTest, isExcluded, normRect, problem, rangeIndexAt, removeRange, setBounds,
  splitAt,
} from "./regions";

const R = (start: number, end: number, crop: RegionRange["crop"] = null, blocks: RegionRange["blocks"] = []): RegionRange =>
  ({ start, end, crop, blocks });

describe("ranges", () => {
  it("finds the range of a frame and treats a no-op range as no region", () => {
    const rs = [R(0, 3, null, [[0, 0, 5, 5]]), R(5, 9)];
    expect(rangeIndexAt(rs, 2)).toBe(0);
    expect(rangeIndexAt(rs, 4)).toBeNull();
    expect(rangeIndexAt(rs, 7)).toBe(1);
    expect(activeRegion(rs, 7)).toBeNull();
    expect(activeRegion(rs, 1)?.blocks).toHaveLength(1);
  });

  it("adds a range from a frame to the end of the gap it sits in", () => {
    const { regions, index } = addRangeFrom([R(0, 3), R(10, 12)], 5, 20);
    expect(regions.map((r) => [r.start, r.end])).toEqual([[0, 3], [5, 9], [10, 12]]);
    expect(index).toBe(1);
    expect(addRangeFrom([], 4, 20).regions).toEqual([R(4, 19)]);
  });

  it("splits a range at a frame, the new part copying the regions", () => {
    const rs = [R(0, 9, [0, 0, 50, 50], [[1, 1, 5, 5]])];
    const { regions, index } = splitAt(rs, 0, 4);
    expect(regions.map((r) => [r.start, r.end])).toEqual([[0, 3], [4, 9]]);
    expect(index).toBe(1);
    expect(regions[1].blocks).toEqual([[1, 1, 5, 5]]);
    expect(regions[1].blocks).not.toBe(regions[0].blocks);
    expect(splitAt(rs, 0, 0).regions).toBe(rs); // splitting at the start is a no-op
  });

  it("removes and re-bounds ranges, clamped to the clip", () => {
    expect(removeRange([R(0, 1), R(2, 3)], 0)).toEqual([R(2, 3)]);
    expect(setBounds([R(2, 5)], 0, -3, 40, 20)).toEqual([R(0, 19)]);
  });

  it("reports the first problem like the server would", () => {
    expect(problem([R(0, 5), R(5, 6)], 100, 100, 20)).toMatch(/overlaps/);
    expect(problem([R(0, 5, [0, 0, 20, 40])], 100, 100, 20)).toMatch(/32/);
    expect(problem([R(0, 5, [0, 0, 40, 40], [[50, 50, 60, 60]])], 100, 100, 20)).toMatch(/outside the crop/);
    expect(problem([R(0, 5, [0, 0, 40, 40], [[10, 10, 20, 20]])], 100, 100, 20)).toBeNull();
  });
});

describe("geometry", () => {
  const region = R(0, 9, [10, 10, 90, 90], [[20, 20, 40, 40]]);

  it("tells excluded points", () => {
    expect(isExcluded(region, 5, 50)).toBe(true);
    expect(isExcluded(region, 30, 30)).toBe(true);
    expect(isExcluded(region, 60, 60)).toBe(false);
    expect(isExcluded(null, 0, 0)).toBe(false);
  });

  it("hit-tests blocks before the crop, handles before insides", () => {
    expect(hitTest(region, 40, 20, 3)).toEqual({ kind: "block", index: 0, part: "delete" });
    expect(hitTest(region, 20, 40, 3)).toEqual({ kind: "block", index: 0, part: "sw" });
    expect(hitTest(region, 30, 30, 3)).toEqual({ kind: "block", index: 0, part: "move" });
    expect(hitTest(region, 90, 90, 3)).toEqual({ kind: "crop", index: 0, part: "se" });
    expect(hitTest(region, 60, 60, 3)).toEqual({ kind: "crop", index: 0, part: "move" });
    expect(hitTest(region, 2, 2, 3)).toBeNull();
  });

  it("drags and resizes inside the frame", () => {
    expect(dragRect([10, 10, 30, 30], "move", 100, 0, 100, 100)).toEqual([80, 10, 100, 30]);
    expect(dragRect([10, 10, 30, 30], "se", 5, 5, 100, 100)).toEqual([10, 10, 35, 35]);
    expect(dragRect([10, 10, 30, 30], "nw", 40, 0, 100, 100)).toEqual([30, 10, 50, 30]); // flips, stays normalised
    expect(normRect([50.7, 10.2], [-5, 40.9], 100, 100)).toEqual([0, 10, 51, 41]);
  });
});

import { regionRuns } from "./regions";

it("splits the clip into excluded and all-clear runs", () => {
  const rs = [R(2, 4, null, [[0, 0, 5, 5]]), R(5, 6), R(7, 8, [0, 0, 40, 40])];
  expect(regionRuns(rs, 10)).toEqual([
    { start: 0, end: 1, excluded: false }, { start: 2, end: 4, excluded: true },
    { start: 5, end: 6, excluded: false }, { start: 7, end: 8, excluded: true }, { start: 9, end: 9, excluded: false },
  ]);
  expect(regionRuns([], 3)).toEqual([{ start: 0, end: 2, excluded: false }]);
});
