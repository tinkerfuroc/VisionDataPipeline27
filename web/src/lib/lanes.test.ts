import { describe, expect, it } from "vitest";
import { ACTIVE_H, GUTTER, LANE_GAP, LANE_H, MAX_LANES, MAX_VIEW, clampScroll, laneAt, layoutLanes, scrollToShow } from "./lanes";

describe("lane layout", () => {
  it("uses the spec's sizes", () => {
    expect([LANE_H, ACTIVE_H, LANE_GAP, GUTTER, MAX_LANES, MAX_VIEW]).toEqual([16, 30, 2, 72, 6, 120]);
  });
  it("stacks lanes in order: the active one 30 px, the others 16 px, 2 px apart", () => {
    expect(layoutLanes([1, 2, 3], 2)).toEqual({
      boxes: [{ id: 1, top: 0, height: 16 }, { id: 2, top: 18, height: 30 }, { id: 3, top: 50, height: 16 }],
      content: 66,
      view: 66,
    });
  });
  it("draws every lane short when no object is active, and the placeholder lane of an empty clip tall", () => {
    expect(layoutLanes([1, 2], null).boxes.map((b) => b.height)).toEqual([16, 16]);
    expect(layoutLanes([null], null)).toEqual({ boxes: [{ id: null, top: 0, height: 30 }], content: 30, view: 30 });
  });
  it("grows up to six lanes, then keeps the view at that height", () => {
    const six = layoutLanes([1, 2, 3, 4, 5, 6], 1);
    expect([six.content, six.view]).toEqual([120, 120]);
    const eight = layoutLanes([1, 2, 3, 4, 5, 6, 7, 8], 8);
    expect([eight.content, eight.view]).toEqual([156, 120]); // 7 × (16 + 2) + 30
  });
});

describe("hit testing and scrolling", () => {
  const eight = layoutLanes([1, 2, 3, 4, 5, 6, 7, 8], 8);
  it("finds the lane under a y position, and none in a gap", () => {
    const three = layoutLanes([1, 2, 3], 2);
    expect(laneAt(three.boxes, 0)?.id).toBe(1);
    expect(laneAt(three.boxes, 15.9)?.id).toBe(1);
    expect(laneAt(three.boxes, 17)).toBeNull();
    expect(laneAt(three.boxes, 18)?.id).toBe(2);
    expect(laneAt(three.boxes, 47.9)?.id).toBe(2);
    expect(laneAt(three.boxes, 70)).toBeNull();
  });
  it("clamps the scroll offset to the lane stack", () => {
    expect(clampScroll(-5, eight)).toBe(0);
    expect(clampScroll(999, eight)).toBe(36); // 156 − 120
    expect(clampScroll(10, layoutLanes([1, 2], 1))).toBe(0);
  });
  it("scrolls just enough to show a lane", () => {
    expect(scrollToShow(0, eight, 8)).toBe(36);
    expect(scrollToShow(36, eight, 1)).toBe(0);
    expect(scrollToShow(10, eight, 3)).toBe(10);
    expect(scrollToShow(10, eight, null)).toBe(10);
  });
});
