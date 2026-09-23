import { describe, expect, it } from "vitest";
import {
  DEFAULT_SIZE, MAX_SIZE, MIN_SIZE, baseLayerData, brushReducer, clampSize, drawStroke, emptyBrush, extendStroke,
  layerToMaskData, rasterize, type BrushContext, type Stroke,
} from "./brush";

const stroke = (points: [number, number][], mode: "paint" | "erase" = "paint", size = 10): Stroke => ({ mode, size, points });

function recorder(): BrushContext & { calls: string[] } {
  const calls: string[] = [];
  const ctx = {
    globalCompositeOperation: "source-over", strokeStyle: "", fillStyle: "", lineWidth: 1, lineCap: "butt", lineJoin: "miter",
    beginPath: () => calls.push("beginPath"),
    moveTo: (x: number, y: number) => calls.push(`moveTo ${x},${y}`),
    lineTo: (x: number, y: number) => calls.push(`lineTo ${x},${y}`),
    arc: (x: number, y: number, r: number) => calls.push(`arc ${x},${y},${r}`),
    stroke: () => calls.push(`stroke ${ctx.globalCompositeOperation} ${ctx.strokeStyle} w${ctx.lineWidth} ${ctx.lineCap}/${ctx.lineJoin}`),
    fill: () => calls.push(`fill ${ctx.globalCompositeOperation} ${ctx.fillStyle}`),
    clearRect: (x: number, y: number, w: number, h: number) => calls.push(`clearRect ${x},${y},${w},${h}`),
    putImageData: (_d: ImageData, x: number, y: number) => calls.push(`putImageData ${x},${y}`),
    calls,
  } as unknown as BrushContext & { calls: string[] };
  return ctx;
}

describe("brush draft", () => {
  it("clamps the size and grows or shrinks it by 20 %", () => {
    expect(emptyBrush()).toEqual({ size: DEFAULT_SIZE, strokes: [] });
    expect(clampSize(0)).toBe(MIN_SIZE);
    expect(clampSize(999)).toBe(MAX_SIZE);
    expect(clampSize(7.6)).toBe(8);
    const d = emptyBrush(50);
    expect(brushReducer(d, { type: "grow", dir: 1 }).size).toBe(60);
    expect(brushReducer(d, { type: "grow", dir: -1 }).size).toBe(42);
    const max = emptyBrush(MAX_SIZE);
    expect(brushReducer(max, { type: "grow", dir: 1 })).toBe(max); // unchanged: same object
    expect(brushReducer(d, { type: "size", size: 50 })).toBe(d);
  });
  it("commits, undoes and clears strokes", () => {
    let d = emptyBrush();
    d = brushReducer(d, { type: "commit", stroke: stroke([[1, 1]]) });
    d = brushReducer(d, { type: "commit", stroke: stroke([[2, 2], [3, 3]], "erase") });
    expect(d.strokes).toHaveLength(2);
    d = brushReducer(d, { type: "undo" });
    expect(d.strokes).toEqual([stroke([[1, 1]])]);
    const cleared = brushReducer(d, { type: "clear" });
    expect(cleared.strokes).toEqual([]);
    expect(brushReducer(cleared, { type: "undo" })).toBe(cleared);
    expect(brushReducer(cleared, { type: "clear" })).toBe(cleared);
  });
  it("extendStroke appends distinct points only", () => {
    const s = stroke([[1, 1]]);
    expect(extendStroke(s, 1, 1)).toBe(false);
    expect(extendStroke(s, 2, 3)).toBe(true);
    expect(s.points).toEqual([[1, 1], [2, 3]]);
  });
});

describe("rasteriser", () => {
  it("draws a one-point stroke as a dot and a polyline with round caps, in the class colour", () => {
    const ctx = recorder();
    drawStroke(ctx, stroke([[4, 5]], "paint", 10), 0, [255, 0, 0]);
    expect(ctx.calls).toEqual(["beginPath", "arc 4.5,5.5,5", "fill source-over rgb(255, 0, 0)"]);
    ctx.calls.length = 0;
    drawStroke(ctx, stroke([[0, 0], [10, 0], [10, 10]], "erase", 6), 0, [255, 0, 0]);
    expect(ctx.calls).toEqual([
      "beginPath", "moveTo 0.5,0.5", "lineTo 10.5,0.5", "lineTo 10.5,10.5",
      "stroke destination-out rgb(255, 0, 0) w6 round/round",
    ]);
  });
  it("draws only the new segment when asked to continue from a point", () => {
    const ctx = recorder();
    drawStroke(ctx, stroke([[0, 0], [10, 0], [10, 10]]), 2, [0, 0, 0]);
    expect(ctx.calls).toEqual(["beginPath", "moveTo 10.5,0.5", "lineTo 10.5,10.5", "stroke source-over rgb(0, 0, 0) w10 round/round"]);
  });
  it("rasterize clears, restores the base and replays every stroke", () => {
    const ctx = recorder();
    const base = { data: new Uint8ClampedArray(4), width: 1, height: 1 } as unknown as ImageData;
    rasterize(ctx, base, [stroke([[1, 1]]), stroke([[2, 2]], "erase")], [1, 2, 3], 8, 6);
    expect(ctx.calls[0]).toBe("clearRect 0,0,8,6");
    expect(ctx.calls[1]).toBe("putImageData 0,0");
    expect(ctx.calls.filter((c) => c.startsWith("fill"))).toEqual(["fill source-over rgb(1, 2, 3)", "fill destination-out rgb(1, 2, 3)"]);
    expect(ctx.globalCompositeOperation).toBe("source-over"); // left in the default state
    ctx.calls.length = 0;
    rasterize(ctx, null, [], [1, 2, 3], 8, 6);
    expect(ctx.calls).toEqual(["clearRect 0,0,8,6"]);
  });
  it("converts a stored mask to a colour layer and a layer back to a mask", () => {
    // 2x1 mask image: left pixel white (in the mask), right pixel black
    const mask = { data: new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]), width: 2, height: 1 };
    expect(Array.from(baseLayerData(mask, [10, 20, 30]))).toEqual([10, 20, 30, 255, 0, 0, 0, 0]);
    const layer = new Uint8ClampedArray([10, 20, 30, 200, 10, 20, 30, 100, 0, 0, 0, 0]);
    const out = layerToMaskData(layer);
    expect(Array.from(out.data)).toEqual([255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
    expect(out.any).toBe(true);
    expect(layerToMaskData(new Uint8ClampedArray(8)).any).toBe(false);
  });
});
