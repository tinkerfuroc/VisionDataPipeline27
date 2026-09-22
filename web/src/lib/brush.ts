import type { Rgb } from "./mask";

/** Brush strokes are recorded in image space so a wheel-zoom mid-stroke cannot distort them. */
export type BrushMode = "paint" | "erase";
export type Stroke = { mode: BrushMode; size: number; points: [number, number][] };
export type BrushDraft = { size: number; strokes: Stroke[] };

export const MIN_SIZE = 2;
export const MAX_SIZE = 200;
export const DEFAULT_SIZE = 24;

export const emptyBrush = (size = DEFAULT_SIZE): BrushDraft => ({ size, strokes: [] });
export const clampSize = (n: number): number => Math.max(MIN_SIZE, Math.min(MAX_SIZE, Math.round(n)));

export type BrushAction =
  | { type: "commit"; stroke: Stroke }
  | { type: "undo" }
  | { type: "clear" }
  | { type: "size"; size: number }
  | { type: "grow"; dir: 1 | -1 };

export function brushReducer(d: BrushDraft, a: BrushAction): BrushDraft {
  switch (a.type) {
    case "commit":
      return { ...d, strokes: [...d.strokes, a.stroke] };
    case "undo":
      return d.strokes.length === 0 ? d : { ...d, strokes: d.strokes.slice(0, -1) };
    case "clear":
      return d.strokes.length === 0 ? d : { ...d, strokes: [] };
    case "size": {
      const size = clampSize(a.size);
      return size === d.size ? d : { ...d, size };
    }
    case "grow":
      return brushReducer(d, { type: "size", size: d.size * (a.dir === 1 ? 1.2 : 1 / 1.2) });
  }
}

/** Append a point to the stroke being drawn; false when it repeats the last point. Mutates: this runs per mouse move. */
export function extendStroke(stroke: Stroke, x: number, y: number): boolean {
  const [px, py] = stroke.points[stroke.points.length - 1];
  if (px === x && py === y) return false;
  stroke.points.push([x, y]);
  return true;
}

/** The part of a 2D context the brush draws with; tests pass a recorder. */
export type BrushContext = Pick<
  CanvasRenderingContext2D,
  | "globalCompositeOperation" | "strokeStyle" | "fillStyle" | "lineWidth" | "lineCap" | "lineJoin"
  | "beginPath" | "moveTo" | "lineTo" | "arc" | "stroke" | "fill" | "clearRect" | "putImageData"
>;

const css = ([r, g, b]: Rgb) => `rgb(${r}, ${g}, ${b})`;

/** Draw `stroke` from its point `from` on (0 = the whole stroke). Paint adds colour, erase cuts alpha. */
export function drawStroke(ctx: BrushContext, stroke: Stroke, from: number, rgb: Rgb): void {
  ctx.globalCompositeOperation = stroke.mode === "paint" ? "source-over" : "destination-out";
  ctx.strokeStyle = css(rgb);
  ctx.fillStyle = css(rgb);
  ctx.lineWidth = stroke.size;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const pts = stroke.points;
  ctx.beginPath();
  if (pts.length === 1) {
    ctx.arc(pts[0][0] + 0.5, pts[0][1] + 0.5, stroke.size / 2, 0, 2 * Math.PI);
    ctx.fill();
    return;
  }
  const start = Math.max(1, from);
  ctx.moveTo(pts[start - 1][0] + 0.5, pts[start - 1][1] + 0.5);
  for (let i = start; i < pts.length; i++) ctx.lineTo(pts[i][0] + 0.5, pts[i][1] + 0.5);
  ctx.stroke();
}

/** Rebuild the layer: the stored mask (already in the class colour) then every stroke in order. */
export function rasterize(
  ctx: BrushContext, base: ImageData | null, strokes: Stroke[], rgb: Rgb, width: number, height: number,
): void {
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, width, height);
  if (base !== null) ctx.putImageData(base, 0, 0);
  for (const s of strokes) drawStroke(ctx, s, 0, rgb);
  ctx.globalCompositeOperation = "source-over";
}

type Pixels = { data: Uint8ClampedArray; width: number; height: number };

/** A stored mask image (white = in the mask) as an opaque class-colour layer on transparent. */
export function baseLayerData(mask: Pixels, [r, g, b]: Rgb): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.data.length);
  for (let i = 0; i < mask.data.length; i += 4) {
    if (mask.data[i] > 127) {
      out[i] = r;
      out[i + 1] = g;
      out[i + 2] = b;
      out[i + 3] = 255;
    }
  }
  return out;
}

/** The layer's alpha as an opaque white-on-black mask image, ready to encode as a PNG for the server. */
export function layerToMaskData(layer: Uint8ClampedArray): { data: Uint8ClampedArray; any: boolean } {
  const data = new Uint8ClampedArray(layer.length);
  let any = false;
  for (let i = 0; i < layer.length; i += 4) {
    const on = layer[i + 3] > 127;
    any ||= on;
    data[i] = data[i + 1] = data[i + 2] = on ? 255 : 0;
    data[i + 3] = 255;
  }
  return { data, any };
}
