import type { Box } from "../api/types";

/** screen = image * scale + (tx, ty) */
export type View = { scale: number; tx: number; ty: number };

export function fitView(imgW: number, imgH: number, boxW: number, boxH: number, margin = 8): View {
  const w = Math.max(1, boxW - 2 * margin);
  const h = Math.max(1, boxH - 2 * margin);
  const scale = Math.min(w / imgW, h / imgH);
  return { scale, tx: (boxW - imgW * scale) / 2, ty: (boxH - imgH * scale) / 2 };
}

export const toImage = (v: View, sx: number, sy: number): [number, number] => [(sx - v.tx) / v.scale, (sy - v.ty) / v.scale];
export const toScreen = (v: View, x: number, y: number): [number, number] => [x * v.scale + v.tx, y * v.scale + v.ty];

export function zoomAt(v: View, sx: number, sy: number, factor: number, minScale: number, maxScale: number): View {
  const scale = Math.max(minScale, Math.min(maxScale, v.scale * factor));
  const [ix, iy] = toImage(v, sx, sy);
  return { scale, tx: sx - ix * scale, ty: sy - iy * scale };
}

export const panBy = (v: View, dx: number, dy: number): View => ({ scale: v.scale, tx: v.tx + dx, ty: v.ty + dy });

export const insideImage = (x: number, y: number, w: number, h: number): boolean => x >= 0 && y >= 0 && x < w && y < h;

/** The server requires integer pixel indices with 0 <= x < w and 0 <= y < h. */
export function clampPoint(x: number, y: number, w: number, h: number): [number, number] {
  return [Math.max(0, Math.min(w - 1, Math.floor(x))), Math.max(0, Math.min(h - 1, Math.floor(y)))];
}

export function normBox(a: [number, number], b: [number, number], w: number, h: number): Box {
  const cx = (v: number) => Math.max(0, Math.min(w, Math.round(v)));
  const cy = (v: number) => Math.max(0, Math.min(h, Math.round(v)));
  const [ax, ay, bx, by] = [cx(a[0]), cy(a[1]), cx(b[0]), cy(b[1])];
  return [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
}

/** Index of the smallest box containing the point (text-prompt candidates overlap; the smallest is the most specific). */
export function pickCandidate(bboxes: readonly Box[], x: number, y: number): number | null {
  let best: number | null = null;
  let bestArea = Infinity;
  bboxes.forEach(([x1, y1, x2, y2], i) => {
    const area = (x2 - x1) * (y2 - y1);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && area < bestArea) {
      best = i;
      bestArea = area;
    }
  });
  return best;
}
