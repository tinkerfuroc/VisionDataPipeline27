export type Pt = [number, number];

/** Dark categorical slots 1–4 of the reference palette, validated (CVD and contrast) on the well colour #1E1F21. */
export const SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"] as const;
/** Grid and axis hairlines: the chrome's `line` token. */
export const GRID = "#4A4D52";
export const TEXT_DIM = "#A3A5AA";
export const TEXT_INK = "#E9E7E2";
export const WELL = "#1E1F21";

export function extent(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return [0, 1];
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  return lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi];
}

export function scale(domain: [number, number], range: [number, number]): (v: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v: number) => r0 + (v - d0) * k;
}

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

/** Ticks inside [min, max] at a 1/2/5 × 10^k step; rounded to kill float noise (0.30000000000000004). */
export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min];
  const step = niceStep((max - min) / count);
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

export function linePath(points: Pt[], x: (v: number) => number, y: (v: number) => number): string {
  return points
    .filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b))
    .map(([a, b], i) => `${i === 0 ? "M" : "L"}${x(a).toFixed(1)},${y(b).toFixed(1)}`)
    .join("");
}

export function nearestIndex(xs: number[], x: number): number {
  let best = -1;
  let dist = Infinity;
  xs.forEach((v, i) => {
    const d = Math.abs(v - x);
    if (d < dist) {
      dist = d;
      best = i;
    }
  });
  return best;
}
