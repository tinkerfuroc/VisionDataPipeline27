import type { Rect, RegionRange } from "../api/types";

export const MIN_CROP = 32;

export type Part = "move" | "nw" | "ne" | "sw" | "se" | "delete";
export type Hit = { kind: "crop" | "block"; index: number; part: Part };

const sorted = (rs: RegionRange[]) => [...rs].sort((a, b) => a.start - b.start);
const copy = (r: RegionRange): RegionRange => ({
  ...r, crop: r.crop === null ? null : [...r.crop] as Rect, blocks: r.blocks.map((b) => [...b] as Rect),
});

export function rangeIndexAt(regions: RegionRange[], frame: number): number | null {
  const i = regions.findIndex((r) => r.start <= frame && frame <= r.end);
  return i < 0 ? null : i;
}

/** the region that applies to `frame`; a range with no crop and no blocks is no region */
export function activeRegion(regions: RegionRange[], frame: number): RegionRange | null {
  const i = rangeIndexAt(regions, frame);
  if (i === null) return null;
  const r = regions[i];
  return r.crop !== null || r.blocks.length > 0 ? r : null;
}

/** a new empty range from `frame` to the end of the gap it sits in (before the next range, or the clip's end) */
export function addRangeFrom(regions: RegionRange[], frame: number, frameCount: number) {
  const next = sorted(regions).find((r) => r.start > frame);
  const added: RegionRange = { start: frame, end: next ? next.start - 1 : frameCount - 1, crop: null, blocks: [] };
  const out = sorted([...regions, added]);
  return { regions: out, index: out.indexOf(added) };
}

/** split range `index` at `frame`: frames before keep its regions, the new part from `frame` starts as a copy */
export function splitAt(regions: RegionRange[], index: number, frame: number) {
  const r = regions[index];
  if (frame <= r.start || frame > r.end) return { regions, index };
  const head = { ...copy(r), end: frame - 1 };
  const tail = { ...copy(r), start: frame };
  const out = [...regions.slice(0, index), head, tail, ...regions.slice(index + 1)];
  return { regions: out, index: index + 1 };
}

export const removeRange = (regions: RegionRange[], index: number) => regions.filter((_, i) => i !== index);

export function setBounds(regions: RegionRange[], index: number, start: number, end: number, frameCount: number) {
  const clamp = (v: number) => Math.max(0, Math.min(frameCount - 1, Math.round(v)));
  return updateRange(regions, index, { start: clamp(start), end: clamp(end) });
}

export function updateRange(regions: RegionRange[], index: number, patch: Partial<RegionRange>) {
  return regions.map((r, i) => (i === index ? { ...r, ...patch } : r));
}

const intersects = (a: Rect, b: Rect) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];

/** the first thing the server would reject (validate_regions), or null */
export function problem(regions: RegionRange[], width: number, height: number, frameCount: number): string | null {
  const rs = sorted(regions);
  for (let i = 0; i < rs.length; i++) {
    const r = rs[i];
    const name = `Region ${r.start}–${r.end}`;
    if (r.start > r.end) return `${name}: start is after end`;
    if (r.start < 0 || r.end >= frameCount) return `${name}: frames must be within 0..${frameCount - 1}`;
    if (i > 0 && r.start <= rs[i - 1].end) return `${name} overlaps region ${rs[i - 1].start}–${rs[i - 1].end}`;
    const rects: [string, Rect][] = [...(r.crop ? [["crop", r.crop] as [string, Rect]] : []), ...r.blocks.map((b) => ["block", b] as [string, Rect])];
    for (const [kind, [x0, y0, x1, y1]] of rects) {
      if (x0 >= x1 || y0 >= y1) return `${name}: the ${kind} is empty`;
      if (x0 < 0 || y0 < 0 || x1 > width || y1 > height) return `${name}: the ${kind} is outside the frame`;
    }
    if (r.crop) {
      if (r.crop[2] - r.crop[0] < MIN_CROP || r.crop[3] - r.crop[1] < MIN_CROP) {
        return `${name}: the crop must be at least ${MIN_CROP}×${MIN_CROP} px`;
      }
      if (r.blocks.some((b) => !intersects(b, r.crop as Rect))) return `${name}: a blocked box lies outside the crop`;
    }
  }
  return null;
}

const inside = (r: Rect, x: number, y: number) => r[0] <= x && x < r[2] && r[1] <= y && y < r[3];

export function isExcluded(region: RegionRange | null, x: number, y: number): boolean {
  if (region === null) return false;
  if (region.crop && !inside(region.crop, x, y)) return true;
  return region.blocks.some((b) => inside(b, x, y));
}

function rectHit(r: Rect, x: number, y: number, tol: number, withDelete: boolean): Part | null {
  const near = (ax: number, ay: number) => Math.abs(x - ax) <= tol && Math.abs(y - ay) <= tol;
  // the × sits just outside the box's NE corner so that corner stays a resize handle
  if (withDelete && Math.abs(x - (r[2] + tol * 1.5)) <= tol && Math.abs(y - (r[1] - tol * 1.5)) <= tol) return "delete";
  if (near(r[0], r[1])) return "nw";
  if (near(r[2], r[1])) return "ne";
  if (near(r[0], r[3])) return "sw";
  if (near(r[2], r[3])) return "se";
  if (inside(r, x, y)) return "move";
  return null;
}

/** what a press at image point (x, y) grabs; `tol` is the handle radius in image pixels. Blocks (topmost last)
 * come before the crop, so a block inside the crop stays reachable. */
export function hitTest(region: RegionRange | null, x: number, y: number, tol: number): Hit | null {
  if (region === null) return null;
  for (let i = region.blocks.length - 1; i >= 0; i--) {
    const part = rectHit(region.blocks[i], x, y, tol, true);
    if (part !== null) return { kind: "block", index: i, part };
  }
  if (region.crop) {
    const part = rectHit(region.crop, x, y, tol, false);
    if (part !== null) return { kind: "crop", index: 0, part };
  }
  return null;
}

/** two corners -> a normalised integer rect clamped to the frame */
export function normRect(a: [number, number], b: [number, number], width: number, height: number): Rect {
  const cx = (v: number) => Math.max(0, Math.min(width, v));
  const cy = (v: number) => Math.max(0, Math.min(height, v));
  return [
    Math.floor(cx(Math.min(a[0], b[0]))), Math.floor(cy(Math.min(a[1], b[1]))),
    Math.ceil(cx(Math.max(a[0], b[0]))), Math.ceil(cy(Math.max(a[1], b[1]))),
  ];
}

/** `rect` moved (whole, kept inside the frame) or resized by one corner by (dx, dy) image pixels */
export function dragRect(rect: Rect, part: Part, dx: number, dy: number, width: number, height: number): Rect {
  const [x0, y0, x1, y1] = rect;
  if (part === "move") {
    const mx = Math.max(-x0, Math.min(width - x1, Math.round(dx)));
    const my = Math.max(-y0, Math.min(height - y1, Math.round(dy)));
    return [x0 + mx, y0 + my, x1 + mx, y1 + my];
  }
  const ax = part === "nw" || part === "sw" ? x1 : x0; // the fixed corner
  const ay = part === "nw" || part === "ne" ? y1 : y0;
  const mx = part === "nw" || part === "sw" ? x0 + dx : x1 + dx;
  const my = part === "nw" || part === "ne" ? y0 + dy : y1 + dy;
  return normRect([ax, ay], [mx, my], width, height);
}

/** the clip as runs of frames that have something excluded (a crop or a block) or are all clear */
export function regionRuns(regions: RegionRange[], frameCount: number) {
  const runs: { start: number; end: number; excluded: boolean }[] = [];
  for (let i = 0; i < frameCount; i++) {
    const excluded = activeRegion(regions, i) !== null;
    const last = runs[runs.length - 1];
    if (last && last.excluded === excluded) last.end = i;
    else runs.push({ start: i, end: i, excluded });
  }
  return runs;
}

/** "crop + 2 blocked", "crop", "1 blocked" or "all clear" for one frame */
export function regionWords(regions: RegionRange[], frame: number): string {
  const r = activeRegion(regions, frame);
  if (r === null) return "all clear";
  const blocked = r.blocks.length > 0 ? `${r.blocks.length} blocked` : "";
  return r.crop ? (blocked ? `crop + ${blocked}` : "crop") : blocked;
}
