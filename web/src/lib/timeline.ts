import type { ClipSummary, FrameState } from "../api/types";

/** A maximal run of consecutive frames in one state; `end` is inclusive. */
export type Run = { state: FrameState; start: number; end: number };

export function runs(states: readonly FrameState[]): Run[] {
  const out: Run[] = [];
  for (let i = 0; i < states.length; i++) {
    const last = out[out.length - 1];
    if (last !== undefined && last.state === states[i]) last.end = i;
    else out.push({ state: states[i], start: i, end: i });
  }
  return out;
}

export function clampFrame(i: number, count: number): number {
  if (count <= 0) return 0;
  return Math.max(0, Math.min(count - 1, Math.round(i)));
}

/** Index of the nearest frame in state `want` strictly beyond `from` in direction `dir`; no wrap-around. */
export function findNext(states: readonly FrameState[], from: number, dir: 1 | -1, want: FrameState): number | null {
  for (let i = from + dir; i >= 0 && i < states.length; i += dir) if (states[i] === want) return i;
  return null;
}

export function counts(states: readonly FrameState[]): Record<FrameState, number> {
  const c: Record<FrameState, number> = { empty: 0, masked: 0, keyframe: 0, deleted: 0 };
  for (const s of states) c[s]++;
  return c;
}

/** Labeled (masked or keyframe) frames as a rounded percentage of the frames that are not deleted. */
export function labeledPercent(states: readonly FrameState[]): number {
  const c = counts(states);
  const live = states.length - c.deleted;
  return live === 0 ? 0 : Math.round((100 * (c.masked + c.keyframe)) / live);
}

/** Labeled frames of one clip summary, as a rounded percentage of its non-deleted frames. */
export function percentLabeled(c: ClipSummary): number {
  const live = c.frame_count - c.deleted_frames.length;
  return live <= 0 ? 0 : Math.round((100 * c.labeled_frames) / live);
}

/** Apply a propagation `frame` event. Keyframes and deleted frames never change; unchanged input is returned as is. */
export function applyFrameEvent(states: FrameState[], frame: number, hasMask: boolean): FrameState[] {
  const current = states[frame];
  if (current === undefined || current === "keyframe" || current === "deleted") return states;
  const next: FrameState = hasMask ? "masked" : "empty";
  if (current === next) return states;
  const copy = states.slice();
  copy[frame] = next;
  return copy;
}

export function frameAtX(x: number, width: number, count: number): number {
  if (width <= 0) return 0;
  return clampFrame(Math.floor((x / width) * count), count);
}
