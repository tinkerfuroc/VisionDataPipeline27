import type { FrameState } from "../api/types";

/** Timeline geometry (spec §9), in CSS px. */
export const LANE_H = 16; // not smaller: the lane names beside it stay text-xs
export const ACTIVE_H = 30;
export const LANE_GAP = 2;
export const GUTTER = 72;
export const MAX_LANES = 6;
/** the lane viewport's height cap: six lanes with the active one among them; beyond that the lanes scroll */
export const MAX_VIEW = ACTIVE_H + (MAX_LANES - 1) * (LANE_H + LANE_GAP);

/** One object's lane: its name ("#2 cup"), class colour, per-frame states, and whether it is hidden on the canvas. */
export type Lane = { id: number; label: string; color: string; states: FrameState[]; hidden: boolean };
/** A lane's place in the stack, from the top of the lane area, before scrolling. `id` null = the empty clip's lane. */
export type LaneBox = { id: number | null; top: number; height: number };
export type LaneLayout = { boxes: LaneBox[]; content: number; view: number };

/** Stack the lanes in the given order: the active lane (and the single placeholder lane) tall, the others short. */
export function layoutLanes(ids: readonly (number | null)[], activeId: number | null): LaneLayout {
  const boxes: LaneBox[] = [];
  let top = 0;
  for (const id of ids) {
    const height = id === null || id === activeId ? ACTIVE_H : LANE_H;
    boxes.push({ id, top, height });
    top += height + LANE_GAP;
  }
  const content = Math.max(0, top - LANE_GAP);
  return { boxes, content, view: Math.min(content, MAX_VIEW) };
}

/** The lane under `y` (lane-area coordinates, scroll applied), or null in a gap or past the last lane. */
export function laneAt(boxes: readonly LaneBox[], y: number): LaneBox | null {
  for (const b of boxes) if (y >= b.top && y < b.top + b.height) return b;
  return null;
}

export function clampScroll(scroll: number, layout: LaneLayout): number {
  return Math.max(0, Math.min(layout.content - layout.view, Math.round(scroll)));
}

/** The smallest change of `scroll` that shows lane `id` entirely; unchanged (but clamped) when it already shows. */
export function scrollToShow(scroll: number, layout: LaneLayout, id: number | null): number {
  const box = layout.boxes.find((b) => b.id === id);
  if (box === undefined) return clampScroll(scroll, layout);
  if (box.top < scroll) return clampScroll(box.top, layout);
  if (box.top + box.height > scroll + layout.view) return clampScroll(box.top + box.height - layout.view, layout);
  return clampScroll(scroll, layout);
}
