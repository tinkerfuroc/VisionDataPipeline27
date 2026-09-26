import type { ClipDetail, FrameState, JobEvent } from "../api/types";
import { applyFrameEvent } from "./timeline";

/** One object's outcome on one frame of a propagation: the server's `{id, has_mask}`. */
export type ObjectFrame = { id: number; hasMask: boolean };
/** A propagation `frame` event, listing only the objects still active on that frame. Events are not in frame
 * order when objects' anchors diverge (the lagging group re-emits earlier frames), so each is applied on its own.
 * `done` is the job's progress (the furthest any object has got), which the page follows rather than `frame`. */
export type FrameEvent = { frame: number; objects: ObjectFrame[]; done?: number };

/** The `frame` event of a propagation job, or null for any other (or malformed) event. The server's top-level
 * `has_mask` is deliberately not read: a frame's clip-level state comes from all the lanes. */
export function parseFrameEvent(event: JobEvent): FrameEvent | null {
  if (event.type !== "frame" || typeof event.frame !== "number" || !Array.isArray(event.objects)) return null;
  const objects: ObjectFrame[] = [];
  for (const item of event.objects as unknown[]) {
    if (item === null || typeof item !== "object") continue;
    const { id, has_mask } = item as { id?: unknown; has_mask?: unknown };
    if (typeof id === "number") objects.push({ id, hasMask: has_mask === true });
  }
  return typeof event.done === "number" ? { frame: event.frame, objects, done: event.done } : { frame: event.frame, objects };
}

export const isLabeled = (s: FrameState | undefined): boolean => s === "masked" || s === "keyframe";

/** A lane with no masks: every frame empty except the clip's deleted ones. */
export function emptyLane(states: readonly FrameState[]): FrameState[] {
  return states.map((s) => (s === "deleted" ? "deleted" : "empty"));
}

/** One object's per-frame states. A "New object" row (`null`) or an id the detail does not know has no masks. */
export function laneOf(detail: ClipDetail, objectId: number | null): FrameState[] {
  const lane = objectId === null ? undefined : detail.object_states[String(objectId)];
  return lane ?? emptyLane(detail.states);
}

/** Frames an object has a mask on (propagated or keyframe). */
export function labeledCount(lane: readonly FrameState[]): number {
  let n = 0;
  for (const s of lane) if (isLabeled(s)) n++;
  return n;
}

/** The clip-level state of one frame from every object's lane, by the server's rule (deleted > keyframe > masked >
 * empty); `fallback` when the clip has no objects. */
export function aggregateAt(lanes: readonly (readonly FrameState[])[], frame: number, fallback: FrameState): FrameState {
  if (lanes.length === 0) return fallback;
  let best: FrameState = "empty";
  for (const lane of lanes) {
    const s = lane[frame];
    if (s === "deleted") return "deleted";
    if (s === "keyframe") best = "keyframe";
    else if (s === "masked" && best === "empty") best = "masked";
  }
  return best;
}

/**
 * Apply propagation frame events: each listed object's lane follows `applyFrameEvent` (keyframes and deleted frames
 * never change; an object not in the clip is ignored), then every frame whose lanes changed gets its clip-level
 * state recomputed from all lanes — an object that did not take part keeps a frame "masked". Returns `detail`
 * itself when nothing changed.
 */
export function patchObjectFrames(detail: ClipDetail, events: readonly FrameEvent[]): ClipDetail {
  let lanes = detail.object_states;
  const touched = new Set<number>();
  for (const e of events) {
    for (const o of e.objects) {
      const key = String(o.id);
      const lane = lanes[key];
      if (lane === undefined) continue;
      const next = applyFrameEvent(lane, e.frame, o.hasMask);
      if (next === lane) continue;
      if (lanes === detail.object_states) lanes = { ...lanes };
      lanes[key] = next;
      touched.add(e.frame);
    }
  }
  if (lanes === detail.object_states) return detail;
  const all = Object.values(lanes);
  let states = detail.states;
  for (const f of touched) {
    const s = aggregateAt(all, f, states[f]);
    if (s === states[f]) continue;
    if (states === detail.states) states = states.slice();
    states[f] = s;
  }
  return { ...detail, object_states: lanes, states };
}
