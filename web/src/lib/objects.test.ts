import { describe, expect, it } from "vitest";
import type { ClipDetail, FrameState, JobEvent } from "../api/types";
import { aggregateAt, emptyLane, labeledCount, laneOf, parseFrameEvent, patchObjectFrames } from "./objects";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);
const obj = (id: number, class_id = 0) => ({ id, class_id, keyframes: [], propagated_ranges: [] });

/** #1 cup: keyframe on 0; #2 bowl: masked on 1; frame 3 deleted */
const detail = (): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 4, created_at: "", deleted_frames: [3], objects: [obj(1), obj(2, 1)],
  },
  states: S("kmed"),
  classes: [{ id: 0, name: "cup", color: "#ff0000" }, { id: 1, name: "bowl", color: "#00ff00" }],
  object_states: { "1": S("keed"), "2": S("emed") },
});

describe("parseFrameEvent", () => {
  it("reads the per-object outcomes of a frame event", () => {
    const e: JobEvent = {
      seq: 3, type: "frame", frame: 7, objects: [{ id: 1, has_mask: true }, { id: 2, has_mask: false }], done: 1, total: 9,
    };
    expect(parseFrameEvent(e)).toEqual({ frame: 7, objects: [{ id: 1, hasMask: true }, { id: 2, hasMask: false }], done: 1 });
  });
  it("ignores other events and malformed entries", () => {
    expect(parseFrameEvent({ seq: 0, type: "progress", done: 0, total: 9 })).toBeNull();
    expect(parseFrameEvent({ seq: 1, type: "frame", frame: 2, has_mask: true })).toBeNull(); // no objects list
    expect(parseFrameEvent({ seq: 1, type: "frame", frame: 2, objects: [null, { has_mask: true }, { id: 4 }] })).toEqual({
      frame: 2, objects: [{ id: 4, hasMask: false }],
    });
  });
});

describe("lanes", () => {
  it("gives each object's lane, and an empty lane keeping the deleted frames for a new object or an unknown id", () => {
    const d = detail();
    expect(laneOf(d, 2)).toBe(d.object_states["2"]);
    expect(laneOf(d, null)).toEqual(S("eeed"));
    expect(laneOf(d, 9)).toEqual(S("eeed"));
    expect(emptyLane(S("kmdm"))).toEqual(S("eede"));
  });
  it("counts an object's labelled frames", () => expect(labeledCount(S("kmmed"))).toBe(3));
  it("aggregates one frame over the lanes: deleted, then keyframe, then masked", () => {
    expect(aggregateAt([S("m"), S("k")], 0, "empty")).toBe("keyframe");
    expect(aggregateAt([S("e"), S("m")], 0, "empty")).toBe("masked");
    expect(aggregateAt([S("e"), S("e")], 0, "masked")).toBe("empty");
    expect(aggregateAt([S("d"), S("d")], 0, "empty")).toBe("deleted");
    expect(aggregateAt([], 0, "deleted")).toBe("deleted");
  });
});

describe("patchObjectFrames", () => {
  it("updates each listed object's lane and recomputes the aggregate for that frame", () => {
    const d = detail();
    const next = patchObjectFrames(d, [{ frame: 2, objects: [{ id: 1, hasMask: true }] }]);
    expect(next.object_states["1"]).toEqual(S("kemd"));
    expect(next.object_states["2"]).toBe(d.object_states["2"]); // an untouched lane keeps its identity
    expect(next.states).toEqual(S("kmmd"));
    expect(d.states).toEqual(S("kmed")); // the input is not mutated
  });
  it("keeps a frame masked while an object that did not take part still has a mask there", () => {
    const withBoth = patchObjectFrames(detail(), [{ frame: 1, objects: [{ id: 1, hasMask: true }] }]);
    const next = patchObjectFrames(withBoth, [{ frame: 1, objects: [{ id: 1, hasMask: false }] }]);
    expect(next.object_states["1"][1]).toBe("empty");
    expect(next.states[1]).toBe("masked"); // #2 still has its mask on frame 1
    const cleared = patchObjectFrames(detail(), [{ frame: 1, objects: [{ id: 2, hasMask: false }] }]);
    expect(cleared.states[1]).toBe("empty");
  });
  it("applies events out of frame order, as a lagging group re-emits earlier frames", () => {
    const next = patchObjectFrames(detail(), [
      { frame: 2, objects: [{ id: 1, hasMask: true }] },
      { frame: 1, objects: [{ id: 1, hasMask: true }] },
      { frame: 2, objects: [{ id: 2, hasMask: true }] },
    ]);
    expect(next.object_states["1"]).toEqual(S("kmmd"));
    expect(next.object_states["2"]).toEqual(S("emmd"));
    expect(next.states).toEqual(S("kmmd"));
  });
  it("never changes keyframes or deleted frames, ignores unknown objects, and returns the same detail when nothing changed", () => {
    const d = detail();
    expect(
      patchObjectFrames(d, [
        { frame: 0, objects: [{ id: 1, hasMask: false }] },
        { frame: 3, objects: [{ id: 2, hasMask: true }] },
        { frame: 2, objects: [{ id: 9, hasMask: true }] },
        { frame: 1, objects: [{ id: 2, hasMask: true }] },
      ]),
    ).toBe(d);
  });
});
