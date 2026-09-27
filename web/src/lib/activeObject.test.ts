import { describe, expect, it } from "vitest";
import type { ClipDetail } from "../api/types";
import { cycle, initialSelection, reconcile, writeTarget } from "./activeObject";

const CLASSES = [{ id: 0, name: "cup", color: "#e6194b" }, { id: 1, name: "bowl", color: "#3cb44b" }];
const make = (ids: number[], classes = CLASSES): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 1, created_at: "", deleted_frames: [],
    objects: ids.map((id) => ({ id, class_id: 0, keyframes: [], propagated_ranges: [] })),
  },
  states: ["empty"],
  classes,
  object_states: Object.fromEntries(ids.map((id) => [String(id), ["empty"]])),
});

describe("initialSelection", () => {
  it("starts on the lowest object id, or on a new object of the first class", () => {
    expect(initialSelection(make([3, 1]))).toEqual({ kind: "object", id: 1 });
    expect(initialSelection(make([]))).toEqual({ kind: "new", classId: 0 });
    expect(initialSelection(make([], []))).toEqual({ kind: "new", classId: null });
  });
});

describe("reconcile", () => {
  it("keeps a selection that is still valid", () => {
    const sel = { kind: "object" as const, id: 2 };
    expect(reconcile(sel, make([1, 2]))).toBe(sel);
    const fresh = { kind: "new" as const, classId: 1 };
    expect(reconcile(fresh, make([1]))).toBe(fresh);
  });
  it("moves off a deleted object to the next id, else the previous one, else a new object", () => {
    expect(reconcile({ kind: "object", id: 2 }, make([1, 3]))).toEqual({ kind: "object", id: 3 });
    expect(reconcile({ kind: "object", id: 3 }, make([1, 2]))).toEqual({ kind: "object", id: 2 });
    expect(reconcile({ kind: "object", id: 1 }, make([]))).toEqual({ kind: "new", classId: 0 });
  });
  it("gives a new object the first class when its class is unknown", () => {
    expect(reconcile({ kind: "new", classId: 7 }, make([]))).toEqual({ kind: "new", classId: 0 });
    expect(reconcile({ kind: "new", classId: null }, make([]))).toEqual({ kind: "new", classId: 0 });
  });
});

describe("cycle", () => {
  const d = make([1, 2, 3]);
  it("steps through the objects in id order, wrapping around", () => {
    expect(cycle({ kind: "object", id: 3 }, d, 1)).toEqual({ kind: "object", id: 1 });
    expect(cycle({ kind: "object", id: 1 }, d, -1)).toEqual({ kind: "object", id: 3 });
    expect(cycle({ kind: "object", id: 1 }, d, 1)).toEqual({ kind: "object", id: 2 });
  });
  it("goes from a new object to the first or the last, and stays put without objects", () => {
    expect(cycle({ kind: "new", classId: 0 }, d, 1)).toEqual({ kind: "object", id: 1 });
    expect(cycle({ kind: "new", classId: 0 }, d, -1)).toEqual({ kind: "object", id: 3 });
    const none = { kind: "new" as const, classId: 0 };
    expect(cycle(none, make([]), 1)).toBe(none);
  });
});

describe("writeTarget", () => {
  it("writes to the active object, or creates the new one with its class", () => {
    expect(writeTarget({ kind: "object", id: 2 })).toEqual({ objectId: 2 });
    expect(writeTarget({ kind: "new", classId: 1 })).toEqual({ objectId: null, classId: 1 });
    expect(writeTarget({ kind: "new", classId: null })).toBeNull();
    expect(writeTarget(null)).toBeNull();
  });
});
