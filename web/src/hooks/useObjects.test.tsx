// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ClipDetail } from "../api/types";
import { useObjects } from "./useObjects";

/** objects as [id, class] pairs; classes 0 = cup, 1 = bowl */
const make = (objects: [number, number][]): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 1, created_at: "", deleted_frames: [],
    objects: objects.map(([id, class_id]) => ({ id, class_id, keyframes: [], propagated_ranges: [] })),
  },
  states: ["empty"],
  classes: [{ id: 0, name: "cup", color: "#e6194b" }, { id: 1, name: "bowl", color: "#3cb44b" }],
  object_states: Object.fromEntries(objects.map(([id]) => [String(id), ["empty"]])),
});
type P = { id: string; d: ClipDetail | null };
const setup = (d: ClipDetail | null) => renderHook(({ id, d: detail }: P) => useObjects(id, detail), { initialProps: { id: "c1", d } });

afterEach(cleanup);

describe("useObjects", () => {
  it("has no selection until the clip loads, then starts on the lowest object id", () => {
    const { result, rerender } = setup(null);
    expect(result.current.selection).toBeNull();
    rerender({ id: "c1", d: make([[2, 1], [1, 0]]) });
    expect(result.current.activeId).toBe(1);
    expect(result.current.active?.class_id).toBe(0);
    expect(result.current.classId).toBe(0);
  });
  it("selects, steps with wrap-around, and starts a new object of the active object's class", () => {
    const { result } = setup(make([[1, 0], [2, 1]]));
    act(() => result.current.select(2));
    expect([result.current.activeId, result.current.classId]).toEqual([2, 1]);
    act(() => result.current.step(1));
    expect(result.current.activeId).toBe(1);
    act(() => result.current.startNew());
    expect(result.current.selection).toEqual({ kind: "new", classId: 0 });
    expect([result.current.activeId, result.current.active]).toEqual([null, null]);
    act(() => result.current.setNewClass(1));
    expect(result.current.classId).toBe(1);
    act(() => result.current.step(-1));
    expect(result.current.activeId).toBe(2);
  });
  it("moves to the next object when the active one disappears, and to a new object when none is left", () => {
    const { result, rerender } = setup(make([[1, 0], [2, 0], [3, 0]]));
    act(() => result.current.select(2));
    rerender({ id: "c1", d: make([[1, 0], [3, 0]]) });
    expect(result.current.activeId).toBe(3);
    rerender({ id: "c1", d: make([]) });
    expect(result.current.selection).toEqual({ kind: "new", classId: 0 });
  });
  it("hides objects for this clip only, and forgets the selection on another clip", () => {
    const { result, rerender } = setup(make([[1, 0], [2, 0]]));
    act(() => result.current.toggleHidden(2));
    expect([...result.current.hidden]).toEqual([2]);
    act(() => result.current.toggleHidden(2));
    expect(result.current.hidden.size).toBe(0);
    act(() => result.current.toggleHidden(1));
    act(() => result.current.select(2));
    rerender({ id: "c2", d: make([[1, 0], [2, 0]]) });
    expect(result.current.hidden.size).toBe(0);
    expect(result.current.activeId).toBe(1);
  });
});
