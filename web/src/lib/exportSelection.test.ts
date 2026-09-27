import { describe, expect, it } from "vitest";
import type { ClipSummary } from "../api/types";
import { classObjectCounts, defaultValClips, exportProblem, splitModeFor } from "./exportSelection";

const clip = (id: string, created_at: string): ClipSummary =>
  ({ id, created_at, project_id: "p", source: { type: "video", name: `${id}.mp4` }, fps: 10, stride: 1, width: 640, height: 480,
     frame_count: 40, deleted_frames: [], original_names: null, objects: [] }) as unknown as ClipSummary;
const clips = [clip("a", "2026-01-01"), clip("b", "2026-01-03"), clip("c", "2026-01-02")];

describe("export selection", () => {
  it("uses the per-clip split for two or more clips and the contiguous split for one", () => {
    expect(splitModeFor(1)).toBe("contiguous");
    expect(splitModeFor(2)).toBe("per_clip");
  });
  it("pre-selects the most recently imported clips for validation, never all of them", () => {
    expect(defaultValClips(clips, ["a", "b", "c"], 0.2)).toEqual(["b"]);
    expect(defaultValClips(clips, ["a", "b", "c"], 0.5)).toEqual(["b", "c"]);
    expect(defaultValClips(clips, ["a", "c"], 0.9)).toEqual(["c"]);
    expect(defaultValClips(clips, ["a", "b"], 0)).toEqual([]);
    expect(defaultValClips(clips, ["a"], 0.2)).toEqual([]);
  });
  it("says what stops an export, or null", () => {
    const ok = { selected: ["a", "b"], formats: ["coco"], classes: [0], classCount: 1, valClips: ["b"], valFraction: 0.2, stride: 1 };
    expect(exportProblem(ok)).toBeNull();
    expect(exportProblem({ ...ok, selected: [] })).toBe("Choose at least one clip.");
    expect(exportProblem({ ...ok, formats: [] })).toBe("Choose at least one format.");
    expect(exportProblem({ ...ok, valClips: ["a", "b"] })).toBe("Leave at least one clip for training.");
    expect(exportProblem({ ...ok, valFraction: 1 })).toBe("The validation share must be below 100%.");
    expect(exportProblem({ ...ok, stride: 0 })).toBe("Keep every Nth frame needs a whole number of 1 or more.");
    expect(exportProblem({ ...ok, classes: [] })).toBe("Choose at least one class.");
    expect(exportProblem({ ...ok, classes: [], classCount: 0 })).toBeNull(); // a project without classes can still export backgrounds
  });

  it("counts each class's objects over the included clips only", () => {
    const withObjects = (c: ClipSummary, classIds: number[]) =>
      ({ ...c, objects: classIds.map((class_id, i) => ({ id: i + 1, class_id, keyframes: [], propagated_ranges: [] })) }) as ClipSummary;
    const list = [withObjects(clips[0], [1, 1, 0]), withObjects(clips[1], [2]), withObjects(clips[2], [2])];
    expect(Object.fromEntries(classObjectCounts(list, ["a", "b"]))).toEqual({ 0: 1, 1: 2, 2: 1 });
    expect(classObjectCounts(list, []).size).toBe(0);
  });
});
