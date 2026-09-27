// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { MaskResult } from "../api/types";
import type { Selection } from "../lib/activeObject";
import { loadBase } from "../lib/brushCanvas";
import type { FrameStore } from "../lib/frameCache";
import type { Rgb } from "../lib/mask";
import { useBrush, type Brush } from "./useBrush";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { setMask: vi.fn() },
}));

vi.mock("../lib/brushCanvas", () => ({
  createLayer: () => ({ canvas: document.createElement("canvas"), ctx: null }),
  loadBase: vi.fn(async () => null),
  encodeLayer: () => ({ pngB64: "UE5H", any: true }),
}));

const store = { image: async () => null } as unknown as FrameStore;
const rgb: Rgb = [255, 0, 0];
const saved = { object_id: 7 } as MaskResult;
const OBJECT_2: Selection = { kind: "object", id: 2 };

type Args = { active: boolean; selection: Selection | null };
const setup = (over: Partial<Args> = {}, onCreated = vi.fn()) =>
  renderHook(
    (a: Args) =>
      useBrush({
        clipId: "c1", frame: 3, width: 4, height: 4, rgb, store, maskUrl: null,
        active: a.active, selection: a.selection, onCreated, apply: vi.fn(),
      }),
    { initialProps: { active: true, selection: OBJECT_2, ...over } },
  );

const stroke = (result: { current: Brush }) => {
  act(() => result.current.start("paint", 0, 0));
  act(() => result.current.extend(1, 1));
  act(() => result.current.end());
};

beforeEach(() => {
  vi.resetAllMocks();
});
afterEach(cleanup);

describe("useBrush", () => {
  it("blocks save while a stroke is in progress, then saves to the active object and clears the draft", async () => {
    vi.mocked(api.setMask).mockResolvedValue(saved);
    const onCreated = vi.fn();
    const { result } = setup({}, onCreated);
    act(() => result.current.start("paint", 0, 0));
    expect(result.current.canSave).toBe(false);
    await act(async () => result.current.save());
    expect(api.setMask).not.toHaveBeenCalled();

    act(() => result.current.end());
    expect(result.current.canSave).toBe(true);
    await act(async () => result.current.save());
    expect(api.setMask).toHaveBeenCalledWith("c1", 3, "UE5H", { objectId: 2 });
    expect(onCreated).not.toHaveBeenCalled();
    expect(result.current.draft.strokes).toHaveLength(0);
    expect(result.current.dirty).toBe(false);
  });

  it("needs a class for a new object, then creates it and reports its id", async () => {
    vi.mocked(api.setMask).mockResolvedValue(saved);
    const onCreated = vi.fn();
    const { result, rerender } = setup({ selection: { kind: "new", classId: null } }, onCreated);
    stroke(result);
    expect(result.current.canSave).toBe(false);
    expect(result.current.saveBlockedReason).toBe("Choose a class for the new object before saving.");

    rerender({ active: true, selection: { kind: "new", classId: 3 } });
    expect(result.current.saveBlockedReason).toBeNull();
    expect(result.current.canSave).toBe(true);
    await act(async () => result.current.save());
    expect(api.setMask).toHaveBeenCalledWith("c1", 3, "UE5H", { objectId: null, classId: 3 });
    expect(onCreated).toHaveBeenCalledWith(7);
  });

  it("keeps committed strokes when the tool becomes inactive, but drops an already-clean draft", () => {
    const { result, rerender } = setup();
    stroke(result);
    expect(result.current.draft.strokes).toHaveLength(1);
    rerender({ active: false, selection: OBJECT_2 });
    expect(result.current.draft.strokes).toHaveLength(1);
    act(() => result.current.discard());
    expect(result.current.draft.strokes).toHaveLength(0);
    rerender({ active: false, selection: OBJECT_2 });
    expect(result.current.draft.strokes).toHaveLength(0);
  });

  it("notifies the canvas once the base mask finishes loading, and start() refuses to paint until then", async () => {
    vi.mocked(loadBase).mockImplementation(async () => {
      await Promise.resolve();
      return { fake: true } as unknown as ImageData;
    });
    const { result } = renderHook(() =>
      useBrush({
        clipId: "c1", frame: 3, width: 4, height: 4, rgb, store, maskUrl: "http://x/mask.png",
        active: true, selection: OBJECT_2, onCreated: vi.fn(), apply: vi.fn(),
      }),
    );
    expect(result.current.loading).toBe(true);
    act(() => result.current.start("paint", 0, 0));
    expect(result.current.dirty).toBe(false); // start() was refused: no stroke began

    const before = result.current.baseVersion;
    await act(async () => {
      await new Promise((r) => setTimeout(r)); // flush the microtask queue past loadBase's await tick
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.baseVersion).toBe(before + 1);
  });

  it("ignores undo while a stroke is in progress, so the in-progress stroke is not lost", () => {
    const { result } = setup();
    act(() => result.current.start("paint", 0, 0));
    act(() => result.current.undo()); // must be a no-op: a stroke is in progress
    act(() => result.current.end());
    expect(result.current.draft.strokes).toHaveLength(1);
    act(() => result.current.undo()); // now that the stroke is committed, undo removes it
    expect(result.current.draft.strokes).toHaveLength(0);
  });
});
