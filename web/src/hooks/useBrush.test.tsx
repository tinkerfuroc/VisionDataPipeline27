// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import type { FrameStore } from "../lib/frameCache";
import type { Rgb } from "../lib/mask";
import { useBrush, type Brush } from "./useBrush";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { setMask: vi.fn() },
}));

vi.mock("../lib/brushCanvas", () => ({
  createLayer: () => ({ canvas: document.createElement("canvas"), ctx: null }),
  loadBase: async () => null,
  encodeLayer: () => ({ pngB64: "UE5H", any: true }),
}));

const store = { image: async () => null } as unknown as FrameStore;
const rgb: Rgb = [255, 0, 0];
const detail = {} as ClipDetail;

type Args = { active: boolean; classId: number | null; needsClass: boolean };
const setup = (over: Partial<Args> = {}, apply = vi.fn()) =>
  renderHook(
    (a: Args) =>
      useBrush({
        clipId: "c1", frame: 3, width: 4, height: 4, rgb, store, maskUrl: null,
        active: a.active, classId: a.classId, needsClass: a.needsClass, apply,
      }),
    { initialProps: { active: true, classId: null, needsClass: false, ...over } },
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
  it("blocks save while a stroke is in progress, then saves and clears the draft once it ends", async () => {
    vi.mocked(api.setMask).mockResolvedValue(detail);
    const { result } = setup();
    act(() => result.current.start("paint", 0, 0));
    expect(result.current.canSave).toBe(false);
    await act(async () => result.current.save());
    expect(api.setMask).not.toHaveBeenCalled();

    act(() => result.current.end());
    expect(result.current.canSave).toBe(true);
    await act(async () => result.current.save());
    expect(api.setMask).toHaveBeenCalledTimes(1);
    expect(result.current.draft.strokes).toHaveLength(0);
    expect(result.current.dirty).toBe(false);
  });

  it("requires a class before the first save, and sends the chosen class once one is set", async () => {
    vi.mocked(api.setMask).mockResolvedValue(detail);
    const { result, rerender } = setup({ needsClass: true, classId: null });
    stroke(result);
    expect(result.current.canSave).toBe(false);
    expect(result.current.saveBlockedReason).toBe("Choose a class before saving the first mask.");

    rerender({ active: true, needsClass: true, classId: 3 });
    expect(result.current.saveBlockedReason).toBe(null);
    expect(result.current.canSave).toBe(true);
    await act(async () => result.current.save());
    expect(api.setMask).toHaveBeenCalledWith("c1", 3, "UE5H", { classId: 3 });
  });

  it("keeps committed strokes when the tool becomes inactive, but drops an already-clean draft", async () => {
    const { result, rerender } = setup();
    stroke(result);
    expect(result.current.draft.strokes).toHaveLength(1);

    rerender({ active: false, classId: null, needsClass: false });
    expect(result.current.draft.strokes).toHaveLength(1);

    act(() => result.current.discard());
    expect(result.current.draft.strokes).toHaveLength(0);
    rerender({ active: false, classId: null, needsClass: false });
    expect(result.current.draft.strokes).toHaveLength(0);
  });
});
