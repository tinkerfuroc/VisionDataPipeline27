// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BrushMode } from "../lib/brush";
import type { FrameStore } from "../lib/frameCache";
import { fitView, toImage } from "../lib/geometry";
import { emptyDraft } from "../lib/promptState";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { FrameCanvas } from "./FrameCanvas";

// jsdom does no layout, so wrap.clientWidth/clientHeight are always 0: fitView(width, height, 0, 0) is exactly the
// view FrameCanvas computes on mount, with no extra stubbing needed.
const store = { image: async () => null, layer: async () => null, preview: async () => null } as unknown as FrameStore;

type BrushSpies = { size: number; layer: null; baseVersion: number; onStart: ReturnType<typeof vi.fn>; onExtend: ReturnType<typeof vi.fn>; onEnd: ReturnType<typeof vi.fn> };
const brushProp = (): BrushSpies => ({ size: 24, layer: null, baseVersion: 0, onStart: vi.fn(), onExtend: vi.fn(), onEnd: vi.fn() });

const renderCanvas = (width: number, height: number, brush: { size: number; layer: null; baseVersion: number; onStart: (mode: BrushMode, x: number, y: number) => void; onExtend: (x: number, y: number) => void; onEnd: () => void }) =>
  render(
    <FrameCanvas
      store={store}
      width={width}
      height={height}
      frameUrl="f"
      maskUrl={null}
      color="#22c55e"
      outline={{ rgb: [34, 197, 94], edge: 5 }}
      tool={null}
      draft={emptyDraft()}
      candidates={[]}
      onPoint={() => {}}
      onBox={() => {}}
      onPickCandidate={() => {}}
      brush={brush}
    />,
  );

beforeEach(() => {
  stubCanvasAndObservers();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("FrameCanvas brush pointer handling", () => {
  it("floors the pointer to an image pixel index when starting a brush stroke (F3)", () => {
    const brush = brushProp();
    const width = 100;
    const height = 100;
    renderCanvas(width, height, brush);
    const canvas = screen.getByLabelText("Frame");
    const [sx, sy] = [37, 52];
    fireEvent.mouseDown(canvas, { clientX: sx, clientY: sy, button: 0 });

    const view = fitView(width, height, 0, 0);
    const [ix, iy] = toImage(view, sx, sy);
    expect(brush.onStart).toHaveBeenCalledWith("paint", Math.floor(ix), Math.floor(iy));
  });

  it("ends a painting drag when a mousemove arrives with no button held (F6)", () => {
    const brush = brushProp();
    renderCanvas(100, 100, brush);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 10, clientY: 10, button: 0 });
    expect(brush.onStart).toHaveBeenCalledTimes(1);

    fireEvent.mouseMove(canvas, { clientX: 20, clientY: 20, buttons: 0 });
    expect(brush.onEnd).toHaveBeenCalledTimes(1);
    expect(brush.onExtend).not.toHaveBeenCalled();
  });
});
