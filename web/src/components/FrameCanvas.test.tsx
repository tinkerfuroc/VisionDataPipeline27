// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FrameStore, MaskPart } from "../lib/frameCache";
import { fitView, toImage } from "../lib/geometry";
import { emptyDraft } from "../lib/promptState";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { FrameCanvas } from "./FrameCanvas";

// jsdom does no layout, so wrap.clientWidth/clientHeight are always 0: fitView(width, height, 0, 0) is exactly the
// view FrameCanvas computes on mount, with no extra stubbing needed.
const store = { image: async () => null, composite: async () => null, preview: async () => null } as unknown as FrameStore;

type Props = ComponentProps<typeof FrameCanvas>;
const baseProps = (over: Partial<Props> = {}): Props => ({
  store, width: 100, height: 100, frameUrl: "f", masks: [], color: "#22c55e", outline: { rgb: [34, 197, 94], edge: 5 },
  tool: null, draft: emptyDraft(), candidates: [], onPoint: () => {}, onBox: () => {}, onPickCandidate: () => {},
  brush: null, ...over,
});
const brushProp = () => ({ size: 24, layer: null, baseVersion: 0, onStart: vi.fn(), onExtend: vi.fn(), onEnd: vi.fn() });

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
    render(<FrameCanvas {...baseProps({ brush })} />);
    const canvas = screen.getByLabelText("Frame");
    const [sx, sy] = [37, 52];
    fireEvent.mouseDown(canvas, { clientX: sx, clientY: sy, button: 0 });
    const [ix, iy] = toImage(fitView(100, 100, 0, 0), sx, sy);
    expect(brush.onStart).toHaveBeenCalledWith("paint", Math.floor(ix), Math.floor(iy));
  });

  it("ends a painting drag when a mousemove arrives with no button held (F6)", () => {
    const brush = brushProp();
    render(<FrameCanvas {...baseProps({ brush })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 10, clientY: 10, button: 0 });
    expect(brush.onStart).toHaveBeenCalledTimes(1);
    fireEvent.mouseMove(canvas, { clientX: 20, clientY: 20, buttons: 0 });
    expect(brush.onEnd).toHaveBeenCalledTimes(1);
    expect(brush.onExtend).not.toHaveBeenCalled();
  });
});

describe("FrameCanvas stored masks", () => {
  it("asks the store for one composite of the given masks, and again only when they change", () => {
    const composite = vi.fn(async () => null);
    const s = { image: async () => null, composite, preview: async () => null } as unknown as FrameStore;
    const parts: MaskPart[] = [
      { url: "/api/clips/c1/objects/1/masks/0?v=0.0", rgb: [1, 2, 3], outline: null },
      { url: "/api/clips/c1/objects/2/masks/0?v=0.0", rgb: [4, 5, 6], outline: { rgb: [9, 9, 9], edge: 3 } },
    ];
    const { rerender } = render(<FrameCanvas {...baseProps({ store: s, masks: parts })} />);
    expect(composite).toHaveBeenCalledWith(parts);
    rerender(<FrameCanvas {...baseProps({ store: s, masks: parts.map((p) => ({ ...p })) })} />); // equal content
    expect(composite).toHaveBeenCalledTimes(1);
    rerender(<FrameCanvas {...baseProps({ store: s, masks: [parts[0]] })} />);
    expect(composite).toHaveBeenLastCalledWith([parts[0]]);
  });
});
