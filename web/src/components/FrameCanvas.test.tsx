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

describe("FrameCanvas regions", () => {
  // A zero-size wrap makes the fitted scale 0.01, so the 6 px handle tolerance would swallow the whole image; give
  // the wrap a 116 px box instead, which fits the 100 px image at scale 1 (margin 8).
  const view = fitView(100, 100, 116, 116);
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 116 });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => 116 });
  });
  afterEach(() => {
    // jsdom defines these on Element.prototype; dropping our overrides restores them
    delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientWidth;
    delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientHeight;
  });
  /** image point -> client point under the fitted view (the canvas rect is at 0,0 in jsdom) */
  const at = (x: number, y: number) => ({ clientX: x * view.scale + view.tx, clientY: y * view.scale + view.ty });
  type R = [number, number, number, number];

  it("a drag on empty space draws a new block", () => {
    const onChange = vi.fn();
    render(<FrameCanvas {...baseProps({ region: null, regionEdit: { draw: "block", onChange } })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { ...at(10, 10), button: 0 });
    fireEvent.mouseMove(canvas, { ...at(30, 40), buttons: 1 });
    fireEvent.mouseUp(canvas, at(30, 40));
    expect(onChange).toHaveBeenCalledWith({ crop: null, blocks: [[10, 10, 30, 40]] });
  });

  it("in crop mode a drag sets the crop", () => {
    const onChange = vi.fn();
    const region = { start: 0, end: 5, crop: null, blocks: [[50, 50, 60, 60]] as R[] };
    render(<FrameCanvas {...baseProps({ region, regionEdit: { draw: "crop", onChange } })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { ...at(5, 5), button: 0 });
    fireEvent.mouseMove(canvas, { ...at(95, 90), buttons: 1 });
    fireEvent.mouseUp(canvas, at(95, 90));
    expect(onChange).toHaveBeenCalledWith({ crop: [5, 5, 95, 90], blocks: [[50, 50, 60, 60]] });
  });

  it("dragging inside a block moves it, and its × deletes it", () => {
    const onChange = vi.fn();
    const region = { start: 0, end: 5, crop: null, blocks: [[20, 20, 40, 40]] as R[] };
    render(<FrameCanvas {...baseProps({ region, regionEdit: { draw: "block", onChange } })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { ...at(30, 30), button: 0 });
    fireEvent.mouseMove(canvas, { ...at(40, 35), buttons: 1 });
    fireEvent.mouseUp(canvas, at(40, 35));
    expect(onChange).toHaveBeenLastCalledWith({ crop: null, blocks: [[30, 25, 50, 45]] });
    // the region prop is unchanged, so the NE corner is still (40, 20); the × sits outside it: 1.5 handle radii (6 screen px) up and to the right
    const tol = 6 / view.scale;
    fireEvent.mouseDown(canvas, { ...at(40 + tol * 1.5, 20 - tol * 1.5), button: 0 });
    fireEvent.mouseUp(canvas, at(40 + tol * 1.5, 20 - tol * 1.5));
    expect(onChange).toHaveBeenLastCalledWith({ crop: null, blocks: [] });
  });

  it("the NE corner of a block resizes it instead of deleting it", () => {
    const onChange = vi.fn();
    const region = { start: 0, end: 5, crop: null, blocks: [[20, 20, 40, 40]] as R[] };
    render(<FrameCanvas {...baseProps({ region, regionEdit: { draw: "block", onChange } })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { ...at(40, 20), button: 0 });
    fireEvent.mouseMove(canvas, { ...at(50, 15), buttons: 1 });
    fireEvent.mouseUp(canvas, at(50, 15));
    expect(onChange).toHaveBeenCalledWith({ crop: null, blocks: [[20, 15, 50, 40]] });
  });

  it("a drag keeps the region and callback it started with when the frame changes mid-drag", () => {
    const original = vi.fn();
    const replacement = vi.fn();
    const region = { start: 0, end: 5, crop: null, blocks: [[20, 20, 40, 40]] as R[] };
    const other = { start: 6, end: 9, crop: [10, 10, 90, 90] as R, blocks: [] as R[] };
    const { rerender } = render(<FrameCanvas {...baseProps({ region, regionEdit: { draw: "block", onChange: original } })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { ...at(30, 30), button: 0 });
    rerender(<FrameCanvas {...baseProps({ region: other, regionEdit: { draw: "block", onChange: replacement } })} />);
    fireEvent.mouseMove(canvas, { ...at(40, 35), buttons: 1 });
    fireEvent.mouseUp(canvas, at(40, 35));
    expect(original).toHaveBeenCalledWith({ crop: null, blocks: [[30, 25, 50, 45]] });
    expect(replacement).not.toHaveBeenCalled();
  });

  it("a positive click in an excluded area calls onExcluded, a negative one still prompts", () => {
    const onPoint = vi.fn();
    const onExcluded = vi.fn();
    const region = { start: 0, end: 5, crop: null, blocks: [[20, 20, 40, 40]] as R[] };
    render(<FrameCanvas {...baseProps({ tool: "clicks", region, onPoint, onExcluded })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { ...at(30, 30), button: 0 });
    fireEvent.mouseUp(canvas, at(30, 30));
    expect(onExcluded).toHaveBeenCalledTimes(1);
    expect(onPoint).not.toHaveBeenCalled();
    fireEvent.mouseDown(canvas, { ...at(30, 30), button: 2 });
    fireEvent.mouseUp(canvas, at(30, 30));
    expect(onPoint).toHaveBeenCalledWith(30, 30, 0);
  });
});
