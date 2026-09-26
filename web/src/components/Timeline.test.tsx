// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FrameState } from "../api/types";
import type { Lane } from "../lib/lanes";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Timeline } from "./Timeline";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);
const states = S("kmmeeedeee");
const lane = (id: number, label: string, st: FrameState[], hidden = false): Lane => ({ id, label, color: "#e6194b", states: st, hidden });
const two = [lane(1, "#1 connector", S("kmmeeedeee")), lane(2, "#2 cup", S("eekmmedeee"))];
// The canvas spans clientX 10..182: a 72 px name gutter, then 10 px per frame, so frame f is centred at 87 + 10f.
const X = (f: number) => 87 + 10 * f;
// Lanes start 6 px down: with #1 active, lane 1 spans y 6..36 and lane 2 y 38..54.
const noLanes = { states, lanes: [] as Lane[], activeId: null, frame: 0 };

/** a 2D context that records every call and every globalAlpha it is given */
function recordCanvas() {
  const calls: { name: string; args: unknown[] }[] = [];
  const alphas: number[] = [];
  const ctx = new Proxy(
    {},
    {
      get: (_t, name) => (...args: unknown[]) => {
        calls.push({ name: String(name), args });
      },
      set: (_t, name, value) => {
        if (name === "globalAlpha") alphas.push(value as number);
        return true;
      },
    },
  );
  HTMLCanvasElement.prototype.getContext = (() => ctx) as unknown as HTMLCanvasElement["getContext"];
  return { calls, alphas };
}

beforeEach(() => {
  stubCanvasAndObservers();
  HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ left: 10, top: 0, width: 172, height: 60 }) as DOMRect;
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Timeline", () => {
  it("seeks on click and while dragging with the left button held", () => {
    const onSeek = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), button: 0, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(5);
    fireEvent.mouseMove(strip, { clientX: X(8), buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(8);
    fireEvent.mouseMove(strip, { clientX: X(1), buttons: 0 });
    expect(onSeek).toHaveBeenCalledTimes(2);
  });
  it("ignores a press in the name gutter", () => {
    const onSeek = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} />);
    fireEvent.mouseDown(screen.getByLabelText("Timeline"), { clientX: 40, button: 0, buttons: 1 });
    expect(onSeek).not.toHaveBeenCalled();
  });
  it("marks a range with Shift-drag instead of seeking, in either direction", () => {
    const onSeek = vi.fn();
    const onRange = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} onRange={onRange} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), button: 0, buttons: 1, shiftKey: true });
    expect(onRange).toHaveBeenLastCalledWith([5, 5]);
    fireEvent.mouseMove(strip, { clientX: X(2), buttons: 1, shiftKey: true });
    expect(onRange).toHaveBeenLastCalledWith([2, 5]);
    fireEvent.mouseUp(strip);
    fireEvent.mouseMove(strip, { clientX: X(8), buttons: 1 }); // a plain drag afterwards seeks again
    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenLastCalledWith(8);
    expect(onRange).toHaveBeenCalledTimes(2);
  });
  it("scrubs normally after a Shift-drag that was released outside the strip", () => {
    const onSeek = vi.fn();
    const onRange = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} onRange={onRange} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), button: 0, buttons: 1, shiftKey: true });
    fireEvent.mouseLeave(strip); // the button is let go somewhere else: the strip never sees a mouseup
    fireEvent.mouseUp(window);
    onRange.mockClear();
    fireEvent.mouseDown(strip, { clientX: X(1), button: 0, buttons: 1 });
    fireEvent.mouseMove(strip, { clientX: X(8), buttons: 1 });
    expect(onRange).not.toHaveBeenCalled();
    expect(onSeek).toHaveBeenLastCalledWith(8);
  });
  it("shows the hovered frame, its state and its thumbnail, and hides them again on leave", () => {
    const { container } = render(<Timeline {...noLanes} thumb={{ url: (i) => `/thumb/${i}`, aspect: 4 / 3 }} onSeek={() => {}} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseMove(strip, { clientX: X(6), buttons: 0 });
    expect(screen.getByText("frame 6")).toBeTruthy();
    expect(screen.getByText("deleted")).toBeTruthy();
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/thumb/6");
    fireEvent.mouseLeave(strip);
    expect(screen.queryByText("frame 6")).toBeNull();
  });
  it("shows a single lane named No objects yet for a clip without objects", () => {
    render(<Timeline {...noLanes} onSeek={() => {}} />);
    expect(screen.getByText("No objects yet")).toBeTruthy();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
  it("is found by its label (it cannot be operated by keyboard — the global shortcuts cover seeking and selection)", () => {
    render(<Timeline {...noLanes} frame={4} onSeek={() => {}} />);
    expect(screen.getByLabelText("Timeline")).toBeTruthy();
  });
});

describe("Timeline lanes", () => {
  it("names each lane in the gutter; clicking a name, or inside another lane, selects that object", () => {
    const onSelect = vi.fn();
    const onSeek = vi.fn();
    render(<Timeline states={states} lanes={two} activeId={1} frame={0} onSeek={onSeek} onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: "#1 connector" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "#2 cup" }));
    expect(onSelect).toHaveBeenLastCalledWith(2);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), clientY: 42, button: 0, buttons: 1 });
    expect(onSelect).toHaveBeenCalledTimes(2);
    expect(onSelect).toHaveBeenLastCalledWith(2);
    expect(onSeek).toHaveBeenLastCalledWith(5); // and it seeks, as a click always did
    fireEvent.mouseDown(strip, { clientX: X(3), clientY: 20, button: 0, buttons: 1 });
    expect(onSelect).toHaveBeenCalledTimes(2); // the active lane: seek only
    expect(onSeek).toHaveBeenLastCalledWith(3);
    fireEvent.mouseDown(strip, { clientX: X(4), clientY: 37, button: 0, buttons: 1 });
    expect(onSelect).toHaveBeenCalledTimes(2); // the gap between two lanes selects nothing
  });
  it("lists every object's state on the hovered frame", () => {
    render(<Timeline states={states} lanes={two} activeId={1} frame={0} onSeek={() => {}} />);
    fireEvent.mouseMove(screen.getByLabelText("Timeline"), { clientX: X(2), clientY: 20, buttons: 0 });
    expect(screen.getByText("frame 2")).toBeTruthy();
    expect(screen.getAllByRole("listitem").map((li) => li.textContent)).toEqual(["#1 connectormasked", "#2 cupkeyframe"]);
  });
  it("hatches a deleted frame across every lane and draws hidden lanes at 40 % opacity", () => {
    Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 172 });
    try {
      const { calls, alphas } = recordCanvas();
      render(<Timeline states={states} lanes={[two[0], { ...two[1], hidden: true }]} activeId={1} frame={0} onSeek={() => {}} />);
      // frame 6 is deleted: its hatching is clipped to one rect per lane, at x = 72 + 6 × 10
      const hatchTops = calls.filter((c) => c.name === "rect" && c.args[0] === 132).map((c) => c.args[1] as number);
      expect([...new Set(hatchTops)].sort((a, b) => a - b)).toEqual([6, 38]);
      expect(alphas).toContain(0.4);
      expect(screen.getByRole("button", { name: "#2 cup" }).style.opacity).toBe("0.4");
      // no onSelect (the selection is locked): the names are disabled
      expect((screen.getByRole("button", { name: "#2 cup" }) as HTMLButtonElement).disabled).toBe(true);
    } finally {
      delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientWidth;
    }
  });
  it("grows to six lanes, then scrolls the lanes and keeps the active one in view", () => {
    const many = (n: number) => Array.from({ length: n }, (_, i) => lane(i + 1, `#${i + 1} part`, S("eeeeeeeeee")));
    const { rerender } = render(<Timeline states={states} lanes={many(3)} activeId={1} frame={0} onSeek={() => {}} />);
    const strip = screen.getByLabelText("Timeline");
    expect(strip.style.height).toBe("90px"); // 6 + (30 + 2 + 16 + 2 + 16) + 18
    rerender(<Timeline states={states} lanes={many(6)} activeId={1} frame={0} onSeek={() => {}} />);
    expect(strip.style.height).toBe("144px"); // 6 + 120 + 18
    rerender(<Timeline states={states} lanes={many(8)} activeId={8} frame={0} onSeek={() => {}} />);
    expect(strip.style.height).toBe("144px"); // capped at six lanes' worth
    // lane 8 sits at 7 × 18 = 126..156 in the stack: scrolled by 156 − 120 = 36 to show it
    expect(screen.getByRole("button", { name: "#8 part" }).style.top).toBe("90px");
    expect(screen.getByRole("button", { name: "#1 part" }).style.top).toBe("-36px");
  });
});
