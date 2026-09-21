// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FrameState } from "../api/types";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Timeline } from "./Timeline";

const states: FrameState[] = ["keyframe", "masked", "masked", "empty", "empty", "empty", "deleted", "empty", "empty", "empty"];

beforeEach(() => {
  stubCanvasAndObservers();
  HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ left: 10, top: 0, width: 100, height: 28 }) as DOMRect;
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Timeline", () => {
  it("seeks on click and while dragging with the left button held", () => {
    const onSeek = vi.fn();
    render(<Timeline states={states} frame={0} color="#e6194b" onSeek={onSeek} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: 65, button: 0, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(5);
    fireEvent.mouseMove(strip, { clientX: 95, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(8);
    fireEvent.mouseMove(strip, { clientX: 20, buttons: 0 });
    expect(onSeek).toHaveBeenCalledTimes(2);
  });
  it("marks a range with Shift-drag instead of seeking, in either direction", () => {
    const onSeek = vi.fn();
    const onRange = vi.fn();
    render(<Timeline states={states} frame={0} color="#e6194b" onSeek={onSeek} onRange={onRange} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: 65, button: 0, buttons: 1, shiftKey: true });
    expect(onRange).toHaveBeenLastCalledWith([5, 5]);
    fireEvent.mouseMove(strip, { clientX: 35, buttons: 1, shiftKey: true });
    expect(onRange).toHaveBeenLastCalledWith([2, 5]);
    fireEvent.mouseUp(strip);
    fireEvent.mouseMove(strip, { clientX: 95, buttons: 1 }); // a plain drag afterwards seeks again
    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenLastCalledWith(8);
    expect(onRange).toHaveBeenCalledTimes(2);
  });
  it("shows the hovered frame, its state and its thumbnail, and hides them again on leave", () => {
    const { container } = render(
      <Timeline states={states} frame={0} color="#e6194b" thumb={{ url: (i) => `/thumb/${i}`, aspect: 4 / 3 }} onSeek={() => {}} />,
    );
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseMove(strip, { clientX: 75, buttons: 0 });
    expect(screen.getByText("frame 6")).toBeTruthy();
    expect(screen.getByText("deleted")).toBeTruthy();
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/thumb/6");
    fireEvent.mouseLeave(strip);
    expect(screen.queryByText("frame 6")).toBeNull();
  });
  it("is found by its label (it cannot be operated by keyboard — the global arrow-key shortcuts cover seeking)", () => {
    render(<Timeline states={states} frame={4} color="#e6194b" onSeek={() => {}} />);
    expect(screen.getByLabelText("Timeline")).toBeTruthy();
  });
});
