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
    render(<Timeline states={states} frame={0} onSeek={onSeek} />);
    const strip = screen.getByRole("slider");
    fireEvent.mouseDown(strip, { clientX: 65, button: 0, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(5);
    fireEvent.mouseMove(strip, { clientX: 95, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(8);
    fireEvent.mouseMove(strip, { clientX: 20, buttons: 0 });
    expect(onSeek).toHaveBeenCalledTimes(2);
  });
  it("exposes the position to assistive technology", () => {
    render(<Timeline states={states} frame={4} onSeek={() => {}} />);
    const strip = screen.getByRole("slider");
    expect(strip.getAttribute("aria-valuenow")).toBe("4");
    expect(strip.getAttribute("aria-valuemax")).toBe("9");
  });
});
