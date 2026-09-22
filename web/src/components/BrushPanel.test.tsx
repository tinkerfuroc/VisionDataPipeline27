// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Brush } from "../hooks/useBrush";
import { emptyBrush } from "../lib/brush";
import { BrushPanel } from "./BrushPanel";

const brush = (over: Partial<Brush> = {}): Brush => ({
  draft: emptyBrush(), dirty: false, layer: null, busy: false, error: null, canSave: false, saveBlockedReason: null,
  setSize: vi.fn(), grow: vi.fn(), start: vi.fn(), extend: vi.fn(), end: vi.fn(), undo: vi.fn(), discard: vi.fn(),
  save: vi.fn(async () => {}), ...over,
});
afterEach(cleanup);

describe("BrushPanel", () => {
  it("shows the size, the hint and disabled buttons while there is nothing to save", () => {
    render(<BrushPanel brush={brush()} frame={12} disabledReason={null} />);
    expect((screen.getByLabelText("Brush size") as HTMLInputElement).value).toBe("24");
    expect(screen.getByText("24 px")).toBeTruthy();
    expect(screen.getByText(/Left-drag to paint, right-drag to erase/)).toBeTruthy();
    expect(screen.getByText("No strokes yet on frame 12.")).toBeTruthy();
    for (const name of [/Save/, /Discard/, /Undo stroke/]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
  });
  it("changes the size from the slider and wires Save, Discard and Undo when dirty", () => {
    const b = brush({ dirty: true, canSave: true, draft: { size: 40, strokes: [{ mode: "paint", size: 40, points: [[1, 1]] }, { mode: "erase", size: 40, points: [[2, 2]] }, { mode: "paint", size: 40, points: [[3, 3]] }] } });
    render(<BrushPanel brush={b} frame={3} disabledReason={null} />);
    fireEvent.change(screen.getByLabelText("Brush size"), { target: { value: "55" } });
    expect(b.setSize).toHaveBeenCalledWith(55);
    expect(screen.getByText("3 strokes on frame 3, not saved.")).toBeTruthy();
    expect(screen.getByText(/Save or discard the brush strokes on frame 3 first/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Save/ }));
    expect(b.save).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Discard/ }));
    expect(b.discard).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Undo stroke/ }));
    expect(b.undo).toHaveBeenCalled();
  });
  it("says why the brush is unavailable, shows saving and the last error", () => {
    const { rerender } = render(<BrushPanel brush={brush()} frame={0} disabledReason="A propagation is running. Stop it to paint." />);
    expect(screen.getByText("A propagation is running. Stop it to paint.")).toBeTruthy();
    expect((screen.getByLabelText("Brush size") as HTMLInputElement).disabled).toBe(true);
    rerender(<BrushPanel brush={brush({ dirty: true, busy: true, error: "mask is 640×480, the clip is 1920×1080" })} frame={0} disabledReason={null} />);
    expect(screen.getByText("Saving…")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toBe("mask is 640×480, the clip is 1920×1080");
    expect((screen.getByRole("button", { name: /Save/ }) as HTMLButtonElement).disabled).toBe(true); // while saving
  });
  it("shows a note and disables Save when a class is needed for the first save, keeping Discard and Undo enabled", () => {
    const b = brush({ dirty: true, canSave: false, saveBlockedReason: "Choose a class before saving the first mask." });
    render(<BrushPanel brush={b} frame={5} disabledReason={null} />);
    expect(screen.getByText("Choose a class before saving the first mask.")).toBeTruthy();
    expect((screen.getByRole("button", { name: /Save/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: /Discard/ }) as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByRole("button", { name: /Undo stroke/ }) as HTMLButtonElement).disabled).toBe(false);
  });
});
