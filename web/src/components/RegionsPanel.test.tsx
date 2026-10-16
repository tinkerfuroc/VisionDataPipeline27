// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RegionsPanel } from "./RegionsPanel";

type Props = ComponentProps<typeof RegionsPanel>;
const props = (over: Partial<Props> = {}): Props => ({
  regions: [], frame: 5, frameCount: 20, width: 100, height: 100, draw: "block", dirty: false, saving: false,
  error: null, onDraw: vi.fn(), onChange: vi.fn(), onSave: vi.fn(), onDiscard: vi.fn(), ...over,
});
afterEach(cleanup);

describe("RegionsPanel", () => {
  it("offers to add a region when the frame has none", () => {
    const p = props();
    render(<RegionsPanel {...p} />);
    expect(screen.getByText("No region on this frame")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Add region from here" }));
    expect(p.onChange).toHaveBeenCalledWith([{ start: 5, end: 19, crop: null, blocks: [] }]);
  });

  it("shows the range, sets its ends from the current frame, splits, clears and removes", () => {
    const regions = [{ start: 0, end: 9, crop: null, blocks: [[1, 1, 5, 5]] as [number, number, number, number][] }];
    const p = props({ regions, frame: 4 });
    render(<RegionsPanel {...p} />);
    expect((screen.getByLabelText("Start frame") as HTMLInputElement).value).toBe("0");
    fireEvent.click(screen.getByRole("button", { name: "Set end to this frame" }));
    expect(p.onChange).toHaveBeenLastCalledWith([{ ...regions[0], end: 4 }]);
    fireEvent.click(screen.getByRole("button", { name: "New range from here" }));
    expect((p.onChange as ReturnType<typeof vi.fn>).mock.lastCall![0].map((r: { start: number; end: number }) => [r.start, r.end])).toEqual([[0, 3], [4, 9]]);
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(p.onChange).toHaveBeenLastCalledWith([{ ...regions[0], crop: null, blocks: [] }]);
    fireEvent.click(screen.getByRole("button", { name: "Remove range" }));
    expect(p.onChange).toHaveBeenLastCalledWith([]);
  });

  it("switches what a drag draws", () => {
    const p = props({ regions: [{ start: 0, end: 9, crop: null, blocks: [] }] });
    render(<RegionsPanel {...p} />);
    fireEvent.click(screen.getByRole("radio", { name: "Crop" }));
    expect(p.onDraw).toHaveBeenCalledWith("crop");
  });

  it("saves only valid, changed regions and shows problems and server errors", () => {
    const bad = [{ start: 0, end: 9, crop: [0, 0, 10, 10] as [number, number, number, number], blocks: [] }];
    const { rerender } = render(<RegionsPanel {...props({ regions: bad, dirty: true })} />);
    expect(screen.getByText(/at least 32/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Save regions" }) as HTMLButtonElement).disabled).toBe(true);
    const p = props({ regions: [{ start: 0, end: 9, crop: null, blocks: [] }], dirty: true, error: "region 0–9 overlaps" });
    rerender(<RegionsPanel {...p} />);
    expect(screen.getByRole("alert").textContent).toContain("overlaps");
    fireEvent.click(screen.getByRole("button", { name: "Save regions" }));
    expect(p.onSave).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(p.onDiscard).toHaveBeenCalled();
  });

  it("commits a typed frame number once, on blur or Enter, not per keystroke", () => {
    const regions = [{ start: 40, end: 120, crop: null, blocks: [] }];
    const p = props({ regions, frame: 50, frameCount: 200 });
    render(<RegionsPanel {...p} />);
    const end = screen.getByLabelText("End frame") as HTMLInputElement;
    fireEvent.change(end, { target: { value: "1" } });
    fireEvent.change(end, { target: { value: "13" } });
    fireEvent.change(end, { target: { value: "130" } });
    expect(p.onChange).not.toHaveBeenCalled();
    fireEvent.blur(end);
    expect(p.onChange).toHaveBeenCalledTimes(1);
    expect(p.onChange).toHaveBeenCalledWith([{ ...regions[0], end: 130 }]);
    fireEvent.change(end, { target: { value: "140" } });
    fireEvent.keyDown(end, { key: "Enter" });
    expect(p.onChange).toHaveBeenLastCalledWith([{ ...regions[0], end: 140 }]);
  });

  it("an emptied frame field commits nothing and snaps back; Escape reverts", () => {
    const regions = [{ start: 40, end: 120, crop: null, blocks: [] }];
    const p = props({ regions, frame: 50, frameCount: 200 });
    render(<RegionsPanel {...p} />);
    const start = screen.getByLabelText("Start frame") as HTMLInputElement;
    fireEvent.change(start, { target: { value: "" } });
    fireEvent.blur(start);
    expect(p.onChange).not.toHaveBeenCalled();
    expect(start.value).toBe("40");
    fireEvent.change(start, { target: { value: "45" } });
    fireEvent.keyDown(start, { key: "Escape" });
    fireEvent.blur(start);
    expect(p.onChange).not.toHaveBeenCalled();
    expect(start.value).toBe("40");
  });

  it("Clear on a range with no crop and no blocks does nothing", () => {
    const p = props({ regions: [{ start: 0, end: 9, crop: null, blocks: [] }], frame: 4 });
    render(<RegionsPanel {...p} />);
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(p.onChange).not.toHaveBeenCalled();
  });

  const two = () => ({
    start: 0, end: 9, crop: [0, 0, 80, 60] as [number, number, number, number],
    blocks: [[1, 2, 11, 22], [30, 40, 60, 50]] as [number, number, number, number][],
  });

  it("lists blocked boxes and the crop with their sizes", () => {
    render(<RegionsPanel {...props({ regions: [two()], frame: 4 })} />);
    expect(screen.getByText("1. 10×20 at (1, 2)")).toBeTruthy();
    expect(screen.getByText("2. 30×10 at (30, 40)")).toBeTruthy();
    expect(screen.getByText("Crop 80×60")).toBeTruthy();
    expect(screen.queryByText("No blocked boxes")).toBeNull();
  });

  it("says so when there are no blocked boxes", () => {
    render(<RegionsPanel {...props({ regions: [{ start: 0, end: 9, crop: null, blocks: [] }], frame: 4 })} />);
    expect(screen.getByText("No blocked boxes")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Remove crop" })).toBeNull();
  });

  it("Delete box 2 removes exactly the second block", () => {
    const r = two();
    const p = props({ regions: [r], frame: 4 });
    render(<RegionsPanel {...p} />);
    fireEvent.click(screen.getByRole("button", { name: "Delete box 2" }));
    expect(p.onChange).toHaveBeenCalledWith([{ ...r, blocks: [r.blocks[0]] }]);
  });

  it("Remove crop sets the crop to null and keeps the blocks", () => {
    const r = two();
    const p = props({ regions: [r], frame: 4 });
    render(<RegionsPanel {...p} />);
    fireEvent.click(screen.getByRole("button", { name: "Remove crop" }));
    expect(p.onChange).toHaveBeenCalledWith([{ ...r, crop: null }]);
  });

  it("hovering or focusing a row highlights it; leaving or blurring clears it", () => {
    const onHighlight = vi.fn();
    render(<RegionsPanel {...props({ regions: [two()], frame: 4, onHighlight })} />);
    const row2 = screen.getByText("2. 30×10 at (30, 40)").closest("li")!;
    fireEvent.mouseEnter(row2);
    expect(onHighlight).toHaveBeenLastCalledWith({ kind: "block", index: 1 });
    fireEvent.mouseLeave(row2);
    expect(onHighlight).toHaveBeenLastCalledWith(null);
    fireEvent.focus(screen.getByRole("button", { name: "Delete box 1" }));
    expect(onHighlight).toHaveBeenLastCalledWith({ kind: "block", index: 0 });
    fireEvent.blur(screen.getByRole("button", { name: "Delete box 1" }));
    expect(onHighlight).toHaveBeenLastCalledWith(null);
    fireEvent.mouseEnter(screen.getByText("Crop 80×60").closest("li")!);
    expect(onHighlight).toHaveBeenLastCalledWith({ kind: "crop" });
  });
});
