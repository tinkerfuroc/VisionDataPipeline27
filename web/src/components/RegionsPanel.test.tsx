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
});
