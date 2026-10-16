// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FrameState } from "../api/types";
import { CleanupPanel } from "./CleanupPanel";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);
type Props = ComponentProps<typeof CleanupPanel>;
const props = (over: Partial<Props> = {}): Props => ({
  frame: 0, states: S("kmmde"), lanes: [S("kmmde")], scope: "active", activeLabel: "#1 cup", range: [0, 4],
  clearBlockedReason: null, lockReason: null, error: null, onScope: vi.fn(), onRange: vi.fn(), onToggleDelete: vi.fn(),
  onPrune: vi.fn(), onClearRange: vi.fn(), onDeleteRange: vi.fn(), onRestoreRange: vi.fn(), ...over,
});
const group = () => within(screen.getByRole("group", { name: "Objects to clear" }));

afterEach(cleanup);

describe("CleanupPanel", () => {
  it("clears the active object by default and offers every object", () => {
    const p = props();
    render(<CleanupPanel {...p} />);
    expect((group().getByRole("radio", { name: "Active object (#1 cup)" }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(group().getByRole("radio", { name: "All objects" }));
    expect(p.onScope).toHaveBeenCalledWith("all");
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frames 0 to 4" }));
    expect(p.onClearRange).toHaveBeenCalledWith(0, 4);
  });
  it("counts the masks and keyframes of every lane it would clear", () => {
    render(<CleanupPanel {...props({ scope: "all", lanes: [S("kmmde"), S("emkde")] })} />);
    expect(screen.getByText("Removes 3 propagated masks, 2 keyframes and any mask under the 1 deleted frame.")).toBeTruthy();
  });
  it("has nothing to clear for a New object row", () => {
    render(<CleanupPanel {...props({ lanes: [], activeLabel: null })} />);
    expect(group().getByRole("radio", { name: "Active object" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Clear masks in frames 0 to 4" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("There is nothing to clear in frames 0 to 4.")).toBeTruthy();
  });
  it("deletes or restores the whole marked range, counting what would change", () => {
    const p = props({ range: [1, 4] }); // states k m m d e: frames 1..4 hold 3 kept frames and 1 deleted
    render(<CleanupPanel {...p} />);
    expect(screen.getByText("In frames 1 to 4: 3 frames to delete, 1 deleted frame to restore.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Delete frames 1 to 4" }));
    expect(p.onDeleteRange).toHaveBeenCalledWith(1, 4);
    fireEvent.click(screen.getByRole("button", { name: "Restore frames 1 to 4" }));
    expect(p.onRestoreRange).toHaveBeenCalledWith(1, 4);
  });
  it("disables what would change nothing, and hides the range buttons with no range marked", () => {
    const { rerender } = render(<CleanupPanel {...props({ range: [1, 2] })} />);
    expect((screen.getByRole("button", { name: "Restore frames 1 to 2" }) as HTMLButtonElement).disabled).toBe(true);
    rerender(<CleanupPanel {...props({ range: [3, 3] })} />);
    expect((screen.getByRole("button", { name: "Delete frame 3" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Restore frame 3" }) as HTMLButtonElement).disabled).toBe(false);
    rerender(<CleanupPanel {...props({ range: null })} />);
    expect(screen.queryByRole("button", { name: /^Delete frames? \d/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Delete frame" })).toBeTruthy(); // the single-frame toggle stays
  });
});
