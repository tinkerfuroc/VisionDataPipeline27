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
  onPrune: vi.fn(), onClearRange: vi.fn(), ...over,
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
});
