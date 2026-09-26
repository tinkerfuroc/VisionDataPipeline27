// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { JobHandle } from "../hooks/useJob";
import { idleJob } from "../lib/jobReducer";
import { PropagatePanel } from "./PropagatePanel";

type Props = ComponentProps<typeof PropagatePanel>;
const job = (over: Partial<JobHandle> = {}): JobHandle => ({ ...idleJob, transport: "ws", ...over });
const props = (over: Partial<Props> = {}): Props => ({
  job: job(), running: false, foreign: false, starting: false, blockedReason: null, follow: true, error: null,
  scope: "all", selectedLabel: "#2 cup", objectName: (id) => (id === 2 ? "#2 cup" : `#${id}`), ownResult: true,
  onScope: vi.fn(), onStart: vi.fn(), onStop: vi.fn(), onFollow: vi.fn(), ...over,
});
const group = () => within(screen.getByRole("group", { name: "Objects to propagate" }));

afterEach(cleanup);

describe("PropagatePanel", () => {
  it("offers All objects and the selected object", () => {
    const p = props();
    render(<PropagatePanel {...p} />);
    expect((group().getByRole("radio", { name: "All objects" }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(group().getByRole("radio", { name: "Selected: #2 cup" }));
    expect(p.onScope).toHaveBeenCalledWith("selected");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    expect(p.onStart).toHaveBeenCalledWith("forward");
  });
  it("disables Selected on a New object row, with the reason, and shows All objects chosen", () => {
    render(<PropagatePanel {...props({ scope: "selected", selectedLabel: null })} />);
    const selected = group().getByRole("radio", { name: "Selected object" }) as HTMLInputElement;
    expect(selected.disabled).toBe(true);
    expect((group().getByRole("radio", { name: "All objects" }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getByText(/A new object has no mask yet/)).toBeTruthy();
  });
  it("names the skipped objects and each lost object when the run ends", () => {
    const result = { processed: 20, written: 30, skipped: [3], lost: { "2": 33, "5": 12 }, lost_at: 12 };
    render(<PropagatePanel {...props({ job: job({ status: "done", result }) })} />);
    expect(screen.getByText("Skipped #3: no mask on the start frame.")).toBeTruthy();
    const lost = screen.getAllByText(/^Lost /).map((el) => el.textContent);
    expect(lost).toEqual([
      "Lost #5 at frame 12. Correct it there and propagate again.",
      "Lost #2 cup at frame 33. Correct it there and propagate again.",
    ]);
  });
  it("names no object of a run this page did not start: its ids are another clip's", () => {
    const result = { processed: 20, written: 30, skipped: [3], lost: { "2": 33 }, lost_at: 33 };
    render(<PropagatePanel {...props({ ownResult: false, job: job({ status: "done", result }) })} />);
    expect(screen.queryByText(/^Skipped/)).toBeNull();
    expect(screen.getAllByText(/^Lost /).map((el) => el.textContent)).toEqual(["Lost an object at frame 33."]);
  });
});
