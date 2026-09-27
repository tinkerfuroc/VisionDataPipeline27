// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Prompting } from "../hooks/usePrompting";
import { emptyDraft } from "../lib/promptState";
import { PromptPanel } from "./PromptPanel";

const prompting = (over: Partial<Prompting> = {}): Prompting => ({
  draft: emptyDraft("clicks"), dispatch: vi.fn(), candidates: [], busy: false, error: null, busyJobId: null,
  canAccept: false, offersAcceptAll: false, canAcceptAll: false, acceptAll: vi.fn(async () => {}),
  accept: vi.fn(async () => {}), clear: vi.fn(), ...over,
});
afterEach(cleanup);

// The tool buttons (and their assertions) live in ToolRail.
describe("PromptPanel", () => {
  it("submits a text phrase with the form", () => {
    const p = prompting({ draft: { ...emptyDraft("text"), phrase: "bolt" } });
    render(<PromptPanel prompting={p} disabledReason={null} target="Accept writes to #1 bolt." />);
    fireEvent.change(screen.getByLabelText("Text prompt"), { target: { value: "red bolt" } });
    expect(p.dispatch).toHaveBeenCalledWith({ type: "phrase", phrase: "red bolt" });
    fireEvent.submit(screen.getByLabelText("Text prompt").closest("form")!);
    expect(p.dispatch).toHaveBeenCalledWith({ type: "submit" });
  });
  it("lists numbered candidates and lets the user choose one", () => {
    const c = { score: 0.91, bbox: [0, 0, 9, 9] as [number, number, number, number], area: 1234, phrase: "bolt", mask_png_b64: "" };
    const p = prompting({ draft: { ...emptyDraft("text"), query: "bolt" }, candidates: [c, { ...c, score: 0.5 }], canAccept: true });
    render(<PromptPanel prompting={p} disabledReason={null} target="Accept writes to #1 bolt." />);
    fireEvent.click(screen.getByRole("button", { name: /^2 score/ }));
    expect(p.dispatch).toHaveBeenCalledWith({ type: "candidate", index: 1 });
  });
  it("enables Accept only when there is something to accept, and wires Accept / Clear / Undo", () => {
    const idle = prompting();
    const { rerender } = render(<PromptPanel prompting={idle} disabledReason={null} target="Accept writes to #1 bolt." />);
    expect((screen.getByRole("button", { name: /Accept/ }) as HTMLButtonElement).disabled).toBe(true);
    const ready = prompting({ canAccept: true });
    rerender(<PromptPanel prompting={ready} disabledReason={null} target="Accept writes to #1 bolt." />);
    fireEvent.click(screen.getByRole("button", { name: /Accept/ }));
    expect(ready.accept).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Clear/ }));
    expect(ready.clear).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Undo/ }));
    expect(ready.dispatch).toHaveBeenCalledWith({ type: "undo" });
  });
  it("lists the clicks by number, with their kind and position, and removes one on request", () => {
    const draft = { ...emptyDraft("clicks"), points: [[412, 288, 1], [50, 60, 0]] as [number, number, 0 | 1][] };
    const p = prompting({ draft, canAccept: true });
    render(<PromptPanel prompting={p} disabledReason={null} target="Accept writes to #1 bolt." />);
    const rows = screen.getAllByRole("listitem");
    expect(rows.map((r) => r.textContent)).toEqual(["1Include412, 288Remove", "2Exclude50, 60Remove"]);
    fireEvent.click(screen.getByRole("button", { name: "Remove click 2" }));
    expect(p.dispatch).toHaveBeenCalledWith({ type: "remove", index: 1 });
    // the list also serves the box tool's refinement clicks, and is absent when there are none
    cleanup();
    render(
      <PromptPanel
        prompting={prompting({ draft: { ...emptyDraft("box"), box: [0, 0, 9, 9], points: [[3, 3, 0]] } })}
        disabledReason={null}
        target="Accept writes to #1 bolt."
      />,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    cleanup();
    render(<PromptPanel prompting={prompting()} disabledReason={null} target="Accept writes to #1 bolt." />);
    expect(screen.queryByRole("listitem")).toBeNull();
  });
  it("shows why prompting is disabled, and the last error", () => {
    render(
      <PromptPanel
        prompting={prompting({ error: "No usable mask" })}
        disabledReason="This frame is deleted."
        target="Accept writes to #1 bolt."
      />,
    );
    expect(screen.getByText("This frame is deleted.")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toBe("No usable mask");
  });

  it("names what Accept writes to, and offers Accept all as new objects on a New object row", () => {
    const c = { score: 0.91, bbox: [0, 0, 9, 9] as [number, number, number, number], area: 1234, phrase: "bolt", mask_png_b64: "" };
    const p = prompting({
      draft: { ...emptyDraft("text"), query: "bolt" }, candidates: [c, c], canAccept: true, offersAcceptAll: true, canAcceptAll: true,
    });
    render(<PromptPanel prompting={p} disabledReason={null} target="Accept creates a new bolt object." />);
    expect(screen.getByText("Accept creates a new bolt object.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Accept selected/ }));
    expect(p.accept).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Accept all as new objects" }));
    expect(p.acceptAll).toHaveBeenCalled();
    cleanup();
    render(<PromptPanel prompting={prompting({ candidates: [c, c], canAccept: true })} disabledReason={null} target="Accept writes to #1 bolt." />);
    expect(screen.queryByRole("button", { name: "Accept all as new objects" })).toBeNull();
  });
});
