// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Prompting } from "../hooks/usePrompting";
import { emptyDraft } from "../lib/promptState";
import { PromptPanel } from "./PromptPanel";

const prompting = (over: Partial<Prompting> = {}): Prompting => ({
  draft: emptyDraft("clicks"), dispatch: vi.fn(), candidates: [], busy: false, error: null, busyJobId: null,
  canAccept: false, accept: vi.fn(async () => {}), clear: vi.fn(), ...over,
});
afterEach(cleanup);

describe("PromptPanel", () => {
  it("switches tools", () => {
    const p = prompting();
    render(<PromptPanel prompting={p} disabledReason={null} />);
    expect(screen.getByRole("button", { name: "Click" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    expect(p.dispatch).toHaveBeenCalledWith({ type: "tool", tool: "box" });
  });
  it("submits a text phrase with the form", () => {
    const p = prompting({ draft: { ...emptyDraft("text"), phrase: "bolt" } });
    render(<PromptPanel prompting={p} disabledReason={null} />);
    fireEvent.change(screen.getByLabelText("Text prompt"), { target: { value: "red bolt" } });
    expect(p.dispatch).toHaveBeenCalledWith({ type: "phrase", phrase: "red bolt" });
    fireEvent.submit(screen.getByLabelText("Text prompt").closest("form")!);
    expect(p.dispatch).toHaveBeenCalledWith({ type: "submit" });
  });
  it("lists numbered candidates and lets the user choose one", () => {
    const c = { score: 0.91, bbox: [0, 0, 9, 9] as [number, number, number, number], area: 1234, phrase: "bolt", mask_png_b64: "" };
    const p = prompting({ draft: { ...emptyDraft("text"), query: "bolt" }, candidates: [c, { ...c, score: 0.5 }], canAccept: true });
    render(<PromptPanel prompting={p} disabledReason={null} />);
    fireEvent.click(screen.getByRole("button", { name: /^2 ·/ }));
    expect(p.dispatch).toHaveBeenCalledWith({ type: "candidate", index: 1 });
  });
  it("enables Accept only when there is something to accept, and wires Accept / Clear / Undo", () => {
    const idle = prompting();
    const { rerender } = render(<PromptPanel prompting={idle} disabledReason={null} />);
    expect((screen.getByRole("button", { name: /Accept/ }) as HTMLButtonElement).disabled).toBe(true);
    const ready = prompting({ canAccept: true });
    rerender(<PromptPanel prompting={ready} disabledReason={null} />);
    fireEvent.click(screen.getByRole("button", { name: /Accept/ }));
    expect(ready.accept).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Clear/ }));
    expect(ready.clear).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Undo/ }));
    expect(ready.dispatch).toHaveBeenCalledWith({ type: "undo" });
  });
  it("shows why prompting is disabled, and the last error", () => {
    render(<PromptPanel prompting={prompting({ error: "No usable mask" })} disabledReason="This frame is deleted." />);
    expect(screen.getByText("This frame is deleted.")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toBe("No usable mask");
    expect((screen.getByRole("button", { name: "Click" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
