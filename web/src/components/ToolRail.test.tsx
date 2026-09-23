// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolRail } from "./ToolRail";

afterEach(cleanup);

describe("ToolRail", () => {
  it("switches tools", () => {
    const onTool = vi.fn();
    render(<ToolRail tool="clicks" disabled={false} brushDisabled={false} locked={false} onTool={onTool} onFit={() => {}} />);
    expect(screen.getByRole("button", { name: "Click" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    expect(onTool).toHaveBeenCalledWith("box");
  });
  it("disables the prompting tools but keeps Fit available", () => {
    const onFit = vi.fn();
    render(<ToolRail tool="clicks" disabled brushDisabled={false} locked={false} onTool={() => {}} onFit={onFit} />);
    expect((screen.getByRole("button", { name: "Click" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Fit" }));
    expect(onFit).toHaveBeenCalled();
  });

  it("offers the Brush tool, which can be disabled on its own, and locks the other tools while painting", () => {
    const onTool = vi.fn();
    const { rerender } = render(<ToolRail tool="clicks" disabled={false} brushDisabled onTool={onTool} onFit={() => {}} locked={false} />);
    expect((screen.getByRole("button", { name: "Brush" }) as HTMLButtonElement).disabled).toBe(true);
    rerender(<ToolRail tool="clicks" disabled={false} brushDisabled={false} onTool={onTool} onFit={() => {}} locked={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Brush" }));
    expect(onTool).toHaveBeenCalledWith("brush");
    rerender(<ToolRail tool="brush" disabled={false} brushDisabled={false} onTool={onTool} onFit={() => {}} locked />);
    expect(screen.getByRole("button", { name: "Brush" }).getAttribute("aria-pressed")).toBe("true");
    for (const name of ["Click", "Box", "Text"]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
    expect((screen.getByRole("button", { name: "Fit" }) as HTMLButtonElement).disabled).toBe(false);
  });
});
