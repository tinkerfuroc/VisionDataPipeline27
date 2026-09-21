// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolRail } from "./ToolRail";

afterEach(cleanup);

describe("ToolRail", () => {
  it("switches tools", () => {
    const onTool = vi.fn();
    render(<ToolRail tool="clicks" disabled={false} onTool={onTool} onFit={() => {}} />);
    expect(screen.getByRole("button", { name: "Click" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    expect(onTool).toHaveBeenCalledWith("box");
  });
  it("disables the prompting tools but keeps Fit available", () => {
    const onFit = vi.fn();
    render(<ToolRail tool="clicks" disabled onTool={() => {}} onFit={onFit} />);
    expect((screen.getByRole("button", { name: "Click" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Fit" }));
    expect(onFit).toHaveBeenCalled();
  });
});
