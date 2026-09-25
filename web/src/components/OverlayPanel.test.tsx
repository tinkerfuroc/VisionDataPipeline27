// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OverlayPanel } from "./OverlayPanel";

afterEach(cleanup);

describe("OverlayPanel", () => {
  it("switches between the class colour and a custom colour", () => {
    const onStyle = vi.fn();
    const { rerender } = render(<OverlayPanel style={{ color: null, thickness: 5 }} classColor="#22c55e" onStyle={onStyle} />);
    const picker = screen.getByLabelText("Outline colour") as HTMLInputElement;
    expect(picker.disabled).toBe(true);
    expect(picker.value).toBe("#22c55e");
    fireEvent.click(screen.getByLabelText("Class colour"));
    expect(onStyle).toHaveBeenLastCalledWith({ color: "#22c55e", thickness: 5 });

    rerender(<OverlayPanel style={{ color: "#22c55e", thickness: 5 }} classColor="#22c55e" onStyle={onStyle} />);
    fireEvent.change(screen.getByLabelText("Outline colour"), { target: { value: "#ffffff" } });
    expect(onStyle).toHaveBeenLastCalledWith({ color: "#ffffff", thickness: 5 });
    fireEvent.click(screen.getByLabelText("Class colour"));
    expect(onStyle).toHaveBeenLastCalledWith({ color: null, thickness: 5 });
  });
  it("sets the thickness down to zero, which reads as off", () => {
    const onStyle = vi.fn();
    const { rerender } = render(<OverlayPanel style={{ color: null, thickness: 5 }} classColor="#22c55e" onStyle={onStyle} />);
    expect(screen.getByRole("button", { name: "Reset" })).toHaveProperty("disabled", true);
    fireEvent.change(screen.getByLabelText("Outline thickness"), { target: { value: "0" } });
    expect(onStyle).toHaveBeenLastCalledWith({ color: null, thickness: 0 });
    rerender(<OverlayPanel style={{ color: "#000000", thickness: 0 }} classColor="#22c55e" onStyle={onStyle} />);
    expect(screen.getByText("off")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(onStyle).toHaveBeenLastCalledWith({ color: null, thickness: 5 });
  });
});
