// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OverlayPanel } from "./OverlayPanel";

afterEach(cleanup);

const base = { classColor: "#22c55e", activeClassName: "cup", classColorError: null };

describe("OverlayPanel", () => {
  it("switches between the class colour and a custom colour", () => {
    const onStyle = vi.fn();
    const { rerender } = render(
      <OverlayPanel {...base} style={{ color: null, thickness: 5 }} onStyle={onStyle} onClassColor={vi.fn()} />,
    );
    const picker = screen.getByLabelText("Outline colour") as HTMLInputElement;
    expect(picker.disabled).toBe(true);
    expect(picker.value).toBe("#22c55e");
    fireEvent.click(screen.getByLabelText("Outline follows class colour"));
    expect(onStyle).toHaveBeenLastCalledWith({ color: "#22c55e", thickness: 5 });

    rerender(<OverlayPanel {...base} style={{ color: "#22c55e", thickness: 5 }} onStyle={onStyle} onClassColor={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Outline colour"), { target: { value: "#ffffff" } });
    expect(onStyle).toHaveBeenLastCalledWith({ color: "#ffffff", thickness: 5 });
    fireEvent.click(screen.getByLabelText("Outline follows class colour"));
    expect(onStyle).toHaveBeenLastCalledWith({ color: null, thickness: 5 });
  });

  it("sets the thickness down to zero, which reads as off", () => {
    const onStyle = vi.fn();
    const { rerender } = render(
      <OverlayPanel {...base} style={{ color: null, thickness: 5 }} onStyle={onStyle} onClassColor={vi.fn()} />,
    );
    expect(screen.getByRole("button", { name: "Reset" })).toHaveProperty("disabled", true);
    fireEvent.change(screen.getByLabelText("Outline thickness"), { target: { value: "0" } });
    expect(onStyle).toHaveBeenLastCalledWith({ color: null, thickness: 0 });
    rerender(<OverlayPanel {...base} style={{ color: "#000000", thickness: 0 }} onStyle={onStyle} onClassColor={vi.fn()} />);
    expect(screen.getByText("off")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(onStyle).toHaveBeenLastCalledWith({ color: null, thickness: 5 });
  });

  it("shows the active class's name and colour, and previews then debounces a change to it", () => {
    const onClassColor = vi.fn();
    render(<OverlayPanel {...base} style={{ color: null, thickness: 5 }} onStyle={vi.fn()} onClassColor={onClassColor} />);
    expect(screen.getByText("cup")).toBeTruthy();
    const input = screen.getByLabelText("Class colour") as HTMLInputElement;
    expect(input.disabled).toBe(false);
    expect(input.value).toBe("#22c55e");

    vi.useFakeTimers();
    fireEvent.change(input, { target: { value: "#111111" } });
    fireEvent.change(input, { target: { value: "#222222" } });
    expect(input.value).toBe("#222222"); // shown locally at once, before the debounce fires
    expect(onClassColor).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(300));
    vi.useRealTimers();
    expect(onClassColor).toHaveBeenCalledTimes(1);
    expect(onClassColor).toHaveBeenCalledWith("#222222");
  });

  it("disables the class colour row, with a reason, when there is no class to colour", () => {
    render(
      <OverlayPanel {...base} activeClassName={null} style={{ color: null, thickness: 5 }} onStyle={vi.fn()}
        onClassColor={vi.fn()} />,
    );
    const input = screen.getByLabelText("Class colour") as HTMLInputElement;
    expect(input.disabled).toBe(true);
    expect(input.title).not.toBe("");
  });

  it("shows a failed class colour save", () => {
    render(
      <OverlayPanel {...base} classColorError="disk full" style={{ color: null, thickness: 5 }} onStyle={vi.fn()}
        onClassColor={vi.fn()} />,
    );
    expect(screen.getByRole("alert").textContent).toBe("disk full");
  });

  it("drops a pending save when the active class changes before the debounce fires", () => {
    const onClassColor = vi.fn();
    const { rerender } = render(
      <OverlayPanel {...base} style={{ color: null, thickness: 5 }} onStyle={vi.fn()} onClassColor={onClassColor} />,
    );
    vi.useFakeTimers();
    fireEvent.change(screen.getByLabelText("Class colour"), { target: { value: "#111111" } });
    rerender(
      <OverlayPanel {...base} activeClassName="connector" classColor="#e6194b" style={{ color: null, thickness: 5 }}
        onStyle={vi.fn()} onClassColor={onClassColor} />,
    );
    expect((screen.getByLabelText("Class colour") as HTMLInputElement).value).toBe("#e6194b");
    act(() => vi.advanceTimersByTime(300));
    vi.useRealTimers();
    expect(onClassColor).not.toHaveBeenCalled();
  });
});
