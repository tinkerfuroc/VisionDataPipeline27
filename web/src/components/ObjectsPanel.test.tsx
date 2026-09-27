// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ClipDetail, FrameState } from "../api/types";
import { ObjectsPanel } from "./ObjectsPanel";

const lane = (labelled: number): FrameState[] =>
  Array.from({ length: 10 }, (_, i) => (i === 0 ? "keyframe" : i < labelled ? "masked" : "empty"));
const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 10, created_at: "", deleted_frames: [],
    // stored out of id order on purpose: the panel lists them by id
    objects: [{ id: 2, class_id: 1, keyframes: [], propagated_ranges: [] }, { id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states: lane(4),
  classes: [{ id: 0, name: "cup", color: "#e6194b" }, { id: 1, name: "bowl", color: "#3cb44b" }],
  object_states: { "1": lane(4), "2": lane(1) },
};
type Props = ComponentProps<typeof ObjectsPanel>;
const props = (over: Partial<Props> = {}): Props => ({
  detail, selection: { kind: "object", id: 1 }, hidden: new Set<number>(), lockReason: null, deleteBlockedReason: null,
  error: null, onSelect: vi.fn(), onToggleHidden: vi.fn(), onClass: vi.fn(), onRelabel: vi.fn(), onNew: vi.fn(),
  onDelete: vi.fn(), onAddClass: vi.fn(async () => {}), ...over,
});
const rows = () => within(screen.getByRole("list", { name: "Objects" }));

afterEach(cleanup);

describe("ObjectsPanel", () => {
  it("lists the objects in id order with class and labelled-frame count; a click selects, the eye hides", () => {
    const p = props({ hidden: new Set([2]) });
    render(<ObjectsPanel {...p} />);
    const names = rows().getAllByRole("button").map((b) => b.getAttribute("aria-label") ?? b.textContent);
    expect(names).toEqual(["#1 cup4 frames", "Hide #1 cup", "#2 bowl1 frame", "Show #2 bowl"]);
    expect(rows().getByRole("button", { name: /^#1 cup/ }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(rows().getByRole("button", { name: /^#2 bowl/ }));
    expect(p.onSelect).toHaveBeenCalledWith(2);
    fireEvent.click(rows().getByRole("button", { name: "Show #2 bowl" }));
    expect(p.onToggleHidden).toHaveBeenCalledWith(2);
  });
  it("picking a class always starts a new object of it, whether or not one is active; it never relabels", () => {
    const p = props();
    const { rerender } = render(<ObjectsPanel {...p} />);
    expect(screen.getByText("Label a class")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Class"), { target: { value: "1" } });
    expect(p.onClass).toHaveBeenCalledWith(1);
    expect(p.onRelabel).not.toHaveBeenCalled();
    rerender(<ObjectsPanel {...p} selection={{ kind: "new", classId: 1 }} />);
    expect(rows().getByText("New object (bowl)", { selector: "div" }).getAttribute("aria-current")).toBe("true");
    expect(screen.getByText("Class of the new object")).toBeTruthy();
    expect((screen.getByRole("button", { name: /^\+ New object/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Delete object" }) as HTMLButtonElement).disabled).toBe(true);
    // no active object: the explicit relabel row does not show
    expect(screen.queryByLabelText("New class of #1")).toBeNull();
  });
  it("relabels the active object only through the explicit Change control, disabled until a different class is chosen", () => {
    const p = props(); // selection: object #1, class 0 (cup)
    render(<ObjectsPanel {...p} />);
    expect(screen.getByText("Change class of #1")).toBeTruthy();
    const select = screen.getByLabelText("New class of #1") as HTMLSelectElement;
    expect(select.value).toBe("0"); // defaults to the active object's own class
    const change = screen.getByRole("button", { name: "Change" }) as HTMLButtonElement;
    expect(change.disabled).toBe(true);
    fireEvent.change(select, { target: { value: "0" } }); // the object's own class: still disabled
    expect(change.disabled).toBe(true);
    fireEvent.change(select, { target: { value: "1" } });
    expect(change.disabled).toBe(false);
    fireEvent.click(change);
    expect(p.onRelabel).toHaveBeenCalledWith(1);
    expect(p.onClass).not.toHaveBeenCalled();
  });
  it("starts a new object and deletes the active one on request", () => {
    const p = props();
    render(<ObjectsPanel {...p} />);
    fireEvent.click(screen.getByRole("button", { name: /^\+ New object/ }));
    expect(p.onNew).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete #1 cup" }));
    expect(p.onDelete).toHaveBeenCalled();
  });
  it("names the Delete button after the active object", () => {
    const p = props({ selection: { kind: "object", id: 2 } });
    render(<ObjectsPanel {...p} />);
    expect(screen.getByRole("button", { name: "Delete #2 bowl" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Delete object" })).toBeNull();
  });
  it("keeps the selection while brush strokes are unsaved, disables the class select, and says why Delete is unavailable", () => {
    render(
      <ObjectsPanel {...props({ lockReason: "Save or discard the brush strokes on frame 0 first.", deleteBlockedReason: "A propagation is running. Stop it to delete an object." })} />,
    );
    expect((rows().getByRole("button", { name: /^#2 bowl/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((rows().getByRole("button", { name: /^#1 cup/ }) as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByRole("button", { name: /^\+ New object/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Delete #1 cup" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText("Class") as HTMLSelectElement).disabled).toBe(true);
    expect((screen.getByLabelText("New class of #1") as HTMLSelectElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Change" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("A propagation is running. Stop it to delete an object.")).toBeTruthy();
  });
  it("tells an empty clip how to start, and adds a class", async () => {
    const p = props({ detail: { ...detail, clip: { ...detail.clip, objects: [] }, object_states: {} }, selection: { kind: "new", classId: 0 } });
    render(<ObjectsPanel {...p} />);
    expect(screen.getByText(/Nothing is labelled yet/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("New class"), { target: { value: " plate " } });
    fireEvent.submit(screen.getByLabelText("New class").closest("form")!);
    await waitFor(() => expect(p.onAddClass).toHaveBeenCalledWith("plate"));
  });
});
