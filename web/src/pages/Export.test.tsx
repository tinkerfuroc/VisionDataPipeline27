// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { Export } from "./Export";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: {
      listProjects: vi.fn(), listClips: vi.fn(), startExport: vi.fn(), getJob: vi.fn(), cancelJob: vi.fn(),
    },
  };
});

class FakeSocket {
  static last: FakeSocket | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(readonly url: string) {
    FakeSocket.last = this;
  }
  close() {}
}

const project: Project = { id: "p1", name: "bench", created_at: "2026-09-20T00:00:00+00:00", classes: [{ id: 0, name: "connector", color: "#e6194b" }] };

const clip = (id: string, name: string, created_at: string): ClipSummary => ({
  id, project_id: project.id, source: { type: "video", name }, fps: 30, stride: 1, width: 640, height: 480,
  frame_count: 40, created_at, deleted_frames: [], objects: [], labeled_frames: 20,
});
const clipA = clip("a", "a.mp4", "2026-01-01");
const clipB = clip("b", "b.mp4", "2026-01-03"); // newest
const clipC = clip("c", "c.mp4", "2026-01-02");
const clips = [clipA, clipB, clipC];

const m = vi.mocked(api);
const renderPage = () =>
  render(
    <MemoryRouter initialEntries={["/projects/p1/export"]}>
      <Routes>
        <Route path="/projects/:projectId/export" element={<Export />} />
      </Routes>
    </MemoryRouter>,
  );

beforeEach(() => {
  vi.stubGlobal("WebSocket", FakeSocket);
  Object.values(m).forEach((fn) => fn.mockReset());
  m.listProjects.mockResolvedValue([project]);
  m.listClips.mockResolvedValue(clips);
  m.cancelJob.mockResolvedValue({ id: "j1", kind: "export", status: "cancelled", result: null, error: null, created_at: "", n_events: 0 });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const formatNames = ["YOLO segmentation", "YOLO detection", "COCO", "Mask images"];

describe("Export page", () => {
  it("selects every clip, pre-selects the newest for validation, and starts the export with exactly what is shown", async () => {
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");

    for (const name of ["a.mp4", "b.mp4", "c.mp4"]) {
      expect((screen.getByRole("checkbox", { name: `Include ${name}` }) as HTMLInputElement).checked).toBe(true);
    }
    const bCard = screen.getByText("b.mp4").closest("li")!;
    expect(within(bCard).getByText("Validation")).toBeTruthy();
    const aCard = screen.getByText("a.mp4").closest("li")!;
    const cCard = screen.getByText("c.mp4").closest("li")!;
    expect(within(aCard).getByText("Training")).toBeTruthy();
    expect(within(cCard).getByText("Training")).toBeTruthy();

    for (const name of formatNames) {
      expect((screen.getByRole("checkbox", { name }) as HTMLInputElement).checked).toBe(true);
    }

    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() =>
      expect(m.startExport).toHaveBeenCalledWith("p1", {
        clips: ["a", "b", "c"],
        formats: ["yolo_seg", "yolo_det", "coco", "masks"],
        options: { split_mode: "per_clip", val_clips: ["b"], val_fraction: 0.2, stride: 1, include_empty: true },
      }),
    );
  });

  it("lets the user move a clip between training and validation", async () => {
    // A running job locks the Start button, so each half of this test (start once, observe the request)
    // uses its own render rather than reusing one page across two starts.
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Use a.mp4 for validation" }));
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() =>
      expect(m.startExport).toHaveBeenCalledWith(
        "p1",
        expect.objectContaining({ options: expect.objectContaining({ val_clips: ["a", "b"] }) }),
      ),
    );

    cleanup();
    m.startExport.mockClear();
    renderPage();
    await screen.findByText("a.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Use a.mp4 for validation" }));
    fireEvent.click(screen.getByRole("button", { name: "Use a.mp4 for training" }));
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() =>
      expect(m.startExport).toHaveBeenCalledWith(
        "p1",
        expect.objectContaining({ options: expect.objectContaining({ val_clips: ["b"] }) }),
      ),
    );
  });

  it("with one clip selected it splits the tail of the clip and sends no validation clips", async () => {
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");

    fireEvent.click(screen.getByRole("checkbox", { name: "Include b.mp4" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Include c.mp4" }));

    expect(screen.getByRole("button", { name: "Export 1 clip" })).toBeTruthy();
    expect(screen.getByText(/last 20% of its frames/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Export 1 clip" }));
    await waitFor(() => expect(m.startExport).toHaveBeenCalled());
    const [, body] = m.startExport.mock.calls[0];
    expect(body.options.split_mode).toBe("contiguous");
    expect(body.options).not.toHaveProperty("val_clips");
  });

  it("says why it cannot start", async () => {
    renderPage();
    await screen.findByText("a.mp4");

    for (const name of formatNames) fireEvent.click(screen.getByRole("checkbox", { name }));

    expect(screen.getByText("Choose at least one format.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Export 3 clips" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows progress, then where the dataset was written", async () => {
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");

    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));

    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ type: "progress", done: 10, total: 40, seq: 0 }) }));
    expect(screen.getByText(/10 of 40/)).toBeTruthy();
    const cancelBtn = screen.getByRole("button", { name: "Cancel" });
    fireEvent.click(cancelBtn);
    expect(m.cancelJob).toHaveBeenCalledWith("j1");

    act(() =>
      FakeSocket.last!.onmessage!({
        data: JSON.stringify({
          type: "done", seq: 1,
          export_id: "e1", path: "/data/exports/e1", formats: ["coco"],
          counts: { train: { images: 32, annotated: 30, empty: 2 }, val: { images: 8, annotated: 8, empty: 0 } },
          linked: true,
        }),
      }),
    );
    expect(await screen.findByDisplayValue("/data/exports/e1")).toBeTruthy();
    expect(screen.getByText(/32 training images/)).toBeTruthy();
    expect(screen.getByText("32 training images, 2 without a mask")).toBeTruthy();
    expect(screen.getByText("8 validation images")).toBeTruthy();
    expect(screen.queryByText(/hardlinks/)).toBeNull();
  });

  it("warns when no clip is left for validation, without blocking the export", async () => {
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");
    expect(screen.queryByText(/no validation set/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Use b.mp4 for training" }));
    expect(screen.getByText(/no validation set/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() => expect(m.startExport).toHaveBeenCalled());
    expect(m.startExport.mock.calls[0][1].options.val_clips).toEqual([]);
  });

  it("says when images had to be copied, and when an export was cancelled", async () => {
    m.startExport.mockResolvedValue({ job_id: "j1" });
    renderPage();
    await screen.findByText("a.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));
    act(() =>
      FakeSocket.last!.onmessage!({
        data: JSON.stringify({
          type: "done", seq: 0, export_id: "e1", path: "/data/exports/e1", formats: ["coco"],
          counts: { train: { images: 4, annotated: 4, empty: 0 }, val: { images: 1, annotated: 1, empty: 0 } }, linked: false,
        }),
      }),
    );
    expect(await screen.findByText(/does not support hardlinks/)).toBeTruthy();

    m.startExport.mockResolvedValue({ job_id: "j2" });
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j2$/));
    expect(screen.queryByText(/does not support hardlinks/)).toBeNull(); // the previous result does not linger
    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ type: "cancelled", seq: 0, cancelled: true }) }));
    expect(await screen.findByText("Export cancelled. Nothing was written.")).toBeTruthy();
  });

  it("says so when the project does not exist", async () => {
    m.listProjects.mockResolvedValue([]);
    renderPage();
    expect((await screen.findByRole("alert")).textContent).toContain("This project no longer exists.");
    expect(screen.getByRole("link", { name: /back to clips/i })).toBeTruthy();
  });

  it("reports a refused start", async () => {
    m.startExport.mockRejectedValueOnce(new ApiError(409, "job abc is still running", "abc"));
    renderPage();
    await screen.findByText("a.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    expect(await screen.findByText("A propagation is running. Wait for it to finish, then export.")).toBeTruthy();

    m.startExport.mockRejectedValueOnce(new Error("network dropped"));
    fireEvent.click(screen.getByRole("button", { name: "Export 3 clips" }));
    expect(await screen.findByText(/network dropped/)).toBeTruthy();
  });
});
