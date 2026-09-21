// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { Clips } from "./Clips";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: {
      listProjects: vi.fn(), createProject: vi.fn(), addClass: vi.fn(), listClips: vi.fn(),
      importFrames: vi.fn(), importVideo: vi.fn(), deleteClip: vi.fn(), getJob: vi.fn(), cancelJob: vi.fn(),
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

const project: Project = {
  id: "demo-abc123", name: "demo", created_at: "2026-09-20T00:00:00+00:00",
  classes: [{ id: 0, name: "connector", color: "#e6194b" }],
};
const clip: ClipSummary = {
  id: "c1", project_id: project.id, source: { type: "video", name: "bench.mp4" }, fps: 30, stride: 1,
  width: 640, height: 480, frame_count: 40, created_at: "", deleted_frames: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  objects: [], labeled_frames: 15,
};
const m = vi.mocked(api);
const renderPage = () => render(<MemoryRouter><Clips /></MemoryRouter>);

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("WebSocket", FakeSocket);
  Object.values(m).forEach((fn) => fn.mockReset());
  m.listProjects.mockResolvedValue([project]);
  m.listClips.mockResolvedValue([clip]);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Clips page", () => {
  it("selects the first project and lists its classes and clips", async () => {
    renderPage();
    expect(await screen.findByText("bench.mp4")).toBeTruthy();
    expect(screen.getByText("connector")).toBeTruthy();
    expect(screen.getByText(/40 frames/).textContent).toContain("50% labeled");
    expect(screen.getByRole("link", { name: /open/i }).getAttribute("href")).toBe("/clips/c1");
    expect(m.listClips).toHaveBeenCalledWith(project.id);
    expect(localStorage.getItem("vsp.project")).toBe(project.id);
  });

  it("creates a project and selects it", async () => {
    const created: Project = { id: "new-1", name: "new", created_at: "", classes: [] };
    m.createProject.mockResolvedValue(created);
    m.listProjects.mockResolvedValueOnce([project]).mockResolvedValue([project, created]);
    m.listClips.mockResolvedValueOnce([clip]).mockResolvedValue([]);
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("New project name"), { target: { value: " new " } });
    fireEvent.click(screen.getByRole("button", { name: "Create project" }));
    await waitFor(() => expect(m.createProject).toHaveBeenCalledWith("new"));
    await waitFor(() => expect(m.listClips).toHaveBeenLastCalledWith("new-1"));
    expect((screen.getByLabelText("Project") as HTMLSelectElement).value).toBe("new-1");
  });

  it("adds a class to the selected project", async () => {
    m.addClass.mockResolvedValue({ id: 1, name: "bolt", color: "#3cb44b" });
    m.listProjects
      .mockResolvedValueOnce([project])
      .mockResolvedValue([{ ...project, classes: [...project.classes, { id: 1, name: "bolt", color: "#3cb44b" }] }]);
    renderPage();
    await screen.findByText("connector");
    fireEvent.change(screen.getByLabelText("New class name"), { target: { value: "bolt" } });
    fireEvent.click(screen.getByRole("button", { name: "Add class" }));
    expect(await screen.findByText("bolt")).toBeTruthy();
    expect(m.addClass).toHaveBeenCalledWith(project.id, "bolt");
  });

  it("starts a frame-directory import, shows its progress and reloads the clips when it is done", async () => {
    m.importFrames.mockResolvedValue({ job_id: "j7" });
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Frame directory"), { target: { value: "~/footage/run1" } });
    fireEvent.change(screen.getByLabelText("Frame directory fps"), { target: { value: "15" } });
    fireEvent.click(screen.getByRole("button", { name: "Import frames" }));
    await waitFor(() => expect(m.importFrames).toHaveBeenCalledWith(project.id, { path: "~/footage/run1", fps: 15, stride: 1 }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j7$/));
    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ seq: 0, type: "progress", done: 12 }) }));
    expect(screen.getByText(/12 frames/)).toBeTruthy();
    const before = m.listClips.mock.calls.length;
    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ seq: 1, type: "done", clip_id: "c2", frame_count: 12 }) }));
    await waitFor(() => expect(m.listClips.mock.calls.length).toBe(before + 1));
    await waitFor(() => expect(screen.queryByText(/12 frames/)).toBeNull());
  });

  it("keeps a failed import visible until it is dismissed", async () => {
    m.importFrames.mockResolvedValue({ job_id: "j8" });
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Frame directory"), { target: { value: "~/nope" } });
    fireEvent.click(screen.getByRole("button", { name: "Import frames" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j8$/));
    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ seq: 0, type: "error", message: "IngestError: no frames" }) }));
    expect(await screen.findByText(/IngestError: no frames/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(/IngestError: no frames/)).toBeNull();
  });

  it("confirms before deleting a clip and shows a 409 instead of throwing", async () => {
    const confirm = vi.fn().mockReturnValueOnce(false).mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    m.deleteClip.mockRejectedValue(new ApiError(409, "job abc is still running", "abc"));
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Delete clip" }));
    expect(m.deleteClip).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete clip" }));
    expect(await screen.findByText(/job abc is still running/)).toBeTruthy();
    expect(screen.getByText("bench.mp4")).toBeTruthy();
  });
});
