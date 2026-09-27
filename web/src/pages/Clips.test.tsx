// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, UploadCancelled, api } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { Clips } from "./Clips";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: {
      listProjects: vi.fn(), createProject: vi.fn(), addClass: vi.fn(), setClassColor: vi.fn(), listClips: vi.fn(),
      importFrames: vi.fn(), uploadClip: vi.fn(), deleteClip: vi.fn(), getJob: vi.fn(), cancelJob: vi.fn(),
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
const project2: Project = { id: "proj-2", name: "second", created_at: "", classes: [] };
const clip2: ClipSummary = { ...clip, id: "c2", project_id: project2.id, source: { type: "video", name: "second.mp4" } };
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
    expect(screen.getByRole("link", { name: "Export dataset" }).getAttribute("href")).toBe(`/projects/${project.id}/export`);
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

  it("shows a picked class colour at once and saves only the final one, ~300ms later", async () => {
    m.setClassColor.mockResolvedValue({ id: 0, name: "connector", color: "#123456" });
    m.listProjects
      .mockResolvedValueOnce([project])
      .mockResolvedValue([{ ...project, classes: [{ id: 0, name: "connector", color: "#123456" }] }]);
    renderPage();
    await screen.findByText("connector");
    const picker = screen.getByLabelText("Colour of connector") as HTMLInputElement;

    vi.useFakeTimers();
    fireEvent.change(picker, { target: { value: "#111111" } });
    fireEvent.change(picker, { target: { value: "#222222" } });
    fireEvent.change(picker, { target: { value: "#123456" } });
    expect(picker.value).toBe("#123456"); // shown locally at once, before the debounce fires
    expect(m.setClassColor).not.toHaveBeenCalled();

    await act(async () => vi.advanceTimersByTime(300));
    vi.useRealTimers();
    expect(m.setClassColor).toHaveBeenCalledTimes(1);
    expect(m.setClassColor).toHaveBeenCalledWith(project.id, 0, "#123456");
    await waitFor(() => expect(m.listProjects).toHaveBeenCalledTimes(2)); // reloaded, as addClass does
  });

  it("shows a class colour save error the way the page already shows errors", async () => {
    m.setClassColor.mockRejectedValue(new Error("disk full"));
    renderPage();
    await screen.findByText("connector");
    vi.useFakeTimers();
    fireEvent.change(screen.getByLabelText("Colour of connector"), { target: { value: "#111111" } });
    await act(async () => vi.advanceTimersByTime(300));
    vi.useRealTimers();
    expect(await screen.findByText("disk full")).toBeTruthy();
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
    // the server only names a job id; the page says what that means for the user
    expect(await screen.findByText("A propagation or an export is running. Wait for it to finish, then delete this clip.")).toBeTruthy();
    expect(screen.queryByText(/job abc/)).toBeNull();
    expect(screen.getByText("bench.mp4")).toBeTruthy();
  });

  it("ignores a slow clip list from the previously selected project", async () => {
    m.listProjects.mockResolvedValue([project, project2]);
    let resolveFirst!: (clips: ClipSummary[]) => void;
    m.listClips.mockImplementation((pid: string) =>
      pid === project.id ? new Promise<ClipSummary[]>((resolve) => { resolveFirst = resolve; }) : Promise.resolve([clip2]),
    );
    renderPage();
    // the first project's clip list never resolves yet, but its classes render independently of it
    await screen.findByText("connector");

    fireEvent.change(screen.getByLabelText("Project"), { target: { value: project2.id } });
    expect(await screen.findByText("second.mp4")).toBeTruthy();

    await act(async () => {
      resolveFirst([clip]);
    });
    expect(screen.getByText("second.mp4")).toBeTruthy();
    expect(screen.queryByText("bench.mp4")).toBeNull();
  });

  it("a double click starts only one frame-directory import and the form resets", async () => {
    let resolveImport!: (result: { job_id: string }) => void;
    m.importFrames.mockImplementation(() => new Promise((resolve) => { resolveImport = resolve; }));
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Frame directory"), { target: { value: "~/footage/run1" } });
    const submit = screen.getByRole("button", { name: "Import frames" }) as HTMLButtonElement;
    fireEvent.click(submit);
    fireEvent.click(submit);
    expect(m.importFrames).toHaveBeenCalledTimes(1);
    expect(submit.disabled).toBe(true);

    await act(async () => {
      resolveImport({ job_id: "j9" });
    });
    expect((screen.getByLabelText("Frame directory") as HTMLInputElement).value).toBe("");
    expect(submit.disabled).toBe(true); // pending has cleared, but the path is empty again
  });

  it("disables the other submit buttons while a request is in flight", async () => {
    let resolveCreate!: (created: Project) => void;
    m.createProject.mockImplementation(() => new Promise((resolve) => { resolveCreate = resolve; }));
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("New project name"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Create project" }));
    await waitFor(() => expect(m.createProject).toHaveBeenCalledWith("x"));
    const addClassBtn = () => screen.getByRole("button", { name: "Add class" }) as HTMLButtonElement;
    const importFramesBtn = () => screen.getByRole("button", { name: "Import frames" }) as HTMLButtonElement;
    const deleteClipBtn = () => screen.getByRole("button", { name: "Delete clip" }) as HTMLButtonElement;
    expect(addClassBtn().disabled).toBe(true);
    expect(importFramesBtn().disabled).toBe(true);
    expect(deleteClipBtn().disabled).toBe(true);

    await act(async () => {
      resolveCreate(project);
    });
    await waitFor(() => expect(addClassBtn().disabled).toBe(false));
    // "Import frames" stays disabled, but now because the path field is empty, not because of pending
    expect(importFramesBtn().disabled).toBe(true);
    expect(deleteClipBtn().disabled).toBe(false);
  });

  it("clears a load error once a later load succeeds", async () => {
    m.listProjects.mockResolvedValue([project, project2]);
    m.listClips.mockRejectedValueOnce(new Error("network error")).mockResolvedValue([clip2]);
    renderPage();
    expect(await screen.findByText(/network error/)).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Project"), { target: { value: project2.id } });
    await screen.findByText("second.mp4");
    expect(screen.queryByText(/network error/)).toBeNull();
  });

  const choose = (...names: string[]) =>
    fireEvent.change(screen.getByLabelText("Footage file"), {
      target: { files: names.map((name) => new File(["abc"], name)) },
    });
  const drop = (...names: string[]) =>
    fireEvent.drop(screen.getByLabelText("Footage file").closest("section")!, {
      dataTransfer: { files: names.map((name) => new File(["abc"], name)) },
    });
  const uploadHandle = (jobId = "j10") => ({ promise: Promise.resolve({ job_id: jobId }), abort: vi.fn() });
  const deferredUpload = () => {
    let progress!: (sent: number, total: number) => void;
    let resolve!: (result: { job_id: string }) => void;
    let reject!: (e: unknown) => void;
    const handle = {
      promise: new Promise<{ job_id: string }>((res, rej) => {
        resolve = res;
        reject = rej;
      }),
      abort: vi.fn(),
    };
    return { handle, capture: (onProgress?: (sent: number, total: number) => void) => (progress = onProgress!), get progress() { return progress; }, resolve, reject };
  };

  it("starts uploading the moment a video is chosen, with no button click, using the current stride and fps", async () => {
    const first = deferredUpload();
    m.uploadClip.mockImplementation((_pid, _file, _opts, onProgress) => {
      first.capture(onProgress);
      return first.handle;
    });
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Upload stride"), { target: { value: "2" } });
    choose("bench.mp4");
    await waitFor(() =>
      expect(m.uploadClip).toHaveBeenCalledWith(project.id, expect.any(File), { stride: 2, fps: 30 }, expect.any(Function)),
    );
    expect(screen.queryByRole("button", { name: "Upload and import" })).toBeNull();
    act(() => first.progress(50, 100));
    expect(screen.getByText("50% of 100 B")).toBeTruthy();
    await act(async () => first.resolve({ job_id: "j10" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j10$/));
    expect((screen.getByLabelText("Footage file") as HTMLInputElement).value).toBe("");
  });

  it("resets the file input after each selection, so choosing the same file again fires onChange", async () => {
    m.uploadClip.mockReturnValue(uploadHandle("j11"));
    renderPage();
    await screen.findByText("bench.mp4");
    const input = screen.getByLabelText("Footage file") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["abc"], "bench.mp4")] } });
    expect(input.value).toBe("");
  });

  it("always offers frames per second, labelled as zip-only, and sends both settings", async () => {
    m.uploadClip.mockReturnValue(uploadHandle("j11"));
    renderPage();
    await screen.findByText("bench.mp4");
    expect(screen.getByText("Frames per second (zip only)")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Zip frames per second"), { target: { value: "12" } });
    fireEvent.change(screen.getByLabelText("Upload stride"), { target: { value: "3" } });
    choose("run.zip");
    await waitFor(() =>
      expect(m.uploadClip).toHaveBeenCalledWith(project.id, expect.any(File), { stride: 3, fps: 12 }, expect.any(Function)),
    );
  });

  it("uploads two chosen files one at a time, in order, both becoming import rows", async () => {
    const first = deferredUpload();
    const second = deferredUpload();
    m.uploadClip
      .mockImplementationOnce((_pid, _file, _opts, onProgress) => {
        first.capture(onProgress);
        return first.handle;
      })
      .mockImplementationOnce((_pid, _file, _opts, onProgress) => {
        second.capture(onProgress);
        return second.handle;
      });
    renderPage();
    await screen.findByText("bench.mp4");
    choose("one.mp4", "two.mp4");
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(1));
    expect(m.uploadClip).toHaveBeenCalledWith(project.id, expect.objectContaining({ name: "one.mp4" }), { stride: 1, fps: 30 }, expect.any(Function));
    expect(screen.getByText(/two\.mp4/)).toBeTruthy(); // waiting

    await act(async () => first.resolve({ job_id: "job-one" }));
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(2));
    expect(m.uploadClip).toHaveBeenCalledWith(project.id, expect.objectContaining({ name: "two.mp4" }), { stride: 1, fps: 30 }, expect.any(Function));

    await act(async () => second.resolve({ job_id: "job-two" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/job-two$/));
  });

  it("moves on to the next file when the first upload fails, without stopping the queue", async () => {
    const first = deferredUpload();
    const second = deferredUpload();
    m.uploadClip
      .mockImplementationOnce((_pid, _file, _opts, onProgress) => {
        first.capture(onProgress);
        return first.handle;
      })
      .mockImplementationOnce((_pid, _file, _opts, onProgress) => {
        second.capture(onProgress);
        return second.handle;
      });
    renderPage();
    await screen.findByText("bench.mp4");
    choose("one.mp4", "two.mp4");
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(1));

    await act(async () => first.reject(new ApiError(413, "the upload is 100 bytes; the limit is 10")));
    expect(await screen.findByText("the upload is 100 bytes; the limit is 10")).toBeTruthy();
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(2));

    await act(async () => second.resolve({ job_id: "job-two" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/job-two$/));

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText("the upload is 100 bytes; the limit is 10")).toBeNull();
  });

  it("skips an invalid file among valid ones, with a note, and still queues the valid ones", async () => {
    m.uploadClip.mockReturnValue(uploadHandle("j12"));
    renderPage();
    await screen.findByText("bench.mp4");
    choose("notes.txt", "good.mp4");
    expect(await screen.findByText("notes.txt: That file is not a video or a .zip of images.")).toBeTruthy();
    await waitFor(() =>
      expect(m.uploadClip).toHaveBeenCalledWith(project.id, expect.objectContaining({ name: "good.mp4" }), { stride: 1, fps: 30 }, expect.any(Function)),
    );
  });

  it("queues every file dropped at once", async () => {
    const first = deferredUpload();
    m.uploadClip.mockImplementation((_pid, _file, _opts, onProgress) => {
      first.capture(onProgress);
      return first.handle;
    });
    renderPage();
    await screen.findByText("bench.mp4");
    drop("a.mp4", "b.mp4", "c.mp4");
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(1));
    expect(screen.getByText(/b\.mp4/)).toBeTruthy();
    expect(screen.getByText(/c\.mp4/)).toBeTruthy();
  });

  it("cancels the current upload without reporting an error, and moves on to the next queued file", async () => {
    const first = deferredUpload();
    m.uploadClip.mockImplementationOnce((_pid, _file, _opts, onProgress) => {
      first.capture(onProgress);
      return first.handle;
    });
    const second = uploadHandle("j13");
    m.uploadClip.mockImplementationOnce(() => second);
    renderPage();
    await screen.findByText("bench.mp4");
    choose("one.mp4", "two.mp4");
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(1));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel upload" }));
    expect(first.handle.abort).toHaveBeenCalledTimes(1);
    await act(async () => first.reject(new UploadCancelled()));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/one\.mp4/)).toBeNull();
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(2));
  });

  it("cancels all queued and running uploads at once", async () => {
    const first = deferredUpload();
    m.uploadClip.mockImplementationOnce((_pid, _file, _opts, onProgress) => {
      first.capture(onProgress);
      return first.handle;
    });
    renderPage();
    await screen.findByText("bench.mp4");
    choose("one.mp4", "two.mp4", "three.mp4");
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Cancel all" }));
    expect(first.handle.abort).toHaveBeenCalledTimes(1);
    await act(async () => first.reject(new UploadCancelled()));
    expect(screen.queryByText(/two\.mp4/)).toBeNull();
    expect(screen.queryByText(/three\.mp4/)).toBeNull();
    expect(m.uploadClip).toHaveBeenCalledTimes(1); // the queue never advances to the cleared files
  });

  it("removes one waiting file from the queue without touching the others", async () => {
    const first = deferredUpload();
    m.uploadClip.mockImplementation((_pid, _file, _opts, onProgress) => {
      first.capture(onProgress);
      return first.handle;
    });
    renderPage();
    await screen.findByText("bench.mp4");
    choose("one.mp4", "two.mp4");
    await waitFor(() => expect(m.uploadClip).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Remove two.mp4 from the queue" }));
    expect(screen.queryByText(/two\.mp4/)).toBeNull();
  });

  it("refuses a file that is neither a video nor a zip", async () => {
    renderPage();
    await screen.findByText("bench.mp4");
    choose("notes.txt");
    expect(await screen.findByText("notes.txt: That file is not a video or a .zip of images.")).toBeTruthy();
    expect(m.uploadClip).not.toHaveBeenCalled();
  });
});
