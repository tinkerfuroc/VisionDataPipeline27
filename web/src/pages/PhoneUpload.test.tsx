// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import { rememberedUploads, startResumableUpload } from "../api/resumable";
import { HealthContext } from "../hooks/useHealth";
import { PhoneUpload } from "./PhoneUpload";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { listProjects: vi.fn(), listUploads: vi.fn(), deleteUpload: vi.fn(), getJob: vi.fn(), logout: vi.fn() },
}));
vi.mock("../api/resumable", async (original) => ({
  ...(await original<typeof import("../api/resumable")>()),
  startResumableUpload: vi.fn(),
  rememberedUploads: vi.fn(() => []),
}));

const m = vi.mocked(api);
const mStart = vi.mocked(startResumableUpload);
const mRemembered = vi.mocked(rememberedUploads);
const projects = [
  { id: "p1", name: "Cups", classes: [], created_at: "" },
  { id: "p2", name: "Boards", classes: [], created_at: "" },
];

function deferred() {
  let resolve!: (v: { job_id: string }) => void;
  let progress!: (sent: number, total: number) => void;
  const handle = { promise: new Promise<{ job_id: string }>((r) => (resolve = r)), abort: vi.fn() };
  mStart.mockImplementationOnce((_p, _f, _o, onProgress) => {
    progress = onProgress!;
    return handle;
  });
  return { handle, resolve: (id: string) => resolve({ job_id: id }), progress: (s: number, t: number) => progress(s, t) };
}

function renderPage(path = "/upload?project=p2") {
  return render(
    <HealthContext.Provider value={{ health: { status: "ok", version: "0", engine: { loaded: true, device: "cuda:0", error: null }, upload: { max_bytes: 1000 } }, unreachable: false }}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/upload" element={<PhoneUpload />} />
        </Routes>
      </MemoryRouter>
    </HealthContext.Provider>,
  );
}

const video = (name = "IMG_0412.MOV", size = 100, lastModified = 5) => new File([new Uint8Array(size)], name, { lastModified, type: "video/quicktime" });
const choose = (label: string, files: File[]) =>
  fireEvent.change(screen.getByLabelText(label), { target: { files } });

beforeEach(() => {
  Object.values(m).forEach((fn) => fn.mockReset());
  mStart.mockReset();
  mRemembered.mockReset();
  mRemembered.mockReturnValue([]);
  m.listProjects.mockResolvedValue(projects);
  m.listUploads.mockResolvedValue([]);
  localStorage.clear();
});
afterEach(cleanup);

describe("PhoneUpload", () => {
  it("picks the project from the link and offers Record and Choose inputs", async () => {
    renderPage();
    await waitFor(() => expect((screen.getByLabelText("Project") as HTMLSelectElement).value).toBe("p2"));
    const record = screen.getByLabelText("Record video") as HTMLInputElement;
    expect(record.accept).toBe("video/*");
    expect(record.getAttribute("capture")).toBe("environment");
    const pick = screen.getByLabelText("Choose videos") as HTMLInputElement;
    expect(pick.multiple).toBe(true);
    expect(pick.accept).toBe("video/*,.zip,application/zip");
  });

  it("uploads a recorded video, shows progress, then links to the imported clip", async () => {
    const up = deferred();
    m.getJob.mockResolvedValue({ id: "j1", kind: "import", status: "done", result: { clip_id: "c9" }, error: null, created_at: "", n_events: 0 });
    renderPage();
    await screen.findByDisplayValue("Boards");
    choose("Record video", [video()]);
    expect(mStart).toHaveBeenCalledWith("p2", expect.any(File), { stride: 1, fps: 30 }, expect.any(Function), undefined);
    act(() => up.progress(40, 100));
    expect(await screen.findByText("40% sent")).toBeTruthy();
    await act(async () => up.resolve("j1"));
    const link = await screen.findByRole("link", { name: "Open the clip" });
    expect(link.getAttribute("href")).toBe("/clips/c9");
  });

  it("uses the stride from Options", async () => {
    deferred();
    renderPage();
    await screen.findByDisplayValue("Boards");
    fireEvent.change(screen.getByLabelText("Keep every Nth frame"), { target: { value: "3" } });
    choose("Choose videos", [video()]);
    expect(mStart.mock.calls[0][2]).toEqual({ stride: 3, fps: 30 });
  });

  it("refuses a file that is not a video or zip, and one over the size limit", async () => {
    renderPage();
    await screen.findByDisplayValue("Boards");
    choose("Choose videos", [new File(["x"], "notes.txt"), video("big.mp4", 2000)]);
    expect(mStart).not.toHaveBeenCalled();
    expect(screen.getAllByRole("alert").length).toBe(2);
  });

  it("uploads several chosen files one after another", async () => {
    const first = deferred();
    deferred();
    m.getJob.mockResolvedValue({ id: "j1", kind: "import", status: "running", result: null, error: null, created_at: "", n_events: 0 });
    renderPage();
    await screen.findByDisplayValue("Boards");
    choose("Choose videos", [video("a.mp4"), video("b.mp4")]);
    expect(mStart).toHaveBeenCalledTimes(1);
    await act(async () => first.resolve("j1"));
    expect(mStart).toHaveBeenCalledTimes(2);
    expect(mStart.mock.calls[1][1].name).toBe("b.mp4");
  });

  it("cancels the running upload", async () => {
    const up = deferred();
    renderPage();
    await screen.findByDisplayValue("Boards");
    choose("Choose videos", [video()]);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel IMG_0412.MOV" }));
    expect(up.handle.abort).toHaveBeenCalled();
  });

  it("lists an unfinished upload this phone remembers and resumes it when the same file is chosen again", async () => {
    const remembered = { upload_id: "u1", project_id: "p2", name: "IMG_0412.MOV", size: 100, lastModified: 5 };
    mRemembered.mockReturnValue([remembered]);
    m.listUploads.mockResolvedValue([{ upload_id: "u1", project_id: "p2", name: "IMG_0412.MOV", size: 100, received: 45, job_id: null }]);
    deferred();
    renderPage();
    expect(await screen.findByText("IMG_0412.MOV — 45% sent — choose it again to continue")).toBeTruthy();
    choose("Choose videos", [video()]);
    expect(mStart).toHaveBeenCalledWith("p2", expect.any(File), { stride: 1, fps: 30 }, expect.any(Function), remembered);
  });

  it("offers Discard for an unfinished upload started elsewhere", async () => {
    m.listUploads.mockResolvedValue([{ upload_id: "u7", project_id: "p2", name: "other.mp4", size: 100, received: 10, job_id: null }]);
    m.deleteUpload.mockResolvedValue(undefined);
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: "Discard other.mp4" }));
    await waitFor(() => expect(m.deleteUpload).toHaveBeenCalledWith("u7"));
  });
});
