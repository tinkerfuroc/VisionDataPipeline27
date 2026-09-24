// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { BackgroundPool, ExportInfo } from "../api/types";
import type { JobHandle } from "../hooks/useJob";
import { useJob } from "../hooks/useJob";
import { AugmentPanel } from "./AugmentPanel";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: { listBackgrounds: vi.fn(), fetchBackgrounds: vi.fn(), augmentExport: vi.fn(), cancelJob: vi.fn() },
  };
});
vi.mock("../hooks/useJob", () => ({ useJob: vi.fn() }));

const m = vi.mocked(api);
const mUseJob = vi.mocked(useJob);

const idle: JobHandle = {
  status: "idle", nextSeq: 0, done: 0, total: null, lastFrame: null, logs: [], result: null, error: null, transport: "ws",
};
const jobs: Record<string, JobHandle> = {};

const exp = (over: Partial<ExportInfo> = {}): ExportInfo => ({
  id: "20260924-040124-8336a8", project_name: "OpticsConverter1", created_at: "", classes: ["Module"],
  train_images: 798, val_images: 252, formats: ["yolo_seg", "masks"], derived_from: null, ...over,
});
const pool = (over: Partial<BackgroundPool> = {}): BackgroundPool => ({
  name: "dtd", label: "DTD textures", images: 0, size_bytes: 625_239_812, fetching: null, ...over,
});

beforeEach(() => {
  Object.values(m).forEach((fn) => fn.mockReset());
  Object.keys(jobs).forEach((k) => delete jobs[k]);
  mUseJob.mockReset();
  mUseJob.mockImplementation((jobId: string | null) => (jobId === null ? idle : (jobs[jobId] ?? idle)));
  m.listBackgrounds.mockResolvedValue([pool()]);
});
afterEach(cleanup);

describe("AugmentPanel", () => {
  it("shows a note and no controls when there is no export", async () => {
    render(<AugmentPanel exportInfo={null} onCreated={vi.fn()} />);
    expect(await screen.findByText("Augmented copies need an export with Mask images.")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows a note and no controls when the export has no masks format", async () => {
    render(<AugmentPanel exportInfo={exp({ formats: ["yolo_seg"] })} onCreated={vi.fn()} />);
    expect(await screen.findByText("Augmented copies need an export with Mask images.")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows a note and no controls for a derived export", async () => {
    render(<AugmentPanel exportInfo={exp({ derived_from: "20260924-040124-8336a8" })} onCreated={vi.fn()} />);
    expect(
      await screen.findByText(
        "This is an augmented copy of 20260924-040124-8336a8. Make copies from the original export.",
      ),
    ).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("downloads the DTD pool when it is empty", async () => {
    m.fetchBackgrounds.mockResolvedValue({ job_id: "fetch1" });
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    const btn = await screen.findByRole("button", { name: "Download DTD textures (625 MB)" });
    fireEvent.click(btn);
    await waitFor(() => expect(m.fetchBackgrounds).toHaveBeenCalledWith("dtd"));
  });

  it("shows download progress and cancels the fetch job", async () => {
    m.fetchBackgrounds.mockResolvedValue({ job_id: "fetch1" });
    jobs.fetch1 = { ...idle, status: "running", done: 37, total: 100 };
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download DTD textures (625 MB)" }));
    expect(await screen.findByText("Downloading… 37 %")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(m.cancelJob).toHaveBeenCalledWith("fetch1"));
  });

  it("follows an in-progress fetch it did not start itself (pool.fetching)", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 0, fetching: "j9" })]);
    jobs.j9 = { ...idle, status: "running", done: 80, total: 200 };
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    expect(await screen.findByText("Downloading… 40 %")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Download DTD textures/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(m.cancelJob).toHaveBeenCalledWith("j9"));
  });

  it("shows the pool note once images are present", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    expect(await screen.findByText("DTD textures · 5,640 images")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Download DTD textures/ })).toBeNull();
  });

  it("makes an augmented copy with the chosen counts", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockResolvedValue({ job_id: "aug1" });
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    fireEvent.change(screen.getByLabelText("Composites per image"), { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    await waitFor(() =>
      expect(m.augmentExport).toHaveBeenCalledWith("20260924-040124-8336a8", {
        per_image: 5, max_objects: 2, seed: 0, pool: "dtd",
      }),
    );
  });

  it("disables Make augmented copy while the pool has no images", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 0 })]);
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByRole("button", { name: "Download DTD textures (625 MB)" });
    expect((screen.getByRole("button", { name: "Make augmented copy" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("calls onCreated once when the augment job finishes", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockResolvedValue({ job_id: "aug1" });
    const onCreated = vi.fn();
    render(<AugmentPanel exportInfo={exp()} onCreated={onCreated} />);
    await screen.findByText("DTD textures · 5,640 images");
    jobs.aug1 = { ...idle, status: "done", result: { export_id: "new" } };
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("new"));
    expect(onCreated).toHaveBeenCalledTimes(1);
  });

  it("shows an alert when the augment job errors", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockResolvedValue({ job_id: "aug1" });
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    jobs.aug1 = { ...idle, status: "error", error: "the source export has no train objects" };
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    expect((await screen.findByRole("alert")).textContent).toBe("the source export has no train objects");
  });

  it("shows an alert when augmentExport itself is refused with a 400", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockRejectedValue(new Error("already an augmented copy; augment the original export"));
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    expect((await screen.findByRole("alert")).textContent).toBe("already an augmented copy; augment the original export");
  });

  it("scopes the augment job per export: starting on A does not block or show progress for B", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockResolvedValue({ job_id: "augA" });
    const expA = exp({ id: "expA" });
    const expB = exp({ id: "expB" });
    const { rerender } = render(<AugmentPanel exportInfo={expA} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    jobs.augA = { ...idle, status: "running", done: 3, total: 10 };
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    expect(await screen.findByText("Making composites… 30 %")).toBeTruthy();

    rerender(<AugmentPanel exportInfo={expB} onCreated={vi.fn()} />);
    expect(screen.queryByText(/Making composites/)).toBeNull();
    const makeB = screen.getByRole("button", { name: "Make augmented copy" }) as HTMLButtonElement;
    expect(makeB.disabled).toBe(false);
    m.augmentExport.mockResolvedValue({ job_id: "augB" });
    fireEvent.click(makeB);
    await waitFor(() =>
      expect(m.augmentExport).toHaveBeenLastCalledWith("expB", { per_image: 3, max_objects: 2, seed: 0, pool: "dtd" }),
    );

    rerender(<AugmentPanel exportInfo={expA} onCreated={vi.fn()} />);
    expect(await screen.findByText("Making composites… 30 %")).toBeTruthy();
  });

  it("calls onCreated exactly once per job, keyed to its own export", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockResolvedValue({ job_id: "augA" });
    const expA = exp({ id: "expA" });
    const expB = exp({ id: "expB" });
    const onCreatedA = vi.fn();
    const { rerender } = render(<AugmentPanel exportInfo={expA} onCreated={onCreatedA} />);
    await screen.findByText("DTD textures · 5,640 images");
    jobs.augA = { ...idle, status: "done", result: { export_id: "newA" } };
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    await waitFor(() => expect(onCreatedA).toHaveBeenCalledWith("newA"));
    expect(onCreatedA).toHaveBeenCalledTimes(1);

    // switching away and back to expA must not re-fire onCreated for the same finished job
    const onCreatedB = vi.fn();
    rerender(<AugmentPanel exportInfo={expB} onCreated={onCreatedB} />);
    rerender(<AugmentPanel exportInfo={expA} onCreated={onCreatedA} />);
    expect(onCreatedA).toHaveBeenCalledTimes(1);
    expect(onCreatedB).not.toHaveBeenCalled();
  });

  it("shows a cancelled note when the augment job is cancelled", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockResolvedValue({ job_id: "aug1" });
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    jobs.aug1 = { ...idle, status: "cancelled", result: { cancelled: true } };
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    expect(await screen.findByText("Cancelled. Nothing was written.")).toBeTruthy();
  });
});
