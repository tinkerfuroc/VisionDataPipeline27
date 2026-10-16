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
  id: "20260924-040124-8336a8", name: null, project_name: "OpticsConverter1", created_at: "", classes: ["Module"],
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
        per_image: 5, max_objects: 2, seed: 0, pool: "dtd", lighting: 0,
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
    expect(await screen.findByText("Making augmented copy… 30 %")).toBeTruthy();

    rerender(<AugmentPanel exportInfo={expB} onCreated={vi.fn()} />);
    expect(screen.queryByText(/Making augmented copy/)).toBeNull();
    const makeB = screen.getByRole("button", { name: "Make augmented copy" }) as HTMLButtonElement;
    expect(makeB.disabled).toBe(false);
    m.augmentExport.mockResolvedValue({ job_id: "augB" });
    fireEvent.click(makeB);
    await waitFor(() =>
      expect(m.augmentExport).toHaveBeenLastCalledWith("expB", { per_image: 3, max_objects: 2, seed: 0, pool: "dtd", lighting: 0 }),
    );

    rerender(<AugmentPanel exportInfo={expA} onCreated={vi.fn()} />);
    expect(await screen.findByText("Making augmented copy… 30 %")).toBeTruthy();
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

  it("makes a lighting-only copy without the texture pool", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 0 })]);
    m.augmentExport.mockResolvedValue({ job_id: "aug1" });
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByRole("button", { name: "Download DTD textures (625 MB)" });
    fireEvent.change(screen.getByLabelText("Composites per image"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Lighting variants per image"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    await waitFor(() =>
      expect(m.augmentExport).toHaveBeenCalledWith("20260924-040124-8336a8", {
        per_image: 0, max_objects: 2, seed: 0, pool: "dtd", lighting: 2,
      }),
    );
  });

  it("disables Make augmented copy when it would add nothing", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    fireEvent.change(screen.getByLabelText("Composites per image"), { target: { value: "0" } });
    expect((screen.getByRole("button", { name: "Make augmented copy" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Choose at least one composite or lighting variant per image.")).toBeTruthy();
  });

  it("sends the chosen seed with the augment request", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    m.augmentExport.mockResolvedValue({ job_id: "aug1" });
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "7" } });
    fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
    await waitFor(() =>
      expect(m.augmentExport).toHaveBeenCalledWith("20260924-040124-8336a8", {
        per_image: 3, max_objects: 2, seed: 7, pool: "dtd", lighting: 0,
      }),
    );
  });

  it("shows an alert when the panel's own fetch job errors, and offers Download again", async () => {
    m.fetchBackgrounds.mockResolvedValue({ job_id: "fetch1" });
    const { rerender } = render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download DTD textures (625 MB)" }));
    await waitFor(() => expect(m.fetchBackgrounds).toHaveBeenCalled());
    jobs.fetch1 = { ...idle, status: "error", error: "the archive holds no images under dtd/images/" };
    m.listBackgrounds.mockResolvedValue([pool({ images: 0, fetching: null })]);
    rerender(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    expect((await screen.findByRole("alert")).textContent).toBe(
      "the archive holds no images under dtd/images/",
    );
    expect(await screen.findByRole("button", { name: "Download DTD textures (625 MB)" })).toBeTruthy();
  });

  it("follows a later foreign fetch once the panel's own fetch job is done", async () => {
    m.fetchBackgrounds.mockResolvedValue({ job_id: "fetch1" });
    const { rerender } = render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download DTD textures (625 MB)" }));
    await waitFor(() => expect(m.fetchBackgrounds).toHaveBeenCalled());

    // fetch1 (this panel's own job) finishes just as another tab (or the CLI) starts a new one; the
    // pool list the "done" transition re-fetches already reports it. The panel must follow j9, not stay
    // pinned to the finished fetch1 (which `ownFetchJobId` would otherwise still hold).
    jobs.fetch1 = { ...idle, status: "done", result: { images: 5640 } };
    jobs.j9 = { ...idle, status: "running", done: 10, total: 100 };
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640, fetching: "j9" })]);
    rerender(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Downloading… 10 %")).toBeTruthy());
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

  it("notes that an export without COCO cuts objects per class region", async () => {
    m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
    render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
    expect(await screen.findByText(/has no COCO format, so composites cut objects per class region/)).toBeTruthy();
    cleanup();
    render(<AugmentPanel exportInfo={exp({ formats: ["yolo_seg", "coco", "masks"] })} onCreated={vi.fn()} />);
    await screen.findByText("DTD textures · 5,640 images");
    expect(screen.queryByText(/has no COCO format/)).toBeNull();
  });

  describe("limits", () => {
    const ready = async () => {
      m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
      m.augmentExport.mockResolvedValue({ job_id: "aug1" });
      render(<AugmentPanel exportInfo={exp()} onCreated={vi.fn()} />);
      await screen.findByText("DTD textures · 5,640 images");
    };
    const limits = (over: Record<string, unknown>) => ({
      max_train_images: null, time_limit_s: null, requested: { composites: 2394, lighting: 0 },
      planned: { composites: 2394, lighting: 0 }, produced: { composites: 2394, lighting: 0 }, stopped_by: null, ...over,
    });

    it("omits both limit keys by default", async () => {
      await ready();
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      await waitFor(() => expect(m.augmentExport).toHaveBeenCalled());
      const body = m.augmentExport.mock.calls[0][1];
      expect("max_train_images" in body).toBe(false);
      expect("time_limit_s" in body).toBe(false);
    });

    it("sends the image cap and the time limit in seconds, and omits a cleared one", async () => {
      await ready();
      fireEvent.change(screen.getByLabelText("At most training images"), { target: { value: "2000" } });
      fireEvent.change(screen.getByLabelText("Stop after minutes"), { target: { value: "10" } });
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      await waitFor(() =>
        expect(m.augmentExport).toHaveBeenLastCalledWith("20260924-040124-8336a8", {
          per_image: 3, max_objects: 2, seed: 0, pool: "dtd", lighting: 0, max_train_images: 2000, time_limit_s: 600,
        }),
      );
      fireEvent.change(screen.getByLabelText("At most training images"), { target: { value: "" } });
      fireEvent.change(screen.getByLabelText("Stop after minutes"), { target: { value: "" } });
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      await waitFor(() => expect(m.augmentExport).toHaveBeenCalledTimes(2));
      const body = m.augmentExport.mock.calls[1][1];
      expect("max_train_images" in body).toBe(false);
      expect("time_limit_s" in body).toBe(false);
    });

    it("shows the server's 400 about the cap in the error spot", async () => {
      await ready();
      m.augmentExport.mockRejectedValue(
        new Error("this export already has 798 training frames; set a limit above that, or leave it empty"),
      );
      fireEvent.change(screen.getByLabelText("At most training images"), { target: { value: "500" } });
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      expect((await screen.findByRole("alert")).textContent).toContain("this export already has 798 training frames");
    });

    it("reports a time-limited run", async () => {
      await ready();
      jobs.aug1 = {
        ...idle, status: "done",
        result: {
          export_id: "new",
          limits: limits({
            time_limit_s: 600, stopped_by: "time",
            planned: { composites: 2394, lighting: 1596 }, produced: { composites: 1200, lighting: 800 },
          }),
        },
      };
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      expect(
        await screen.findByText(
          "Stopped at the 10-minute limit: 1,200 composites and 800 lighting copies (of 2,394 and 1,596 asked for).",
        ),
      ).toBeTruthy();
    });

    it("reports an image-capped run", async () => {
      await ready();
      jobs.aug1 = {
        ...idle, status: "done",
        result: {
          export_id: "new",
          limits: limits({
            max_train_images: 2000, stopped_by: "images",
            produced: { composites: 800, lighting: 402 },
          }),
        },
      };
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      expect(await screen.findByText("Capped at 2,000 training images: 800 composites and 402 lighting copies.")).toBeTruthy();
    });

    it("shows no limit line when nothing stopped the run", async () => {
      m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
      m.augmentExport.mockResolvedValue({ job_id: "aug1" });
      const onCreated = vi.fn();
      render(<AugmentPanel exportInfo={exp()} onCreated={onCreated} />);
      await screen.findByText("DTD textures · 5,640 images");
      jobs.aug1 = { ...idle, status: "done", result: { export_id: "new", limits: limits({}) } };
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      await waitFor(() => expect(onCreated).toHaveBeenCalledWith("new"));
      expect(screen.queryByText(/Stopped at the|Capped at/)).toBeNull();
    });

    it("pins the time-limit line's 'asked for' numbers to planned, not requested", async () => {
      await ready();
      jobs.aug1 = {
        ...idle, status: "done",
        result: {
          export_id: "new",
          limits: limits({
            time_limit_s: 90, stopped_by: "time",
            requested: { composites: 5000, lighting: 3000 }, planned: { composites: 3000, lighting: 1000 },
            produced: { composites: 10, lighting: 4 },
          }),
        },
      };
      fireEvent.click(screen.getByRole("button", { name: "Make augmented copy" }));
      expect(
        await screen.findByText("Stopped at the 2-minute limit: 10 composites and 4 lighting copies (of 3,000 and 1,000 asked for)."),
      ).toBeTruthy();
    });

    it("says how a selected derived copy was cut, from its persisted limits", async () => {
      const copy = exp({
        id: "copy", derived_from: "20260924-040124-8336a8",
        augment_limits: limits({
          max_train_images: 2000, stopped_by: "images",
          planned: { composites: 721, lighting: 481 }, produced: { composites: 721, lighting: 481 },
        }),
      });
      render(<AugmentPanel exportInfo={copy} onCreated={vi.fn()} />);
      expect(await screen.findByText(/This is an augmented copy of 20260924-040124-8336a8/)).toBeTruthy();
      expect(screen.getByText("Capped at 2,000 training images: 721 composites and 481 lighting copies.")).toBeTruthy();
    });

    it("says nothing extra for a derived copy nothing cut short", async () => {
      render(<AugmentPanel exportInfo={exp({ derived_from: "src", augment_limits: limits({}) })} onCreated={vi.fn()} />);
      await screen.findByText(/This is an augmented copy of src/);
      expect(screen.queryByText(/Stopped at the|Capped at/)).toBeNull();
    });

    it("keeps the minutes field within 1 to 1440 whole minutes", async () => {
      await ready();
      const field = screen.getByLabelText("Stop after minutes") as HTMLInputElement;
      expect(field.max).toBe("1440");
      expect(field.step).toBe("1");
      const make = screen.getByRole("button", { name: "Make augmented copy" }) as HTMLButtonElement;
      for (const bad of ["0.1", "1500", "0"]) {
        fireEvent.change(field, { target: { value: bad } });
        expect(screen.getByText("Stop after takes 1 to 1440 minutes.")).toBeTruthy();
        expect(make.disabled).toBe(true);
      }
      fireEvent.change(field, { target: { value: "2.6" } });
      expect(screen.queryByText(/takes 1 to 1440 minutes/)).toBeNull();
      fireEvent.click(make);
      await waitFor(() => expect(m.augmentExport).toHaveBeenCalled());
      expect(m.augmentExport.mock.calls[0][1].time_limit_s).toBe(180); // rounded to whole minutes
    });

    it("clears the limit fields when another source export is selected", async () => {
      m.listBackgrounds.mockResolvedValue([pool({ images: 5640 })]);
      const { rerender } = render(<AugmentPanel exportInfo={exp({ id: "expA" })} onCreated={vi.fn()} />);
      await screen.findByText("DTD textures · 5,640 images");
      fireEvent.change(screen.getByLabelText("At most training images"), { target: { value: "2000" } });
      fireEvent.change(screen.getByLabelText("Stop after minutes"), { target: { value: "10" } });
      fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "7" } });
      rerender(<AugmentPanel exportInfo={exp({ id: "expB", train_images: 3000 })} onCreated={vi.fn()} />);
      await waitFor(() => expect((screen.getByLabelText("At most training images") as HTMLInputElement).value).toBe(""));
      expect((screen.getByLabelText("Stop after minutes") as HTMLInputElement).value).toBe("");
      expect((screen.getByLabelText("Seed") as HTMLInputElement).value).toBe("7"); // the generic options carry over
    });
  });
});
