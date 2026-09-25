import { useCallback, useEffect, useState } from "react";
import { api } from "../api/rest";
import type { BackgroundPool, ExportInfo, JobEvent } from "../api/types";
import { useJob } from "../hooks/useJob";

const POOL = "dtd";
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));
const isBusy = (status: string) => status === "pending" || status === "running";
const pct = (done: number, total: number | null) => (total ? Math.round((done / total) * 100) : 0);

type Props = { exportInfo: ExportInfo | null; onCreated: (exportId: string) => void };

function Note({ text }: { text: string }) {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Augmented copy</h2>
      <p className="note">{text}</p>
    </section>
  );
}

export function AugmentPanel({ exportInfo, onCreated }: Props) {
  const [pools, setPools] = useState<BackgroundPool[]>([]);
  const [poolsError, setPoolsError] = useState<string | null>(null);
  // A fetch this panel itself started; while null, a fetch reported by the server (`pool.fetching`, started by
  // the CLI, another tab, or before a reload) is followed instead, so the Download button never fires a 409.
  const [ownFetchJobId, setOwnFetchJobId] = useState<string | null>(null);
  const [fetchPhase, setFetchPhase] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  // Augment jobs are scoped per source export, so switching the selected export never shows another
  // export's progress or disables its own Make button.
  const [augmentJobs, setAugmentJobs] = useState<Record<string, string>>({});
  const [augmentErrors, setAugmentErrors] = useState<Record<string, string>>({});
  // Last job id per export whose "done" was already reported to `onCreated`, so a re-render (or switching
  // back to an export whose job already finished) never calls it twice for the same job.
  const [handled, setHandled] = useState<Record<string, string>>({});
  const [perImage, setPerImage] = useState(3);
  const [maxObjects, setMaxObjects] = useState(2);
  const [seed, setSeed] = useState(0);

  const loadPools = useCallback(() => {
    api.listBackgrounds().then(setPools, (e) => setPoolsError(message(e)));
  }, []);
  useEffect(() => {
    loadPools();
  }, [loadPools]);

  const pool = pools.find((p) => p.name === POOL) ?? null;
  const fetchJobId = ownFetchJobId ?? pool?.fetching ?? null;
  const augmentJobId = exportInfo !== null ? (augmentJobs[exportInfo.id] ?? null) : null;

  const fetchJob = useJob(
    fetchJobId,
    useCallback((e: JobEvent) => {
      if (e.type === "progress" && typeof e.phase === "string") setFetchPhase(e.phase);
    }, []),
  );
  const augmentJob = useJob(augmentJobId);

  // Any terminal fetch job (own or one this panel is merely following) may have changed the pool, so
  // always refresh it. Only the panel's OWN job also clears `ownFetchJobId`: once it does, a later
  // `pool.fetching` reported by the server (the CLI, another tab, or a reload) is followed again instead
  // of staying stuck on this finished job's id. Its error is captured into `fetchError` -- the same state
  // the alert below already renders -- since clearing the id would otherwise drop it from `fetchJob` too.
  useEffect(() => {
    if (!["done", "cancelled", "error"].includes(fetchJob.status)) return;
    loadPools();
    if (ownFetchJobId === null) return;
    if (fetchJob.status === "error") setFetchError(fetchJob.error ?? "the download failed");
    setOwnFetchJobId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchJob.status, fetchJob.error, ownFetchJobId, loadPools]);

  useEffect(() => {
    if (exportInfo === null || augmentJobId === null) return;
    if (augmentJob.status !== "done" || handled[exportInfo.id] === augmentJobId) return;
    const id = augmentJob.result?.export_id;
    if (typeof id === "string") {
      const exportId = exportInfo.id;
      setHandled((h) => ({ ...h, [exportId]: augmentJobId }));
      onCreated(id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exportInfo?.id, augmentJobId, augmentJob.status, augmentJob.result, handled, onCreated]);

  if (exportInfo === null) return <Note text="Augmented copies need an export with Mask images." />;
  if (exportInfo.derived_from !== null) {
    return (
      <Note text={`This is an augmented copy of ${exportInfo.derived_from}. Make copies from the original export.`} />
    );
  }
  if (!exportInfo.formats.includes("masks")) return <Note text="Augmented copies need an export with Mask images." />;

  const poolReady = pool !== null && pool.images > 0;
  const fetchRunning = isBusy(fetchJob.status);
  const augmentRunning = isBusy(augmentJob.status);
  const augmentError = augmentErrors[exportInfo.id] ?? null;

  const onDownload = () => {
    setFetchError(null);
    setFetchPhase(null);
    api.fetchBackgrounds(POOL).then(
      (r) => setOwnFetchJobId(r.job_id),
      (e) => setFetchError(message(e)),
    );
  };
  const onCancelFetch = () => {
    if (fetchJobId !== null) void api.cancelJob(fetchJobId);
  };
  const onCancelAugment = () => {
    if (augmentJobId !== null) void api.cancelJob(augmentJobId);
  };
  const onMake = () => {
    if (!poolReady || augmentRunning) return;
    const exportId = exportInfo.id;
    setAugmentErrors((errs) => {
      const { [exportId]: _dropped, ...rest } = errs;
      return rest;
    });
    api.augmentExport(exportId, { per_image: perImage, max_objects: maxObjects, seed, pool: POOL }).then(
      (r) => setAugmentJobs((j) => ({ ...j, [exportId]: r.job_id })),
      (e) => setAugmentErrors((errs) => ({ ...errs, [exportId]: message(e) })),
    );
  };

  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Augmented copy</h2>
      {poolsError !== null && <p role="alert" className="alert">{poolsError}</p>}
      {pool !== null &&
        (fetchRunning ? (
          <div className="flex flex-col gap-1">
            <progress value={fetchJob.done} max={fetchJob.total ?? 1} />
            <div className="flex items-center justify-between gap-2 text-sm text-dim">
              <span>
                {fetchPhase === "extract" ? "Extracting" : "Downloading"}… {pct(fetchJob.done, fetchJob.total)} %
              </span>
              <button type="button" className="btn" onClick={onCancelFetch}>Cancel</button>
            </div>
          </div>
        ) : poolReady ? (
          <p className="note">
            {pool.label} · {pool.images.toLocaleString()} images
          </p>
        ) : (
          <button type="button" className="btn" onClick={onDownload}>
            Download DTD textures ({Math.round(pool.size_bytes / 1e6)} MB)
          </button>
        ))}
      {fetchError !== null && <p role="alert" className="alert">{fetchError}</p>}
      {fetchJob.status === "cancelled" && <p className="note">Cancelled. Nothing was downloaded.</p>}

      <label className="flex items-center justify-between gap-2 text-sm text-dim">
        Composites per image
        <input aria-label="Composites per image" type="number" className="field w-20" min={1} max={10}
          value={perImage} onChange={(e) => setPerImage(Math.min(10, Math.max(1, Number(e.target.value) || 1)))} />
      </label>
      <label className="flex items-center justify-between gap-2 text-sm text-dim">
        Objects per image
        <input aria-label="Objects per image" type="number" className="field w-20" min={1} max={3}
          value={maxObjects} onChange={(e) => setMaxObjects(Math.min(3, Math.max(1, Number(e.target.value) || 1)))} />
      </label>
      <label className="flex items-center justify-between gap-2 text-sm text-dim">
        Seed
        <input aria-label="Seed" type="number" className="field w-20" min={0}
          value={seed} onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))} />
      </label>

      {augmentRunning && (
        <div className="flex flex-col gap-1">
          <progress value={augmentJob.done} max={augmentJob.total ?? 1} />
          <div className="flex items-center justify-between gap-2 text-sm text-dim">
            <span>Making composites… {pct(augmentJob.done, augmentJob.total)} %</span>
            <button type="button" className="btn" onClick={onCancelAugment}>Cancel</button>
          </div>
        </div>
      )}
      <button type="button" className="btn btn-on" disabled={!poolReady || augmentRunning} onClick={onMake}>
        Make augmented copy
      </button>
      {augmentError !== null && <p role="alert" className="alert">{augmentError}</p>}
      {augmentJob.status === "error" && augmentJob.error !== null && (
        <p role="alert" className="alert">{augmentJob.error}</p>
      )}
      {augmentJob.status === "cancelled" && <p className="note">Cancelled. Nothing was written.</p>}
    </section>
  );
}
