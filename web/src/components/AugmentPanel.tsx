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
  const [fetchJobId, setFetchJobId] = useState<string | null>(null);
  const [fetchPhase, setFetchPhase] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [augmentJobId, setAugmentJobId] = useState<string | null>(null);
  const [augmentError, setAugmentError] = useState<string | null>(null);
  const [perImage, setPerImage] = useState(3);
  const [maxObjects, setMaxObjects] = useState(2);
  const [handled, setHandled] = useState<string | null>(null); // last augment job id whose "done" we already reported

  const loadPools = useCallback(() => {
    api.listBackgrounds().then(setPools, (e) => setPoolsError(message(e)));
  }, []);
  useEffect(() => {
    loadPools();
  }, [loadPools]);

  const fetchJob = useJob(
    fetchJobId,
    useCallback((e: JobEvent) => {
      if (e.type === "progress" && typeof e.phase === "string") setFetchPhase(e.phase);
    }, []),
  );
  const augmentJob = useJob(augmentJobId);

  useEffect(() => {
    if (fetchJob.status === "done" || fetchJob.status === "cancelled") loadPools();
  }, [fetchJob.status, loadPools]);

  useEffect(() => {
    if (augmentJob.status !== "done" || augmentJobId === null || handled === augmentJobId) return;
    const id = augmentJob.result?.export_id;
    if (typeof id === "string") {
      setHandled(augmentJobId);
      onCreated(id);
    }
  }, [augmentJob.status, augmentJob.result, augmentJobId, handled, onCreated]);

  if (exportInfo === null) return <Note text="Augmented copies need an export with Mask images." />;
  if (exportInfo.derived_from !== null) {
    return (
      <Note text={`This is an augmented copy of ${exportInfo.derived_from}. Make copies from the original export.`} />
    );
  }
  if (!exportInfo.formats.includes("masks")) return <Note text="Augmented copies need an export with Mask images." />;

  const pool = pools.find((p) => p.name === POOL) ?? null;
  const poolReady = pool !== null && pool.images > 0;
  const fetchRunning = isBusy(fetchJob.status);
  const augmentRunning = isBusy(augmentJob.status);

  const onDownload = () => {
    setFetchError(null);
    setFetchPhase(null);
    api.fetchBackgrounds(POOL).then(
      (r) => setFetchJobId(r.job_id),
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
    setAugmentError(null);
    api.augmentExport(exportInfo.id, { per_image: perImage, max_objects: maxObjects, seed: 0, pool: POOL }).then(
      (r) => setAugmentJobId(r.job_id),
      (e) => setAugmentError(message(e)),
    );
  };

  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Augmented copy</h2>
      {poolsError !== null && <p role="alert" className="alert">{poolsError}</p>}
      {pool !== null &&
        (poolReady ? (
          <p className="note">
            {pool.label} · {pool.images.toLocaleString()} images
          </p>
        ) : fetchRunning ? (
          <div className="flex flex-col gap-1">
            <progress value={fetchJob.done} max={fetchJob.total ?? 1} />
            <div className="flex items-center justify-between gap-2 text-sm text-dim">
              <span>
                {fetchPhase === "extract" ? "Extracting" : "Downloading"}… {pct(fetchJob.done, fetchJob.total)} %
              </span>
              <button type="button" className="btn" onClick={onCancelFetch}>Cancel</button>
            </div>
          </div>
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
