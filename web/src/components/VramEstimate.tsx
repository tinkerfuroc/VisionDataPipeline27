import { type KeyboardEvent, useState } from "react";
import { api } from "../api/rest";
import type { Run, TrainStatus, VramEstimate } from "../api/types";
import { useJob } from "../hooks/useJob";
import { type VramFit, vramBusyReason, vramFit, vramLabel } from "../lib/train";

const FIT_TEXT: Record<VramFit, string> = { fits: "fits", tight: "tight", no: "doesn't fit" };
const FIT_CLASS: Record<VramFit, string> = { fits: "text-dim", tight: "text-ink", no: "text-danger" };
const gibFree = (bytes: number) => `${(bytes / 2 ** 30).toFixed(1)} GiB`;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

function FitCell({ fit }: { fit: VramFit | null }) {
  return <td className={fit === null ? "text-dim" : FIT_CLASS[fit]}>{fit === null ? "—" : FIT_TEXT[fit]}</td>;
}

type Props = {
  status: TrainStatus;
  runs: Run[];
  /** the model to estimate for; null = none picked yet */
  model: string | null;
  imgsz: number;
  gpu: number;
  /** a row click sets the batch field; omitted when the batch is locked (Continue keeps the run's own) */
  onPickBatch?: (batch: number) => void;
};

/** "Estimate VRAM" next to the batch field: measures (or recalls) VRAM per batch size for the model, image size and
 * GPU, then shows each against that GPU's free memory now, and with SAM3 paused when SAM3 shares the GPU. An answer
 * is only shown for the model/image size/GPU it was asked for. */
export function VramEstimatePanel({ status, runs, model, imgsz, gpu, onPickBatch }: Props) {
  const key = `${model}|${imgsz}|${gpu}`;
  const [asked, setAsked] = useState<{ key: string; jobId: string } | null>(null);
  const [requestError, setRequestError] = useState<{ key: string; message: string } | null>(null);
  const [requesting, setRequesting] = useState(false);
  const job = useJob(asked?.jobId ?? null);
  const current = asked !== null && asked.key === key;
  const running = requesting || (current && (job.status === "pending" || job.status === "running"));
  const busy = vramBusyReason(status, runs, gpu);
  const disabledReason = model === null ? "Pick a model first." : busy;

  const estimate = async (remeasure: boolean) => {
    if (model === null) return;
    setRequesting(true);
    setRequestError(null);
    try {
      const { job_id } = await api.vramEstimate({ model, imgsz, gpu, ...(remeasure ? { remeasure: true } : {}) });
      setAsked({ key, jobId: job_id });
    } catch (e) {
      setAsked(null);
      setRequestError({ key, message: message(e) });
    } finally {
      setRequesting(false);
    }
  };

  const result = current && job.status === "done" ? (job.result as unknown as VramEstimate) : null;
  const error = requestError?.key === key ? requestError.message : current && job.status === "error" ? job.error : null;
  const g = status.gpus.find((gg) => gg.index === gpu);
  const free = g?.memory_free ?? result?.free_bytes ?? null;
  const sam3 = status.sam3_gpu === gpu && status.sam3_vram_bytes != null ? status.sam3_vram_bytes : null;
  const freePaused = free !== null && sam3 !== null ? free + sam3 : null;

  return (
    <div className="flex flex-col gap-1.5 text-sm">
      <button type="button" className="btn self-end" disabled={disabledReason !== null || running}
        title={disabledReason ?? undefined} onClick={() => void estimate(false)}>
        Estimate VRAM
      </button>
      {running && (
        <p className="note">
          {current && job.total !== null ? `Measuring… ${job.done} of ${job.total} batch sizes` : "Measuring…"}
        </p>
      )}
      {error !== null && <p role="alert" className="alert">{error}</p>}
      {result !== null && (
        <>
          <table className="w-full text-left">
            <caption className="note text-left">
              {result.model} at {result.imgsz} px on cuda:{result.gpu}
              {result.cached ? ", measured earlier" : ""}
              {onPickBatch !== undefined ? " · click a row to use that batch" : ""}
            </caption>
            <thead>
              <tr className="text-dim">
                <th>Batch</th>
                <th>VRAM</th>
                <th>cuda:{gpu} now{free !== null ? ` (${gibFree(free)} free)` : ""}</th>
                {freePaused !== null && <th>With SAM3 paused ({gibFree(freePaused)})</th>}
              </tr>
            </thead>
            <tbody>
              {result.estimates.map((e) => {
                const pick = onPickBatch;
                const fit = vramFit(e.bytes, free);
                const row = pick === undefined ? {} : {
                  "aria-label": `Use batch ${e.batch}: ${vramLabel(e.bytes)}, ${fit === null ? "unknown" : FIT_TEXT[fit]}`,
                  tabIndex: 0,
                  className: "cursor-pointer hover:bg-line",
                  onClick: () => pick(e.batch),
                  onKeyDown: (ev: KeyboardEvent) => {
                    if (ev.key === "Enter" || ev.key === " ") {
                      ev.preventDefault();
                      pick(e.batch);
                    }
                  },
                };
                return (
                  <tr key={e.batch} {...row}>
                    <td>{e.batch}</td>
                    <td>{vramLabel(e.bytes)}</td>
                    <FitCell fit={fit} />
                    {freePaused !== null && <FitCell fit={vramFit(e.bytes, freePaused)} />}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <button type="button" className="self-start text-xs text-dim underline" disabled={disabledReason !== null || running}
            title={disabledReason ?? undefined} onClick={() => void estimate(true)}>
            Re-measure
          </button>
        </>
      )}
    </div>
  );
}
