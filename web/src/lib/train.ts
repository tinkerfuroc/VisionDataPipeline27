import type { ExportInfo, GpuQueue, MetricRow, Run, RunBest, StartMode, TrainStatus } from "../api/types";
import { exportLabel } from "./exportName";
import type { Pt } from "./chart";

export const MASK_MAP = "metrics/mAP50-95(M)";
export const MASK_MAP50 = "metrics/mAP50(M)";
export const BOX_MAP = "metrics/mAP50-95(B)";

/** Ids of the runs whose `best[key]` equals the maximum over all runs that have one. */
export function bestRunIds(runs: Run[], key: keyof RunBest): Set<string> {
  const scored = runs.filter((r) => r.best !== null && typeof r.best[key] === "number");
  if (scored.length === 0) return new Set();
  const top = Math.max(...scored.map((r) => r.best![key] as number));
  return new Set(scored.filter((r) => Math.abs((r.best![key] as number) - top) < 1e-9).map((r) => r.id));
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds - h * 3600) / 60);
  return `${h} h ${String(m).padStart(2, "0")} min`;
}

export function defaultGpu(status: Pick<TrainStatus, "gpus" | "default_gpu">): number | null {
  return status.default_gpu ?? status.gpus[0]?.index ?? null;
}

/** e.g. "cuda:0 · RTX 3080 Ti · default · ~34 TFLOPS · 9.1 GiB free · SAM3". The default tag comes right after the
 * name so a closed, narrow select still shows it; TFLOPS is an estimate (NVML's max clock). Options are already
 * server-sorted fastest first; this only formats one of them. */
export function gpuOptionLabel(g: GpuQueue, status: Pick<TrainStatus, "default_gpu" | "sam3_gpu">): string {
  const parts = [`cuda:${g.index}`, g.name.replace(/^NVIDIA GeForce /, "")];
  if (g.index === status.default_gpu) parts.push("default");
  parts.push(`~${Math.round(g.tflops)} TFLOPS`);
  if (g.memory_free !== null) parts.push(`${(g.memory_free / 2 ** 30).toFixed(1)} GiB free`);
  if (g.index === status.sam3_gpu) parts.push("SAM3");
  return parts.join(" · ");
}

export function polygonPoints(flat: number[], w: number, h: number): string {
  const out: string[] = [];
  for (let i = 0; i + 1 < flat.length; i += 2) out.push(`${+(flat[i] * w).toFixed(1)},${+(flat[i + 1] * h).toFixed(1)}`);
  return out.join(" ");
}

export function metricSeries(rows: MetricRow[], key: string): Pt[] {
  return rows.filter((r) => typeof r[key] === "number").map((r) => [r.epoch, r[key]]);
}

export function statusText(run: Run): string {
  switch (run.status) {
    case "queued":
      return "Queued";
    case "running": {
      const p = run.progress;
      if (p === null || p.phase === "setup") return "Starting";
      if (p.phase === "val") return `Validating epoch ${p.epoch}`;
      if (p.phase === "predict") return "Predicting val images";
      return `Epoch ${p.epoch}/${p.epochs}`;
    }
    case "done":
      return run.completed_early ? `Done (stopped at epoch ${run.epochs_done} of ${run.params.epochs})` : "Done";
    case "error":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    case "interrupted":
      return "Interrupted";
  }
}

export const runLabel = (run: Pick<Run, "id" | "model">) => `${run.model} · ${run.id.slice(-4)}`;

export function startProblem(o: { available: boolean; error: string | null; exportId: string | null; picked: number }): string | null {
  if (!o.available) return o.error ?? "Training is not available on this server.";
  if (o.exportId === null) return "Choose an export to train on.";
  if (o.picked === 0) return "Tick at least one model.";
  return null;
}

/** Runs offered as a "Start from" source in the New-runs form: any run that can be continued or
 * fine-tuned, in the order `runs` is already in (the server lists them newest first). */
export function checkpointSources(runs: Run[]): Run[] {
  return runs.filter((r) => r.continuable || r.finetunable);
}

/** "yolo26s-seg · <project/export label> · epoch 70/200 · <status>", for a "Start from" option. */
export function sourceRunLabel(run: Run, exports: ExportInfo[]): string {
  const exp = exports.find((e) => e.id === run.export_id);
  const label = exp !== undefined && exp.name !== null ? exportLabel(exp) : run.project_name;
  return `${run.model} · ${label} · epoch ${run.epochs_done}/${run.params.epochs} · ${statusText(run)}`;
}

export const continueModeLabel = (run: Run): string =>
  `Continue (from epoch ${run.epochs_done} of ${run.params.epochs}: same optimizer and schedule, remaining epochs on the new data)`;

export const finetuneModeLabel = (run: Run): string =>
  `Fine-tune (fresh schedule from best.pt, epoch ${run.best?.epoch ?? run.epochs_done})`;

/** Why `mode` cannot be used to train from `run`, or null when it can. */
export function modeDisabledReason(run: Run, mode: StartMode): string | null {
  if (mode === "continue") {
    return run.continuable ? null : "only a stopped run (cancelled, interrupted or error) with last.pt can be continued";
  }
  return run.finetunable ? null : "needs a best.pt (no improving epoch recorded yet)";
}

/** The default mode for a newly chosen source run: Continue when it is allowed, else Fine-tune. */
export const defaultMode = (run: Run): StartMode => (run.continuable ? "continue" : "finetune");

/** Whether `exp`'s classes are compatible with `run`'s; a run with unknown classes (no data.yaml
 * yet, e.g. still queued) never blocks -- the server itself would 409 on an actual mismatch. */
export function classesCompatible(exp: ExportInfo, run: Run): boolean {
  if (run.classes == null) return true;
  return exp.classes.length === run.classes.length && exp.classes.every((c, i) => c === run.classes![i]);
}

/** The Runs table's small note under a run started from a checkpoint's model, e.g. "continued from
 * 20260927-0722…" or "fine-tuned from 20260927-0722…"; null for an ordinary run. */
export function startModeNote(run: Run): string | null {
  if (run.source_run == null || run.start_mode == null) return null;
  const short = run.source_run.slice(0, 13);
  return run.start_mode === "continue" ? `continued from ${short}…` : `fine-tuned from ${short}…`;
}

/** The run detail header's version of `startModeNote`: for "continue", also names the epoch
 * training switched to the new data at. */
export function startModeHeaderNote(run: Run): string | null {
  const note = startModeNote(run);
  if (note === null) return null;
  return run.start_mode === "continue" ? `${note} at epoch ${run.continued_from_epoch ?? "?"}` : note;
}
