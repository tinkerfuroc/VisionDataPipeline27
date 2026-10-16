import type { ExportInfo, GpuQueue, MetricRow, Run, RunBest, RunSpeed, StartMode, TrainParams, TrainStatus } from "../api/types";
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
  // a stopped run's `best` is null (its best epoch is only summarised when a run finishes), so name the epoch only when known
  `Fine-tune (fresh schedule from best.pt${run.best != null ? `, epoch ${run.best.epoch}` : ""})`;

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

export type VramFit = "fits" | "tight" | "no" | "unknown";

/** An estimate against free memory: "tight" within 10 % of it. A batch that did not fit when measured needs more
 * than `over_bytes`: it doesn't fit in `free` up to that, and is "unknown" (worth re-measuring) beyond. Null when free
 * memory is unknown. */
export function vramFit(e: { bytes: number | null; over_bytes?: number | null }, free: number | null): VramFit | null {
  if (free === null) return null;
  if (e.bytes === null) return e.over_bytes != null && free > e.over_bytes ? "unknown" : "no";
  if (e.bytes > free) return "no";
  return e.bytes > free * 0.9 ? "tight" : "fits";
}

export const vramLabel = (bytes: number | null): string =>
  bytes === null ? "—" : `about ${(bytes / 2 ** 30).toFixed(1)} GiB`;

/** Why "Estimate VRAM" is disabled for `gpu`, mirroring the server's 409: anything queued or running there. */
export function vramBusyReason(status: Pick<TrainStatus, "gpus">, runs: Run[], gpu: number): string | null {
  const g = status.gpus.find((gg) => gg.index === gpu);
  if (g === undefined) return null;
  const why = "; estimate VRAM once it is idle (measuring now would be wrong and could make the run run out of memory).";
  if (g.running !== null) return `cuda:${gpu} is ${g.running_kind === "onnx" ? "exporting ONNX" : "training"}${why}`;
  if (g.queue.length > 0) return `cuda:${gpu} has a run queued${why}`;
  if (runs.some((r) => r.status === "running" && r.gpu === gpu)) return `cuda:${gpu} is training${why}`;
  return null;
}

const count = (n: number) => n.toLocaleString("en-US");

/** The params as sent to the server: an empty (all) images-per-epoch size is left out rather than sent as null. */
export function sendableParams(params: TrainParams): TrainParams {
  const out = { ...params };
  if (out.train_images_per_epoch == null) delete out.train_images_per_epoch;
  if (out.val_images_per_epoch == null) delete out.val_images_per_epoch;
  return out;
}

/** The newest measured throughput (`runs` come newest first) of a run with this model and image size on this GPU. */
export function measuredSpeed(runs: Run[], model: string, imgsz: number, gpu: number): RunSpeed | null {
  const hit = runs.find((r) => r.model === model && r.params.imgsz === imgsz && r.gpu === gpu && r.speed?.train_images_per_s != null);
  return hit?.speed ?? null;
}

/** The Train form's estimate under the images-per-epoch fields: the epoch size (with minutes when a run has measured
 * this model/image size/GPU), how many full passes over the train set the epochs make, and the validation per epoch. */
export function epochEstimate(o: {
  trainImages: number; valImages: number; epochs: number; trainPerEpoch: number | null | undefined;
  valPerEpoch: number | null | undefined; speed: RunSpeed | null;
}): string[] {
  const n = Math.min(o.trainPerEpoch ?? o.trainImages, o.trainImages);
  const m = Math.min(o.valPerEpoch ?? o.valImages, o.valImages);
  const minutes = o.speed?.train_images_per_s ? ` (≈ ${formatDuration(n / o.speed.train_images_per_s)})` : "";
  const passes = Math.round((o.epochs * n / Math.max(o.trainImages, 1)) * 10) / 10;
  const valTime = o.speed?.val_images_per_s ? ` (≈ ${formatDuration(m / o.speed.val_images_per_s)})` : "";
  const val = m < o.valImages ? `${count(m)} of ${count(o.valImages)}` : `all ${count(o.valImages)}`;
  return [
    `1 epoch = ${count(n)} images${minutes} · ${o.epochs} epochs ≈ ${passes} full passes`,
    `Validation: ${val} images per epoch${valTime}`,
  ];
}

/** The run detail's description of its sampled epochs; null for whole epochs. */
export function samplingNote(run: Run): string | null {
  const s = run.sampling;
  if (s == null) return null;
  const parts: string[] = [];
  if (s.train_images_per_epoch != null) {
    parts.push(
      `Each epoch trains on the next ${count(s.train_images_per_epoch)} of ${count(s.train_images)} images ` +
      `(warmup ${s.warmup_epochs} epochs, mosaic off for the last ${s.close_mosaic}).`,
    );
  }
  if (s.val_images_per_epoch != null) {
    parts.push(
      `Per-epoch validation uses a fixed ${count(s.val_images_per_epoch)} of ${count(s.val_images)} val images; ` +
      `the final metrics are on all ${count(s.val_images)}.`,
    );
  }
  return parts.join(" ");
}

/** "val subset of 400" for a run validated per epoch on a subset (its per-epoch charts), else null. */
export const valSubsetNote = (run: Run): string | null =>
  run.sampling?.val_images_per_epoch != null ? `val subset of ${count(run.sampling.val_images_per_epoch)}` : null;

/** "Final, best.pt (epoch 12) on all 252 val images: mask mAP50-95 0.712 · …" for a run whose reported metrics come
 * from the final evaluation on the whole val split (validated per epoch on a subset); null otherwise. */
export function finalMetricsNote(run: Run): string | null {
  const b = run.best;
  if (b == null || b.val_images == null) return null;
  const f = (v: number | null) => (v === null ? "—" : v.toFixed(3));
  return `Final, best.pt (epoch ${b.epoch}) on all ${count(b.val_images)} val images: mask mAP50-95 ${f(b.mask_map50_95)} · ` +
    `mask mAP50 ${f(b.mask_map50)} · box mAP50-95 ${f(b.box_map50_95)}`;
}
