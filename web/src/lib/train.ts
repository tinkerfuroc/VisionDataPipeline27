import type { GpuQueue, MetricRow, Run, RunBest, TrainStatus } from "../api/types";
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

/** e.g. "cuda:0 · RTX 3080 Ti · est. 34 TFLOPS · 9.1 GiB free · SAM3 · default". Options are already
 * server-sorted fastest first; this only formats one of them. */
export function gpuOptionLabel(g: GpuQueue, status: Pick<TrainStatus, "default_gpu" | "sam3_gpu">): string {
  const parts = [`cuda:${g.index}`, g.name.replace(/^NVIDIA GeForce /, ""), `est. ${Math.round(g.tflops)} TFLOPS`];
  if (g.memory_free !== null) parts.push(`${(g.memory_free / 2 ** 30).toFixed(1)} GiB free`);
  if (g.index === status.sam3_gpu) parts.push("SAM3");
  let label = parts.join(" · ");
  if (g.index === status.default_gpu) label += " · default";
  return label;
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
      return "Done";
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
