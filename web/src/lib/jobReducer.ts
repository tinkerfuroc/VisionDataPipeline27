import type { JobEvent, JobSnapshot, JobStatus } from "../api/types";

export type JobView = {
  status: JobStatus | "idle";
  /** seq of the next event we have not seen; lower seqs are replays */
  nextSeq: number;
  done: number;
  /** unknown for imports: a video's frame count is not known until decoding ends */
  total: number | null;
  lastFrame: number | null;
  logs: string[];
  result: Record<string, unknown> | null;
  error: string | null;
};

export const idleJob: JobView = {
  status: "idle", nextSeq: 0, done: 0, total: null, lastFrame: null, logs: [], result: null, error: null,
};

export const isTerminal = (s: JobView["status"]): boolean => s === "done" || s === "error" || s === "cancelled";

export type JobAction =
  | { type: "reset" }
  | { type: "event"; event: JobEvent }
  | { type: "snapshot"; snapshot: JobSnapshot };

function resultOf(event: JobEvent): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(event)) if (k !== "seq" && k !== "type") out[k] = v;
  return out;
}

export function jobReducer(view: JobView, action: JobAction): JobView {
  if (action.type === "reset") return idleJob;

  if (action.type === "snapshot") {
    if (isTerminal(view.status)) return view;
    const s = action.snapshot;
    if (!isTerminal(s.status)) return view.status === "running" ? view : { ...view, status: "running" };
    return { ...view, status: s.status, result: s.result, error: s.error };
  }

  const e = action.event;
  if (e.seq < view.nextSeq) return view;
  const base: JobView = { ...view, nextSeq: e.seq + 1 };
  if (isTerminal(view.status)) return base;
  base.status = "running";
  const done = typeof e.done === "number" ? e.done : base.done;
  const total = typeof e.total === "number" ? e.total : base.total;
  switch (e.type) {
    case "progress":
      return { ...base, done, total };
    case "frame":
      return { ...base, done, total, lastFrame: typeof e.frame === "number" ? e.frame : base.lastFrame };
    case "log":
      return { ...base, logs: [...base.logs, String(e.message ?? "")].slice(-50) };
    case "done":
    case "cancelled":
      return { ...base, status: e.type, result: resultOf(e) };
    case "error":
      return { ...base, status: "error", error: String(e.message ?? "the job failed") };
  }
}
