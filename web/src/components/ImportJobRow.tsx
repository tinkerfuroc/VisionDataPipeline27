import { useEffect, useRef } from "react";
import { api } from "../api/rest";
import { useJob } from "../hooks/useJob";
import { isTerminal } from "../lib/jobReducer";

type Props = {
  jobId: string;
  label: string;
  /** called once when the job reaches a terminal state */
  onFinished: (jobId: string, ok: boolean) => void;
  onDismiss: (jobId: string) => void;
};

export function ImportJobRow({ jobId, label, onFinished, onDismiss }: Props) {
  const job = useJob(jobId);
  const reported = useRef(false);

  useEffect(() => {
    if (isTerminal(job.status) && !reported.current) {
      reported.current = true;
      onFinished(jobId, job.status === "done");
    }
  }, [job.status, jobId, onFinished]);

  const running = !isTerminal(job.status);
  return (
    <li className="flex items-center gap-3 rounded border border-line bg-panel px-3 py-1.5 text-sm">
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {running && <span className="font-cond text-dim">{job.done} frames…</span>}
      {job.status === "cancelled" && <span className="font-cond text-dim">Cancelled</span>}
      {job.status === "error" && <span className="text-danger">{job.error}</span>}
      {running ? (
        <button type="button" className="btn" onClick={() => void api.cancelJob(jobId).catch(() => {})}>
          Cancel
        </button>
      ) : (
        <button type="button" className="btn" onClick={() => onDismiss(jobId)}>
          Dismiss
        </button>
      )}
    </li>
  );
}
