import type { Direction } from "../api/types";
import type { JobHandle } from "../hooks/useJob";

type Props = {
  job: JobHandle;
  running: boolean;
  /** true while `job` is a propagation this page did not start (attached from a 409 naming another clip's job) */
  foreign: boolean;
  /** true while a start request (including its interactive retries) is in flight and has no job yet to show */
  starting: boolean;
  /** why propagation cannot start from here, or null */
  blockedReason: string | null;
  follow: boolean;
  error: string | null;
  onStart: (direction: Direction) => void;
  onStop: () => void;
  onFollow: (follow: boolean) => void;
};

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);

export function PropagatePanel({ job, running, foreign, starting, blockedReason, follow, error, onStart, onStop, onFollow }: Props) {
  const blocked = running || starting || blockedReason !== null;
  const lostAt = num(job.result?.lost_at);
  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Propagate</h2>
      <div className="flex gap-1">
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => onStart("backward")}>◀ Backward</button>
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => onStart("both")}>Both</button>
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => onStart("forward")}>Forward ▶</button>
        <button type="button" className="btn btn-danger" disabled={!running} onClick={onStop}>Stop</button>
      </div>
      {!running && blockedReason !== null && <p className="note">{blockedReason}</p>}
      <label className="flex items-center gap-1.5 text-xs text-dim">
        <input type="checkbox" checked={follow} onChange={(e) => onFollow(e.target.checked)} />
        Follow the propagation
      </label>

      {running && (
        <>
          <p className="note text-ink">
            {foreign
              ? "The GPU is busy with a propagation that was not started from this page. Stop cancels it."
              : "Propagating. Stop keeps the masks written so far."}
          </p>
          <progress value={job.total === null ? undefined : job.done} max={job.total ?? undefined} />
          <p className="note">
            {job.done} of {job.total ?? "?"} frames{job.transport === "poll" ? ". Live updates dropped; checking every second." : ""}
          </p>
        </>
      )}
      {job.logs.length > 0 && <p className="note">{job.logs[job.logs.length - 1]}</p>}
      {job.status === "done" && (
        <p className="note text-ink">
          Done: {num(job.result?.processed) ?? 0} frames processed, {num(job.result?.written) ?? 0} masks written.
        </p>
      )}
      {job.status === "cancelled" && <p className="note text-ink">Stopped. The masks written so far were kept.</p>}
      {lostAt !== null && (
        <p className="note text-ink">Lost the object at frame {lostAt}. Correct it there and propagate again.</p>
      )}
      {job.status === "error" && <p role="alert" className="text-xs text-danger">{job.error}</p>}
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
    </section>
  );
}
