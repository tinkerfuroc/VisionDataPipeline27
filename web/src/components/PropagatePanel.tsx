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

const button = "rounded bg-neutral-700 px-3 py-1 text-sm hover:bg-neutral-600 disabled:opacity-40";
const num = (v: unknown): number | null => (typeof v === "number" ? v : null);

export function PropagatePanel({ job, running, foreign, starting, blockedReason, follow, error, onStart, onStop, onFollow }: Props) {
  const blocked = running || starting || blockedReason !== null;
  const lostAt = num(job.result?.lost_at);
  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Propagate</h2>
      <div className="flex gap-1">
        <button type="button" className={button} disabled={blocked} onClick={() => onStart("backward")}>◀ Backward</button>
        <button type="button" className={button} disabled={blocked} onClick={() => onStart("both")}>Both</button>
        <button type="button" className={button} disabled={blocked} onClick={() => onStart("forward")}>Forward ▶</button>
        <button type="button" className={`${button} ml-auto bg-red-800 hover:bg-red-700`} disabled={!running} onClick={onStop}>Stop</button>
      </div>
      {!running && blockedReason !== null && <p className="text-xs text-neutral-400">{blockedReason}</p>}
      <label className="flex items-center gap-1 text-xs">
        <input type="checkbox" checked={follow} onChange={(e) => onFollow(e.target.checked)} />
        Follow the propagation
      </label>

      {running && (
        <>
          <p className="text-xs text-amber-400">
            {foreign
              ? "The GPU is busy with a propagation that was not started from this page. Stop cancels it."
              : "Propagating — prompting is paused. Stop keeps the masks written so far."}
          </p>
          <progress className="w-full" value={job.total === null ? undefined : job.done} max={job.total ?? undefined} />
          <p className="text-xs tabular-nums text-neutral-300">
            {job.done} / {job.total ?? "?"} frames{job.transport === "poll" ? " (live updates lost — polling)" : ""}
          </p>
        </>
      )}
      {job.logs.length > 0 && <p className="text-xs text-neutral-400">{job.logs[job.logs.length - 1]}</p>}
      {job.status === "done" && (
        <p className="text-xs text-emerald-400">
          Done: {num(job.result?.processed) ?? 0} frames processed, {num(job.result?.written) ?? 0} masks written.
        </p>
      )}
      {job.status === "cancelled" && <p className="text-xs text-amber-400">Stopped. The masks written so far were kept.</p>}
      {lostAt !== null && (
        <p className="text-xs text-amber-400">Lost the object at frame {lostAt} — correct it there and propagate again.</p>
      )}
      {job.status === "error" && <p role="alert" className="text-xs text-red-400">{job.error}</p>}
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}
    </section>
  );
}
