import type { Direction } from "../api/types";
import type { JobHandle } from "../hooks/useJob";

/** "All objects" (every object with a mask on the start frame) or the selected object alone. */
export type PropagateScope = "all" | "selected";

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
  scope: PropagateScope;
  /** "#2 cup" for the active object; null on a "New object" row, where only "All objects" is possible */
  selectedLabel: string | null;
  /** how an object id in the job result is named ("#2 cup", or "#2" once it is gone) */
  objectName: (id: number) => string;
  onScope: (scope: PropagateScope) => void;
  /** the job was started from this page, so the ids in its result are this clip's objects and can be named */
  ownResult: boolean;
  onStart: (direction: Direction) => void;
  onStop: () => void;
  onFollow: (follow: boolean) => void;
};

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
const ids = (v: unknown): number[] => (Array.isArray(v) ? v.filter((x): x is number => typeof x === "number") : []);
/** the result's `lost` (object id, a string key in JSON → the frame it was lost on), nearest frame first */
const lostList = (v: unknown): [number, number][] => {
  if (v === null || typeof v !== "object") return [];
  return Object.entries(v as Record<string, unknown>)
    .filter((e): e is [string, number] => typeof e[1] === "number")
    .map(([id, f]): [number, number] => [Number(id), f])
    .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
};

export function PropagatePanel(p: Props) {
  const { job, running, foreign, starting, blockedReason, follow, error, selectedLabel } = p;
  const blocked = running || starting || blockedReason !== null;
  const scope: PropagateScope = selectedLabel === null ? "all" : p.scope;
  // another page's job names another clip's objects: say only where it lost one
  const skipped = p.ownResult ? ids(job.result?.skipped) : [];
  const lost = p.ownResult ? lostList(job.result?.lost) : [];
  const lostAt = p.ownResult ? null : num(job.result?.lost_at);
  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Propagate</h2>
      <fieldset className="flex flex-col gap-1" disabled={running || starting}>
        <legend className="sr-only">Objects to propagate</legend>
        <label className="flex items-center gap-1.5 text-xs">
          <input type="radio" name="propagate-scope" checked={scope === "all"} onChange={() => p.onScope("all")} />
          All objects
        </label>
        <label className="flex items-center gap-1.5 text-xs">
          <input type="radio" name="propagate-scope" checked={scope === "selected"} disabled={selectedLabel === null}
            onChange={() => p.onScope("selected")} />
          {selectedLabel === null ? "Selected object" : `Selected: ${selectedLabel}`}
        </label>
        {selectedLabel === null && <p className="note">A new object has no mask yet: select an object to propagate it alone.</p>}
      </fieldset>
      <div className="flex gap-1">
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => p.onStart("backward")}>◀ Backward</button>
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => p.onStart("both")}>Both</button>
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => p.onStart("forward")}>Forward ▶</button>
        <button type="button" className="btn btn-danger" disabled={!running} onClick={p.onStop}>Stop</button>
      </div>
      {!running && blockedReason !== null && <p className="note">{blockedReason}</p>}
      <label className="flex items-center gap-1.5 text-xs text-dim">
        <input type="checkbox" checked={follow} onChange={(e) => p.onFollow(e.target.checked)} />
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
      {skipped.length > 0 && <p className="note">Skipped {skipped.map(p.objectName).join(", ")}: no mask on the start frame.</p>}
      {lost.map(([id, f]) => (
        <p key={id} className="note text-ink">
          Lost {p.objectName(id)} at frame {f}. Correct it there and propagate again.
        </p>
      ))}
      {lostAt !== null && <p className="note text-ink">Lost an object at frame {lostAt}.</p>}
      {job.status === "error" && <p role="alert" className="text-xs text-danger">{job.error}</p>}
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
    </section>
  );
}
