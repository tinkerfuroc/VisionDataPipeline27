import type { FrameState } from "../api/types";

/** Frames marked for clearing, inclusive. Start may exceed end while the numbers are being typed. */
export type Range = [number, number];

/** what "Clear masks" removes: the active object's masks (default) or every object's */
export type ClearScope = "active" | "all";

type Props = {
  frame: number;
  /** the whole clip's states: Delete / Restore, and the deleted frames a clear covers */
  states: FrameState[];
  /** the lanes "Clear masks" acts on: [the active object's lane], every object's lane, or none (a New object row) */
  lanes: FrameState[][];
  scope: ClearScope;
  /** "#1 cup"; null on a "New object" row */
  activeLabel: string | null;
  onScope: (scope: ClearScope) => void;
  /** null: nothing is marked, so clearing acts on the current frame only */
  range: Range | null;
  /** why clearing is unavailable (a job is running), or null */
  clearBlockedReason: string | null;
  /** unsaved brush strokes: Clear, Delete/Restore and Prune are all disabled with this reason */
  lockReason: string | null;
  error: string | null;
  onRange: (range: Range | null) => void;
  onToggleDelete: () => void;
  onPrune: () => void;
  onClearRange: (start: number, end: number) => void;
};

export function CleanupPanel({
  frame, states, lanes, scope, activeLabel, onScope, range, clearBlockedReason, lockReason, error, onRange,
  onToggleDelete, onPrune, onClearRange,
}: Props) {
  const last = Math.max(0, states.length - 1);
  const clampIndex = (v: string) => Math.max(0, Math.min(last, Math.floor(Number(v)) || 0));
  const [start, end] = range ?? [frame, frame];
  const valid = start <= end;
  // What the button would actually remove. Keyframes go too (clearing is the only way to remove one), so they are
  // counted out loud. A deleted frame may still hold a mask (deleting only flags the frame) and the server clears
  // those too, so they keep the button live.
  const slice = (lane: FrameState[]) => (valid ? lane.slice(start, end + 1) : []);
  const count = (want: FrameState) => lanes.reduce((n, lane) => n + slice(lane).filter((s) => s === want).length, 0);
  const masks = count("masked");
  const keyframes = count("keyframe");
  // deleted frames are the clip's (deleted in every lane); they only matter when some object is cleared
  const deleted = lanes.length === 0 ? 0 : slice(states).filter((s) => s === "deleted").length;
  const where = start === end ? `frame ${start}` : `frames ${start} to ${end}`;
  const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  const parts = [
    ...(masks > 0 ? [`${plural(masks, "propagated mask", "propagated masks")}`] : []),
    ...(keyframes > 0 ? [plural(keyframes, "keyframe", "keyframes")] : []),
    ...(deleted > 0 ? [`any mask under the ${plural(deleted, "deleted frame", "deleted frames")}`] : []),
  ];
  const outcome = !valid
    ? "The first frame must not come after the last."
    : parts.length === 0
      ? `There is nothing to clear in ${where}.`
      : `Removes ${parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`}.`;

  // The error sits outside the fold, because Del works (and can fail) while the panel is closed.
  return (
    <section className="flex flex-col gap-2">
      <details className="group" open>
        <summary className="heading flex items-center gap-1.5 marker:content-none">
          <svg aria-hidden width="10" height="10" viewBox="0 0 10 10" className="fill-dim group-open:rotate-90">
            <path d="M3 1.5 7.5 5 3 8.5z" />
          </svg>
          Cleanup
        </summary>
        <div className="mt-2 flex flex-col gap-2">
          <p className="note">
            To clear masks from a stretch of frames, mark it first: hold <kbd>Shift</kbd> and drag along the timeline,
            or press <kbd>[</kbd> and <kbd>]</kbd> on its first and last frame.
          </p>
          <div className="flex items-center gap-1">
            <input aria-label="Range start" type="number" min={0} max={last} className="field w-16" value={start}
              onChange={(e) => onRange([clampIndex(e.target.value), end])} />
            <span className="text-dim">to</span>
            <input aria-label="Range end" type="number" min={0} max={last} className="field w-16" value={end}
              onChange={(e) => onRange([start, clampIndex(e.target.value)])} />
            {range !== null && <button type="button" className="btn ml-auto" onClick={() => onRange(null)}>Unmark</button>}
          </div>
          <fieldset className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            <legend className="sr-only">Objects to clear</legend>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="clear-scope" checked={scope === "active"} onChange={() => onScope("active")} />
              {activeLabel === null ? "Active object" : `Active object (${activeLabel})`}
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="clear-scope" checked={scope === "all"} onChange={() => onScope("all")} />
              All objects
            </label>
          </fieldset>
          <button type="button" className="btn" disabled={!valid || parts.length === 0 || clearBlockedReason !== null || lockReason !== null}
            onClick={() => onClearRange(start, end)}>
            Clear masks in {where}
          </button>
          <p className="note">{lockReason ?? clearBlockedReason ?? outcome}</p>

          <div className="mt-1 flex gap-1">
            <button type="button" aria-keyshortcuts="Delete" className="btn flex-1" disabled={lockReason !== null} onClick={onToggleDelete}>
              {states[frame] === "deleted" ? "Restore frame" : "Delete frame"} <kbd aria-hidden>Del</kbd>
            </button>
            <button type="button" className="btn flex-1" disabled={lockReason !== null} onClick={onPrune}>Prune from here</button>
          </div>
          <p className="note">Deleted frames are left out of the export. Prune deletes frame {frame} and every frame after it.</p>
        </div>
      </details>
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
    </section>
  );
}
