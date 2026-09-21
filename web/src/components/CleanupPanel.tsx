import type { FrameState } from "../api/types";

/** Frames marked for clearing, inclusive. Start may exceed end while the numbers are being typed. */
export type Range = [number, number];

type Props = {
  frame: number;
  states: FrameState[];
  /** null: nothing is marked, so clearing acts on the current frame only */
  range: Range | null;
  /** why clearing is unavailable (a job is running), or null */
  clearBlockedReason: string | null;
  error: string | null;
  onRange: (range: Range | null) => void;
  onToggleDelete: () => void;
  onPrune: () => void;
  onClearRange: (start: number, end: number) => void;
};

export function CleanupPanel({ frame, states, range, clearBlockedReason, error, onRange, onToggleDelete, onPrune, onClearRange }: Props) {
  const last = Math.max(0, states.length - 1);
  const clampIndex = (v: string) => Math.max(0, Math.min(last, Math.floor(Number(v)) || 0));
  const [start, end] = range ?? [frame, frame];
  const valid = start <= end;
  // what the button would actually remove: keyframes are kept, so only propagated masks count
  const masks = valid ? states.slice(start, end + 1).filter((s) => s === "masked").length : 0;
  const where = start === end ? `frame ${start}` : `frames ${start} to ${end}`;
  const outcome = !valid
    ? "The first frame must not come after the last."
    : masks === 0
      ? `There are no propagated masks in ${where}.`
      : `Removes ${masks} propagated ${masks === 1 ? "mask" : "masks"}. Keyframes are kept.`;

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
          <button type="button" className="btn" disabled={!valid || masks === 0 || clearBlockedReason !== null}
            onClick={() => onClearRange(start, end)}>
            Clear masks in {where}
          </button>
          <p className="note">{clearBlockedReason ?? outcome}</p>

          <div className="mt-1 flex gap-1">
            <button type="button" aria-keyshortcuts="Delete" className="btn flex-1" onClick={onToggleDelete}>
              {states[frame] === "deleted" ? "Restore frame" : "Delete frame"} <kbd aria-hidden>Del</kbd>
            </button>
            <button type="button" className="btn flex-1" onClick={onPrune}>Prune from here</button>
          </div>
          <p className="note">Deleted frames are left out of the export. Prune deletes frame {frame} and every frame after it.</p>
        </div>
      </details>
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
    </section>
  );
}
