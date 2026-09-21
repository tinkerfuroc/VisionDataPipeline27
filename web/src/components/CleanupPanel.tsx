import { useState } from "react";
import type { FrameState } from "../api/types";

type Props = {
  frame: number;
  frameCount: number;
  state: FrameState;
  /** why clear-range is unavailable (a job is running), or null */
  clearBlockedReason: string | null;
  error: string | null;
  onToggleDelete: () => void;
  onPrune: () => void;
  onClearRange: (start: number, end: number) => void;
};


export function CleanupPanel({ frame, frameCount, state, clearBlockedReason, error, onToggleDelete, onPrune, onClearRange }: Props) {
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const last = Math.max(0, frameCount - 1);
  const clampIndex = (v: string) => Math.max(0, Math.min(last, Math.floor(Number(v)) || 0));
  const valid = start <= end;

  // Folded by default: cleanup is the rare step, and folding it keeps the inspector from scrolling on short screens.
  // The error sits outside the fold, because Del works (and can fail) while the panel is closed.
  return (
    <section className="flex flex-col gap-2">
      <details className="group">
        <summary className="heading flex items-center gap-1.5 marker:content-none">
          <svg aria-hidden width="10" height="10" viewBox="0 0 10 10" className="fill-dim group-open:rotate-90">
            <path d="M3 1.5 7.5 5 3 8.5z" />
          </svg>
          Cleanup
        </summary>
        <div className="mt-2 flex flex-col gap-2">
          <div className="flex gap-1">
            <button type="button" aria-keyshortcuts="Delete" className="btn flex-1" onClick={onToggleDelete}>
              {state === "deleted" ? "Restore frame" : "Delete frame"} <kbd aria-hidden>Del</kbd>
            </button>
            <button type="button" className="btn flex-1" onClick={onPrune}>Prune from here</button>
          </div>
          <p className="note">Deleted frames are left out of the export. Prune deletes frame {frame} and every frame after it.</p>

          <div className="flex items-center gap-1">
            <input aria-label="Range start" type="number" min={0} max={last} className="field w-16" value={start}
              onChange={(e) => setStart(clampIndex(e.target.value))} />
            <button type="button" className="btn" title="Start the range at this frame" onClick={() => setStart(frame)}>From here</button>
            <span className="text-dim">to</span>
            <input aria-label="Range end" type="number" min={0} max={last} className="field w-16" value={end}
              onChange={(e) => setEnd(clampIndex(e.target.value))} />
            <button type="button" className="btn" title="End the range at this frame" onClick={() => setEnd(frame)}>To here</button>
          </div>
          <button type="button" className="btn" disabled={!valid || clearBlockedReason !== null} onClick={() => onClearRange(start, end)}>
            Clear masks in range
          </button>
          <p className="note">{clearBlockedReason ?? "Removes propagated masks in the range. Keyframes are kept."}</p>
        </div>
      </details>
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
    </section>
  );
}
