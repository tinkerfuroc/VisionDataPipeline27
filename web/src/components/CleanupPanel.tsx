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

const button = "rounded bg-neutral-700 px-3 py-1 text-sm hover:bg-neutral-600 disabled:opacity-40";
const field = "w-20 rounded border border-neutral-600 bg-neutral-800 px-2 py-1";

export function CleanupPanel({ frame, frameCount, state, clearBlockedReason, error, onToggleDelete, onPrune, onClearRange }: Props) {
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const last = Math.max(0, frameCount - 1);
  const clampIndex = (v: string) => Math.max(0, Math.min(last, Math.floor(Number(v)) || 0));
  const valid = start <= end;

  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Cleanup</h2>
      <div className="flex gap-1">
        <button type="button" className={button} onClick={onToggleDelete}>
          {state === "deleted" ? "Restore frame (Del)" : "Delete frame (Del)"}
        </button>
        <button type="button" className={button} onClick={onPrune}>Prune from here</button>
      </div>
      <p className="text-xs text-neutral-400">Deleted frames are left out of the export. Prune deletes frame {frame} and everything after it.</p>

      <div className="flex flex-wrap items-center gap-1">
        <input aria-label="Range start" type="number" min={0} max={last} className={field} value={start}
          onChange={(e) => setStart(clampIndex(e.target.value))} />
        <button type="button" className={button} onClick={() => setStart(frame)}>[ here</button>
        <span>–</span>
        <input aria-label="Range end" type="number" min={0} max={last} className={field} value={end}
          onChange={(e) => setEnd(clampIndex(e.target.value))} />
        <button type="button" className={button} onClick={() => setEnd(frame)}>here ]</button>
      </div>
      <button type="button" className={button} disabled={!valid || clearBlockedReason !== null} onClick={() => onClearRange(start, end)}>
        Clear masks in range
      </button>
      <p className="text-xs text-neutral-400">{clearBlockedReason ?? "Removes propagated masks in the range; keyframes are kept."}</p>
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}
    </section>
  );
}
