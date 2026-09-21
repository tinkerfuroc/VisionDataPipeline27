import type { ReactNode } from "react";
import type { FrameState } from "../api/types";
import { counts, labeledPercent } from "../lib/timeline";

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** The same marks the timeline draws, so these counts double as its legend. */
function Swatch({ state, color }: { state: FrameState; color: string }) {
  const marks: Record<FrameState, ReactNode> = {
    keyframe: <path d="M6 0.5 11.5 6 6 11.5 0.5 6Z" className="fill-key" />,
    masked: <rect x="0" y="2" width="12" height="8" fill={color} />,
    empty: <rect x="0.5" y="2.5" width="11" height="7" fill="none" className="stroke-dim" />,
    deleted: (
      <>
        <rect x="0.5" y="2.5" width="11" height="7" fill="none" className="stroke-line" />
        <path d="M1 9.5 5 2.5M5 9.5 9 2.5M9 9.5 11.5 5" fill="none" className="stroke-dim" />
      </>
    ),
  };
  return (
    <svg aria-hidden width="12" height="12" viewBox="0 0 12 12" className="shrink-0">
      {marks[state]}
    </svg>
  );
}

export function ClipStats({ states, color }: { states: FrameState[]; color: string }) {
  const c = counts(states);
  const items: [FrameState, string][] = [
    ["keyframe", plural(c.keyframe, "keyframe")],
    ["masked", `${c.masked} masked`],
    ["empty", `${c.empty} empty`],
    ["deleted", `${c.deleted} deleted`],
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-cond text-sm text-dim">
      {items.map(([state, text]) => (
        <span key={state} className="flex items-center gap-1.5">
          <Swatch state={state} color={color} />
          {text}
        </span>
      ))}
      <span className="text-ink">{labeledPercent(states)}% labeled</span>
    </div>
  );
}
