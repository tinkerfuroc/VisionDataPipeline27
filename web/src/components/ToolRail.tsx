import type { ReactNode } from "react";
import type { Tool } from "../lib/promptState";

export type RailTool = Tool | "brush" | "regions";

type Props = {
  tool: RailTool;
  /** prompting is unavailable (no engine, a job running, a deleted frame, playback); Fit still works */
  disabled: boolean;
  /** the Brush tool is unavailable on its own (a job running, a deleted frame, playback, or no clip) */
  brushDisabled: boolean;
  /** unsaved brush strokes: every tool but Brush is disabled */
  locked: boolean;
  onTool: (tool: RailTool) => void;
  onFit: () => void;
};

const icon = (children: ReactNode) => (
  <svg aria-hidden width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
    {children}
  </svg>
);

const TOOLS: [RailTool, string, ReactNode][] = [
  ["clicks", "Click", icon(<><circle cx="9" cy="9" r="5.5" /><circle cx="9" cy="9" r="1.5" fill="currentColor" stroke="none" /></>)],
  ["box", "Box", icon(<rect x="2.5" y="4" width="13" height="10" strokeDasharray="3 2" />)],
  ["text", "Text", icon(<path d="M4 4.5h10M9 4.5v9.5" />)],
  ["brush", "Brush", icon(<><path d="M3 14.5c2.5 0 3-1.2 3.4-2.6.4-1.3 1.9-1.6 2.7-.8s.5 2.3-.8 2.7C6.9 14.2 5.7 14.5 3 14.5z" /><path d="M8.6 10.6 14.5 3.5" /></>)],
  ["regions", "Regions", icon(<><rect x="2.5" y="2.5" width="13" height="13" strokeDasharray="2 2" /><rect x="6" y="6" width="6" height="6" fill="currentColor" stroke="none" opacity="0.5" /></>)],
];

const item = "flex h-12 w-12 flex-col items-center justify-center gap-0.5 rounded font-cond text-[0.6875rem] font-medium leading-none";
const idle = "text-dim hover:bg-raise hover:text-ink disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent";

/** The prompting tools, the brush and the regions tool (which needs no engine), between the frame they act on and the inspector. */
export function ToolRail({ tool, disabled, brushDisabled, locked, onTool, onFit }: Props) {
  return (
    <nav aria-label="Tools" className="flex w-14 shrink-0 flex-col items-center gap-1 border-l border-line bg-panel py-2">
      {TOOLS.map(([id, label, glyph]) => {
        const off = locked ? id !== tool : id === "brush" ? brushDisabled : id === "regions" ? false : disabled;
        return (
          <button key={id} type="button" disabled={off} aria-pressed={tool === id}
            className={`${item} ${tool === id && !off ? "bg-ink text-well" : idle}`} onClick={() => onTool(id)}>
            {glyph}
            {label}
          </button>
        );
      })}
      <button type="button" title="Fit the frame to the view" className={`${item} ${idle} mt-auto`} onClick={onFit}>
        {icon(<path d="M2.5 6.5v-4h4M11.5 2.5h4v4M15.5 11.5v4h-4M6.5 15.5h-4v-4" />)}
        Fit
      </button>
    </nav>
  );
}
