import type { ReactNode } from "react";
import type { Tool } from "../lib/promptState";

type Props = {
  tool: Tool;
  /** prompting is unavailable (no engine, a job running, a deleted frame, playback); Fit still works */
  disabled: boolean;
  onTool: (tool: Tool) => void;
  onFit: () => void;
};

const icon = (children: ReactNode) => (
  <svg aria-hidden width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
    {children}
  </svg>
);

const TOOLS: [Tool, string, ReactNode][] = [
  ["clicks", "Click", icon(<><circle cx="9" cy="9" r="5.5" /><circle cx="9" cy="9" r="1.5" fill="currentColor" stroke="none" /></>)],
  ["box", "Box", icon(<rect x="2.5" y="4" width="13" height="10" strokeDasharray="3 2" />)],
  ["text", "Text", icon(<path d="M4 4.5h10M9 4.5v9.5" />)],
];

const item = "flex h-12 w-12 flex-col items-center justify-center gap-0.5 rounded font-cond text-[0.6875rem] font-medium leading-none";
const idle = "text-dim hover:bg-raise hover:text-ink disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent";

/** The prompting tools, between the frame they act on and the inspector. */
export function ToolRail({ tool, disabled, onTool, onFit }: Props) {
  return (
    <nav aria-label="Tools" className="flex w-14 shrink-0 flex-col items-center gap-1 border-l border-line bg-panel py-2">
      {TOOLS.map(([id, label, glyph]) => (
        <button key={id} type="button" disabled={disabled} aria-pressed={tool === id}
          className={`${item} ${tool === id && !disabled ? "bg-ink text-well" : idle}`} onClick={() => onTool(id)}>
          {glyph}
          {label}
        </button>
      ))}
      <button type="button" title="Fit the frame to the view" className={`${item} ${idle} mt-auto`} onClick={onFit}>
        {icon(<path d="M2.5 6.5v-4h4M11.5 2.5h4v4M15.5 11.5v4h-4M6.5 15.5h-4v-4" />)}
        Fit
      </button>
    </nav>
  );
}
