import type { FormEvent } from "react";
import type { Prompting } from "../hooks/usePrompting";
import type { Tool } from "../lib/promptState";

type Props = { prompting: Prompting; disabledReason: string | null };

const TOOLS: [Tool, string, string][] = [
  ["clicks", "Click", "Left click = include, right click = exclude. Each click refreshes the preview."],
  ["box", "Box", "Drag a box around the object, then refine with clicks."],
  ["text", "Text", "Describe the object, then pick one of the numbered candidates."],
];
const base = "rounded px-3 py-1 text-sm disabled:opacity-40";

export function PromptPanel({ prompting, disabledReason }: Props) {
  const { draft, dispatch, candidates, busy, error, canAccept, accept, clear } = prompting;
  const disabled = disabledReason !== null;
  const hint = TOOLS.find(([tool]) => tool === draft.tool)?.[2];

  const submit = (e: FormEvent) => {
    e.preventDefault();
    dispatch({ type: "submit" });
  };

  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Prompt</h2>
      <div className="flex gap-1">
        {TOOLS.map(([tool, label]) => (
          <button key={tool} type="button" disabled={disabled} aria-pressed={draft.tool === tool}
            className={`${base} ${draft.tool === tool ? "bg-sky-700" : "bg-neutral-700"}`}
            onClick={() => dispatch({ type: "tool", tool })}>
            {label}
          </button>
        ))}
      </div>
      {disabled ? <p className="text-xs text-amber-400">{disabledReason}</p> : <p className="text-xs text-neutral-400">{hint}</p>}

      {draft.tool === "text" && (
        <form className="flex gap-1" onSubmit={submit}>
          <input aria-label="Text prompt" disabled={disabled} value={draft.phrase} placeholder="e.g. red connector"
            className="min-w-0 flex-1 rounded border border-neutral-600 bg-neutral-800 px-2 py-1"
            onChange={(e) => dispatch({ type: "phrase", phrase: e.target.value })} />
          <button type="submit" disabled={disabled || draft.phrase.trim() === ""} className={`${base} bg-neutral-700`}>Find</button>
        </form>
      )}

      {candidates.length > 1 && (
        <ul className="flex flex-col gap-1">
          {candidates.map((c, i) => (
            <li key={i}>
              <button type="button" aria-pressed={draft.candidate === i}
                className={`${base} w-full text-left ${draft.candidate === i ? "bg-amber-600" : "bg-neutral-700"}`}
                onClick={() => dispatch({ type: "candidate", index: i })}>
                {i + 1} · score {c.score.toFixed(2)} · {c.area} px
              </button>
            </li>
          ))}
        </ul>
      )}
      {candidates.length === 1 && (
        <p className="text-xs text-neutral-400">Preview: {candidates[0].area} px, score {candidates[0].score.toFixed(2)}</p>
      )}
      {busy && <p className="text-xs text-neutral-400">Predicting…</p>}
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}

      <div className="flex gap-1">
        <button type="button" disabled={!canAccept} className={`${base} bg-emerald-700`} onClick={() => void accept()}>Accept (Enter)</button>
        <button type="button" disabled={disabled} className={`${base} bg-neutral-700`} onClick={clear}>Clear (Esc)</button>
        <button type="button" disabled={disabled} className={`${base} bg-neutral-700`} onClick={() => dispatch({ type: "undo" })}>Undo (Z)</button>
      </div>
    </section>
  );
}
