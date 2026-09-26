import type { FormEvent } from "react";
import type { Prompting } from "../hooks/usePrompting";
import type { Tool } from "../lib/promptState";

type Props = { prompting: Prompting; disabledReason: string | null; /** what Accept does, as a sentence naming the active object */ target: string };

// The tools themselves live in the ToolRail beside the frame; this panel is what the chosen tool needs.
const HINTS: Record<Tool, string> = {
  clicks: "Left-click the object to include it, right-click to exclude. Every click updates the preview.",
  box: "Drag a box around the object. Add clicks afterwards to refine it.",
  text: "Describe the object, then choose one of the numbered candidates.",
};

export function PromptPanel({ prompting, disabledReason, target }: Props) {
  const { draft, dispatch, candidates, busy, error, canAccept, offersAcceptAll, canAcceptAll, acceptAll, accept, clear } = prompting;
  const disabled = disabledReason !== null;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    dispatch({ type: "submit" });
  };

  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Prompt</h2>
      {disabled ? <p className="note text-ink">{disabledReason}</p> : <p className="note">{HINTS[draft.tool]}</p>}
      {!disabled && <p className="note text-ink">{target}</p>}

      {draft.tool === "text" && (
        <form className="flex gap-1" onSubmit={submit}>
          <input aria-label="Text prompt" disabled={disabled} value={draft.phrase} placeholder="red connector"
            className="field flex-1" onChange={(e) => dispatch({ type: "phrase", phrase: e.target.value })} />
          <button type="submit" disabled={disabled || draft.phrase.trim() === ""} className="btn">Find</button>
        </form>
      )}

      {candidates.length > 1 && (
        <ul className="flex flex-col gap-1">
          {candidates.map((c, i) => (
            <li key={i}>
              <button type="button" aria-pressed={draft.candidate === i}
                className={`btn w-full justify-start gap-3 ${draft.candidate === i ? "btn-on" : ""}`}
                onClick={() => dispatch({ type: "candidate", index: i })}>
                <span className="w-4 text-center font-semibold">{i + 1}</span>
                <span>score {c.score.toFixed(2)}</span>
                <span className="ml-auto opacity-70">{c.area} px</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {draft.points.length > 0 && (
        // Numbered like the dots on the frame, so a stray click can be found and removed without undoing the rest.
        <ul aria-label="Clicks" className="flex flex-col gap-1">
          {draft.points.map(([x, y, label], i) => (
            <li key={`${x},${y},${label},${i}`} className="flex items-center gap-2 text-xs">
              <span aria-hidden className={`w-5 rounded-full text-center font-cond font-semibold text-well ${label === 1 ? "bg-[#22c55e]" : "bg-[#ef4444]"}`}>
                {i + 1}
              </span>
              <span className="w-14">{label === 1 ? "Include" : "Exclude"}</span>
              <span className="flex-1 text-dim">{x}, {y}</span>
              <button type="button" className="btn" aria-label={`Remove click ${i + 1}`} disabled={disabled}
                onClick={() => dispatch({ type: "remove", index: i })}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
      {candidates.length === 1 && (
        <p className="note">Preview covers {candidates[0].area} px, score {candidates[0].score.toFixed(2)}.</p>
      )}
      {busy && <p className="note">Predicting…</p>}
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}

      <div className="flex gap-1">
        <button type="button" disabled={!canAccept} aria-keyshortcuts="Enter"
          className={`btn flex-1 ${canAccept ? "btn-on" : ""}`} onClick={() => void accept()}>
          {candidates.length > 1 ? "Accept selected" : "Accept"} <kbd aria-hidden>Enter</kbd>
        </button>
        <button type="button" disabled={disabled} aria-keyshortcuts="Escape" className="btn" onClick={clear}>
          Clear <kbd aria-hidden>Esc</kbd>
        </button>
        <button type="button" disabled={disabled} aria-keyshortcuts="Z" className="btn" onClick={() => dispatch({ type: "undo" })}>
          Undo <kbd aria-hidden>Z</kbd>
        </button>
      </div>
      {offersAcceptAll && (
        <button type="button" disabled={!canAcceptAll} className="btn" onClick={() => void acceptAll()}>
          Accept all as new objects
        </button>
      )}
    </section>
  );
}
