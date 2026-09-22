import type { Brush } from "../hooks/useBrush";
import { MAX_SIZE, MIN_SIZE } from "../lib/brush";

type Props = { brush: Brush; frame: number; disabledReason: string | null };

export function BrushPanel({ brush, frame, disabledReason }: Props) {
  const { draft, dirty, busy, error } = brush;
  const disabled = disabledReason !== null;
  const n = draft.strokes.length;
  const status = busy
    ? "Saving…"
    : !dirty
      ? `No strokes yet on frame ${frame}.`
      : `${n} ${n === 1 ? "stroke" : "strokes"} on frame ${frame}, not saved.`;
  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Brush</h2>
      {disabled ? (
        <p className="note text-ink">{disabledReason}</p>
      ) : (
        <p className="note">Left-drag to paint, right-drag to erase. The brush edits this frame's mask; Save makes the frame a keyframe.</p>
      )}
      <label className="flex items-center gap-2 text-xs">
        <span className="w-16">Brush size</span>
        <input aria-label="Brush size" type="range" min={MIN_SIZE} max={MAX_SIZE} value={draft.size} disabled={disabled}
          className="flex-1" onChange={(e) => brush.setSize(Number(e.target.value))} />
        <span className="w-12 text-right text-dim">{draft.size} px</span>
      </label>
      <p className="note">
        <kbd aria-hidden>-</kbd> <kbd aria-hidden>=</kbd> shrink and grow the brush.
      </p>
      <p className="note text-ink">{status}</p>
      {dirty && !busy && <p className="note">Save or discard the brush strokes on frame {frame} first to do anything else.</p>}
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
      <div className="flex gap-1">
        <button type="button" disabled={!dirty || busy || disabled} aria-keyshortcuts="Enter"
          className={`btn flex-1 ${dirty && !busy ? "btn-on" : ""}`} onClick={() => void brush.save()}>
          Save <kbd aria-hidden>Enter</kbd>
        </button>
        <button type="button" disabled={!dirty || busy} aria-keyshortcuts="Escape" className="btn" onClick={brush.discard}>
          Discard <kbd aria-hidden>Esc</kbd>
        </button>
        <button type="button" disabled={!dirty || busy} aria-keyshortcuts="Z" className="btn" onClick={brush.undo}>
          Undo stroke <kbd aria-hidden>Z</kbd>
        </button>
      </div>
    </section>
  );
}
