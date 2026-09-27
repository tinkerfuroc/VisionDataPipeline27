import { useEffect, useState, type FormEvent } from "react";
import type { ClipDetail } from "../api/types";
import type { Selection } from "../lib/activeObject";
import { classColor, className, labeledCount, laneOf, objectLabel, sortedObjects } from "../lib/objects";

type Props = {
  detail: ClipDetail;
  selection: Selection;
  hidden: ReadonlySet<number>;
  /** unsaved brush strokes: the active row cannot change (the strokes belong to it) */
  lockReason: string | null;
  /** why the active object cannot be deleted now (a propagation is running), or null */
  deleteBlockedReason: string | null;
  error: string | null;
  onSelect: (id: number) => void;
  onToggleHidden: (id: number) => void;
  /** picking a class always starts a new object of it (ignored when it is already the active object's class);
   * relabelling an existing object is the separate, explicit "Change" control below */
  onClass: (classId: number) => void;
  /** relabels the active object at once (the "Change" button); never called while there is no active object */
  onRelabel: (classId: number) => void;
  onNew: () => void;
  onDelete: () => void;
  onAddClass: (name: string) => Promise<void>;
};

const glyph = (d: string) => (
  <svg aria-hidden width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" />
    <path d={d} />
  </svg>
);
const EYE = glyph("M9.5 8a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z");
const EYE_OFF = glyph("M2.5 13.5 13.5 2.5");

/** One row per object (spec §8): swatch, "#id class", labelled-frame count, and an eye that hides it from the frame. */
export function ObjectsPanel(p: Props) {
  const { detail, selection } = p;
  const [name, setName] = useState("");
  const [addError, setAddError] = useState<string | null>(null);
  const locked = p.lockReason !== null;
  const objects = sortedObjects(detail);
  const active = selection.kind === "object" ? (objects.find((o) => o.id === selection.id) ?? null) : null;
  const classId = active?.class_id ?? (selection.kind === "new" ? selection.classId : null);
  // the explicit relabel select's own choice, reset to the active object's own class whenever the active object changes
  const [relabelTo, setRelabelTo] = useState<number | null>(null);
  useEffect(() => setRelabelTo(active?.class_id ?? null), [active?.id]);
  const relabelValue = relabelTo ?? active?.class_id ?? null;

  const add = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (trimmed === "") return;
    setAddError(null);
    p.onAddClass(trimmed).then(() => setName(""), (err: unknown) => setAddError(err instanceof Error ? err.message : String(err)));
  };

  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Objects</h2>
      <ul aria-label="Objects" className="flex flex-col gap-1">
        {objects.map((o) => {
          const isActive = active?.id === o.id;
          const label = objectLabel(detail, o);
          const n = labeledCount(laneOf(detail, o.id));
          const isHidden = p.hidden.has(o.id);
          return (
            <li key={o.id} className="flex items-center gap-1">
              <button type="button" aria-pressed={isActive} disabled={locked && !isActive}
                className={`btn flex-1 justify-start gap-2 ${isActive ? "btn-on" : ""}`} onClick={() => p.onSelect(o.id)}>
                <span aria-hidden className="h-3.5 w-3.5 shrink-0 rounded-sm border border-line" style={{ background: classColor(detail, o.class_id) }} />
                <span className="truncate">{label}</span>
                <span className="ml-auto text-xs opacity-70">{n} {n === 1 ? "frame" : "frames"}</span>
              </button>
              <button type="button" className="btn px-2" aria-pressed={isHidden} aria-label={`${isHidden ? "Show" : "Hide"} ${label}`}
                title={isHidden ? "Hidden from the frame (display only)" : "Hide from the frame (display only)"}
                onClick={() => p.onToggleHidden(o.id)}>
                {isHidden ? EYE_OFF : EYE}
              </button>
            </li>
          );
        })}
        {selection.kind === "new" && (
          <li>
            <div aria-current="true" className="btn btn-on w-full justify-start gap-2">
              <span aria-hidden className="h-3.5 w-3.5 shrink-0 rounded-sm border border-dashed border-line" style={{ background: classColor(detail, classId) }} />
              New object ({className(detail, classId) ?? "choose a class"})
            </div>
          </li>
        )}
      </ul>
      <label className="flex items-center gap-2 text-xs">
        <span className="w-28 shrink-0">{active !== null ? "Label a class" : "Class of the new object"}</span>
        <select aria-label="Class" className="field flex-1" disabled={locked} value={classId ?? ""} onChange={(e) => p.onClass(Number(e.target.value))}>
          {classId === null && <option value="">Choose a class</option>}
          {detail.classes.map((c) => (
            <option key={c.id} value={c.id}>{c.name}</option>
          ))}
        </select>
      </label>
      {active !== null && (
        <label className="flex items-center gap-2 text-xs">
          <span className="w-28 shrink-0">{`Change class of #${active.id}`}</span>
          <select aria-label={`New class of #${active.id}`} className="field flex-1" disabled={locked}
            value={relabelValue ?? ""} onChange={(e) => setRelabelTo(Number(e.target.value))}>
            {detail.classes.map((c) => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
          <button type="button" className="btn" disabled={locked || relabelValue === active.class_id}
            onClick={() => p.onRelabel(relabelValue as number)}>
            Change
          </button>
        </label>
      )}
      <div className="flex gap-1">
        <button type="button" className="btn flex-1" aria-keyshortcuts="N" disabled={locked || selection.kind === "new"} onClick={p.onNew}>
          + New object <kbd aria-hidden>N</kbd>
        </button>
        <button type="button" className="btn btn-danger" disabled={active === null || locked || p.deleteBlockedReason !== null}
          aria-label={active !== null ? `Delete ${objectLabel(detail, active)}` : undefined} onClick={p.onDelete}>
          Delete object
        </button>
      </div>
      {active !== null && p.deleteBlockedReason !== null && <p className="note">{p.deleteBlockedReason}</p>}
      <p className="note">
        <kbd>O</kbd> / <kbd>Shift+O</kbd> next / previous object · <kbd>N</kbd> new object
      </p>
      {objects.length === 0 && <p className="note">Nothing is labelled yet. Prompt on any frame, then accept the mask.</p>}
      {objects.length > 0 && selection.kind === "new" && (
        <p className="note">The new object is created when you accept or paint its first mask.</p>
      )}
      <form className="flex gap-1" onSubmit={add}>
        <input aria-label="New class" className="field flex-1" placeholder="New class" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="submit" className="btn">Add</button>
      </form>
      {p.error !== null && <p role="alert" className="text-xs text-danger">{p.error}</p>}
      {addError !== null && <p role="alert" className="text-xs text-danger">{addError}</p>}
    </section>
  );
}
