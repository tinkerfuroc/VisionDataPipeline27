import { useCallback, useEffect, useMemo, useState } from "react";
import type { ClipDetail, LabelObject } from "../api/types";
import { cycle, initialSelection, reconcile, type Selection } from "../lib/activeObject";

export type ObjectsHandle = {
  /** the active row; null until the clip loads */
  selection: Selection | null;
  /** the active object's id; null on a "New object" row */
  activeId: number | null;
  active: LabelObject | null;
  /** the active object's class, or the class the new object will get */
  classId: number | null;
  /** objects hidden from the canvas (display only, for this page session and clip) */
  hidden: ReadonlySet<number>;
  select: (id: number) => void;
  step: (dir: 1 | -1) => void;
  startNew: () => void;
  setNewClass: (classId: number) => void;
  toggleHidden: (id: number) => void;
};

export function useObjects(clipId: string, detail: ClipDetail | null): ObjectsHandle {
  const [chosen, setChosen] = useState<Selection | null>(null);
  const [hidden, setHidden] = useState<ReadonlySet<number>>(() => new Set());

  useEffect(() => {
    setChosen(null);
    setHidden(new Set());
  }, [clipId]);

  // Derived, not stored: the moment a detail without the chosen object (or class) arrives, the selection moves on.
  const selection = useMemo(
    () => (detail === null ? null : reconcile(chosen ?? initialSelection(detail), detail)),
    [chosen, detail],
  );
  const active =
    selection?.kind === "object" ? (detail?.clip.objects.find((o) => o.id === selection.id) ?? null) : null;
  const classId = selection === null ? null : selection.kind === "object" ? (active?.class_id ?? null) : selection.classId;

  const select = useCallback((id: number) => setChosen({ kind: "object", id }), []);
  const step = useCallback(
    (dir: 1 | -1) => {
      if (detail !== null && selection !== null) setChosen(cycle(selection, detail, dir));
    },
    [detail, selection],
  );
  const startNew = useCallback(() => setChosen({ kind: "new", classId }), [classId]);
  const setNewClass = useCallback((id: number) => setChosen({ kind: "new", classId: id }), []);
  const toggleHidden = useCallback(
    (id: number) =>
      setHidden((h) => {
        const next = new Set(h);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }),
    [],
  );

  return { selection, activeId: active?.id ?? null, active, classId, hidden, select, step, startNew, setNewClass, toggleHidden };
}
