import type { ClipDetail, WriteTarget } from "../api/types";

/** The active row of the Objects panel: an existing object, or "New object" — which only comes into existence when
 * its first mask is accepted or painted, with `classId`. */
export type Selection = { kind: "object"; id: number } | { kind: "new"; classId: number | null };

export function sortedIds(detail: ClipDetail): number[] {
  return detail.clip.objects.map((o) => o.id).sort((a, b) => a - b);
}

/** A freshly opened clip: its lowest object id, or a new object of the first class when it has none. */
export function initialSelection(detail: ClipDetail): Selection {
  const ids = sortedIds(detail);
  return ids.length > 0 ? { kind: "object", id: ids[0] } : { kind: "new", classId: detail.classes[0]?.id ?? null };
}

/**
 * Keep `sel` valid for `detail`: a deleted object hands over to the next id (else the previous one, else "New
 * object"), and a new object whose class is unknown gets the first class. A valid selection is returned as is.
 * Ids are never reused, so a stale id can only ever mean "deleted".
 */
export function reconcile(sel: Selection, detail: ClipDetail): Selection {
  const ids = sortedIds(detail);
  if (sel.kind === "object") {
    if (ids.includes(sel.id)) return sel;
    const next = ids.find((id) => id > sel.id) ?? ids[ids.length - 1];
    return next !== undefined ? { kind: "object", id: next } : { kind: "new", classId: detail.classes[0]?.id ?? null };
  }
  if (sel.classId !== null && detail.classes.some((c) => c.id === sel.classId)) return sel;
  const classId = detail.classes[0]?.id ?? null;
  return classId === sel.classId ? sel : { kind: "new", classId };
}

/** `O` / `Shift+O`: the next / previous object in id order, wrapping; from "New object", the first / last. */
export function cycle(sel: Selection, detail: ClipDetail, dir: 1 | -1): Selection {
  const ids = sortedIds(detail);
  if (ids.length === 0) return sel;
  if (sel.kind === "new") return { kind: "object", id: dir === 1 ? ids[0] : ids[ids.length - 1] };
  const i = ids.indexOf(sel.id);
  const j = i === -1 ? 0 : (i + dir + ids.length) % ids.length;
  return { kind: "object", id: ids[j] };
}

/** Where an accept or a brush save writes; null while a new object has no class (or before the clip loads). */
export function writeTarget(selection: Selection | null): WriteTarget | null {
  if (selection === null) return null;
  if (selection.kind === "object") return { objectId: selection.id };
  return selection.classId === null ? null : { objectId: null, classId: selection.classId };
}
