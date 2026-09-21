import { useState, type FormEvent } from "react";
import type { ClipDetail } from "../api/types";

type Props = {
  detail: ClipDetail;
  classId: number | null;
  onClassId: (id: number) => void;
  onAddClass: (name: string) => Promise<void>;
};

export function ObjectPanel({ detail, classId, onClassId, onAddClass }: Props) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const object = detail.clip.objects[0] ?? null;
  const pending = object !== null && classId !== null && classId !== object.class_id;

  const add = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (trimmed === "") return;
    setError(null);
    onAddClass(trimmed).then(() => setName(""), (err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  };

  const color = detail.classes.find((c) => c.id === classId)?.color ?? null;

  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Object</h2>
      <div className="flex items-center gap-2">
        <span aria-hidden className="h-4 w-4 shrink-0 rounded-sm border border-line" style={{ background: color ?? "transparent" }} />
        <select aria-label="Class" className="field flex-1" value={classId ?? ""} onChange={(e) => onClassId(Number(e.target.value))}>
          {classId === null && <option value="">Choose a class</option>}
          {detail.classes.map((c) => (
            <option key={c.id} value={c.id}>{c.name}</option>
          ))}
        </select>
      </div>
      <form className="flex gap-1" onSubmit={add}>
        <input aria-label="New class" className="field flex-1" placeholder="New class" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="submit" className="btn">Add</button>
      </form>
      {pending && <p className="note text-ink">The object changes class when you next accept a mask.</p>}
      {object === null && <p className="note">Nothing is labelled yet. Prompt on any frame, then accept the mask.</p>}
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
    </section>
  );
}
