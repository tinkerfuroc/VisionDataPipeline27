import { useState, type FormEvent } from "react";
import type { ClipDetail } from "../api/types";

type Props = {
  detail: ClipDetail;
  classId: number | null;
  onClassId: (id: number) => void;
  onAddClass: (name: string) => Promise<void>;
};

const field = "min-w-0 flex-1 rounded border border-neutral-600 bg-neutral-800 px-2 py-1";

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

  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Object</h2>
      <select aria-label="Class" className={field} value={classId ?? ""} onChange={(e) => onClassId(Number(e.target.value))}>
        {classId === null && <option value="">(choose a class)</option>}
        {detail.classes.map((c) => (
          <option key={c.id} value={c.id}>{c.name}</option>
        ))}
      </select>
      <form className="flex gap-1" onSubmit={add}>
        <input aria-label="New class" className={field} placeholder="New class" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="submit" className="rounded bg-neutral-700 px-3 py-1">Add</button>
      </form>
      {pending && <p className="text-xs text-amber-400">The new class is applied with the next Accept.</p>}
      {object === null && <p className="text-xs text-neutral-400">No object yet — prompt on any frame and accept the mask.</p>}
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}
    </section>
  );
}
