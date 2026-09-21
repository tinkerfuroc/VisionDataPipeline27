import type { ClipDetail } from "../api/types";
import { counts, labeledPercent } from "../lib/timeline";

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function ClipInfo({ detail }: { detail: ClipDetail }) {
  const c = counts(detail.states);
  return (
    <section className="text-sm">
      <h2 className="truncate font-semibold">{detail.clip.source.name}</h2>
      <p className="text-xs text-neutral-400">
        {detail.clip.width}×{detail.clip.height} · {detail.clip.fps} fps · stride {detail.clip.stride}
      </p>
      <p className="mt-1 text-xs text-neutral-300">
        {plural(c.keyframe, "keyframe")} · {c.masked} masked · {c.empty} empty · {c.deleted} deleted ·{" "}
        {labeledPercent(detail.states)}% labeled
      </p>
    </section>
  );
}
