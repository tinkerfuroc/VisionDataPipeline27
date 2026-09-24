import { useEffect, useRef, useState } from "react";
import { api, exportImageUrl } from "../api/rest";
import type { ExportInfo, ExportSample } from "../api/types";
import { GalleryCard } from "./GalleryCard";

const VISIBLE = 4;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function CompositeSamples({ exportInfo }: { exportInfo: ExportInfo }) {
  const [seed, setSeed] = useState(0);
  const [samples, setSamples] = useState<ExportSample[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [first, setFirst] = useState(0);
  const strip = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setSamples(null);
    setError(null);
    setFirst(0);
    api.exportSamples(exportInfo.id, 12, seed).then(
      (s) => {
        if (!cancelled) setSamples(s);
      },
      (e) => {
        if (!cancelled) setError(message(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [exportInfo.id, seed]);

  const total = samples?.length ?? 0;
  const page = (d: number) => {
    const next = Math.max(0, Math.min(first + d * VISIBLE, Math.max(0, total - 1)));
    setFirst(next);
    const card = strip.current?.children[next] as HTMLElement | undefined;
    card?.scrollIntoView?.({ inline: "start", block: "nearest", behavior: "smooth" });
  };
  const last = Math.min(first + VISIBLE, total);

  return (
    <section className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="heading">Composite samples · {exportInfo.id}</h2>
        <span className="flex-1" />
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>Shuffle</button>
      </div>
      {error !== null && <p role="alert" className="alert">{error}</p>}
      {error === null && samples === null && <p className="note">Loading samples…</p>}
      {samples !== null && samples.length === 0 && <p className="note">No synthetic train images yet.</p>}
      {samples !== null && samples.length > 0 && (
        <div className="flex items-center gap-2">
          <button type="button" className="btn self-center" aria-label="Previous four" disabled={first === 0}
            onClick={() => page(-1)}>‹</button>
          <div ref={strip}
            className="grid min-w-0 flex-1 snap-x snap-mandatory auto-cols-[calc((100%-2.25rem)/4)] grid-flow-col gap-3 overflow-x-auto pb-1">
            {samples.map((s) => (
              <GalleryCard key={s.name} src={exportImageUrl(exportInfo.id, "train", s.name)} name={s.name}
                width={s.width} height={s.height} groundTruth={s.ground_truth} predictions={[]} showGt
                showPred={false} caption={<span className="text-ink">{s.name}</span>} />
            ))}
          </div>
          <button type="button" className="btn self-center" aria-label="Next four" disabled={last >= total}
            onClick={() => page(1)}>›</button>
        </div>
      )}
    </section>
  );
}
