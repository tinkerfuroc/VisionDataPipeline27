import { useEffect, useState } from "react";
import { api, valImageUrl } from "../api/rest";
import type { PredictionPage, Run } from "../api/types";
import { SERIES_COLORS, TEXT_INK, WELL } from "../lib/chart";
import { polygonPoints } from "../lib/train";

const PRED = SERIES_COLORS[0];

/** A finished run's predictions on the val images: ground truth outlined in ink, predictions filled in blue. */
export function PredictionPreview({ run }: { run: Run }) {
  const [index, setIndex] = useState(0);
  const [page, setPage] = useState<PredictionPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showGt, setShowGt] = useState(true);
  const [showPred, setShowPred] = useState(true);

  useEffect(() => setIndex(0), [run.id]);
  useEffect(() => {
    let closed = false;
    api.predictions(run.id, index, 1).then(
      (p) => !closed && (setPage(p), setError(null)),
      (e: unknown) => !closed && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      closed = true;
    };
  }, [run.id, index]);

  const total = page?.total ?? 0;
  const step = (d: number) => setIndex((i) => (total === 0 ? 0 : (i + d + total) % total));
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (error !== null) return <p role="alert" className="alert">{error}</p>;
  const item = page?.items[0];
  if (!item) return <p className="note">Loading predictions…</p>;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <button type="button" className="btn" aria-label="Previous image" onClick={() => step(-1)}>‹</button>
        <span className="font-cond text-dim">{index + 1} / {total} · {item.name}</span>
        <button type="button" className="btn" aria-label="Next image" onClick={() => step(1)}>›</button>
        <label className="flex items-center gap-1.5 text-dim">
          <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm border-2" style={{ borderColor: TEXT_INK }} />
          Ground truth
        </label>
        <label className="flex items-center gap-1.5 text-dim">
          <input type="checkbox" checked={showPred} onChange={(e) => setShowPred(e.target.checked)} />
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: PRED }} />
          Predictions (conf ≥ {page?.conf ?? 0.25})
        </label>
        <span className="note">{item.predictions.length} predicted, {item.ground_truth.length} labeled</span>
      </div>
      <div className="relative w-full overflow-hidden rounded border border-line bg-well">
        <img src={valImageUrl(run.export_id, item.name)} alt={item.name} className="block w-full" />
        <svg viewBox={`0 0 ${item.width} ${item.height}`} className="absolute inset-0 h-full w-full" aria-hidden>
          {showPred && item.predictions.flatMap((inst, i) =>
            inst.polygons.map((poly, j) => (
              <polygon key={`p${i}-${j}`} points={polygonPoints(poly, item.width, item.height)} fill={PRED} fillOpacity={0.35}
                stroke={PRED} strokeWidth={2} vectorEffect="non-scaling-stroke" />
            )))}
          {showGt && item.ground_truth.flatMap((inst, i) =>
            inst.polygons.map((poly, j) => (
              <polygon key={`g${i}-${j}`} points={polygonPoints(poly, item.width, item.height)} fill="none" stroke={TEXT_INK}
                strokeWidth={2} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
            )))}
          {showPred && item.predictions.map((inst, i) =>
            inst.score !== undefined && inst.polygons[0]?.length >= 2 ? (
              <text key={`s${i}`} x={inst.polygons[0][0] * item.width} y={inst.polygons[0][1] * item.height - 6}
                fontSize={Math.max(12, item.width / 60)} fill={TEXT_INK} stroke={WELL} strokeWidth={3} paintOrder="stroke">
                {inst.score.toFixed(2)}
              </text>
            ) : null)}
        </svg>
      </div>
    </div>
  );
}
