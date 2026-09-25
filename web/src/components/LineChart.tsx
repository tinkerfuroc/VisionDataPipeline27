import { useState } from "react";
import { GRID, TEXT_DIM, TEXT_INK, WELL, extent, linePath, nearestIndex, niceTicks, scale, type Pt } from "../lib/chart";

export type Series = { name: string; color: string; points: Pt[] };

const W = 520;
const H = 200;
const M = { top: 12, right: 44, bottom: 26, left: 40 };

/**
 * A per-epoch line chart: 2 px lines, hairline grid, a legend only for two or more series, the last value written
 * at the end of a single line, and a crosshair + tooltip on hover. Text wears the ink tokens, never a series colour.
 */
export function LineChart({ title, series, yFormat = (v: number) => v.toFixed(2) }: {
  title: string;
  series: Series[];
  yFormat?: (v: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const x0 = xs[0] ?? 0;
  const x1 = Math.max(xs[xs.length - 1] ?? 1, x0 + 1);
  const [lo, hi] = extent(series.flatMap((s) => s.points.map((p) => p[1])));
  const yTicks = niceTicks(lo, hi, 4);
  const yDomain: [number, number] = [Math.min(lo, yTicks[0] ?? lo), Math.max(hi, yTicks[yTicks.length - 1] ?? hi)];
  const x = scale([x0, x1], [M.left, W - M.right]);
  const y = scale(yDomain, [H - M.bottom, M.top]);
  const xTicks = niceTicks(x0, x1, 5).filter(Number.isInteger);

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.ownerSVGElement?.getBoundingClientRect();
    const px = box && box.width > 0 ? ((e.clientX - box.left) * W) / box.width : e.clientX;
    const i = nearestIndex(xs, x0 + ((px - M.left) / (W - M.left - M.right)) * (x1 - x0));
    setHover(i < 0 ? null : xs[i]);
  };

  return (
    <figure className="flex min-w-0 flex-col gap-1 rounded border border-line bg-well p-2">
      <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 font-cond text-sm">
        <span className="font-semibold text-ink">{title}</span>
        {series.length >= 2 &&
          series.map((s) => (
            <span key={s.name} data-legend className="flex items-center gap-1 text-dim">
              <span aria-hidden className="inline-block h-0.5 w-3 rounded" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
      </figcaption>
      {xs.length === 0 ? (
        <p className="note px-2 py-6">No epochs yet.</p>
      ) : (
        <div className="relative">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${title} by epoch`}>
            {yTicks.map((t) => (
              <g key={`y${t}`}>
                <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={1} />
                <text x={M.left - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill={TEXT_DIM}>
                  {yFormat(t)}
                </text>
              </g>
            ))}
            {xTicks.map((t) => (
              <text key={`x${t}`} x={x(t)} y={H - 8} textAnchor="middle" fontSize={11} fill={TEXT_DIM}>
                {t}
              </text>
            ))}
            {series.map((s) =>
              s.points.length === 1 ? (
                <circle key={s.name} data-series={s.name} cx={x(s.points[0][0])} cy={y(s.points[0][1])} r={3} fill={s.color} />
              ) : (
                <path key={s.name} data-series={s.name} d={linePath(s.points, x, y)} fill="none" stroke={s.color}
                  strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              ),
            )}
            {series.length === 1 && series[0].points.length > 0 && (() => {
              const last = series[0].points[series[0].points.length - 1];
              return (
                <text x={x(last[0]) + 6} y={y(last[1]) + 4} fontSize={11} fill={TEXT_INK}>
                  {yFormat(last[1])}
                </text>
              );
            })()}
            {hover !== null && (
              <g>
                <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={H - M.bottom} stroke={TEXT_DIM} strokeWidth={1} />
                {series.map((s) => {
                  const p = s.points.find((q) => q[0] === hover);
                  return p ? <circle key={s.name} cx={x(p[0])} cy={y(p[1])} r={4} fill={s.color} stroke={WELL} strokeWidth={2} /> : null;
                })}
              </g>
            )}
            <rect data-hit x={M.left} y={M.top} width={W - M.left - M.right} height={H - M.top - M.bottom} fill="transparent"
              onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
          </svg>
          {hover !== null && (
            <div role="tooltip" className="pointer-events-none absolute top-1 rounded border border-line bg-panel px-2 py-1 font-cond text-xs"
              style={{ left: `${(x(hover) / W) * 100}%`, transform: x(hover) > W / 2 ? "translateX(calc(-100% - 8px))" : "translateX(8px)" }}>
              <div className="text-dim">Epoch {hover}</div>
              {series.map((s) => {
                const p = s.points.find((q) => q[0] === hover);
                return p ? (
                  <div key={s.name} className="flex items-center gap-1.5 whitespace-nowrap text-ink">
                    <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
                    {s.name} {yFormat(p[1])}
                  </div>
                ) : null;
              })}
            </div>
          )}
        </div>
      )}
    </figure>
  );
}
