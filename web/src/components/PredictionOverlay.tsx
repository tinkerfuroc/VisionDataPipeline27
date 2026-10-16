import type { Instance } from "../api/types";
import { SERIES_COLORS, TEXT_INK, WELL } from "../lib/chart";
import { polygonPoints } from "../lib/train";

const PRED = SERIES_COLORS[0];

function boxRect(box: number[], w: number, h: number, key: string, style: { stroke: string; strokeDasharray?: string }) {
  const [x1, y1, x2, y2] = box;
  return (
    <rect key={key} x={x1 * w} y={y1 * h} width={(x2 - x1) * w} height={(y2 - y1) * h} fill="none" strokeWidth={2}
      vectorEffect="non-scaling-stroke" {...style} />
  );
}

type Props = {
  width: number;
  height: number;
  groundTruth: Instance[];
  predictions: Instance[];
  showGt: boolean;
  showPred: boolean;
  scores: boolean;
};

/** Ground truth outlined in ink (dashed), predictions filled in blue; an absolutely positioned SVG over the image. A
 * detection run's instances are boxes (`box`): drawn as rectangle outlines, never filled like masks. */
export function PredictionOverlay({ width, height, groundTruth, predictions, showGt, showPred, scores }: Props) {
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      className="absolute inset-0 h-full w-full"
      aria-hidden
    >
      {showPred && predictions.flatMap((inst, i) =>
        inst.box !== undefined ? [boxRect(inst.box, width, height, `p${i}`, { stroke: PRED })] : inst.polygons.map((poly, j) => (
          <polygon key={`p${i}-${j}`} points={polygonPoints(poly, width, height)} fill={PRED} fillOpacity={0.35}
            stroke={PRED} strokeWidth={2} vectorEffect="non-scaling-stroke" />
        )))}
      {showGt && groundTruth.flatMap((inst, i) =>
        inst.box !== undefined ? [boxRect(inst.box, width, height, `g${i}`, { stroke: TEXT_INK, strokeDasharray: "6 4" })]
          : inst.polygons.map((poly, j) => (
          <polygon key={`g${i}-${j}`} points={polygonPoints(poly, width, height)} fill="none" stroke={TEXT_INK}
            strokeWidth={2} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
        )))}
      {scores && showPred && predictions.map((inst, i) =>
        inst.score !== undefined && inst.polygons[0]?.length >= 2 ? (
          <text key={`s${i}`} x={inst.polygons[0][0] * width} y={inst.polygons[0][1] * height - 6}
            fontSize={Math.max(12, width / 60)} fill={TEXT_INK} stroke={WELL} strokeWidth={3} paintOrder="stroke">
            {inst.score.toFixed(2)}
          </text>
        ) : null)}
    </svg>
  );
}
