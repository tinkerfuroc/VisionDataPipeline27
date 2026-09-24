import type { RunDetail } from "../api/types";
import { SERIES_COLORS as C } from "../lib/chart";
import { BOX_MAP, MASK_MAP, MASK_MAP50, metricSeries, runLabel } from "../lib/train";
import { LineChart } from "./LineChart";

const loss = (v: number) => v.toFixed(3);

export function MetricCharts({ run }: { run: RunDetail }) {
  const rows = run.metrics;
  const s = (name: string, key: string, color: string) => ({ name, color, points: metricSeries(rows, key) });
  return (
    <div className="grid gap-3 xl:grid-cols-2">
      <LineChart title="Mask mAP" series={[s("mAP50-95", MASK_MAP, C[0]), s("mAP50", MASK_MAP50, C[1])]} />
      <LineChart title="Box mAP50-95" series={[s("Box mAP50-95", BOX_MAP, C[0])]} />
      <LineChart title="Training loss" yFormat={loss}
        series={[s("box", "train/box_loss", C[0]), s("seg", "train/seg_loss", C[1]), s("cls", "train/cls_loss", C[2])]} />
      <LineChart title="Validation loss" yFormat={loss}
        series={[s("box", "val/box_loss", C[0]), s("seg", "val/seg_loss", C[1]), s("cls", "val/cls_loss", C[2])]} />
    </div>
  );
}

/** Mask mAP50-95 of up to four runs; a run keeps its colour slot for as long as it stays in the comparison. */
export function CompareChart({ runs }: { runs: RunDetail[] }) {
  return (
    <LineChart title="Mask mAP50-95, compared"
      series={runs.slice(0, C.length).map((r, i) => ({ name: runLabel(r), color: C[i], points: metricSeries(r.metrics, MASK_MAP) }))} />
  );
}
