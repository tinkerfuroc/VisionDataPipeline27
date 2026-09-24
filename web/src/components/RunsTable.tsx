import type { Run, RunBest } from "../api/types";
import { weightsUrl } from "../api/rest";
import { bestRunIds, formatDuration, statusText } from "../lib/train";

const METRICS: { key: keyof RunBest; label: string }[] = [
  { key: "mask_map50", label: "Mask mAP50" },
  { key: "mask_map50_95", label: "Mask mAP50-95" },
  { key: "box_map50_95", label: "Box mAP50-95" },
];
const MAX_COMPARE = 4;
const stop = (e: React.SyntheticEvent) => e.stopPropagation();

type Props = {
  runs: Run[];
  selectedId: string | null;
  compareIds: string[];
  onSelect: (id: string) => void;
  onToggleCompare: (id: string) => void;
  onCancel: (id: string) => void;
  onResume: (id: string) => void;
  onOnnx: (id: string) => void;
  onDelete: (id: string) => void;
};

function OnnxCell({ run, onOnnx }: { run: Run; onOnnx: (id: string) => void }) {
  if (run.has_onnx) return <a className="btn" href={weightsUrl(run.id, "best.onnx")} download>best.onnx</a>;
  if (run.onnx === "queued" || run.onnx === "running") return <span className="note">Exporting…</span>;
  if (run.status !== "done") return null;
  const failed = run.onnx?.startsWith("error") ?? false;
  return (
    <button type="button" className="btn" title={failed ? run.onnx ?? undefined : undefined} onClick={() => onOnnx(run.id)}>
      {failed ? "Retry ONNX" : "Export ONNX"}
    </button>
  );
}

export function RunsTable({ runs, selectedId, compareIds, onSelect, onToggleCompare, onCancel, onResume, onOnnx, onDelete }: Props) {
  const best = Object.fromEntries(METRICS.map((m) => [m.key, bestRunIds(runs, m.key)]));
  if (runs.length === 0) return <p className="note">No training runs yet. Start one from the panel on the right.</p>;
  return (
    <div className="overflow-x-auto rounded border border-line">
      <table className="w-full text-sm">
        <thead className="bg-panel font-cond text-dim">
          <tr className="text-left">
            <th className="px-2 py-1.5 font-medium">Compare</th>
            <th className="px-2 py-1.5 font-medium">Model</th>
            <th className="px-2 py-1.5 font-medium">Export</th>
            <th className="px-2 py-1.5 font-medium">GPU</th>
            <th className="px-2 py-1.5 font-medium">Aug</th>
            <th className="px-2 py-1.5 font-medium">Status</th>
            <th className="px-2 py-1.5 text-right font-medium">Epochs</th>
            {METRICS.map((m) => (
              <th key={m.key} className="px-2 py-1.5 text-right font-medium">{m.label}</th>
            ))}
            <th className="px-2 py-1.5 text-right font-medium">Best epoch</th>
            <th className="px-2 py-1.5 text-right font-medium">Time</th>
            <th className="px-2 py-1.5 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const comparing = compareIds.includes(r.id);
            const active = r.status === "queued" || r.status === "running";
            return (
              <tr key={r.id} aria-selected={r.id === selectedId} onClick={() => onSelect(r.id)}
                className={`cursor-pointer border-t border-line ${r.id === selectedId ? "bg-raise" : "hover:bg-panel"}`}>
                <td className="px-2 py-1.5" onClick={stop}>
                  <input type="checkbox" aria-label={`Compare ${r.model}`} checked={comparing}
                    disabled={!comparing && compareIds.length >= MAX_COMPARE} onChange={() => onToggleCompare(r.id)} />
                </td>
                <td className="px-2 py-1.5 font-medium">{r.model}</td>
                <td className="px-2 py-1.5 text-dim">{r.project_name}</td>
                <td className="px-2 py-1.5 font-cond text-dim">cuda:{r.gpu}</td>
                <td className="px-2 py-1.5 text-dim">{r.params.augment ?? "default"}</td>
                <td className="px-2 py-1.5">
                  <div className="flex flex-col gap-1">
                    <span className={r.status === "error" ? "text-danger" : ""}>{statusText(r)}</span>
                    {r.status === "running" && r.progress !== null && r.progress.batches > 0 && (
                      <progress value={r.progress.batch} max={r.progress.batches} />
                    )}
                  </div>
                </td>
                <td className="px-2 py-1.5 text-right font-cond">{r.epochs_done}/{r.params.epochs}</td>
                {METRICS.map((m) => {
                  const v = r.best?.[m.key];
                  const text = typeof v === "number" ? v.toFixed(3) : "—";
                  return (
                    <td key={m.key} className="px-2 py-1.5 text-right font-cond">
                      {best[m.key].has(r.id) ? <strong className="font-semibold text-ink">{text}</strong> : <span className="text-dim">{text}</span>}
                    </td>
                  );
                })}
                <td className="px-2 py-1.5 text-right font-cond text-dim">{r.best?.epoch ?? "—"}</td>
                <td className="px-2 py-1.5 text-right font-cond text-dim">{formatDuration(r.best?.train_seconds)}</td>
                <td className="px-2 py-1.5" onClick={stop}>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {r.has_best && <a className="btn" href={weightsUrl(r.id, "best.pt")} download>best.pt</a>}
                    <OnnxCell run={r} onOnnx={onOnnx} />
                    {active && <button type="button" className="btn" onClick={() => onCancel(r.id)}>Cancel</button>}
                    {r.resumable && <button type="button" className="btn" onClick={() => onResume(r.id)}>Resume</button>}
                    {!active && r.onnx !== "queued" && r.onnx !== "running" && (
                      <button type="button" className="btn btn-danger" onClick={() => onDelete(r.id)}>Delete</button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
