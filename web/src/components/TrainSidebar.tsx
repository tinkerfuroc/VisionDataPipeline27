import { useEffect, useState } from "react";
import { api } from "../api/rest";
import type { AugmentMode, AugParams, ExportInfo, GpuQueue, Run, StartRunsRequest, TrainParams, TrainStatus } from "../api/types";
import { exportLabel } from "../lib/exportName";
import { defaultGpu, startProblem, statusText } from "../lib/train";
import { AugmentPanel } from "./AugmentPanel";

type ModelPick = { checked: boolean; gpu: number };
const PARAM_FIELDS: { key: "epochs" | "imgsz" | "batch" | "patience"; label: string; min: number; max: number; step: number }[] = [
  { key: "epochs", label: "Epochs", min: 1, max: 1000, step: 1 },
  { key: "imgsz", label: "Image size (px)", min: 320, max: 1920, step: 32 },
  { key: "batch", label: "Batch", min: 1, max: 64, step: 1 },
  { key: "patience", label: "Early-stop patience", min: 0, max: 1000, step: 1 },
];
const AUG_FIELDS: { key: keyof Omit<AugParams, "photometric">; min: number; max: number; step: number }[] = [
  { key: "degrees", min: 0, max: 180, step: 1 },
  { key: "shear", min: 0, max: 45, step: 1 },
  { key: "perspective", min: 0, max: 0.001, step: 0.0001 },
  { key: "flipud", min: 0, max: 1, step: 0.05 },
  { key: "fliplr", min: 0, max: 1, step: 0.05 },
  { key: "scale", min: 0, max: 0.9, step: 0.05 },
  { key: "translate", min: 0, max: 0.9, step: 0.05 },
  { key: "hsv_h", min: 0, max: 0.1, step: 0.005 },
  { key: "hsv_s", min: 0, max: 1, step: 0.05 },
  { key: "hsv_v", min: 0, max: 1, step: 0.05 },
  { key: "mosaic", min: 0, max: 1, step: 0.05 },
  { key: "mixup", min: 0, max: 1, step: 0.05 },
  { key: "copy_paste", min: 0, max: 1, step: 0.05 },
];
const gib = (bytes: number) => `${Math.round(bytes / 2 ** 30)} GiB`;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Runs training on `g` that this server process is not itself running (e.g. a `vsp train` CLI
 * process sharing the same data root): present in the `runs` prop but not `g`'s own `running` slot. */
const foreignRunning = (g: GpuQueue, runs: Run[]) =>
  runs.filter((r) => r.status === "running" && r.gpu === g.index && r.id !== g.running);

/** A note for the New-runs form when `gpu` is already busy training, whether owned by this server
 * process (queues behind it) or by another process entirely (foreign; risk of running out of VRAM). */
function gpuBusyNote(status: TrainStatus, runs: Run[], gpu: number): string | null {
  const g = status.gpus.find((gg) => gg.index === gpu);
  if (g === undefined) return null;
  if (g.running !== null && g.running_kind === "train") {
    const model = runs.find((r) => r.id === g.running)?.model ?? "a run";
    return `cuda:${gpu} is already training ${model}; this run will queue behind it.`;
  }
  if (foreignRunning(g, runs).length > 0) {
    return `cuda:${gpu} is busy with a vsp train run in another process; two runs on one GPU may run out of memory.`;
  }
  return null;
}

type Props = {
  status: TrainStatus | null;
  exports: ExportInfo[];
  runs: Run[];
  exportId: string | null;
  onExportChange: (id: string) => void;
  onExportsChanged: () => void;
  onStart: (body: StartRunsRequest) => Promise<void>;
  onCancel: (id: string) => void;
};

export function TrainSidebar({ status, exports, runs, exportId, onExportChange, onExportsChanged, onStart, onCancel }: Props) {
  const [picks, setPicks] = useState<Record<string, ModelPick>>({});
  const [showAll, setShowAll] = useState(false);
  const [params, setParams] = useState<TrainParams | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  const selectedExport = exports.find((e) => e.id === exportId) ?? null;

  useEffect(() => {
    if (status === null || Object.keys(picks).length > 0) return;
    const gpu = defaultGpu(status) ?? 0;
    setPicks(Object.fromEntries(status.models.map((m) => [m, { checked: status.default_models.includes(m), gpu }])));
    setParams(status.defaults);
  }, [status, picks]);

  // Resets whenever a different export is selected, or that export's own name changes (e.g. after a rename).
  useEffect(() => {
    setDraftName(selectedExport?.name ?? "");
  }, [selectedExport?.id, selectedExport?.name]);

  if (status === null || params === null) return <aside className="note p-3">Loading…</aside>;

  const onAugmentChange = (mode: AugmentMode) =>
    setParams((p) => (p === null ? p : { ...p, augment: mode, aug: mode === "custom" ? (p.aug ?? status.aug_presets.strong) : null }));
  const setAug = (patch: Partial<AugParams>) =>
    setParams((p) => (p === null || !p.aug ? p : { ...p, aug: { ...p.aug, ...patch } }));

  const visible = showAll ? status.models : status.models.filter((m) => status.default_models.includes(m) || picks[m]?.checked);
  const chosen = status.models.filter((m) => picks[m]?.checked);
  const onSam3 = status.sam3_gpu !== null && chosen.some((m) => picks[m].gpu === status.sam3_gpu);
  const problem = startProblem({ available: status.available, error: status.error, exportId, picked: chosen.length });
  const byId = new Map(runs.map((r) => [r.id, r]));

  const start = async () => {
    if (problem !== null || starting || exportId === null) return;
    setStarting(true);
    setStartError(null);
    try {
      await onStart({ export_id: exportId, runs: chosen.map((m) => ({ model: m, gpu: picks[m].gpu })), params });
    } catch (e) {
      setStartError(message(e));
    } finally {
      setStarting(false);
    }
  };

  const rename = async () => {
    if (selectedExport === null) return;
    setRenaming(true);
    setRenameError(null);
    try {
      await api.renameExport(selectedExport.id, draftName.trim() === "" ? null : draftName.trim());
      onExportsChanged();
    } catch (e) {
      setRenameError(message(e));
    } finally {
      setRenaming(false);
    }
  };

  return (
    <aside className="flex shrink-0 flex-col gap-5 border-t border-line bg-panel p-3 md:w-96 md:overflow-y-auto md:border-l md:border-t-0">
      <section className="flex flex-col gap-2">
        <h2 className="heading">New runs</h2>
        <label className="flex flex-col gap-1 text-sm text-dim">
          Export
          <select aria-label="Export" className="field" value={exportId ?? ""} onChange={(e) => onExportChange(e.target.value)}>
            {exports.length === 0 && <option value="">No yolo_seg export yet</option>}
            {exports.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name !== null ? exportLabel(e) : e.project_name} · {e.id} ·{" "}
                {e.train_images} train / {e.val_images} val
              </option>
            ))}
          </select>
        </label>
        {selectedExport !== null && (
          <>
            <form className="flex gap-1" onSubmit={(ev) => { ev.preventDefault(); void rename(); }}>
              <input aria-label="Rename export" className="field flex-1" maxLength={80} placeholder="Name"
                value={draftName} onChange={(e) => setDraftName(e.target.value)} />
              <button type="submit" className="btn" disabled={renaming || draftName.trim() === (selectedExport.name ?? "")}>
                Save
              </button>
            </form>
            {renameError !== null && <p role="alert" className="text-xs text-danger">{renameError}</p>}
          </>
        )}
        <AugmentPanel exportInfo={selectedExport}
          onCreated={(id) => {
            onExportsChanged();
            onExportChange(id);
          }} />
        <fieldset className="flex flex-col gap-1.5">
          <legend className="mb-1 text-sm text-dim">Models</legend>
          {visible.map((m) => (
            <div key={m} className="flex items-center justify-between gap-2 text-sm">
              <label className="flex items-center gap-2">
                <input type="checkbox" aria-label={m} checked={picks[m]?.checked ?? false}
                  onChange={(e) => setPicks((p) => ({ ...p, [m]: { ...p[m], checked: e.target.checked } }))} />
                {m}
              </label>
              <select aria-label={`GPU for ${m}`} className="field py-0.5" value={picks[m]?.gpu ?? 0}
                onChange={(e) => setPicks((p) => ({ ...p, [m]: { ...p[m], gpu: Number(e.target.value) } }))}>
                {status.gpus.map((g) => (
                  <option key={g.index} value={g.index}>
                    cuda:{g.index} {g.name}{g.index === status.sam3_gpu ? " (shared with SAM3)" : ""}
                  </option>
                ))}
              </select>
            </div>
          ))}
          <button type="button" className="btn self-start" onClick={() => setShowAll((v) => !v)}>
            {showAll ? "Fewer models" : "More models"}
          </button>
        </fieldset>
        {onSam3 && (
          <p className="note text-ink">
            SAM3 runs on cuda:{status.sam3_gpu}. Training there shares its memory; labeling or training may run out of VRAM.
          </p>
        )}
        {[...new Set(chosen.map((m) => picks[m].gpu))].map((gpu) => {
          const note = gpuBusyNote(status, runs, gpu);
          return note !== null ? (
            <p key={gpu} className="note text-ink">{note}</p>
          ) : null;
        })}
        {PARAM_FIELDS.map((f) => (
          <label key={f.key} className="flex items-center justify-between gap-2 text-sm text-dim">
            {f.label}
            <input aria-label={f.label} type="number" className="field w-24" min={f.min} max={f.max} step={f.step}
              value={params[f.key]} onChange={(e) => setParams({ ...params, [f.key]: Number(e.target.value) || f.min })} />
          </label>
        ))}
        <label className="flex flex-col gap-1 text-sm text-dim">
          Augmentation
          <select aria-label="Augmentation" className="field" value={params.augment ?? "default"}
            onChange={(e) => onAugmentChange(e.target.value as AugmentMode)}>
            <option value="default">Default</option>
            <option value="strong">Strong</option>
            <option value="custom">Custom</option>
          </select>
        </label>
        {params.augment === "custom" && params.aug && (
          <fieldset className="grid grid-cols-2 gap-x-2 gap-y-1.5">
            <legend className="sr-only">Custom augmentation</legend>
            {AUG_FIELDS.map((f) => (
              <label key={f.key} className="flex items-center justify-between gap-1 text-sm text-dim">
                {f.key}
                <input aria-label={f.key} type="number" className="field w-20" min={f.min} max={f.max} step={f.step}
                  value={params.aug![f.key]} onChange={(e) => setAug({ [f.key]: Number(e.target.value) || f.min })} />
              </label>
            ))}
            <label className="col-span-2 flex items-center gap-2 text-sm text-dim">
              <input type="checkbox" checked={params.aug.photometric} onChange={(e) => setAug({ photometric: e.target.checked })} />
              Blur, noise, JPEG, contrast
            </label>
          </fieldset>
        )}
        {problem !== null && <p className="note text-ink">{problem}</p>}
        {startError !== null && <p role="alert" className="alert">{startError}</p>}
        <button type="button" className="btn btn-on" disabled={problem !== null || starting} onClick={() => void start()}>
          Start {chosen.length} run{chosen.length === 1 ? "" : "s"}
        </button>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="heading">GPU queues</h2>
        {status.gpus.length === 0 && <p className="note">No CUDA GPU is visible to the server.</p>}
        {status.gpus.map((g) => {
          const current = g.running ? byId.get(g.running) : undefined;
          const foreign = foreignRunning(g, runs);
          return (
            <div key={g.index} className="flex flex-col gap-1.5 rounded border border-line p-2 text-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">cuda:{g.index} {g.name}</span>
                <span className="font-cond text-dim">{gib(g.memory_total)}{g.index === status.sam3_gpu ? " · SAM3" : ""}</span>
              </div>
              {current === undefined && g.queue.length === 0 && foreign.length === 0 && <span className="note">Idle</span>}
              {current !== undefined && (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center justify-between gap-2">
                    <span>
                      {current.model} · {g.running_kind === "onnx" ? "exporting ONNX" : statusText(current)}
                    </span>
                    {g.running_kind === "train" && (
                      <button type="button" className="btn" onClick={() => onCancel(current.id)}>Cancel</button>
                    )}
                  </div>
                  {g.running_kind === "train" && current.progress !== null && current.progress.batches > 0 && (
                    <progress value={current.progress.batch} max={current.progress.batches} />
                  )}
                </div>
              )}
              {foreign.map((r) => (
                <div key={r.id} className="flex flex-col gap-1">
                  <span>{r.model} · {statusText(r)} · running in another process (vsp train)</span>
                  {r.progress !== null && r.progress.batches > 0 && (
                    <progress value={r.progress.batch} max={r.progress.batches} />
                  )}
                </div>
              ))}
              {g.queue.map((id, i) => (
                <div key={id} className="flex items-center justify-between gap-2 text-dim">
                  <span>{i + 1}. {byId.get(id)?.model ?? id}</span>
                  <button type="button" className="btn" aria-label={`Cancel queued ${byId.get(id)?.model ?? id}`} onClick={() => onCancel(id)}>
                    Cancel
                  </button>
                </div>
              ))}
            </div>
          );
        })}
      </section>
    </aside>
  );
}
