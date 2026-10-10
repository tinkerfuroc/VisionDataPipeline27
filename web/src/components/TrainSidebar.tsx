import { useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import type {
  AugmentMode, AugParams, ExportInfo, FromCheckpointRequest, GpuQueue, Run, StartMode, StartRunsRequest, TrainParams,
  TrainStatus,
} from "../api/types";
import { exportLabel } from "../lib/exportName";
import {
  checkpointSources, classesCompatible, continueModeLabel, defaultGpu, defaultMode, finetuneModeLabel,
  gpuOptionLabel, modeDisabledReason, sourceRunLabel, startProblem, statusText,
} from "../lib/train";
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
const gbFree = (bytes: number) => `${(bytes / 2 ** 30).toFixed(1)} GiB free`;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Runs training on `g` that this server process is not itself running (e.g. a `vdp train` CLI
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
    return `cuda:${gpu} is busy with a vdp train run in another process; two runs on one GPU may run out of memory.`;
  }
  return null;
}

/** "Stopped at epoch N of M on cuda:X (<name>)" — the GPU name is skipped if that GPU is no longer listed. */
function stoppedAt(run: Run, status: TrainStatus): string {
  const g = status.gpus.find((gg) => gg.index === run.gpu);
  const name = g !== undefined ? ` (${g.name.replace(/^NVIDIA GeForce /, "")})` : "";
  return `Stopped at epoch ${run.epochs_done} of ${run.params.epochs} on cuda:${run.gpu}${name}.`;
}

/** The Mark-complete button's note: the best epoch so far (and its mask mAP50-95, when known --
 * a stopped run's own `best` is null until it is actually marked complete, but this still shows
 * it if a future caller ever supplies one) before the fixed description of what the action does. */
function markCompleteNote(best: Run["best"]): string {
  if (best === null) return "Keeps the best epoch so far, runs the validation predictions on that GPU, then marks the run done.";
  const mask = best.mask_map50_95 !== null ? `, mask mAP50-95 ${best.mask_map50_95.toFixed(3)}` : "";
  return `Keeps the best epoch so far (epoch ${best.epoch}${mask}), runs the validation predictions on that GPU, then marks the run done.`;
}

/** The right sidebar's "Stopped run" section for a selected run that is resumable, completable,
 * or both: its own GPU choice (defaulting to the fastest idle one), the same SAM3/busy notes as
 * the New-runs form, and one button per action it is eligible for, each naming exactly which GPU
 * it will act on. Its own component (rather than inline in `TrainSidebar`) so its state can key
 * off `run.id` and reset cleanly when the selection changes. */
function StoppedRunSection({
  status, runs, run, focus, onResume, onComplete, onTrainOnNewData,
}: {
  status: TrainStatus;
  runs: Run[];
  run: Run;
  /** changes whenever the table's Resume/Mark complete button is pressed */
  focus: number;
  onResume: (id: string, gpu: number) => Promise<void>;
  onComplete: (id: string, gpu: number) => Promise<void>;
  onTrainOnNewData: (id: string) => void;
}) {
  const [gpu, setGpu] = useState(defaultGpu(status) ?? status.gpus[0]?.index ?? 0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const onSam3 = status.sam3_gpu !== null && gpu === status.sam3_gpu;
  const note = gpuBusyNote(status, runs, gpu);
  // the table's Resume/Mark complete button opens this section; bring it into view even if the sidebar was scrolled down
  const ref = useRef<HTMLElement>(null);
  useEffect(() => ref.current?.scrollIntoView?.({ block: "nearest" }), [focus]);

  const act = (action: (id: string, gpu: number) => Promise<void>) => async () => {
    setBusy(true);
    setError(null);
    try {
      await action(run.id, gpu);
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section ref={ref} className="flex flex-col gap-2">
      <h2 className="heading">Stopped {run.model} run</h2>
      <p className="note">{stoppedAt(run, status)}</p>
      <label className="flex flex-col gap-1 text-sm text-dim">
        GPU
        <select aria-label="Resume GPU" className="field" value={gpu} onChange={(e) => setGpu(Number(e.target.value))}>
          {status.gpus.map((g) => (
            <option key={g.index} value={g.index}>{gpuOptionLabel(g, status)}</option>
          ))}
        </select>
      </label>
      {onSam3 && (
        <p className="note text-ink">
          SAM3 runs on cuda:{status.sam3_gpu}. Training there shares its memory; labeling or training may run out of VRAM.
        </p>
      )}
      {note !== null && <p className="note text-ink">{note}</p>}
      {error !== null && <p role="alert" className="alert">{error}</p>}
      <div className="flex flex-wrap gap-2">
        {run.resumable && (
          <button type="button" className="btn btn-on" disabled={busy} onClick={() => void act(onResume)()}>
            Resume on cuda:{gpu}
          </button>
        )}
        {run.completable && (
          <button type="button" className="btn btn-on" disabled={busy} onClick={() => void act(onComplete)()}>
            Mark complete on cuda:{gpu}
          </button>
        )}
        <button type="button" className="btn" onClick={() => onTrainOnNewData(run.id)}>Train on new data</button>
      </div>
      {run.completable && <p className="note">{markCompleteNote(run.best)}</p>}
    </section>
  );
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
  selectedRun: Run | null;
  onResume: (id: string, gpu: number) => Promise<void>;
  onComplete: (id: string, gpu: number) => Promise<void>;
  /** bumped by the table's Resume/Mark complete button: scrolls the Stopped run section into view */
  stoppedFocus?: number;
  /** New-runs form's "Start from": null = "Pretrained weights" (today's behaviour). */
  sourceRunId: string | null;
  onSourceRunChange: (id: string | null) => void;
  onStartFromCheckpoint: (sourceId: string, body: FromCheckpointRequest) => Promise<void>;
  /** the Stopped-run section's and the run detail header's "Train on new data" buttons: sets the
   * New-runs form's "Start from" to that run and scrolls the form into view. */
  onTrainOnNewData: (id: string) => void;
  /** bumped by `onTrainOnNewData`: scrolls the New-runs section into view */
  newRunsFocus?: number;
};

export function TrainSidebar({
  status, exports, runs, exportId, onExportChange, onExportsChanged, onStart, onCancel, selectedRun, onResume,
  onComplete, stoppedFocus = 0, sourceRunId, onSourceRunChange, onStartFromCheckpoint, onTrainOnNewData,
  newRunsFocus = 0,
}: Props) {
  const [picks, setPicks] = useState<Record<string, ModelPick>>({});
  const [batchGpu, setBatchGpu] = useState<number | null>(null);
  const [perModelGpu, setPerModelGpu] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [params, setParams] = useState<TrainParams | null>(null);
  const [mode, setMode] = useState<StartMode>("continue");
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const newRunsRef = useRef<HTMLElement>(null);

  const selectedExport = exports.find((e) => e.id === exportId) ?? null;
  const sourceRun = runs.find((r) => r.id === sourceRunId) ?? null;

  useEffect(() => {
    if (status === null || Object.keys(picks).length > 0) return;
    const gpu = defaultGpu(status) ?? 0;
    setPicks(Object.fromEntries(status.models.map((m) => [m, { checked: status.default_models.includes(m), gpu }])));
    setBatchGpu(gpu);
    setParams(status.defaults);
  }, [status, picks]);

  // Resets whenever a different export is selected, or that export's own name changes (e.g. after a rename).
  useEffect(() => {
    setDraftName(selectedExport?.name ?? "");
  }, [selectedExport?.id, selectedExport?.name]);

  // A newly chosen "Start from" run picks its own default mode: Continue when allowed, else Fine-tune.
  useEffect(() => {
    if (sourceRun !== null) setMode(defaultMode(sourceRun));
  }, [sourceRun?.id]);

  // `onTrainOnNewData` (the Stopped-run section's and the run detail header's own buttons) brings
  // this whole New-runs section into view, even if the sidebar was scrolled past it.
  useEffect(() => {
    newRunsRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [newRunsFocus]);

  if (status === null || params === null) return <aside className="note p-3">Loading…</aside>;

  const onAugmentChange = (mode: AugmentMode) =>
    setParams((p) => (p === null ? p : { ...p, augment: mode, aug: mode === "custom" ? (p.aug ?? status.aug_presets.strong) : null }));
  const setAug = (patch: Partial<AugParams>) =>
    setParams((p) => (p === null || !p.aug ? p : { ...p, aug: { ...p.aug, ...patch } }));

  const gpuOf = (m: string) => (perModelGpu ? (picks[m]?.gpu ?? batchGpu ?? 0) : (batchGpu ?? 0));
  const togglePerModelGpu = (checked: boolean) => {
    setPerModelGpu(checked);
    // Revealing the per-model selects seeds every one of them from the current batch choice.
    if (checked) setPicks((p) => Object.fromEntries(Object.entries(p).map(([m, v]) => [m, { ...v, gpu: batchGpu ?? v.gpu }])));
  };

  const visible = showAll ? status.models : status.models.filter((m) => status.default_models.includes(m) || picks[m]?.checked);
  const chosen = status.models.filter((m) => picks[m]?.checked);
  const onSam3 = status.sam3_gpu !== null && chosen.some((m) => gpuOf(m) === status.sam3_gpu);
  const problem = startProblem({ available: status.available, error: status.error, exportId, picked: chosen.length });
  const byId = new Map(runs.map((r) => [r.id, r]));

  // Continue keeps the source's own params, so they are shown but disabled; Fine-tune (or the
  // ordinary Pretrained-weights flow) edits `params` as normal.
  const paramsLocked = sourceRun !== null && mode === "continue";
  const displayParams = paramsLocked && sourceRun !== null ? sourceRun.params : params;
  const exportMismatch = sourceRun !== null && selectedExport !== null && !classesCompatible(selectedExport, sourceRun);
  const fromCheckpointProblem: string | null =
    !status.available ? (status.error ?? "Training is not available on this server.")
      : exportId === null ? "Choose an export to train on."
        : exportMismatch ? "This export's classes do not match the source run's classes."
          : null;
  const effectiveProblem = sourceRun !== null ? fromCheckpointProblem : problem;

  const start = async () => {
    if (effectiveProblem !== null || starting || exportId === null) return;
    setStarting(true);
    setStartError(null);
    try {
      if (sourceRun !== null) {
        await onStartFromCheckpoint(sourceRun.id, {
          export_id: exportId, mode, gpu: batchGpu ?? 0, ...(mode === "finetune" ? { params } : {}),
        });
      } else {
        await onStart({ export_id: exportId, runs: chosen.map((m) => ({ model: m, gpu: gpuOf(m) })), params });
      }
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
      {selectedRun !== null && (selectedRun.resumable || selectedRun.completable) && (
        <StoppedRunSection key={selectedRun.id} status={status} runs={runs} run={selectedRun} focus={stoppedFocus}
          onResume={onResume} onComplete={onComplete} onTrainOnNewData={onTrainOnNewData} />
      )}
      <section ref={newRunsRef} className="flex flex-col gap-2">
        <h2 className="heading">New runs</h2>
        <label className="flex flex-col gap-1 text-sm text-dim">
          Start from
          <select aria-label="Start from" className="field" value={sourceRunId ?? ""}
            onChange={(e) => onSourceRunChange(e.target.value === "" ? null : e.target.value)}>
            <option value="">Pretrained weights</option>
            {checkpointSources(runs).map((r) => (
              <option key={r.id} value={r.id}>{sourceRunLabel(r, exports)}</option>
            ))}
          </select>
        </label>
        {sourceRun !== null && (
          <fieldset className="flex flex-col gap-1.5">
            <legend className="mb-1 text-sm text-dim">Mode</legend>
            {(["continue", "finetune"] as const).map((m) => {
              const reason = modeDisabledReason(sourceRun, m);
              return (
                <label key={m} className="flex flex-col gap-0.5 text-sm">
                  <span className="flex items-start gap-2">
                    <input type="radio" aria-label={m === "continue" ? "Continue" : "Fine-tune"} name="start-mode"
                      checked={mode === m} disabled={reason !== null} onChange={() => setMode(m)} />
                    <span>{m === "continue" ? continueModeLabel(sourceRun) : finetuneModeLabel(sourceRun)}</span>
                  </span>
                  {reason !== null && <span className="note pl-6">Disabled: {reason}.</span>}
                </label>
              );
            })}
          </fieldset>
        )}
        {sourceRun !== null && <p className="text-sm text-dim">Model: {sourceRun.model}</p>}
        <label className="flex flex-col gap-1 text-sm text-dim">
          Export
          <select aria-label="Export" className="field" value={exportId ?? ""} onChange={(e) => onExportChange(e.target.value)}>
            {exports.length === 0 && <option value="">No yolo_seg export yet</option>}
            {exports.map((e) => {
              const incompatible = sourceRun !== null && !classesCompatible(e, sourceRun);
              return (
                <option key={e.id} value={e.id} disabled={incompatible}>
                  {e.name !== null ? exportLabel(e) : e.project_name} · {e.id} ·{" "}
                  {e.train_images} train / {e.val_images} val{incompatible ? " — different classes" : ""}
                </option>
              );
            })}
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
        <label className="flex flex-col gap-1 text-sm text-dim">
          GPU
          <select aria-label="GPU" className="field" value={batchGpu ?? 0} onChange={(e) => setBatchGpu(Number(e.target.value))}>
            {status.gpus.map((g) => (
              <option key={g.index} value={g.index}>{gpuOptionLabel(g, status)}</option>
            ))}
          </select>
        </label>
        {sourceRun === null && (
          <label className="flex items-center gap-2 text-sm text-dim">
            <input type="checkbox" checked={perModelGpu} onChange={(e) => togglePerModelGpu(e.target.checked)} />
            Different GPU per model
          </label>
        )}
        {sourceRun === null && (
          <fieldset className="flex flex-col gap-1.5">
            <legend className="mb-1 text-sm text-dim">Models</legend>
            {visible.map((m) => (
              <div key={m} className="flex items-center justify-between gap-2 text-sm">
                <label className="flex items-center gap-2">
                  <input type="checkbox" aria-label={m} checked={picks[m]?.checked ?? false}
                    onChange={(e) => setPicks((p) => ({ ...p, [m]: { ...p[m], checked: e.target.checked } }))} />
                  {m}
                </label>
                {perModelGpu && (
                  <select aria-label={`GPU for ${m}`} className="field py-0.5" value={picks[m]?.gpu ?? batchGpu ?? 0}
                    onChange={(e) => setPicks((p) => ({ ...p, [m]: { ...p[m], gpu: Number(e.target.value) } }))}>
                    {status.gpus.map((g) => (
                      <option key={g.index} value={g.index}>{gpuOptionLabel(g, status)}</option>
                    ))}
                  </select>
                )}
              </div>
            ))}
            <button type="button" className="btn self-start" onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Fewer models" : "More models"}
            </button>
          </fieldset>
        )}
        {sourceRun === null && onSam3 && (
          <p className="note text-ink">
            SAM3 runs on cuda:{status.sam3_gpu}. Training there shares its memory; labeling or training may run out of VRAM.
          </p>
        )}
        {sourceRun === null && [...new Set(chosen.map((m) => gpuOf(m)))].map((gpu) => {
          const note = gpuBusyNote(status, runs, gpu);
          return note !== null ? (
            <p key={gpu} className="note text-ink">{note}</p>
          ) : null;
        })}
        {sourceRun !== null && status.sam3_gpu !== null && (batchGpu ?? 0) === status.sam3_gpu && (
          <p className="note text-ink">
            SAM3 runs on cuda:{status.sam3_gpu}. Training there shares its memory; labeling or training may run out of VRAM.
          </p>
        )}
        {sourceRun !== null && gpuBusyNote(status, runs, batchGpu ?? 0) !== null && (
          <p className="note text-ink">{gpuBusyNote(status, runs, batchGpu ?? 0)}</p>
        )}
        {paramsLocked && <p className="note">Continue keeps the run's own settings.</p>}
        {PARAM_FIELDS.map((f) => (
          <label key={f.key} className="flex items-center justify-between gap-2 text-sm text-dim">
            {f.label}
            <input aria-label={f.label} type="number" className="field w-24" min={f.min} max={f.max} step={f.step}
              disabled={paramsLocked}
              value={displayParams[f.key]} onChange={(e) => setParams({ ...params, [f.key]: Number(e.target.value) || f.min })} />
          </label>
        ))}
        <label className="flex flex-col gap-1 text-sm text-dim">
          Augmentation
          <select aria-label="Augmentation" className="field" value={displayParams.augment ?? "default"} disabled={paramsLocked}
            onChange={(e) => onAugmentChange(e.target.value as AugmentMode)}>
            <option value="default">Default</option>
            <option value="strong">Strong</option>
            <option value="custom">Custom</option>
          </select>
        </label>
        {displayParams.augment === "custom" && displayParams.aug && (
          <fieldset className="grid grid-cols-2 gap-x-2 gap-y-1.5">
            <legend className="sr-only">Custom augmentation</legend>
            {AUG_FIELDS.map((f) => (
              <label key={f.key} className="flex items-center justify-between gap-1 text-sm text-dim">
                {f.key}
                <input aria-label={f.key} type="number" className="field w-20" min={f.min} max={f.max} step={f.step}
                  disabled={paramsLocked}
                  value={displayParams.aug![f.key]} onChange={(e) => setAug({ [f.key]: Number(e.target.value) || f.min })} />
              </label>
            ))}
            <label className="col-span-2 flex items-center gap-2 text-sm text-dim">
              <input type="checkbox" checked={displayParams.aug.photometric} disabled={paramsLocked}
                onChange={(e) => setAug({ photometric: e.target.checked })} />
              Blur, noise, JPEG, contrast
            </label>
          </fieldset>
        )}
        {effectiveProblem !== null && <p className="note text-ink">{effectiveProblem}</p>}
        {startError !== null && <p role="alert" className="alert">{startError}</p>}
        <button type="button" className="btn btn-on" disabled={effectiveProblem !== null || starting} onClick={() => void start()}>
          {sourceRun !== null
            ? (mode === "continue" ? "Continue on new data" : "Fine-tune on new data")
            : `Start ${chosen.length} run${chosen.length === 1 ? "" : "s"}`}
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
                <span className="font-cond text-dim">
                  {g.memory_free !== null ? `${gbFree(g.memory_free)} / ` : ""}{gib(g.memory_total)}
                  {g.index === status.sam3_gpu ? " · SAM3" : ""}
                </span>
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
                  <span>{r.model} · {statusText(r)} · running in another process (vdp train)</span>
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
