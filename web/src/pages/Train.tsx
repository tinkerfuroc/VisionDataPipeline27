import { useCallback, useContext, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/rest";
import type { FromCheckpointRequest, RunDetail, StartRunsRequest } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { CompareChart, MetricCharts } from "../components/MetricCharts";
import { CompositeSamples } from "../components/CompositeSamples";
import { PredictionGallery } from "../components/PredictionGallery";
import { RunsTable } from "../components/RunsTable";
import { TrainSidebar } from "../components/TrainSidebar";
import { HealthContext } from "../hooks/useHealth";
import { usePoll } from "../hooks/usePoll";
import { exportLabel } from "../lib/exportName";
import { finalMetricsNote, samplingNote, startModeHeaderNote, statusText } from "../lib/train";

const POLL_MS = 2000;
const MAX_COMPARE = 4;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

const back = (
  <Link to="/" aria-label="Back to clips" className="flex items-center gap-1 font-cond text-dim hover:text-ink">
    <span aria-hidden>‹</span> Clips
  </Link>
);

export function Train() {
  const status = usePoll(() => api.trainStatus(), POLL_MS, []);
  const exports = usePoll(() => api.listExports(), 15000, []);
  const runs = usePoll(() => api.listRuns(), POLL_MS, []);
  const [pickedId, setPickedId] = useState<string | null>(null);
  // bumped by the table's Resume/Mark complete button, so the sidebar brings its Stopped run section into view
  // even for the selected run
  const [stoppedAsk, setStoppedAsk] = useState(0);
  const [exportId, setExportId] = useState<string | null>(null);
  // The New-runs form's "Start from"; set by the Stopped-run section's or the run detail header's
  // "Train on new data" button, which also bumps `newRunsFocus` to scroll that form into view.
  const [sourceRunId, setSourceRunId] = useState<string | null>(null);
  const [newRunsFocus, setNewRunsFocus] = useState(0);
  // Slot index (0-3) a compared run is drawn in never changes while the run stays compared: a newly added run
  // takes the lowest free slot, and removing a run (toggle or delete) frees its slot for the next addition.
  const [compareSlots, setCompareSlots] = useState<(string | null)[]>(Array(MAX_COMPARE).fill(null));
  const [actionError, setActionError] = useState<string | null>(null);

  const compareIds = compareSlots.filter((id): id is string => id !== null);

  const exportsList = exports.data ?? [];
  const selectedExportId = exportId ?? exportsList[0]?.id ?? null;
  const selectedExport = exportsList.find((e) => e.id === selectedExportId) ?? null;

  const list = runs.data ?? [];
  const selectedId = pickedId ?? list.find((r) => r.status === "running")?.id ?? list[0]?.id ?? null;
  const selected = list.find((r) => r.id === selectedId) ?? null;
  const selectedRunExport = selected !== null ? exportsList.find((e) => e.id === selected.export_id) : undefined;
  // Same ruling as the Runs table: an unnamed export falls back to the run's own project name, not its id.
  const selectedRunLabel =
    selected === null ? "" : selectedRunExport !== undefined && selectedRunExport.name !== null
      ? exportLabel(selectedRunExport) : selected.project_name;
  const detail = usePoll<RunDetail | null>(
    () => (selectedId ? api.getRun(selectedId) : Promise.resolve(null)),
    selected?.status === "running" ? POLL_MS : null,
    [selectedId, selected?.status],
  );
  const anyCompareRunning = list.some((r) => compareIds.includes(r.id) && r.status === "running");
  const compared = usePoll<{ run: RunDetail; slot: number }[]>(
    () =>
      Promise.all(
        compareSlots.map((id, slot) => (id === null ? null : api.getRun(id).then((run) => ({ run, slot })))),
      ).then((entries) => entries.filter((e): e is { run: RunDetail; slot: number } => e !== null)),
    anyCompareRunning ? 5000 : null,
    [compareSlots.join(",")],
  );

  const act = useCallback(
    (fn: () => Promise<unknown>) => {
      setActionError(null);
      fn().then(
        () => {
          runs.refresh();
          status.refresh();
        },
        (e: unknown) => setActionError(message(e)),
      );
    },
    [runs, status],
  );

  // A start that paused SAM3 re-polls health at once, so the header's banner (and Label's state) shows the pause.
  const { refresh: refreshHealth } = useContext(HealthContext);
  const afterPause = (pauseSam3: boolean | undefined) => {
    if (pauseSam3) refreshHealth?.();
  };
  const onStart = async (body: StartRunsRequest) => {
    const { run_ids } = await api.startRuns(body);
    if (run_ids.length > 0) setPickedId(run_ids[0]);
    runs.refresh();
    status.refresh();
    afterPause(body.pause_sam3);
  };
  // Left to reject on failure (the sidebar's Stopped run section shows the error itself, like Start does).
  const onResume = async (id: string, gpu: number, pauseSam3?: boolean) => {
    await (pauseSam3 ? api.resumeRun(id, gpu, true) : api.resumeRun(id, gpu));
    runs.refresh();
    status.refresh();
    afterPause(pauseSam3);
  };
  const onComplete = async (id: string, gpu: number, pauseSam3?: boolean) => {
    await (pauseSam3 ? api.completeRun(id, gpu, true) : api.completeRun(id, gpu));
    runs.refresh();
    status.refresh();
    afterPause(pauseSam3);
  };
  const onStartFromCheckpoint = async (sourceId: string, body: FromCheckpointRequest) => {
    const { run_id } = await api.trainFromCheckpoint(sourceId, body);
    setPickedId(run_id);
    runs.refresh();
    status.refresh();
    afterPause(body.pause_sam3);
  };
  const onTrainOnNewData = (id: string) => {
    setSourceRunId(id);
    setNewRunsFocus((n) => n + 1);
  };
  const releaseSlot = (id: string) => setCompareSlots((slots) => slots.map((s) => (s === id ? null : s)));
  const onDelete = (id: string) => {
    if (!window.confirm("Delete this run and its weights?")) return;
    act(() => api.deleteRun(id));
    releaseSlot(id);
    if (pickedId === id) setPickedId(null);
  };
  const toggleCompare = (id: string) =>
    setCompareSlots((slots) => {
      const at = slots.indexOf(id);
      if (at !== -1) {
        const next = [...slots];
        next[at] = null;
        return next;
      }
      const free = slots.indexOf(null);
      if (free === -1) return slots; // all slots taken; RunsTable already disables further picks
      const next = [...slots];
      next[free] = id;
      return next;
    });

  const shown = detail.data !== null && detail.data.id === selectedId ? detail.data : null;

  return (
    <div className="flex h-full flex-col">
      <AppHeader>
        {back}
        <h1 className="truncate font-medium">Train</h1>
      </AppHeader>
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto md:flex-row md:overflow-hidden">
        <div className="flex min-w-0 flex-1 flex-col gap-5 p-4 md:overflow-y-auto">
          {(runs.error ?? status.error ?? actionError) !== null && (
            <p role="alert" className="alert">{runs.error ?? status.error ?? actionError}</p>
          )}
          {selectedExport !== null && selectedExport.derived_from !== null && (
            <CompositeSamples exportInfo={selectedExport} />
          )}
          <section className="flex flex-col gap-2">
            <h2 className="heading">Runs</h2>
            <RunsTable runs={list} selectedId={selectedId} compareIds={compareIds} exports={exportsList} onSelect={setPickedId}
              onToggleCompare={toggleCompare} onCancel={(id) => act(() => api.cancelRun(id))}
              onAskStopped={(id) => { setPickedId(id); setStoppedAsk((n) => n + 1); }}
              onOnnx={(id) => act(() => api.exportOnnx(id))} onDelete={onDelete} />
          </section>

          {compareIds.length >= 2 && (
            <section className="flex flex-col gap-2">
              <h2 className="heading">Comparison</h2>
              {compared.error !== null && <p role="alert" className="alert">{compared.error}</p>}
              {compared.data !== null && <CompareChart runs={compared.data} />}
            </section>
          )}

          {selected !== null && (
            <section className="flex flex-col gap-2">
              <h2 className="heading flex flex-wrap items-center gap-2">
                <span>
                  {selected.model} on {selectedRunLabel} · {statusText(selected)}
                  {startModeHeaderNote(selected) !== null && ` · ${startModeHeaderNote(selected)}`}
                </span>
                {selected.has_best && (
                  <button type="button" className="btn" onClick={() => onTrainOnNewData(selected.id)}>
                    Train on new data
                  </button>
                )}
              </h2>
              {samplingNote(selected) !== null && <p className="note">{samplingNote(selected)}</p>}
              {finalMetricsNote(selected) !== null && <p className="text-sm">{finalMetricsNote(selected)}</p>}
              {selected.error !== null && (
                <pre className="alert max-h-40 overflow-auto whitespace-pre-wrap font-cond text-xs">{selected.error}</pre>
              )}
              {shown !== null ? <MetricCharts run={shown} /> : <p className="note">Loading metrics…</p>}
            </section>
          )}

          {selected !== null && (
            <section className="flex flex-col gap-2">
              <h2 className="heading">Validation predictions</h2>
              {selected.status === "done" ? (
                <PredictionGallery run={selected} />
              ) : (
                <p className="note">Predictions appear when the run finishes.</p>
              )}
            </section>
          )}
        </div>
        <TrainSidebar status={status.data} exports={exportsList} runs={list} exportId={selectedExportId}
          onExportChange={setExportId} onExportsChanged={exports.refresh} onStart={onStart}
          onCancel={(id) => act(() => api.cancelRun(id))} selectedRun={selected} onResume={onResume}
          onComplete={onComplete} stoppedFocus={stoppedAsk} sourceRunId={sourceRunId} onSourceRunChange={setSourceRunId}
          onStartFromCheckpoint={onStartFromCheckpoint} onTrainOnNewData={onTrainOnNewData} newRunsFocus={newRunsFocus} />
      </div>
    </div>
  );
}
