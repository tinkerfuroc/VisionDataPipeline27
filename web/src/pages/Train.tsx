import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/rest";
import type { RunDetail, StartRunsRequest } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { CompareChart, MetricCharts } from "../components/MetricCharts";
import { PredictionGallery } from "../components/PredictionGallery";
import { RunsTable } from "../components/RunsTable";
import { TrainSidebar } from "../components/TrainSidebar";
import { usePoll } from "../hooks/usePoll";
import { statusText } from "../lib/train";

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
  // Slot index (0-3) a compared run is drawn in never changes while the run stays compared: a newly added run
  // takes the lowest free slot, and removing a run (toggle or delete) frees its slot for the next addition.
  const [compareSlots, setCompareSlots] = useState<(string | null)[]>(Array(MAX_COMPARE).fill(null));
  const [actionError, setActionError] = useState<string | null>(null);

  const compareIds = compareSlots.filter((id): id is string => id !== null);

  const list = runs.data ?? [];
  const selectedId = pickedId ?? list.find((r) => r.status === "running")?.id ?? list[0]?.id ?? null;
  const selected = list.find((r) => r.id === selectedId) ?? null;
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

  const onStart = async (body: StartRunsRequest) => {
    const { run_ids } = await api.startRuns(body);
    if (run_ids.length > 0) setPickedId(run_ids[0]);
    runs.refresh();
    status.refresh();
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
          <section className="flex flex-col gap-2">
            <h2 className="heading">Runs</h2>
            <RunsTable runs={list} selectedId={selectedId} compareIds={compareIds} onSelect={setPickedId}
              onToggleCompare={toggleCompare} onCancel={(id) => act(() => api.cancelRun(id))}
              onResume={(id) => act(() => api.resumeRun(id))} onOnnx={(id) => act(() => api.exportOnnx(id))} onDelete={onDelete} />
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
              <h2 className="heading">
                {selected.model} on {selected.project_name} · {statusText(selected)}
              </h2>
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
        <TrainSidebar status={status.data} exports={exports.data ?? []} runs={list} onStart={onStart}
          onCancel={(id) => act(() => api.cancelRun(id))} />
      </div>
    </div>
  );
}
