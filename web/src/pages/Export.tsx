import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, api, thumbUrl } from "../api/rest";
import type { ClipSummary, ExportFormat, ExportRequest, ExportResult, Project } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { useJob } from "../hooks/useJob";
import { slug } from "../lib/exportName";
import { classObjectCounts, defaultValClips, exportProblem, splitModeFor } from "../lib/exportSelection";
import { isTerminal } from "../lib/jobReducer";
import { percentLabeled } from "../lib/timeline";

const FORMATS: { id: ExportFormat; label: string; dir: string }[] = [
  { id: "yolo_seg", label: "YOLO segmentation", dir: "yolo_seg/" },
  { id: "yolo_det", label: "YOLO detection", dir: "yolo_det/" },
  { id: "coco", label: "COCO", dir: "coco/" },
  { id: "masks", label: "Mask images", dir: "masks/" },
];
const ALL_FORMAT_IDS = FORMATS.map((f) => f.id);

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

const back = (
  <Link to="/" aria-label="Back to clips" className="flex items-center gap-1 font-cond text-dim hover:text-ink">
    <span aria-hidden>‹</span> Clips
  </Link>
);

export function Export() {
  const { projectId = "" } = useParams();

  const [projects, setProjects] = useState<Project[] | null>(null);
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [includedIds, setIncludedIds] = useState<Set<string>>(new Set());
  const [pickedVal, setPickedVal] = useState<Set<string>>(new Set()); // only meaningful once `touched`
  const [touched, setTouched] = useState(false);
  const [formatIds, setFormatIds] = useState<Set<ExportFormat>>(new Set(ALL_FORMAT_IDS));
  const [valSharePercent, setValSharePercent] = useState(20);
  const [stride, setStride] = useState(1);
  const [includeEmpty, setIncludeEmpty] = useState(true);
  // excluded rather than ticked, so every class — including one added later — starts ticked
  const [excludedClasses, setExcludedClasses] = useState<Set<number>>(new Set());
  const [name, setName] = useState("");

  const [jobId, setJobId] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const startingRef = useRef(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true; // StrictMode runs effects twice; set it again on the second mount
    return () => {
      mounted.current = false;
    };
  }, []);
  const [starting, setStarting] = useState(false);

  const fail = useCallback((e: unknown) => setLoadError(message(e)), []);

  // F1: a request counter guards against a slow response arriving after this page moved on to another project id.
  const clipsSeq = useRef(0);
  const loadClips = useCallback(async (pid: string) => {
    const seq = ++clipsSeq.current;
    const list = await api.listClips(pid);
    if (seq !== clipsSeq.current) return;
    setClips(list);
    setIncludedIds(new Set(list.map((c) => c.id)));
    setTouched(false);
    setExcludedClasses(new Set());
  }, []);

  useEffect(() => {
    let closed = false;
    api.listProjects().then(
      (list) => {
        if (!closed) setProjects(list);
      },
      (e: unknown) => {
        if (!closed) fail(e);
      },
    );
    return () => {
      closed = true;
    };
  }, [fail]);

  useEffect(() => {
    loadClips(projectId).catch(fail);
  }, [projectId, loadClips, fail]);

  const project = projects?.find((p) => p.id === projectId) ?? null;

  // The default validation clips follow the newest-first rule while the user has not hand-picked a role;
  // once they have, only the checkbox that includes/excludes a clip still changes what is actually sent.
  // Derived during render, not set from an effect: an effect would leave one render with no validation clip at all.
  const defaultVal = useMemo(() => {
    const included = clips.filter((c) => includedIds.has(c.id)).map((c) => c.id);
    return new Set(defaultValClips(clips, included, valSharePercent / 100));
  }, [clips, includedIds, valSharePercent]);
  const valIds = touched ? pickedVal : defaultVal;

  const orderedIncluded = useMemo(() => clips.filter((c) => includedIds.has(c.id)).map((c) => c.id), [clips, includedIds]);
  const orderedVal = useMemo(
    () => clips.filter((c) => includedIds.has(c.id) && valIds.has(c.id)).map((c) => c.id),
    [clips, includedIds, valIds],
  );
  const mode = splitModeFor(orderedIncluded.length);
  const orderedFormats = useMemo(() => FORMATS.filter((f) => formatIds.has(f.id)).map((f) => f.id), [formatIds]);
  const classes = useMemo(() => [...(project?.classes ?? [])].sort((a, b) => a.id - b.id), [project]);
  const objectCounts = useMemo(() => classObjectCounts(clips, orderedIncluded), [clips, orderedIncluded]);
  const chosenClasses = classes.filter((c) => !excludedClasses.has(c.id)).map((c) => c.id);

  const problem = exportProblem({
    selected: orderedIncluded, formats: orderedFormats, classes: chosenClasses, classCount: classes.length,
    valClips: orderedVal, valFraction: valSharePercent / 100, stride,
  });

  const job = useJob(jobId);
  const running = jobId !== null && !isTerminal(job.status);
  const result = job.status === "done" ? ((job.result as unknown as ExportResult) ?? null) : null;

  const toggleIncluded = (id: string) => {
    setIncludedIds((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const toggleRole = (id: string) => {
    const next = new Set(valIds); // the first hand-picked role starts from the defaults that were on screen
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setPickedVal(next);
    setTouched(true);
  };
  const toggleFormat = (id: ExportFormat) => {
    setFormatIds((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const toggleClass = (id: number) => {
    setExcludedClasses((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const startExport = () => {
    if (startingRef.current || running || problem !== null) return;
    startingRef.current = true;
    setStarting(true);
    setStartError(null);
    const options: ExportRequest["options"] = {
      split_mode: mode, val_fraction: valSharePercent / 100, stride, include_empty: includeEmpty,
      classes: chosenClasses.length === classes.length ? null : chosenClasses,
      ...(mode === "per_clip" ? { val_clips: orderedVal } : {}),
    };
    api.startExport(projectId, {
      clips: orderedIncluded,
      formats: orderedFormats,
      options,
      name: name.trim() === "" ? null : name.trim(),
    }).then(
      (r) => {
        startingRef.current = false;
        if (!mounted.current) return; // left the page while the request was in flight
        setStarting(false);
        setJobId(r.job_id);
      },
      (e: unknown) => {
        startingRef.current = false;
        if (!mounted.current) return;
        setStarting(false);
        if (e instanceof ApiError && e.status === 409) {
          setStartError("A propagation is running. Wait for it to finish, then export.");
        } else {
          setStartError(message(e));
        }
      },
    );
  };

  if (projects !== null && project === null) {
    return (
      <div className="flex h-full flex-col">
        <AppHeader>{back}</AppHeader>
        <p role="alert" className="alert m-4">
          This project no longer exists.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <AppHeader>
        {back}
        {project !== null && <h1 className="truncate font-medium">Export {project.name}</h1>}
      </AppHeader>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto md:flex-row md:overflow-hidden">
        <div className="flex min-w-0 flex-1 flex-col gap-4 p-4 md:overflow-y-auto">
          {loadError !== null && (
            <p role="alert" className="alert">
              {loadError}
            </p>
          )}
          <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {clips.map((c) => {
              const included = includedIds.has(c.id);
              const inVal = valIds.has(c.id);
              return (
                <li key={c.id} className="overflow-hidden rounded border border-line bg-panel">
                  <img src={thumbUrl(c.id, 0)} alt="" className="aspect-video w-full bg-well object-contain" />
                  <div className="flex flex-col gap-1 p-3 text-sm">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        aria-label={`Include ${c.source.name}`}
                        checked={included}
                        onChange={() => toggleIncluded(c.id)}
                      />
                      <span className="truncate font-medium">{c.source.name}</span>
                    </label>
                    <span className="font-cond text-dim">
                      {c.frame_count} frames, {percentLabeled(c)}% labeled
                    </span>
                    {!included && <span className="text-xs text-dim">Not exported</span>}
                    {included && mode === "per_clip" && (
                      <div className="mt-1 flex items-center gap-1.5">
                        <span className={`rounded px-1.5 py-0.5 font-cond text-xs ${inVal ? "btn-on" : ""}`}>
                          {inVal ? "Validation" : "Training"}
                        </span>
                        <button type="button" className="btn" onClick={() => toggleRole(c.id)}>
                          {inVal ? `Use ${c.source.name} for training` : `Use ${c.source.name} for validation`}
                        </button>
                      </div>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>

        <aside className="flex shrink-0 flex-col gap-5 border-t border-line bg-panel p-3 md:w-80 md:overflow-y-auto md:border-l md:border-t-0">
          <section className="flex flex-col gap-2">
            <h2 className="heading">Formats</h2>
            {FORMATS.map((f) => (
              <label key={f.id} className="flex flex-col gap-0.5 text-sm">
                <span className="flex items-center gap-2">
                  <input type="checkbox" aria-label={f.label} checked={formatIds.has(f.id)} onChange={() => toggleFormat(f.id)} />
                  {f.label}
                </span>
                <span className="note">{f.dir}</span>
              </label>
            ))}
          </section>

          <section className="flex flex-col gap-2">
            <h2 className="heading">Classes</h2>
            {classes.length === 0 && <p className="note">This project has no classes yet.</p>}
            {classes.map((c) => {
              const n = objectCounts.get(c.id) ?? 0;
              return (
                <label key={c.id} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    aria-label={`Class ${c.name}`}
                    checked={!excludedClasses.has(c.id)}
                    onChange={() => toggleClass(c.id)}
                  />
                  <span aria-hidden className="inline-block h-3 w-3 shrink-0 rounded-sm" style={{ backgroundColor: c.color }} />
                  <span className="min-w-0 flex-1 truncate">{c.name}</span>
                  <span className="font-cond text-dim">
                    {n} object{n === 1 ? "" : "s"}
                  </span>
                </label>
              );
            })}
            <p className="note">
              Objects of unticked classes are left out; a frame left without objects follows "Include frames without a
              mask". The exported classes are numbered 0, 1, … in this order.
            </p>
          </section>

          <section className="flex flex-col gap-2">
            <h2 className="heading">Split</h2>
            <p className="note">
              {mode === "per_clip"
                ? "Whole clips go to training or validation, so neighbouring frames never end up on both sides. Choose the validation clips on the left."
                : `With one clip, the last ${valSharePercent}% of its exported frames become the validation set.`}
            </p>
            {mode === "per_clip" && orderedVal.length === 0 && (
              <p className="note text-ink">
                No clip is marked for validation, so this dataset will have no validation set. Mark one on the left if you
                want one.
              </p>
            )}
            <label className="flex items-center justify-between gap-2 text-sm text-dim">
              Validation share (%)
              <input
                aria-label="Validation share (%)"
                type="number"
                min={0}
                max={99}
                className="field w-16"
                value={valSharePercent}
                onChange={(e) => setValSharePercent(Math.max(0, Math.min(99, Number(e.target.value) || 0)))}
              />
            </label>
          </section>

          <section className="flex flex-col gap-2">
            <h2 className="heading">Frames</h2>
            <label className="flex items-center justify-between gap-2 text-sm text-dim">
              Keep every Nth frame
              <input
                aria-label="Keep every Nth frame"
                type="number"
                min={1}
                className="field w-16"
                value={stride}
                onChange={(e) => setStride(Math.max(1, Number(e.target.value) || 1))}
              />
            </label>
            <label className="flex items-center gap-2 text-sm text-dim">
              <input type="checkbox" checked={includeEmpty} onChange={(e) => setIncludeEmpty(e.target.checked)} />
              Include frames without a mask
            </label>
            <p className="note">They teach the model what the background looks like.</p>
          </section>

          <section className="flex flex-col gap-2">
            <h2 className="heading">Start</h2>
            <label className="flex flex-col gap-1 text-sm text-dim">
              Dataset name
              <input aria-label="Dataset name" className="field" maxLength={80} placeholder="Optional, e.g. Cups v2"
                value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <p className="note">
              Folder: {slug(name) !== "" ? slug(name) : "a timestamp, e.g. 20260926-104433-…"}
              {slug(name) !== "" ? " (a number is added if it is taken)" : ""}
            </p>
            {problem !== null && <p className="note text-ink">{problem}</p>}
            {startError !== null && (
              <p role="alert" className="alert">
                {startError}
              </p>
            )}
            <button type="button" className="btn btn-on" disabled={problem !== null || running || starting} onClick={startExport}>
              Export {orderedIncluded.length} clip{orderedIncluded.length === 1 ? "" : "s"}
            </button>

            {running && (
              <>
                <progress value={job.total === null ? undefined : job.done} max={job.total ?? undefined} />
                <p className="note">
                  {job.done} of {job.total ?? "?"}
                </p>
                <button type="button" className="btn" onClick={() => jobId !== null && api.cancelJob(jobId).catch(() => {})}>
                  Cancel
                </button>
              </>
            )}

            {job.status === "done" && result !== null && (
              <div className="flex flex-col gap-1.5">
                <h3 className="heading">Dataset written</h3>
                {result.name && <p className="text-sm text-ink">{result.name}</p>}
                {/* a textarea, so a long path wraps instead of being cut off; focus selects it for copying */}
                <textarea aria-label="Dataset folder" readOnly rows={3} className="field resize-none break-all font-cond"
                  value={result.path} onFocus={(e) => e.target.select()} />
                <p className="note text-ink">
                  {result.counts.train.images} training images
                  {result.counts.train.empty > 0 ? `, ${result.counts.train.empty} without a mask` : ""}
                </p>
                <p className="note text-ink">
                  {result.counts.val.images} validation images
                  {result.counts.val.empty > 0 ? `, ${result.counts.val.empty} without a mask` : ""}
                </p>
                {!result.linked && (
                  <p className="note">This disk does not support hardlinks, so each format holds its own copy of the images.</p>
                )}
              </div>
            )}
            {job.status === "cancelled" && <p className="note text-ink">Export cancelled. Nothing was written.</p>}
            {job.status === "error" && (
              <p role="alert" className="alert">
                {job.error}
              </p>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}
