import { useCallback, useContext, useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { startResumableUpload } from "../api/resumable";
import { ApiError, UploadCancelled, api, thumbUrl, type UploadHandle } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { ImportJobRow } from "../components/ImportJobRow";
import { PhoneLink } from "../components/PhoneLink";
import { HealthContext } from "../hooks/useHealth";
import { percentLabeled } from "../lib/timeline";
import { formatBytes, uploadPercent } from "../lib/upload";
import {
  cancelAll as cancelAllQueued,
  cancelCurrent,
  dismissFailed as dismissFailedQueued,
  emptyUploadQueue,
  failCurrent,
  finishCurrent,
  queueFiles,
  removeWaiting,
  startNext,
  updateProgress,
  type UploadQueueState,
} from "../lib/uploadQueue";

const PROJECT_KEY = "vdp.project";

export function Clips() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<string | null>(() => localStorage.getItem(PROJECT_KEY));
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [imports, setImports] = useState<{ id: string; label: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const { health } = useContext(HealthContext);
  const maxBytes = health?.upload?.max_bytes ?? null;

  const [projectName, setProjectName] = useState("");
  const [className, setClassName] = useState("");
  const [fileInputKey, setFileInputKey] = useState(0);
  const [stride, setStride] = useState(1);
  const [fps, setFps] = useState(30);
  const [dragging, setDragging] = useState(false);
  const [skipped, setSkipped] = useState<{ name: string; reason: string }[]>([]);
  // The queue is mutated in a ref (so a burst of choose/finish/fail calls composes synchronously, without waiting
  // on React's render cycle to see the previous update) and mirrored into state to trigger a render.
  const queueRef = useRef<UploadQueueState>(emptyUploadQueue);
  const [queue, setQueueView] = useState<UploadQueueState>(emptyUploadQueue);
  const setQueue = useCallback((update: (q: UploadQueueState) => UploadQueueState) => {
    queueRef.current = update(queueRef.current);
    setQueueView(queueRef.current);
  }, []);
  const nextUploadId = useRef(0);
  // Identity guard (F2-style): a running upload's own id, so a stale XHR callback (an abort or a late response
  // for a file the queue has already moved on from) never touches the queue's current state.
  const uploadRef = useRef<{ id: number; handle: UploadHandle } | null>(null);
  const [dirPath, setDirPath] = useState("");
  const [dirFps, setDirFps] = useState(30);
  const [dirStride, setDirStride] = useState(1);
  // A class colour is shown at once (the native picker fires onChange continuously while dragging) and saved
  // ~300 ms after the last change; keyed by class id, and reset on a project switch (ids restart at 0 per project).
  const [colorPreview, setColorPreview] = useState<Record<number, string>>({});
  const colorTimers = useRef<Map<number, number>>(new Map());

  const fail = useCallback((e: unknown) => setError(e instanceof Error ? e.message : String(e)), []);

  const loadProjects = useCallback(async () => {
    const list = await api.listProjects();
    setProjects(list);
    setProjectId((current) => (list.some((p) => p.id === current) ? current : (list[0]?.id ?? null)));
  }, []);

  // F1: a request counter guards against a slow response from a project we have since navigated away from.
  const clipsSeq = useRef(0);
  const loadClips = useCallback(async (pid: string) => {
    const seq = ++clipsSeq.current;
    const list = await api.listClips(pid);
    if (seq === clipsSeq.current) setClips(list);
  }, []);

  useEffect(() => {
    loadProjects()
      .then(() => setError(null))
      .catch(fail);
  }, [loadProjects, fail]);

  useEffect(() => {
    if (projectId === null) {
      setClips([]);
      return;
    }
    localStorage.setItem(PROJECT_KEY, projectId);
    setClips([]);
    loadClips(projectId)
      .then(() => setError(null))
      .catch(fail);
  }, [projectId, loadClips, fail]);

  // A class id from the previous project must never be read as this project's preview.
  useEffect(() => {
    colorTimers.current.forEach((t) => window.clearTimeout(t));
    colorTimers.current.clear();
    setColorPreview({});
  }, [projectId]);
  useEffect(() => () => colorTimers.current.forEach((t) => window.clearTimeout(t)), []);

  const project = projects.find((p) => p.id === projectId) ?? null;

  // F2: a single in-flight flag stops a double click (or click-while-pending) from starting a second request.
  const guarded = (fn: () => Promise<void>) => (e?: FormEvent) => {
    e?.preventDefault();
    if (pending) return;
    setError(null);
    setPending(true);
    fn()
      .catch(fail)
      .finally(() => setPending(false));
  };

  const createProject = guarded(async () => {
    const name = projectName.trim();
    if (name === "") return;
    const created = await api.createProject(name);
    setProjectName("");
    await loadProjects();
    setProjectId(created.id);
  });

  const addClass = guarded(async () => {
    const name = className.trim();
    if (name === "" || project === null) return;
    await api.addClass(project.id, name);
    setClassName("");
    await loadProjects();
  });

  // Not `guarded`: several swatches can be mid-debounce at once, and none of them should disable Add class.
  const changeClassColor = (projectId: string, classId: number, color: string) => {
    setColorPreview((p) => ({ ...p, [classId]: color }));
    const existing = colorTimers.current.get(classId);
    if (existing !== undefined) window.clearTimeout(existing);
    colorTimers.current.set(
      classId,
      window.setTimeout(() => {
        colorTimers.current.delete(classId);
        api
          .setClassColor(projectId, classId, color)
          .then(() => loadProjects())
          .catch(fail);
      }, 300),
    );
  };

  // Not `guarded`: an upload lasts minutes, and the in-flight lock would disable the rest of the page for its
  // whole duration. One upload at a time is enough, so `uploadRef`'s id is the guard instead of `pending`.
  // Started imperatively (not from an effect keyed on `queue`) so a burst of calls — queueing more files while
  // one finishes — never races against React's render/effect scheduling.
  const startNextUpload = () => {
    const started = startNext(queueRef.current);
    if (started === null) return;
    setQueue(() => started.state);
    const item = started.item;
    const handle = startResumableUpload(item.projectId, item.file, { stride: item.stride, fps: item.fps }, (sent, total) => {
      if (uploadRef.current?.id === item.id) setQueue((q) => updateProgress(q, item.id, sent, total));
    });
    uploadRef.current = { id: item.id, handle };
    handle.promise.then(
      ({ job_id }) => {
        if (uploadRef.current?.id !== item.id) return;
        uploadRef.current = null;
        setQueue((q) => finishCurrent(q, item.id));
        setImports((list) => [...list, { id: job_id, label: item.file.name }]);
        startNextUpload();
      },
      (e: unknown) => {
        if (uploadRef.current?.id !== item.id) return;
        uploadRef.current = null;
        if (e instanceof UploadCancelled) {
          setQueue((q) => cancelCurrent(q, item.id)); // the user asked for it: it just disappears
        } else {
          setQueue((q) => failCurrent(q, item.id, e instanceof Error ? e.message : String(e)));
        }
        startNextUpload();
      },
    );
  };

  const pickFiles = (files: File[]) => {
    if (project === null || files.length === 0) return;
    const projectId = project.id;
    let rejected: { name: string; reason: string }[] = [];
    // `setQueue` mutates `queueRef` synchronously, so `rejected` is populated before it is read below.
    setQueue((q) => {
      const result = queueFiles(q, files, { projectId, stride, fps, maxBytes }, () => nextUploadId.current++);
      rejected = result.skipped;
      return result.state;
    });
    setSkipped(rejected);
    startNextUpload();
  };

  const cancelUpload = () => {
    const running = uploadRef.current;
    if (running === null) return;
    uploadRef.current = null; // the abort's rejection is ignored by the identity guard above
    running.handle.abort();
    setQueue((q) => cancelCurrent(q, running.id));
    startNextUpload();
  };

  const cancelAllUploads = () => {
    const running = uploadRef.current;
    uploadRef.current = null;
    running?.handle.abort();
    setQueue((q) => cancelAllQueued(q));
  };

  const removeQueuedUpload = (id: number) => setQueue((q) => removeWaiting(q, id));
  const dismissFailedUpload = (id: number) => setQueue((q) => dismissFailedQueued(q, id));

  const importFrames = guarded(async () => {
    const path = dirPath.trim();
    if (path === "" || project === null) return;
    const { job_id } = await api.importFrames(project.id, { path, fps: dirFps, stride: dirStride });
    setImports((list) => [...list, { id: job_id, label: path }]);
    setDirPath("");
  });

  const dismiss = useCallback((jobId: string) => setImports((list) => list.filter((j) => j.id !== jobId)), []);
  const finished = useCallback(
    (jobId: string, ok: boolean) => {
      if (!ok) return;
      dismiss(jobId);
      if (projectId !== null) loadClips(projectId).catch(fail);
    },
    [dismiss, projectId, loadClips, fail],
  );

  // window.confirm runs before guarded() touches `pending`, so a declined confirm never sets the
  // in-flight lock (which, being cleared via a promise `finally`, could not clear in time for a second,
  // synchronous click on the same button — e.g. decline once, then confirm).
  const deleteClip = (clip: ClipSummary) => {
    if (!window.confirm(`Delete "${clip.source.name}" and all of its masks?`)) return;
    guarded(async () => {
      try {
        await api.deleteClip(clip.id);
      } catch (e) {
        // 409: a propagation or an export is using the clips; the server only names the job's id
        if (e instanceof ApiError && e.status === 409)
          throw new Error("A propagation or an export is running. Wait for it to finish, then delete this clip.");
        throw e;
      }
      if (projectId !== null) await loadClips(projectId);
    })();
  };

  return (
    <div className="flex h-full flex-col">
      <AppHeader>
        <label className="flex items-center gap-2 font-cond text-dim">
          Project
          <select aria-label="Project" className="field py-0.5" value={projectId ?? ""} onChange={(e) => setProjectId(e.target.value || null)}>
            {projects.length === 0 && <option value="">None yet</option>}
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <form className="flex items-center gap-1" onSubmit={createProject}>
          <input aria-label="New project name" className="field w-44 py-0.5" placeholder="New project name" value={projectName}
            onChange={(e) => setProjectName(e.target.value)} />
          <button type="submit" className="btn py-0.5" disabled={pending}>Create project</button>
        </form>
      </AppHeader>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto md:flex-row md:overflow-hidden">
        <div className="flex min-w-0 flex-1 flex-col gap-4 p-4 md:overflow-y-auto">
          {error !== null && (
            <p role="alert" className="alert">
              {error}
            </p>
          )}
          {project === null && <p className="text-sm text-dim">Name a project in the bar above to start labelling.</p>}

          {imports.length > 0 && (
            <ul className="flex flex-col gap-1">
              {imports.map((j) => (
                <ImportJobRow key={j.id} jobId={j.id} label={j.label} onFinished={finished} onDismiss={dismiss} />
              ))}
            </ul>
          )}

          {project !== null && (
            <section>
              <h1 className="mb-3 font-cond text-xl font-semibold">Clips</h1>
              {clips.length === 0 && (
                <p className="text-sm text-dim">This project has no clips yet. Upload a video or a zip of images, or import a folder of frames.</p>
              )}
              <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                {clips.map((c) => (
                  <li key={c.id} className="overflow-hidden rounded border border-line bg-panel">
                    <img src={thumbUrl(c.id, 0)} alt="" className="aspect-video w-full bg-well object-contain" />
                    <progress aria-label="Labeled" className="block h-0.5 rounded-none" value={percentLabeled(c)} max={100} />
                    <div className="flex flex-col gap-1 p-3 text-sm">
                      <span className="truncate font-medium">{c.source.name}</span>
                      <span className="font-cond text-dim">
                        {c.frame_count} frames, {c.width}×{c.height}, {c.fps} fps, {percentLabeled(c)}% labeled
                      </span>
                      <div className="mt-1 flex gap-1">
                        <Link to={`/clips/${c.id}`} className="btn btn-on flex-1">Open</Link>
                        <button type="button" aria-label="Delete clip" disabled={pending} className="btn hover:text-danger"
                          onClick={() => deleteClip(c)}>
                          Delete
                        </button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>

        {project !== null && (
          <aside className="flex shrink-0 flex-col gap-5 border-t border-line bg-panel p-3 md:w-80 md:overflow-y-auto md:border-l md:border-t-0">
            {clips.length > 0 && (
              <Link to={`/projects/${project.id}/export`} className="btn btn-on">
                Export dataset
              </Link>
            )}
            <section className="flex flex-col gap-2">
              <h2 className="heading">Classes</h2>
              {project.classes.length > 0 && (
                <ul className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
                  {project.classes.map((c) => (
                    <li key={c.id} className="flex items-center gap-1.5">
                      <input
                        aria-label={`Colour of ${c.name}`}
                        type="color"
                        value={colorPreview[c.id] ?? c.color}
                        className="h-7 w-12 cursor-pointer rounded-sm"
                        onChange={(e) => changeClassColor(project.id, c.id, e.target.value)}
                      />
                      {c.name}
                    </li>
                  ))}
                </ul>
              )}
              <form className="flex gap-1" onSubmit={addClass}>
                <input aria-label="New class name" className="field flex-1" placeholder="New class name" value={className}
                  onChange={(e) => setClassName(e.target.value)} />
                <button type="submit" className="btn" disabled={pending}>Add class</button>
              </form>
              <p className="note">A class is what you label, such as a connector. Each gets its own mask colour.</p>
            </section>

            <section
              className={`flex flex-col gap-2 rounded border p-2 ${dragging ? "border-ink" : "border-transparent"}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                pickFiles(Array.from(e.dataTransfer.files ?? []));
              }}
            >
              <h2 className="heading">Upload footage</h2>
              <label className="flex items-center justify-between gap-2 text-sm text-dim">
                Keep every Nth frame
                <input
                  aria-label="Upload stride"
                  type="number"
                  min={1}
                  className="field w-16"
                  value={stride}
                  onChange={(e) => setStride(Math.max(1, Number(e.target.value) || 1))}
                />
              </label>
              <label className="flex items-center justify-between gap-2 text-sm text-dim">
                Frames per second (zip only)
                <input
                  aria-label="Zip frames per second"
                  type="number"
                  min={1}
                  step="any"
                  className="field w-16"
                  value={fps}
                  onChange={(e) => setFps(Math.max(1, Number(e.target.value) || 30))}
                />
              </label>
              <input
                key={fileInputKey}
                aria-label="Footage file"
                type="file"
                multiple
                accept="video/*,.zip,application/zip"
                className="text-sm text-dim file:btn file:mr-2 file:border-0"
                onChange={(e) => {
                  pickFiles(Array.from(e.target.files ?? []));
                  setFileInputKey((k) => k + 1); // reset so choosing the same file(s) again still fires onChange
                }}
              />
              <PhoneLink projectId={project.id} />
              {queue.current === null && queue.waiting.length === 0 && (
                <p className="note">Drop one or more videos or zips of images here, or choose them above.</p>
              )}
              {skipped.length > 0 && (
                <ul className="flex flex-col gap-0.5">
                  {skipped.map((s, i) => (
                    <li key={i} className="note text-ink">
                      {s.name}: {s.reason}
                    </li>
                  ))}
                </ul>
              )}
              {queue.current !== null && (
                <div className="flex flex-col gap-1">
                  <p className="note text-ink">{queue.current.file.name}</p>
                  <progress value={queue.current.sent} max={queue.current.total || undefined} />
                  <p className="note">
                    {uploadPercent(queue.current.sent, queue.current.total) ?? 0}% of {formatBytes(queue.current.total)}
                  </p>
                </div>
              )}
              {queue.waiting.length > 0 && (
                <ul className="flex flex-col gap-1">
                  {queue.waiting.map((item) => (
                    <li key={item.id} className="flex items-center justify-between gap-2 text-sm">
                      <span className="truncate">
                        {item.file.name} · {formatBytes(item.file.size)}
                      </span>
                      <button
                        type="button"
                        aria-label={`Remove ${item.file.name} from the queue`}
                        className="btn shrink-0 px-3"
                        onClick={() => removeQueuedUpload(item.id)}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {queue.failed.length > 0 && (
                <ul className="flex flex-col gap-1">
                  {queue.failed.map((item) => (
                    <li key={item.id} className="flex flex-col gap-0.5 rounded border border-line p-1.5">
                      <span className="truncate text-sm">{item.file.name}</span>
                      <p className="note text-ink">{item.message}</p>
                      <button type="button" className="btn self-start" onClick={() => dismissFailedUpload(item.id)}>
                        Dismiss
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {(queue.current !== null || queue.waiting.length > 0) && (
                <div className="flex gap-1">
                  {queue.current !== null && (
                    <button type="button" className="btn" onClick={cancelUpload}>
                      Cancel upload
                    </button>
                  )}
                  {(queue.current !== null ? 1 : 0) + queue.waiting.length > 1 && (
                    <button type="button" className="btn" onClick={cancelAllUploads}>
                      Cancel all
                    </button>
                  )}
                </div>
              )}
            </section>

            <form className="flex flex-col gap-2" onSubmit={importFrames}>
              <h2 className="heading">Import a folder of frames</h2>
              <input aria-label="Frame directory" className="field" placeholder="~/footage/run1" value={dirPath}
                onChange={(e) => setDirPath(e.target.value)} />
              <p className="note">A path on the server. Frames are read in numeric file-name order.</p>
              <label className="flex items-center justify-between gap-2 text-sm text-dim">
                Frames per second
                <input aria-label="Frame directory fps" type="number" min={1} step="any" className="field w-16"
                  value={dirFps} onChange={(e) => setDirFps(Math.max(1, Number(e.target.value) || 30))} />
              </label>
              <label className="flex items-center justify-between gap-2 text-sm text-dim">
                Keep every Nth frame
                <input aria-label="Frame directory stride" type="number" min={1} className="field w-16"
                  value={dirStride} onChange={(e) => setDirStride(Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <button type="submit" className="btn" disabled={dirPath.trim() === "" || pending}>Import frames</button>
            </form>
          </aside>
        )}
      </div>
    </div>
  );
}
