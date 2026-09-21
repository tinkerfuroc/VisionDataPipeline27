import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, thumbUrl } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { ImportJobRow } from "../components/ImportJobRow";

const PROJECT_KEY = "vsp.project";

function percentLabeled(c: ClipSummary): number {
  const live = c.frame_count - c.deleted_frames.length;
  return live <= 0 ? 0 : Math.round((100 * c.labeled_frames) / live);
}

export function Clips() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<string | null>(() => localStorage.getItem(PROJECT_KEY));
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [imports, setImports] = useState<{ id: string; label: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const [projectName, setProjectName] = useState("");
  const [className, setClassName] = useState("");
  const [video, setVideo] = useState<File | null>(null);
  const [videoInputKey, setVideoInputKey] = useState(0);
  const [videoStride, setVideoStride] = useState(1);
  const [dirPath, setDirPath] = useState("");
  const [dirFps, setDirFps] = useState(30);
  const [dirStride, setDirStride] = useState(1);

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

  const importVideo = guarded(async () => {
    if (video === null || project === null) return;
    const { job_id } = await api.importVideo(project.id, video, videoStride);
    setImports((list) => [...list, { id: job_id, label: video.name }]);
    setVideo(null);
    setVideoInputKey((k) => k + 1);
  });

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
      await api.deleteClip(clip.id);
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
                <p className="text-sm text-dim">This project has no clips yet. Import a video or a folder of frames to add one.</p>
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
            <section className="flex flex-col gap-2">
              <h2 className="heading">Classes</h2>
              {project.classes.length > 0 && (
                <ul className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
                  {project.classes.map((c) => (
                    <li key={c.id} className="flex items-center gap-1.5">
                      <span aria-hidden className="inline-block h-3 w-3 rounded-sm" style={{ background: c.color }} />
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

            <form className="flex flex-col gap-2" onSubmit={importVideo}>
              <h2 className="heading">Import a video</h2>
              <input key={videoInputKey} aria-label="Video file" type="file" accept="video/*,.mp4,.mov,.avi,.mkv"
                className="text-sm text-dim file:btn file:mr-2 file:border-0" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
              <label className="flex items-center justify-between gap-2 text-sm text-dim">
                Keep every Nth frame
                <input aria-label="Video stride" type="number" min={1} className="field w-16" value={videoStride}
                  onChange={(e) => setVideoStride(Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <button type="submit" className="btn" disabled={video === null || pending}>Import video</button>
            </form>

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
