import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, thumbUrl } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { ImportJobRow } from "../components/ImportJobRow";

const PROJECT_KEY = "vsp.project";
const input = "rounded border border-neutral-600 bg-neutral-800 px-2 py-1 text-sm";
const button = "rounded bg-sky-700 px-3 py-1 text-sm hover:bg-sky-600 disabled:opacity-40";

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

  const [projectName, setProjectName] = useState("");
  const [className, setClassName] = useState("");
  const [video, setVideo] = useState<File | null>(null);
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

  const loadClips = useCallback(async (pid: string) => setClips(await api.listClips(pid)), []);

  useEffect(() => {
    loadProjects().catch(fail);
  }, [loadProjects, fail]);

  useEffect(() => {
    if (projectId === null) {
      setClips([]);
      return;
    }
    localStorage.setItem(PROJECT_KEY, projectId);
    loadClips(projectId).catch(fail);
  }, [projectId, loadClips, fail]);

  const project = projects.find((p) => p.id === projectId) ?? null;

  const guarded = (fn: () => Promise<void>) => (e?: FormEvent) => {
    e?.preventDefault();
    setError(null);
    fn().catch(fail);
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
  });

  const importFrames = guarded(async () => {
    const path = dirPath.trim();
    if (path === "" || project === null) return;
    const { job_id } = await api.importFrames(project.id, { path, fps: dirFps, stride: dirStride });
    setImports((list) => [...list, { id: job_id, label: path }]);
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

  const deleteClip = (clip: ClipSummary) =>
    guarded(async () => {
      if (!window.confirm(`Delete "${clip.source.name}" and all of its masks?`)) return;
      await api.deleteClip(clip.id);
      if (projectId !== null) await loadClips(projectId);
    })();

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-6 overflow-y-auto p-4">
      {error !== null && (
        <p role="alert" className="rounded bg-red-900 px-3 py-2 text-sm">
          {error}
        </p>
      )}

      <section className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col text-xs text-neutral-400">
          Project
          <select aria-label="Project" className={input} value={projectId ?? ""} onChange={(e) => setProjectId(e.target.value || null)}>
            {projects.length === 0 && <option value="">(none yet)</option>}
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <form className="flex items-end gap-2" onSubmit={createProject}>
          <input aria-label="New project name" className={input} placeholder="New project name" value={projectName}
            onChange={(e) => setProjectName(e.target.value)} />
          <button type="submit" className={button}>Create project</button>
        </form>
      </section>

      {project !== null && (
        <>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-300">Classes</h2>
            <div className="flex flex-wrap items-center gap-2">
              {project.classes.map((c) => (
                <span key={c.id} className="flex items-center gap-1 rounded bg-neutral-800 px-2 py-1 text-sm">
                  <span className="inline-block h-3 w-3 rounded-full" style={{ background: c.color }} />
                  {c.name}
                </span>
              ))}
              <form className="flex items-center gap-2" onSubmit={addClass}>
                <input aria-label="New class name" className={input} placeholder="New class name" value={className}
                  onChange={(e) => setClassName(e.target.value)} />
                <button type="submit" className={button}>Add class</button>
              </form>
            </div>
          </section>

          <section className="grid gap-4 md:grid-cols-2">
            <form className="flex flex-col gap-2 rounded border border-neutral-700 p-3" onSubmit={importVideo}>
              <h2 className="text-sm font-semibold text-neutral-300">Import video</h2>
              <input aria-label="Video file" type="file" accept="video/*,.mp4,.mov,.avi,.mkv" className="text-sm"
                onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
              <label className="flex items-center gap-2 text-xs text-neutral-400">
                Keep every Nth frame
                <input aria-label="Video stride" type="number" min={1} className={`${input} w-20`} value={videoStride}
                  onChange={(e) => setVideoStride(Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <button type="submit" className={button} disabled={video === null}>Import video</button>
            </form>

            <form className="flex flex-col gap-2 rounded border border-neutral-700 p-3" onSubmit={importFrames}>
              <h2 className="text-sm font-semibold text-neutral-300">Import frame directory</h2>
              <input aria-label="Frame directory" className={input} placeholder="~/footage/run1 (a path on the server)"
                value={dirPath} onChange={(e) => setDirPath(e.target.value)} />
              <div className="flex gap-4 text-xs text-neutral-400">
                <label className="flex items-center gap-2">
                  fps
                  <input aria-label="Frame directory fps" type="number" min={1} step="any" className={`${input} w-20`}
                    value={dirFps} onChange={(e) => setDirFps(Math.max(1, Number(e.target.value) || 30))} />
                </label>
                <label className="flex items-center gap-2">
                  Keep every Nth frame
                  <input aria-label="Frame directory stride" type="number" min={1} className={`${input} w-20`}
                    value={dirStride} onChange={(e) => setDirStride(Math.max(1, Number(e.target.value) || 1))} />
                </label>
              </div>
              <button type="submit" className={button} disabled={dirPath.trim() === ""}>Import frames</button>
            </form>
          </section>

          {imports.length > 0 && (
            <ul className="flex flex-col gap-2">
              {imports.map((j) => (
                <ImportJobRow key={j.id} jobId={j.id} label={j.label} onFinished={finished} onDismiss={dismiss} />
              ))}
            </ul>
          )}

          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-300">Clips</h2>
            {clips.length === 0 && <p className="text-sm text-neutral-400">No clips yet — import a video or a frame directory.</p>}
            <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {clips.map((c) => (
                <li key={c.id} className="overflow-hidden rounded border border-neutral-700 bg-neutral-800">
                  <img src={thumbUrl(c.id, 0)} alt="" className="aspect-video w-full bg-black object-contain" />
                  <div className="flex flex-col gap-1 p-3 text-sm">
                    <span className="truncate font-medium">{c.source.name}</span>
                    <span className="text-xs text-neutral-400">
                      {c.frame_count} frames · {c.width}×{c.height} · {c.fps} fps · {percentLabeled(c)}% labeled
                    </span>
                    <div className="mt-1 flex gap-2">
                      <Link to={`/clips/${c.id}`} className={button}>Open</Link>
                      <button type="button" aria-label="Delete clip" className="rounded bg-neutral-700 px-3 py-1 text-sm hover:bg-red-800"
                        onClick={() => deleteClip(c)}>
                        Delete
                      </button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
