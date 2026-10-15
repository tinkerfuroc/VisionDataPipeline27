import { useCallback, useContext, useEffect, useRef, useState, type ChangeEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ApiError, UploadCancelled, api, type UploadHandle } from "../api/rest";
import { forgetUpload, rememberedUploads, sameFile, startResumableUpload, type RememberedUpload } from "../api/resumable";
import type { Project, UploadInfo } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { HealthContext } from "../hooks/useHealth";
import { useOnline } from "../hooks/useOnline";
import { useWakeLock } from "../hooks/useWakeLock";
import { formatBytes, uploadPercent, uploadProblem } from "../lib/upload";

const PROJECT_KEY = "vdp.project";
const POLL_MS = 2000;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

type RowStatus = "waiting" | "sending" | "importing" | "done" | "error";
type Row = { key: number; name: string; status: RowStatus; sent: number; total: number; message?: string; clipId?: string };
type Queued = { key: number; file: File; projectId: string; stride: number; fps: number; resume?: RememberedUpload };

/** The phone page: record or choose videos and send them to a project, resumably. */
export function PhoneUpload() {
  const [params] = useSearchParams();
  const { health } = useContext(HealthContext);
  const maxBytes = health?.upload?.max_bytes ?? null;
  const online = useOnline();
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [projectId, setProjectId] = useState<string | null>(() => params.get("project") ?? localStorage.getItem(PROJECT_KEY));
  const [stride, setStride] = useState(1);
  const [fps, setFps] = useState(30);
  const [rows, setRows] = useState<Row[]>([]);
  const [unfinished, setUnfinished] = useState<UploadInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const queue = useRef<Queued[]>([]);
  const current = useRef<{ key: number; handle: UploadHandle } | null>(null);
  const nextKey = useRef(1);
  const alive = useRef(true);
  const timers = useRef(new Set<ReturnType<typeof setTimeout>>());
  const claimed = useRef(new Set<string>());
  useEffect(() => {
    alive.current = true;
    const pending = timers.current;
    return () => {
      alive.current = false;
      pending.forEach(clearTimeout);
      pending.clear();
    };
  }, []);

  useEffect(() => {
    api.listProjects().then(
      (list) => {
        setProjects(list);
        setProjectId((id) => (id !== null && list.some((p) => p.id === id) ? id : (list[0]?.id ?? null)));
      },
      (e) => setError(message(e)),
    );
  }, []);
  useEffect(() => {
    if (projectId !== null) localStorage.setItem(PROJECT_KEY, projectId);
  }, [projectId]);

  const loadUnfinished = useCallback(() => {
    if (projectId === null) return;
    api.listUploads(projectId).then(
      (list) => {
        // Expired or finished on the server: nothing left to resume. (Not while one is being sent: it may be newer than this list.)
        if (current.current === null) {
          for (const r of rememberedUploads()) {
            if (r.project_id === projectId && !list.some((u) => u.upload_id === r.upload_id)) forgetUpload(r.upload_id);
          }
        }
        setUnfinished(list);
      },
      () => setUnfinished([]),
    );
  }, [projectId]);
  useEffect(loadUnfinished, [loadUnfinished]);

  const patch = (key: number, change: Partial<Row>) =>
    setRows((rs) => rs.map((r) => (r.key === key ? { ...r, ...change } : r)));

  const watchJob = (key: number, jobId: string) => {
    const later = () => {
      if (!alive.current) return;
      const id = setTimeout(() => {
        timers.current.delete(id);
        tick();
      }, POLL_MS);
      timers.current.add(id);
    };
    const tick = () => {
      if (!alive.current) return;
      api.getJob(jobId).then(
        (snap) => {
          if (snap.status === "done") {
            const clipId = snap.result?.clip_id;
            patch(key, { status: "done", clipId: typeof clipId === "string" ? clipId : undefined });
          } else if (snap.status === "error" || snap.status === "cancelled") {
            patch(key, { status: "error", message: snap.error ?? "the import was cancelled" });
          } else {
            later();
          }
        },
        (e: unknown) => {
          if (e instanceof ApiError && e.status === 404) {
            patch(key, { status: "error", message: "the import is no longer known to the server" });
          } else {
            later();
          }
        },
      );
    };
    tick();
  };

  const startNext = () => {
    if (current.current !== null) return;
    const item = queue.current.shift();
    if (item === undefined) return;
    patch(item.key, { status: "sending" });
    const handle = startResumableUpload(
      item.projectId, item.file, { stride: item.stride, fps: item.fps },
      (sent, total) => patch(item.key, { sent, total }), item.resume,
    );
    current.current = { key: item.key, handle };
    const release = () => {
      if (item.resume) claimed.current.delete(item.resume.upload_id);
    };
    handle.promise.then(
      ({ job_id }) => {
        release();
        current.current = null;
        patch(item.key, { status: "importing" });
        watchJob(item.key, job_id);
        loadUnfinished();
        startNext();
      },
      (e: unknown) => {
        release();
        current.current = null;
        if (e instanceof UploadCancelled) setRows((rs) => rs.filter((r) => r.key !== item.key));
        else patch(item.key, { status: "error", message: message(e) });
        loadUnfinished();
        startNext();
      },
    );
  };

  const addFiles = (files: File[]) => {
    if (projectId === null) return;
    const remembered = rememberedUploads().filter(
      (r) => r.project_id === projectId && unfinished.some((u) => u.upload_id === r.upload_id),
    );
    const added: Row[] = [];
    for (const file of files) {
      const key = nextKey.current++;
      const problem = uploadProblem({ name: file.name, size: file.size, stride, fps, maxBytes });
      if (problem !== null) {
        added.push({ key, name: file.name, status: "error", sent: 0, total: file.size, message: problem });
        continue;
      }
      added.push({ key, name: file.name, status: "waiting", sent: 0, total: file.size });
      const resume = remembered.find((r) => !claimed.current.has(r.upload_id) && sameFile(r, file));
      if (resume) claimed.current.add(resume.upload_id);
      queue.current.push({ key, file, projectId, stride, fps, resume });
    }
    setRows((rs) => [...rs, ...added]);
    startNext();
  };

  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    addFiles(Array.from(e.target.files ?? []));
    e.target.value = "";
  };

  const cancel = (key: number) => {
    if (current.current?.key === key) {
      current.current.handle.abort();
      return;
    }
    const dropped = queue.current.find((q) => q.key === key);
    if (dropped?.resume) claimed.current.delete(dropped.resume.upload_id);
    queue.current = queue.current.filter((q) => q.key !== key);
    setRows((rs) => rs.filter((r) => r.key !== key));
  };

  const discard = (uploadId: string) => {
    api.deleteUpload(uploadId).then(
      () => {
        forgetUpload(uploadId);
        loadUnfinished();
      },
      (e) => setError(message(e)),
    );
  };

  const busy = rows.some((r) => r.status === "sending" || r.status === "waiting");
  useWakeLock(busy);

  const mine = new Set(rememberedUploads().map((r) => r.upload_id));
  const inProgress = new Set(rows.filter((r) => r.status === "sending" || r.status === "waiting").map((r) => `${r.name}/${r.total}`));
  const resumable = unfinished.filter((u) => u.job_id === null && !inProgress.has(`${u.name}/${u.size}`));

  const statusText = (r: Row): string => {
    if (r.status === "waiting") return "Waiting";
    if (r.status === "sending") return `${uploadPercent(r.sent, r.total) ?? 0}% sent${online ? "" : " — paused: offline"}`;
    if (r.status === "importing") return "Importing…";
    if (r.status === "done") return "Done";
    return r.message ?? "Failed";
  };

  return (
    <div className="flex h-full flex-col">
      <AppHeader />
      <div className="flex min-h-0 flex-1 justify-center overflow-y-auto">
        <div className="flex w-full max-w-md flex-col gap-4 p-4">
          <h1 className="heading">Upload videos</h1>
          {error !== null && <p role="alert" className="alert">{error}</p>}
          <label className="flex flex-col gap-1 text-sm text-dim">
            Project
            <select aria-label="Project" className="field" value={projectId ?? ""} onChange={(e) => setProjectId(e.target.value)}>
              {(projects ?? []).map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          {projects !== null && projects.length === 0 && (
            <p className="note">No projects yet. Create one on the Clips page first.</p>
          )}
          <label className="btn btn-on py-4 text-center text-base">
            Record video
            <input aria-label="Record video" type="file" accept="video/*" capture="environment" className="sr-only"
              disabled={projectId === null} onChange={onPick} />
          </label>
          <label className="btn py-4 text-center text-base">
            Choose videos
            <input aria-label="Choose videos" type="file" accept="video/*,.zip,application/zip" multiple className="sr-only"
              disabled={projectId === null} onChange={onPick} />
          </label>
          <details className="text-sm text-dim">
            <summary className="cursor-pointer">Options</summary>
            <div className="mt-2 flex flex-col gap-2">
              <label className="flex items-center justify-between gap-2">
                Keep every Nth frame
                <input aria-label="Keep every Nth frame" type="number" min={1} className="field w-20" value={stride}
                  onChange={(e) => setStride(Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <label className="flex items-center justify-between gap-2">
                Frames per second (zip only)
                <input aria-label="Frames per second (zip only)" type="number" min={1} step="any" className="field w-20" value={fps}
                  onChange={(e) => setFps(Math.max(1, Number(e.target.value) || 30))} />
              </label>
            </div>
          </details>
          {resumable.length > 0 && (
            <section className="flex flex-col gap-2">
              <h2 className="heading">Unfinished uploads</h2>
              {resumable.map((u) =>
                mine.has(u.upload_id) ? (
                  <div key={u.upload_id} className="flex items-center justify-between gap-2">
                    <span className="note truncate text-ink">
                      {`${u.name} — ${uploadPercent(u.received, u.size) ?? 0}% sent — choose it again to continue`}
                    </span>
                    <button type="button" className="btn" aria-label={`Discard ${u.name}`} onClick={() => discard(u.upload_id)}>
                      Discard
                    </button>
                  </div>
                ) : (
                  <div key={u.upload_id} className="flex items-center justify-between gap-2">
                    <span className="note truncate">{`${u.name} — ${uploadPercent(u.received, u.size) ?? 0}% sent (another device)`}</span>
                    <button type="button" className="btn" aria-label={`Discard ${u.name}`} onClick={() => discard(u.upload_id)}>
                      Discard
                    </button>
                  </div>
                ),
              )}
            </section>
          )}
          {rows.length > 0 && (
            <ul className="flex flex-col gap-2">
              {rows.map((r) => (
                <li key={r.key} className="flex flex-col gap-1 rounded border border-line bg-panel p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-ink">{r.name}</span>
                    <span className="shrink-0 text-sm text-dim">{formatBytes(r.total)}</span>
                  </div>
                  {r.status === "sending" && <progress value={r.sent} max={r.total || 1} aria-label={`${r.name} progress`} />}
                  {r.status === "error" ? (
                    <p role="alert" className="alert">{statusText(r)}</p>
                  ) : (
                    <p className="note">{statusText(r)}</p>
                  )}
                  {(r.status === "sending" || r.status === "waiting") && (
                    <button type="button" className="btn" aria-label={`Cancel ${r.name}`} onClick={() => cancel(r.key)}>
                      Cancel
                    </button>
                  )}
                  {r.status === "done" && r.clipId !== undefined && (
                    <Link to={`/clips/${r.clipId}`} className="text-ink underline">
                      Open the clip
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
