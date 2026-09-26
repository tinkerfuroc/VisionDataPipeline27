import type {
  AugmentRequest, BackgroundPool, Candidate, ClassDef, ClipDetail, ClipSummary, Direction, ExportInfo, ExportRequest, ExportSample, Health, JobSnapshot, ObjectSelection, PredictionFilter, PredictionPage, Project, PromptSpec, Run, RunDetail, StartRunsRequest, TrainStatus,
} from "./types";

/** The `job_id` the server reports while a prompt/accept/clear-range call holds the GPU lease. */
export const INTERACTIVE = "interactive";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly jobId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** `detail` is a string for the server's own errors and a list of {loc, msg} for FastAPI validation errors. */
export function detailText(body: unknown, fallback: string): string {
  if (body === null || typeof body !== "object" || !("detail" in body)) return fallback;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e: { loc?: unknown[]; msg?: unknown }) => {
        const loc = Array.isArray(e?.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
        const msg = String(e?.msg ?? "invalid");
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  return fallback;
}

/** The one place a failed response becomes an ApiError; `request` and `uploadClip` both use it. */
export function apiError(status: number, body: unknown, statusText: string, jobId: string | null = null): ApiError {
  return new ApiError(status, detailText(body, `${status} ${statusText}`), jobId);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (r.status === 204) return undefined as T;
  let body: unknown = null;
  try {
    body = await r.json();
  } catch {
    /* not JSON: keep null */
  }
  if (!r.ok) {
    const jobId =
      body !== null && typeof body === "object" && typeof (body as { job_id?: unknown }).job_id === "string"
        ? (body as { job_id: string }).job_id
        : null;
    throw apiError(r.status, body, r.statusText, jobId);
  }
  return body as T;
}

export class UploadCancelled extends Error {
  constructor() {
    super("upload cancelled");
    this.name = "UploadCancelled";
  }
}

export type UploadHandle = { promise: Promise<{ job_id: string }>; abort: () => void };

/**
 * Upload one file as the raw request body. XHR, not fetch: only XHR reports upload progress and can abort a
 * request that is still being sent, which is the whole point of this path over an SSH tunnel.
 */
export function uploadClip(
  projectId: string,
  file: File,
  options: { stride: number; fps: number },
  onProgress?: (sent: number, total: number) => void,
): UploadHandle {
  const url =
    `/api/projects/${projectId}/clips/upload` +
    `?name=${encodeURIComponent(file.name)}&stride=${options.stride}&fps=${options.fps}`;
  const xhr = new XMLHttpRequest();
  const promise = new Promise<{ job_id: string }>((resolve, reject) => {
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded, e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON: keep null */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as { job_id: string });
        return;
      }
      const jobId =
        body !== null && typeof body === "object" && typeof (body as { job_id?: unknown }).job_id === "string"
          ? (body as { job_id: string }).job_id
          : null;
      reject(apiError(xhr.status, body, xhr.statusText, jobId));
    };
    xhr.onerror = () => reject(new Error("The upload failed. Check that the connection is still up."));
    xhr.onabort = () => reject(new UploadCancelled());
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** A propagate request loses to an in-flight preview for a few milliseconds; that 409 is not an error. */
export async function retryInteractive<T>(
  fn: () => Promise<T>,
  attempts = 10,
  delayMs = 150,
  sleep: (ms: number) => Promise<void> = defaultSleep,
): Promise<T> {
  for (let i = 1; ; i++) {
    try {
      return await fn();
    } catch (e) {
      const retry = e instanceof ApiError && e.status === 409 && e.jobId === INTERACTIVE && i < attempts;
      if (!retry) throw e;
      await sleep(delayMs);
    }
  }
}

const post = (body?: unknown): RequestInit =>
  body === undefined
    ? { method: "POST" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };

const patch = (body: unknown): RequestInit => ({
  method: "PATCH",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const frameUrl = (clipId: string, idx: number) => `/api/clips/${clipId}/frames/${idx}`;
export const thumbUrl = (clipId: string, idx: number) => `/api/clips/${clipId}/thumbs/${idx}`;
/** One object's stored mask on one frame; `version` busts the cache when the mask is rewritten in place. */
export const objectMaskUrl = (clipId: string, objectId: number, idx: number, version: string) =>
  `/api/clips/${clipId}/objects/${objectId}/masks/${idx}?v=${version}`;
export const weightsUrl = (runId: string, file: "best.pt" | "best.onnx") => `/api/train/runs/${runId}/weights/${file}`;
export const exportImageUrl = (exportId: string, split: "train" | "val", name: string) =>
  `/api/exports/${exportId}/images/${split}/${encodeURIComponent(name)}`;
export const valImageUrl = (exportId: string, name: string) => exportImageUrl(exportId, "val", name);
export const wsUrl = (path: string) =>
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}${path}`;

export const api = {
  health: () => request<Health>("/api/health"),

  listProjects: () => request<Project[]>("/api/projects"),
  createProject: (name: string) => request<Project>("/api/projects", post({ name })),
  getProject: (id: string) => request<Project>(`/api/projects/${id}`),
  addClass: (projectId: string, name: string) => request<ClassDef>(`/api/projects/${projectId}/classes`, post({ name })),
  listClips: (projectId: string) => request<ClipSummary[]>(`/api/projects/${projectId}/clips`),

  importFrames: (projectId: string, body: { path: string; fps: number; stride: number }) =>
    request<{ job_id: string }>(`/api/projects/${projectId}/clips/import-frames`, post(body)),
  uploadClip,

  getClip: (id: string) => request<ClipDetail>(`/api/clips/${id}`),
  deleteClip: (id: string) => request<void>(`/api/clips/${id}`, { method: "DELETE" }),

  prompt: (clipId: string, frame: number, prompt: PromptSpec) =>
    request<{ candidates: Candidate[] }>(`/api/clips/${clipId}/prompt`, post({ frame, prompt })),
  accept: (clipId: string, frame: number, prompt: PromptSpec, opts: { classId?: number; candidate?: number } = {}) =>
    request<ClipDetail>(
      `/api/clips/${clipId}/accept`,
      post({
        frame,
        prompt,
        ...(opts.classId === undefined ? {} : { class_id: opts.classId }),
        candidate: opts.candidate ?? 0,
      }),
    ),
  setMask: (clipId: string, frame: number, maskPngB64: string, opts: { classId?: number } = {}) =>
    request<ClipDetail>(
      `/api/clips/${clipId}/mask`,
      post({ frame, mask_png_b64: maskPngB64, ...(opts.classId === undefined ? {} : { class_id: opts.classId }) }),
    ),
  propagate: (clipId: string, fromFrame: number, direction: Direction, objects: ObjectSelection = "all") =>
    request<{ job_id: string }>(`/api/clips/${clipId}/propagate`, post({ from_frame: fromFrame, direction, objects })),

  deleteFrame: (clipId: string, idx: number) => request<ClipDetail>(`/api/clips/${clipId}/frames/${idx}/delete`, post()),
  restoreFrame: (clipId: string, idx: number) => request<ClipDetail>(`/api/clips/${clipId}/frames/${idx}/restore`, post()),
  prune: (clipId: string, from: number) => request<ClipDetail>(`/api/clips/${clipId}/prune`, post({ from })),
  clearRange: (clipId: string, start: number, end: number, objects: ObjectSelection = "all") =>
    request<ClipDetail>(`/api/clips/${clipId}/clear-range`, post({ start, end, objects })),
  /** relabel at once; allowed during a propagation (it touches no mask) */
  relabelObject: (clipId: string, objectId: number, classId: number) =>
    request<ClipDetail>(`/api/clips/${clipId}/objects/${objectId}`, patch({ class_id: classId })),
  /** 409 while a propagation or an export runs */
  deleteObject: (clipId: string, objectId: number) =>
    request<ClipDetail>(`/api/clips/${clipId}/objects/${objectId}`, { method: "DELETE" }),

  getJob: (id: string) => request<JobSnapshot>(`/api/jobs/${id}`),
  cancelJob: (id: string) => request<JobSnapshot>(`/api/jobs/${id}`, { method: "DELETE" }),

  startExport: (projectId: string, body: ExportRequest) =>
    request<{ job_id: string }>(`/api/projects/${projectId}/export`, post(body)),

  trainStatus: () => request<TrainStatus>("/api/train/status"),
  listExports: () => request<ExportInfo[]>("/api/exports"),
  renameExport: (exportId: string, name: string | null) =>
    request<ExportInfo>(`/api/exports/${exportId}`, patch({ name })),
  listRuns: () => request<Run[]>("/api/train/runs"),
  getRun: (id: string) => request<RunDetail>(`/api/train/runs/${id}`),
  startRuns: (body: StartRunsRequest) => request<{ run_ids: string[] }>("/api/train/runs", post(body)),
  cancelRun: (id: string) => request<Run>(`/api/train/runs/${id}/cancel`, post()),
  resumeRun: (id: string) => request<Run>(`/api/train/runs/${id}/resume`, post()),
  exportOnnx: (id: string) => request<Run>(`/api/train/runs/${id}/onnx`, post()),
  deleteRun: (id: string) => request<void>(`/api/train/runs/${id}`, { method: "DELETE" }),
  predictions: (id: string, offset: number, limit: number, filter: PredictionFilter = "all") =>
    request<PredictionPage>(`/api/train/runs/${id}/predictions?offset=${offset}&limit=${limit}&filter=${filter}`),

  listBackgrounds: () => request<BackgroundPool[]>("/api/backgrounds"),
  fetchBackgrounds: (name: string) => request<{ job_id: string }>(`/api/backgrounds/${name}/fetch`, post()),
  augmentExport: (exportId: string, body: AugmentRequest) =>
    request<{ job_id: string }>(`/api/exports/${exportId}/augment`, post(body)),
  exportSamples: (exportId: string, limit: number, seed: number) =>
    request<ExportSample[]>(`/api/exports/${exportId}/samples?limit=${limit}&seed=${seed}`),
};
