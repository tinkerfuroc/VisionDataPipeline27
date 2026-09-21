import type {
  Candidate, ClassDef, ClipDetail, ClipSummary, Direction, ExportRequest, Health, JobSnapshot, Project, PromptSpec,
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
    throw new ApiError(r.status, detailText(body, `${r.status} ${r.statusText}`), jobId);
  }
  return body as T;
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

export const frameUrl = (clipId: string, idx: number) => `/api/clips/${clipId}/frames/${idx}`;
export const thumbUrl = (clipId: string, idx: number) => `/api/clips/${clipId}/thumbs/${idx}`;
export const maskUrl = (clipId: string, idx: number, version: string) => `/api/clips/${clipId}/masks/${idx}?v=${version}`;
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
  importVideo: (projectId: string, file: File, stride: number) => {
    const form = new FormData();
    form.append("file", file);
    form.append("stride", String(stride));
    // no Content-Type header: the browser must add the multipart boundary itself
    return request<{ job_id: string }>(`/api/projects/${projectId}/clips/import-video`, { method: "POST", body: form });
  },

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
  propagate: (clipId: string, fromFrame: number, direction: Direction) =>
    request<{ job_id: string }>(`/api/clips/${clipId}/propagate`, post({ from_frame: fromFrame, direction })),

  deleteFrame: (clipId: string, idx: number) => request<ClipDetail>(`/api/clips/${clipId}/frames/${idx}/delete`, post()),
  restoreFrame: (clipId: string, idx: number) => request<ClipDetail>(`/api/clips/${clipId}/frames/${idx}/restore`, post()),
  prune: (clipId: string, from: number) => request<ClipDetail>(`/api/clips/${clipId}/prune`, post({ from })),
  clearRange: (clipId: string, start: number, end: number) =>
    request<ClipDetail>(`/api/clips/${clipId}/clear-range`, post({ start, end })),

  getJob: (id: string) => request<JobSnapshot>(`/api/jobs/${id}`),
  cancelJob: (id: string) => request<JobSnapshot>(`/api/jobs/${id}`, { method: "DELETE" }),

  startExport: (projectId: string, body: ExportRequest) =>
    request<{ job_id: string }>(`/api/projects/${projectId}/export`, post(body)),
};
