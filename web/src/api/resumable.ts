import { isLoginRequired, toLogin } from "./auth";
import { ApiError, UploadCancelled, api, apiError, type UploadHandle } from "./rest";

/** The server's chunk limit (server/vdp/ingest/resumable.py CHUNK_SIZE). */
export const CHUNK_SIZE = 8 * 1024 * 1024;
/** Waits between tries of a failed chunk; the last one repeats. */
export const RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 16000, 30000];
const STORE_KEY = "vdp.uploads";

/** How the uploader sizes chunks and waits; tests replace these. */
export const uploaderEnv = {
  chunkSize: CHUNK_SIZE,
  /** Resolves once the browser is online (at once if it already is). */
  online: (signal?: AbortSignal): Promise<void> =>
    navigator.onLine
      ? Promise.resolve()
      : new Promise<void>((resolve) => {
          const done = () => {
            window.removeEventListener("online", done);
            signal?.removeEventListener("abort", done);
            resolve();
          };
          window.addEventListener("online", done);
          signal?.addEventListener("abort", done);
        }),
  /** Resolves after `ms`, or earlier when the browser comes back online or the page becomes visible again. */
  waitToRetry: (ms: number): Promise<void> =>
    new Promise<void>((resolve) => {
      const done = () => {
        clearTimeout(timer);
        window.removeEventListener("online", done);
        document.removeEventListener("visibilitychange", onVisible);
        resolve();
      };
      const onVisible = () => {
        if (document.visibilityState === "visible") done();
      };
      const timer = setTimeout(done, ms);
      window.addEventListener("online", done);
      document.addEventListener("visibilitychange", onVisible);
    }),
};

/** What this browser remembers about an upload it started, to resume it after a reload. */
export type RememberedUpload = { upload_id: string; project_id: string; name: string; size: number; lastModified: number };

export function rememberedUploads(): RememberedUpload[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORE_KEY) ?? "[]");
    return Array.isArray(parsed) ? (parsed as RememberedUpload[]) : [];
  } catch {
    return [];
  }
}

function saveRemembered(list: RememberedUpload[]): void {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(list));
  } catch {
    /* storage unavailable: resuming after a reload just will not be offered */
  }
}

function rememberUpload(r: RememberedUpload): void {
  saveRemembered([...rememberedUploads().filter((x) => x.upload_id !== r.upload_id), r]);
}

export function forgetUpload(uploadId: string): void {
  saveRemembered(rememberedUploads().filter((x) => x.upload_id !== uploadId));
}

/** The same file as when the upload started: same name, size and modified time. */
export const sameFile = (r: RememberedUpload, file: File): boolean =>
  r.name === file.name && r.size === file.size && r.lastModified === file.lastModified;

/** One chunk. A 409 is not an error: it carries the server's count, which the caller continues from. */
async function putChunk(
  uploadId: string,
  offset: number,
  chunk: Blob,
  signal?: AbortSignal,
): Promise<{ received: number; job_id?: string }> {
  const r = await fetch(`/api/uploads/${uploadId}?offset=${offset}`, {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: chunk,
    signal,
  });
  let body: unknown = null;
  try {
    body = await r.json();
  } catch {
    /* not JSON: keep null */
  }
  if (isLoginRequired(r.status, body)) {
    toLogin();
    return new Promise(() => {}); // the page is leaving
  }
  const received = (body as { received?: unknown } | null)?.received;
  if (r.status === 409 && typeof received === "number") {
    const job = (body as { job_id?: unknown }).job_id;
    return typeof job === "string" ? { received, job_id: job } : { received };
  }
  if (!r.ok) throw apiError(r.status, body, r.statusText);
  return body as { received: number; job_id?: string };
}

const retryable = (e: unknown) => !(e instanceof ApiError) || e.status >= 500;

/**
 * Upload `file` to the project in chunks, retrying a failed chunk until it goes through (see RETRY_DELAYS_MS) and
 * pausing while offline. With `resume` (an upload this browser remembers, for the same file), it continues from the
 * server's count; if the server has expired it, it starts over. Resolves with the import job's id.
 */
export function startResumableUpload(
  projectId: string,
  file: File,
  options: { stride: number; fps: number },
  onProgress?: (sent: number, total: number) => void,
  resume?: RememberedUpload,
): UploadHandle {
  let cancelled = false;
  const controller = new AbortController();
  let uploadId: string | null = null;
  let rejectCancelled!: (e: unknown) => void;
  const cancelledPromise = new Promise<never>((_, reject) => (rejectCancelled = reject));

  const begin = async (): Promise<{ received: number; job_id: string | null }> => {
    if (resume !== undefined) {
      try {
        const info = await api.getUpload(resume.upload_id);
        uploadId = resume.upload_id;
        return { received: info.received, job_id: info.job_id };
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 404)) throw e;
        forgetUpload(resume.upload_id); // expired on the server: start over
      }
    }
    const created = await api.createUpload(projectId, { name: file.name, size: file.size, ...options });
    uploadId = created.upload_id;
    rememberUpload({ upload_id: uploadId, project_id: projectId, name: file.name, size: file.size, lastModified: file.lastModified });
    return { received: created.received, job_id: null };
  };

  const waitAfterFailure = (failures: number) =>
    uploaderEnv.waitToRetry(RETRY_DELAYS_MS[Math.min(failures, RETRY_DELAYS_MS.length - 1)]);

  const run = async (): Promise<{ job_id: string }> => {
    const start = await begin();
    let received = start.received;
    const id = uploadId as string;
    if (cancelled) {
      forgetUpload(id);
      void api.deleteUpload(id).catch(() => {});
      throw new UploadCancelled();
    }
    if (start.job_id !== null) {
      forgetUpload(id); // already complete on the server
      onProgress?.(file.size, file.size);
      return { job_id: start.job_id };
    }
    onProgress?.(received, file.size);
    let failures = 0;
    for (;;) {
      await Promise.race([uploaderEnv.online(controller.signal), cancelledPromise]);
      if (cancelled) throw new UploadCancelled();
      const end = Math.min(received + uploaderEnv.chunkSize, file.size);
      try {
        const reply = await putChunk(id, received, file.slice(received, end), controller.signal);
        const progressed = reply.received !== received;
        received = reply.received;
        onProgress?.(received, file.size);
        if (reply.job_id !== undefined) {
          forgetUpload(id);
          return { job_id: reply.job_id };
        }
        if (progressed) failures = 0;
        else await waitAfterFailure(failures++); // a 409 that moved nothing: do not spin
      } catch (e) {
        if (cancelled) throw new UploadCancelled(); // an aborted fetch is a cancel, not a failure
        if (!retryable(e)) throw e;
        await waitAfterFailure(failures++);
      }
    }
  };

  const promise = Promise.race([run(), cancelledPromise]);
  return {
    promise,
    abort: () => {
      if (cancelled) return;
      cancelled = true;
      controller.abort();
      rejectCancelled(new UploadCancelled());
      if (uploadId !== null) {
        forgetUpload(uploadId);
        void api.deleteUpload(uploadId).catch(() => {});
      } // else run() deletes the upload as soon as the create call answers
    },
  };
}
