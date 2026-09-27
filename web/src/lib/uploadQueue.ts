import { uploadProblem } from "./upload";

/** A file that has been validated and snapshotted (project id, stride, fps) at the moment it was queued. */
export type QueuedUpload = {
  id: number;
  file: File;
  projectId: string;
  stride: number;
  fps: number;
};

export type RunningUpload = QueuedUpload & { sent: number; total: number };
export type FailedUpload = QueuedUpload & { message: string };

export type UploadQueueState = {
  current: RunningUpload | null;
  waiting: QueuedUpload[];
  failed: FailedUpload[];
};

export const emptyUploadQueue: UploadQueueState = { current: null, waiting: [], failed: [] };

export type SkippedUpload = { name: string; reason: string };

/**
 * Validates each file with the same rules the picker used to gate a single choice, snapshotting the project id,
 * stride and fps given for every file that queues (a later change to those fields affects only files chosen
 * later). An invalid file is reported in `skipped` and never added to the queue; the rest still queue.
 */
export function queueFiles(
  state: UploadQueueState,
  files: File[],
  snapshot: { projectId: string; stride: number; fps: number; maxBytes: number | null },
  nextId: () => number,
): { state: UploadQueueState; skipped: SkippedUpload[] } {
  const skipped: SkippedUpload[] = [];
  const queued: QueuedUpload[] = [];
  for (const file of files) {
    const problem = uploadProblem({
      name: file.name,
      size: file.size,
      stride: snapshot.stride,
      fps: snapshot.fps,
      maxBytes: snapshot.maxBytes,
    });
    if (problem !== null) {
      skipped.push({ name: file.name, reason: problem });
      continue;
    }
    queued.push({ id: nextId(), file, projectId: snapshot.projectId, stride: snapshot.stride, fps: snapshot.fps });
  }
  return { state: { ...state, waiting: [...state.waiting, ...queued] }, skipped };
}

/** Moves the next waiting file into `current`, if nothing is running and something is waiting. */
export function startNext(state: UploadQueueState): { state: UploadQueueState; item: QueuedUpload } | null {
  if (state.current !== null || state.waiting.length === 0) return null;
  const [item, ...rest] = state.waiting;
  return { state: { ...state, current: { ...item, sent: 0, total: item.file.size }, waiting: rest }, item };
}

export function updateProgress(state: UploadQueueState, id: number, sent: number, total: number): UploadQueueState {
  if (state.current === null || state.current.id !== id) return state;
  return { ...state, current: { ...state.current, sent, total } };
}

/** A finished (successfully uploaded) file just leaves the queue; the caller adds its own import job row. */
export function finishCurrent(state: UploadQueueState, id: number): UploadQueueState {
  if (state.current === null || state.current.id !== id) return state;
  return { ...state, current: null };
}

export function failCurrent(state: UploadQueueState, id: number, message: string): UploadQueueState {
  if (state.current === null || state.current.id !== id) return state;
  const { id: itemId, file, projectId, stride, fps } = state.current;
  return { ...state, current: null, failed: [...state.failed, { id: itemId, file, projectId, stride, fps, message }] };
}

/** A cancelled file just disappears, same as today: no failed entry. */
export function cancelCurrent(state: UploadQueueState, id: number): UploadQueueState {
  if (state.current === null || state.current.id !== id) return state;
  return { ...state, current: null };
}

export function cancelAll(state: UploadQueueState): UploadQueueState {
  return { ...state, current: null, waiting: [] };
}

export function removeWaiting(state: UploadQueueState, id: number): UploadQueueState {
  return { ...state, waiting: state.waiting.filter((i) => i.id !== id) };
}

export function dismissFailed(state: UploadQueueState, id: number): UploadQueueState {
  return { ...state, failed: state.failed.filter((i) => i.id !== id) };
}
