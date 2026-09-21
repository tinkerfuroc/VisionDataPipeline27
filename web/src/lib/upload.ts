export type ImportKind = "video" | "zip";

/** The same extension rule the server applies to the `name` query parameter. */
const VIDEO_SUFFIXES = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".mts", ".m2ts"];

export function importKind(name: string): ImportKind | null {
  const lower = name.toLowerCase();
  if (lower.endsWith(".zip")) return "zip";
  return VIDEO_SUFFIXES.some((suffix) => lower.endsWith(suffix)) ? "video" : null;
}

export type UploadDraft = { name: string | null; size: number; stride: number; fps: number; maxBytes: number | null };

export function uploadProblem(draft: UploadDraft): string | null {
  if (draft.name === null) return "Choose a video or a .zip of images.";
  const kind = importKind(draft.name);
  if (kind === null) return "That file is not a video or a .zip of images.";
  if (!Number.isInteger(draft.stride) || draft.stride < 1) return "Keep every Nth frame needs a whole number of 1 or more.";
  if (kind === "zip" && !(draft.fps > 0)) return "Frames per second must be above 0.";
  // Checked before the transfer starts: a server that refuses mid-body may reset the connection, and the
  // browser would show a network error instead of "too large". The server enforces the same limit anyway.
  if (draft.maxBytes !== null && draft.size > draft.maxBytes)
    return `That file is ${formatBytes(draft.size)}; the limit is ${formatBytes(draft.maxBytes)}.`;
  return null;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[i]}`;
}

export function uploadPercent(sent: number, total: number): number | null {
  return total > 0 ? Math.floor((sent / total) * 100) : null;
}
