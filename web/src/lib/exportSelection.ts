import type { ClipSummary, SplitMode } from "../api/types";

/** Whole clips go to train or val when there are several; one clip is split at its tail. Never per frame. */
export const splitModeFor = (selectedCount: number): SplitMode => (selectedCount >= 2 ? "per_clip" : "contiguous");

/** The most recently imported ceil(fraction × n) of the selected clips, but never all of them (spec §8). */
export function defaultValClips(clips: ClipSummary[], selected: string[], fraction: number): string[] {
  if (selected.length < 2 || fraction <= 0) return [];
  const chosen = clips.filter((c) => selected.includes(c.id));
  const newestFirst = [...chosen].sort((a, b) => (a.created_at === b.created_at ? b.id.localeCompare(a.id) : b.created_at.localeCompare(a.created_at)));
  const n = Math.min(chosen.length - 1, Math.ceil(fraction * chosen.length));
  return newestFirst.slice(0, n).map((c) => c.id);
}

type Draft = { selected: string[]; formats: string[]; valClips: string[]; valFraction: number; stride: number };

export function exportProblem(d: Draft): string | null {
  if (d.selected.length === 0) return "Choose at least one clip.";
  if (d.formats.length === 0) return "Choose at least one format.";
  if (!(d.valFraction >= 0 && d.valFraction < 1)) return "The validation share must be below 100%.";
  if (!Number.isInteger(d.stride) || d.stride < 1) return "Keep every Nth frame needs a whole number of 1 or more.";
  if (splitModeFor(d.selected.length) === "per_clip" && d.valClips.filter((id) => d.selected.includes(id)).length >= d.selected.length)
    return "Leave at least one clip for training.";
  return null;
}
