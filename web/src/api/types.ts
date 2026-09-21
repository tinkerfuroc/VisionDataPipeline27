export type ClassDef = { id: number; name: string; color: string };
export type Project = { id: string; name: string; created_at: string; classes: ClassDef[] };

export type PromptType = "text" | "clicks" | "box";
/** [x, y, label] — label 1 = positive, 0 = negative */
export type Point = [number, number, 0 | 1];
/** [x1, y1, x2, y2] in image pixels */
export type Box = [number, number, number, number];
export type PromptSpec = { type: PromptType; phrase?: string; points?: Point[]; box?: Box };

export type Keyframe = {
  frame_idx: number;
  prompt: { type: PromptType; payload: Record<string, unknown> };
  created_at: string;
};
export type LabelObject = {
  id: number;
  class_id: number;
  keyframes: Keyframe[];
  propagated_ranges: [number, number][];
};
export type Clip = {
  id: string;
  project_id: string;
  source: { type: "video" | "frame_dir"; name: string };
  fps: number;
  stride: number;
  width: number;
  height: number;
  frame_count: number;
  created_at: string;
  original_names?: Record<string, string>;
  deleted_frames: number[];
  objects: LabelObject[];
};
export type ClipSummary = Clip & { labeled_frames: number };
export type FrameState = "empty" | "masked" | "keyframe" | "deleted";
export type ClipDetail = { clip: Clip; states: FrameState[]; classes: ClassDef[] };

export type Candidate = {
  score: number;
  bbox: Box;
  area: number;
  phrase: string | null;
  mask_png_b64: string;
};

export type Direction = "forward" | "backward" | "both";
export type JobStatus = "pending" | "running" | "done" | "error" | "cancelled";
export type JobSnapshot = {
  id: string;
  kind: string;
  status: JobStatus;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  n_events: number;
};
export type JobEvent = {
  seq: number;
  type: "progress" | "frame" | "log" | "done" | "cancelled" | "error";
  [key: string]: unknown;
};

export type Health = {
  status: string;
  version: string;
  engine: { loaded: boolean; device: string | null; error: string | null };
};
