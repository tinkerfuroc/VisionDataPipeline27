export type ClassDef = { id: number; name: string; color: string };
export type Project = { id: string; name: string; created_at: string; classes: ClassDef[] };

export type PromptType = "text" | "clicks" | "box" | "paint";
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

export type ExportFormat = "yolo_seg" | "yolo_det" | "coco" | "masks";
export type SplitMode = "per_clip" | "contiguous";
export type ExportRequest = {
  clips: string[];
  formats: ExportFormat[];
  options: { split_mode: SplitMode; val_clips?: string[]; val_fraction: number; stride: number; include_empty: boolean };
};
export type SplitCounts = { images: number; annotated: number; empty: number };
export type ExportResult = { export_id: string; path: string; formats: ExportFormat[]; counts: { train: SplitCounts; val: SplitCounts }; linked: boolean };

export type Health = {
  status: string;
  version: string;
  engine: { loaded: boolean; device: string | null; error: string | null };
  /** Absent on an older server: the GUI then skips its own size check and lets the server refuse. */
  upload?: { max_bytes: number };
};

export type TrainParams = { epochs: number; imgsz: number; batch: number; patience: number; workers: number; seed: number };
export type GpuQueue = {
  index: number;
  name: string;
  memory_total: number;
  running: string | null;
  running_kind: "train" | "onnx" | null;
  queue: string[];
};
export type TrainStatus = {
  available: boolean;
  error: string | null;
  gpus: GpuQueue[];
  sam3_gpu: number | null;
  models: string[];
  default_models: string[];
  defaults: TrainParams;
};
export type ExportInfo = {
  id: string;
  project_name: string;
  created_at: string;
  classes: string[];
  train_images: number;
  val_images: number;
};
export type RunStatus = "queued" | "running" | "done" | "error" | "cancelled" | "interrupted";
export type RunBest = {
  epoch: number;
  mask_map50: number | null;
  mask_map50_95: number | null;
  box_map50: number | null;
  box_map50_95: number | null;
  train_seconds: number | null;
};
export type RunProgress = { phase: "setup" | "train" | "val" | "predict"; epoch: number; epochs: number; batch: number; batches: number };
export type Run = {
  id: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  export_id: string;
  project_name: string;
  model: string;
  gpu: number;
  params: TrainParams;
  status: RunStatus;
  error: string | null;
  resume: boolean;
  onnx: string | null;
  best: RunBest | null;
  epochs_done: number;
  progress: RunProgress | null;
  resumable: boolean;
  has_best: boolean;
  has_onnx: boolean;
};
export type MetricRow = { epoch: number } & Record<string, number>;
export type RunDetail = Run & { metrics: MetricRow[] };
export type Instance = { class_id: number; score?: number; polygons: number[][] };
export type PredictionItem = { name: string; width: number; height: number; predictions: Instance[]; ground_truth: Instance[] };
export type PredictionPage = { total: number; conf: number | null; items: PredictionItem[] };
export type StartRunsRequest = { export_id: string; runs: { model: string; gpu: number }[]; params: TrainParams };
