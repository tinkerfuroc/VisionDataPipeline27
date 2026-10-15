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
  /** the next object id the server hands out (ids are never reused); absent or 0 in older clip files */
  next_object_id?: number;
};
export type ClipSummary = Clip & { labeled_frames: number };
export type FrameState = "empty" | "masked" | "keyframe" | "deleted";
export type ClipDetail = {
  clip: Clip;
  /** the whole clip per frame (any object): deleted > keyframe > masked > empty */
  states: FrameState[];
  classes: ClassDef[];
  /** each object's own per-frame states, keyed by the object id as a string (JSON keys); a deleted frame is
   * "deleted" in every lane */
  object_states: Record<string, FrameState[]>;
};
/** every object ("all"), or the listed object ids (never empty) */
export type ObjectSelection = "all" | number[];
/** where a mask is written: an existing object, or a new object of `classId` created by that very write */
export type WriteTarget = { objectId: number } | { objectId: null; classId: number };
/** `/accept`: the ids written, one per accepted candidate when several became new objects */
export type AcceptResult = ClipDetail & { object_ids: number[] };
/** `/mask`: the id of the object written */
export type MaskResult = ClipDetail & { object_id: number };

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
  options: {
    split_mode: SplitMode; val_clips?: string[]; val_fraction: number; stride: number; include_empty: boolean;
    /** Project class ids to export; null = every class. */
    classes: number[] | null;
  };
  name?: string | null;
};
export type SplitCounts = { images: number; annotated: number; empty: number };
export type ExportResult = {
  export_id: string;
  name: string | null;
  path: string;
  formats: ExportFormat[];
  counts: { train: SplitCounts; val: SplitCounts };
  linked: boolean;
};

export type Health = {
  status: string;
  version: string;
  engine: { loaded: boolean; device: string | null; error: string | null };
  /** Absent on an older server: the GUI then skips its own size check and lets the server refuse. */
  upload?: { max_bytes: number };
  /** Absent on an older server: treated as login off. */
  auth?: { enabled: boolean };
};

export type AugmentMode = "default" | "strong" | "custom";
export type AugParams = {
  degrees: number;
  shear: number;
  perspective: number;
  flipud: number;
  fliplr: number;
  scale: number;
  translate: number;
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  mosaic: number;
  mixup: number;
  copy_paste: number;
  photometric: boolean;
  /** Absent on runs started before lighting existed. */
  lighting?: boolean;
};
export type TrainParams = {
  epochs: number;
  imgsz: number;
  batch: number;
  patience: number;
  workers: number;
  seed: number;
  /** Absent on an older run's stored params: the GUI treats that as "default". */
  augment?: AugmentMode;
  aug?: AugParams | null;
};
export type GpuQueue = {
  index: number;
  name: string;
  memory_total: number;
  /** Free VRAM in bytes, via NVML; null when NVML could not report it for this GPU. */
  memory_free: number | null;
  /** "8.6"-style compute capability. */
  capability: string;
  sm_count: number;
  /** Estimated FP32 TFLOPS (SMs × cores/SM × max clock), used to rank GPUs fastest first. */
  tflops: number;
  running: string | null;
  running_kind: "train" | "onnx" | null;
  queue: string[];
};
export type TrainStatus = {
  available: boolean;
  error: string | null;
  /** Best (fastest estimated) first. */
  gpus: GpuQueue[];
  /** The fastest GPU that is not busy (fastest overall if every GPU is busy); null without GPUs. */
  default_gpu: number | null;
  sam3_gpu: number | null;
  models: string[];
  default_models: string[];
  defaults: TrainParams;
  aug_presets: { default: AugParams; strong: AugParams };
};
export type ExportInfo = {
  id: string;
  name: string | null;
  project_name: string;
  created_at: string;
  classes: string[];
  train_images: number;
  val_images: number;
  formats: string[];
  derived_from: string | null;
};
export type BackgroundPool = { name: string; label: string; images: number; size_bytes: number; fetching: string | null };
export type ExportSample = { name: string; width: number; height: number; ground_truth: Instance[] };
export type AugmentRequest = { per_image: number; max_objects: number; seed: number; pool: string; lighting: number };
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
  completable: boolean;
  /** True only for a run finished via "Mark complete" (its best epoch so far, not a full run). */
  completed_early?: boolean;
  has_best: boolean;
  has_onnx: boolean;
  /** This run's classes, ordered by yolo id; null when unknown (no data.yaml yet and its export is
   * gone). Absent on an older server. */
  classes?: string[] | null;
  /** The run this one continues or fine-tunes, null for an ordinary run. Absent on an older server. */
  source_run?: string | null;
  start_mode?: "continue" | "finetune" | null;
  /** "Continue" only: how many epochs the source run had already done. Absent on an older server. */
  continued_from_epoch?: number | null;
  /** Eligible as a "continue" source (= `resumable`). Absent on an older server. */
  continuable?: boolean;
  /** Eligible as a "fine-tune" source (= `has_best`). Absent on an older server. */
  finetunable?: boolean;
};
export type MetricRow = { epoch: number } & Record<string, number>;
export type RunDetail = Run & { metrics: MetricRow[] };
export type Instance = { class_id: number; score?: number; polygons: number[][] };
export type PredictionFilter = "all" | "missed" | "worst";
export type PredictionItem = {
  index: number;
  name: string;
  width: number;
  height: number;
  predictions: Instance[];
  ground_truth: Instance[];
  iou: number | null;
};
export type PredictionPage = {
  total: number;
  conf: number | null;
  counts: { all: number; missed: number; worst: number };
  items: PredictionItem[];
};
export type StartRunsRequest = { export_id: string; runs: { model: string; gpu: number }[]; params: TrainParams };
export type StartMode = "continue" | "finetune";
export type FromCheckpointRequest = { export_id: string; mode: StartMode; gpu: number; params?: TrainParams };
