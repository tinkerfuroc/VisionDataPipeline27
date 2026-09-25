# YOLO-seg training — design

Date: 2026-09-24. Status: approved in chat, spec written for the implementation plan.
Amends the main design (`2026-09-20-vision-sampler-design.md`), which listed model training as a non-goal and
dropped `ultralytics`. Training now lives in this repo: a CLI command, a set of API routes and a Train page.

## Goal

Train Ultralytics YOLO segmentation models on a `yolo_seg` export and compare them, without leaving the tool. The
first use is export `20260924-040124-8336a8` (project OpticsConverter1, one class `Module`, 960×544 frames, 798
train frames from 4 clips, 252 val frames from 1 held-out clip), trained as **yolo26s-seg, yolo26m-seg,
yolo11s-seg and yolo11m-seg**.

## Decisions taken with the user

- Model families: **YOLO26 and YOLO11**, small and medium by default (n/l/x selectable).
- Interface: **CLI and a GUI page**.
- GPUs: **one queue per GPU**. A run is placed on a GPU by the user (default: the GPU SAM3 is not on); runs on one
  GPU go one at a time, the two GPUs run in parallel. Choosing SAM3's GPU is allowed with a VRAM warning.
- Train page offers: launching and queueing with cancel, **live metric curves**, a **run comparison table**,
  **weights download** (best.pt, and ONNX on request), and a **val prediction preview** against ground truth.

## Rejected alternatives

- Training on a thread inside the server (reusing `JobManager`): a CUDA or dataloader crash would take the labeling
  server down, cancel cannot interrupt Ultralytics cleanly, and dataloader workers would fork a threaded process.
- An external queue (Celery/RQ): a broker for one machine.

## Facts verified before writing this spec (2026-09-24, on this machine)

- `ultralytics==8.4.161` installs beside the pinned `torch 2.14.0+cu126` once a uv override removes its
  `opencv-python` requirement; `opencv-python-headless` provides the same `cv2`. Only the headless build remains.
- The exporter's `data.yaml` says `path: .`, which Ultralytics resolves against the **process CWD**, not the yaml's
  directory. Training must use a data.yaml with an absolute `path`.
- Ultralytics downloads `yolo26n-seg.pt` etc. into the CWD.
- `results.csv` is rewritten after every epoch (columns `epoch,time,train/box_loss,train/seg_loss,train/cls_loss,…,
  metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M),
  val/…,lr/…`; YOLO26 adds `l1_loss`/`sem_loss`). `on_fit_epoch_end` fires once more after the final validation of
  `best.pt` with a repeated epoch number, so it is not a clean per-epoch source; `results.csv` is.
- yolo26n-seg, imgsz 960, batch 8 uses ~3.5 GiB on the 2080 Ti.

## Dependencies

`pyproject.toml` gains `ultralytics>=8.4.161,<8.5`, `onnx`, `onnxslim`, `onnxruntime` as regular dependencies, and
`[tool.uv] override-dependencies = ["opencv-python; sys_platform == 'never'"]`. The worker sets
`YOLO_AUTOINSTALL=False` (Ultralytics must never pip-install into the uv env) and `settings.update({"sync": False})`.

## Storage

```
data/
├── runs/<run_id>/
│   ├── run.json           # request + state; the source of truth for a run
│   ├── data.yaml          # written by the worker: absolute path to the export's yolo_seg/
│   ├── worker.log         # the subprocess's stdout+stderr
│   ├── progress.json      # {epoch, epochs, batch, batches, phase} — overwritten atomically, throttled to 1/s
│   ├── predictions.json   # val predictions of best.pt, written after training
│   └── train/             # Ultralytics' own output dir: weights/{best,last}.pt, results.csv, args.yaml
│       └── weights/best.onnx   # after an ONNX export
└── train_weights/         # CWD of every worker: pretrained checkpoints are downloaded (and cached) here
```

`run_id` = `<YYYYmmdd-HHMMSS>-<model>-<4 hex>`, e.g. `20260924-120301-yolo26s-seg-3f9a`.

`run.json`:

```json
{
  "id": "…", "created_at": "…", "started_at": null, "finished_at": null,
  "export_id": "20260924-040124-8336a8", "project_name": "OpticsConverter1", "model": "yolo26s-seg", "gpu": 1,
  "params": {"epochs": 100, "imgsz": 960, "batch": 8, "patience": 30, "workers": 8, "seed": 0},
  "status": "queued",            // queued | running | done | error | cancelled | interrupted
  "error": null,
  "pid": null,                   // worker process id while running (process-group leader)
  "resume": false,               // true when the next start must resume from train/weights/last.pt
  "onnx": null,                  // null | "queued" | "running" | "done" | "error: <msg>"
  "best": null                   // filled at the end: {epoch, mask_map50, mask_map50_95, box_map50, box_map50_95}
}
```

`run.json` is written atomically (temp file + rename) and only by the server's scheduler, or by the CLI when no
server is involved. The worker never writes it; it reports through its exit code, `progress.json`, `results.csv`,
`predictions.json` and a final `result.json` (`{best: {...}}` or `{error: "..."}`).

## Components (`server/vsp/train/`)

### `models.py` — request validation and constants

- `MODELS`: `yolo26{n,s,m,l,x}-seg`, `yolo11{n,s,m,l,x}-seg`. `DEFAULT_MODELS`: the four s/m names.
- `TrainParams` (pydantic): `epochs` 1–1000 (default 100), `imgsz` 320–1920 and a multiple of 32 (default 960),
  `batch` 1–64 (default 8), `patience` 0–1000 (default 30), `workers` 0–16 (default 8), `seed` (default 0).
- `find_export(data_root, export_id) -> ExportInfo`: reads `exports/<id>/export.json` and requires `yolo_seg/` with a
  `data.yaml`; 404 when missing, 400 when the export has no yolo_seg format. `ExportInfo` holds the id, project name,
  created_at, class names, train/val image counts.
- `list_exports(data_root)`: every export with a yolo_seg format, newest first.

### `worker.py` — the subprocess (`python -m vsp.train.worker <run_dir> [--onnx]`)

Runs with `CUDA_VISIBLE_DEVICES=<gpu>` (so it always uses `device=0`) and `cwd=data/train_weights/`.

Train mode:
1. Read `run.json`. Write `data.yaml` = the export's data.yaml with `path` replaced by the absolute yolo_seg dir.
2. If `resume`: `YOLO(train/weights/last.pt).train(resume=True)`. Else `YOLO(f"{model}.pt").train(data=…,
   epochs, imgsz, batch, patience, workers, seed, project=<run_dir>, name="train", exist_ok=True, device=0,
   plots=True)`. Ultralytics defaults otherwise (COCO-pretrained, default augmentation, SGD/auto optimizer).
3. Callbacks write `progress.json`: `on_train_epoch_start` (phase `train`, epoch), `on_train_batch_end` (batch,
   throttled to one write per second), `on_val_start` (phase `val`).
4. After training: predict every val image with `best.pt` (`imgsz` as trained, `conf=0.25`, `retina_masks=True`,
   `stream=True`) and write `predictions.json` = `{images: [{name, width, height, instances: [{class_id, score,
   polygons: [[x, y, …] normalized]]}]}]}`. Then `result.json` with `best` = the `results.csv` row with the highest
   `metrics/mAP50-95(M)`.
5. Exit 0. Any exception: write `result.json {error}` and exit 1.

SIGTERM: Ultralytics is interrupted by a `KeyboardInterrupt` raised from a SIGTERM handler; the worker writes
`result.json {cancelled: true}` and exits 2. `last.pt` from the last finished epoch stays, so a cancelled run can be
resumed.

ONNX mode (`--onnx`): `YOLO(best.pt).export(format="onnx", imgsz=<trained>, dynamic=False, simplify=True)`; the file
lands next to `best.pt` as `best.onnx`. `result.json` is not touched; exit code only.

### `scheduler.py` — `TrainScheduler`

- One FIFO queue and one daemon thread per visible GPU (queried once at startup with `torch.cuda`; names and total
  memory are cached).
- A queue item is a run id plus a task kind (`train` | `onnx`). The thread pops it, sets `status=running` /
  `onnx=running`, starts the worker with `subprocess.Popen([sys.executable, "-m", "vsp.train.worker", …],
  start_new_session=True, stdout=worker.log, stderr=STDOUT, env=…)`, waits, then reads `result.json` and exit code:
  0 → `done` + `best`; 2 or a cancel request → `cancelled`; else `error` with the message from `result.json` or the
  last 20 lines of `worker.log`.
- `submit(export_id, [(model, gpu)], params) -> [run_id]` creates the run dirs and `run.json` (queued) and enqueues.
- `cancel(run_id)`: a queued item is removed and marked `cancelled`; a running one gets SIGTERM to its process group,
  then SIGKILL after 30 s.
- `resume(run_id)`: allowed for `cancelled`, `interrupted` and `error` runs that have `last.pt`; sets `resume=true`,
  `status=queued`, enqueues on the run's GPU.
- `export_onnx(run_id)`: allowed for `done` runs; `onnx=queued`, enqueued on the run's GPU (the queue keeps one
  process per GPU, training or export).
- `delete(run_id)`: only for terminal runs; removes the run dir.
- Startup recovery: scan `data/runs/*/run.json`; `running` → `interrupted` (the worker died with the old server:
  workers run in their own session but the scheduler cannot re-attach, so it kills a stale process group recorded
  as `pid` in `run.json` if `/proc/<pid>/cmdline` shows it is that run's worker); `queued` → enqueued again in
  `created_at` order, and a queue entry is launched only if `run.json` still says it is queued; `onnx` of
  `queued`/`running` → `null`.
- `snapshot()`: per GPU `{index, name, memory_total, running: run_id|null, queue: [run_id]}`.
- Thread safety: one lock guards the queues and `run.json` writes.
- The worker command is injectable (`worker_cmd`) so tests substitute a fake worker script.

### `results.py` — reading a run

- `read_metrics(run_dir) -> list[dict]`: parse `train/results.csv` (strip header spaces, float values, `epoch` int).
- `read_progress(run_dir)`, `read_predictions(run_dir)`.
- `read_ground_truth(export_dir, name) -> list[{class_id, polygons}]` from `yolo_seg/labels/val/<stem>.txt`.
- `best_row(metrics)`: highest `metrics/mAP50-95(M)`, ties → later epoch.

### CLI

`uv run vsp train --export <id> [--model M]… [--gpu N] [--epochs --imgsz --batch --patience --workers --seed]`
(defaults: the four default models, GPU = first GPU SAM3 would not pick). Runs in the foreground: creates the run
dirs through the same `TrainScheduler` code, started without restart recovery (`start(recover=False)`), and waits in
the foreground until its runs finish. Prints each run's per-epoch mask mAP from `results.csv` and ends with the
comparison table. Ctrl-C cancels the current run and marks the rest `cancelled` (a CLI has no queue to come back to).
`--list` prints the yolo_seg exports. CLI runs appear in the GUI because they live in `data/runs/`; a CLI must not
be started on a GPU the server is also training on (not enforced; documented).

## API (`server/vsp/api/train.py`)

| Route | Result |
|---|---|
| `GET /api/train/status` | `{available, error, gpus: [{index, name, memory_total, running, queue}], sam3_gpu, models, default_models, defaults: TrainParams}`; `available=false` with the import error when `ultralytics` cannot be imported |
| `GET /api/exports` | yolo_seg exports: `[{id, project_name, created_at, classes, train_images, val_images}]` |
| `GET /api/exports/{id}/images/val/{name}` | a val image of the export (for the preview); names are validated against the directory listing |
| `POST /api/train/runs` | body `{export_id, runs: [{model, gpu}], params}` → `201 {run_ids}`; 422 invalid params (FastAPI validation), 400 unknown model / GPU / empty `runs`, 404 export |
| `GET /api/train/runs` | all runs, newest first: `run.json` + `progress` (when running) |
| `GET /api/train/runs/{id}` | `run.json` + `progress` + `metrics` (all rows of results.csv) |
| `POST /api/train/runs/{id}/cancel` | 409 when already terminal |
| `POST /api/train/runs/{id}/resume` | 409 when not resumable |
| `POST /api/train/runs/{id}/onnx` | 409 unless `done`, or when already queued/running |
| `DELETE /api/train/runs/{id}` | 409 unless terminal |
| `GET /api/train/runs/{id}/weights/{best.pt,best.onnx}` | file download; 404 when absent |
| `GET /api/train/runs/{id}/predictions?offset=&limit=` | `{total, items: [{name, width, height, predictions, ground_truth}]}` (limit ≤ 50) |
| `GET /api/train/runs/{id}/log` | last 200 lines of `worker.log` as text |

`sam3_gpu` is the index parsed from the engine's `device` (`cuda:N`), or null. Training never takes the `gpu`
exclusive key of `JobManager`: that key serialises SAM3 work on one device, while a training run is a separate
process whose only shared resource is VRAM.

## Train page (`/train`)

Linked from the header ("Train"). Layout as every page: main area left, all controls in the right sidebar at a
comfortable scale. For augmentation presets and augmented copies, see `2026-09-24-augmentation-and-prediction-strip-design.md` (B2, B3).

**Sidebar**
- *New runs*: export select (label: project name · id · train/val counts); a checklist of models (the four defaults
  checked; "More models" reveals n/l/x) each with a GPU select (default: the non-SAM3 GPU; the SAM3 GPU is labelled
  "shared with SAM3" and selecting it shows a VRAM warning); epochs, image size, batch, patience inputs; **Start**
  (disabled with no model checked or `available=false`, whose error is shown instead).
- *GPU queues*: per GPU, name and memory, the running run (model, epoch x/y, batch progress bar) with **Cancel**, and
  queued runs with **Cancel**. Also lists, without a Cancel button, any run on that GPU owned by another process
  (a `vsp train` CLI run sharing this data root, not this server's own scheduler) — the server only ever launches
  workers of its own, so it cannot cancel someone else's.

**Main area**
- *Runs* table (the comparison): model, export, GPU, status, epochs done/total, best mask mAP50, mask mAP50-95,
  box mAP50-95, best epoch, train time, and actions (best.pt download, ONNX: Export / Exporting… / download,
  Resume, Delete). Best value per metric column in bold. Clicking a row selects it.
- *Curves* of the selected run: four small inline-SVG line charts over epochs — mask mAP50-95 and mAP50, box
  mAP50-95, train losses (box/seg/cls), val losses. When two or more runs are selected via row checkboxes, the mask
  mAP50-95 chart overlays them (one line per run, each keeping a fixed colour slot per run; lowest free of four,
  released when the run leaves the comparison). No chart library.
- *Preview* of the selected `done` run: the val image with ground truth outlined (dashed, ink) and predictions
  filled in one colour (series slot 1, blue) with a score label, ◀ ▶ through val images (and `←/→` keys), a toggle
  for GT / predictions. Before `done`: "Predictions appear when the run finishes."

Polling: `GET /api/train/status` and `GET /api/train/runs` every 2 s while the page is visible; the selected run's
detail every 2 s while it is running, once otherwise.

## Error handling

- `ultralytics` import failure → `available=false`, POST returns 503 with the error.
- Worker out of memory / crash → run `error` with the tail of `worker.log`; the GPU queue moves on.
- Server restart mid-run → `interrupted`, resumable.
- Deleting an export that runs reference: not prevented; the run dir keeps its own data.yaml, a queued run then
  fails with the worker's missing-path error.

## Testing

- `models.py`: param bounds, imgsz multiple of 32, export lookup (missing / no yolo_seg / ok), `list_exports`.
- `results.py`: results.csv parsing with header spaces and YOLO26 extra columns, `best_row`, GT label parsing.
- `scheduler.py` with a fake worker script (reads `run.json`, writes `progress.json`/`results.csv`/`result.json`,
  exits with a code taken from the params; sleeps until SIGTERM when asked): one-at-a-time per GPU and parallel
  across GPUs, cancel queued, cancel running (→ `cancelled`), error with log tail, resume flow, onnx task, startup
  recovery (running → interrupted, queued re-enqueued), delete refuses non-terminal.
- `worker.py`: data.yaml rewriting and predictions serialisation as pure functions.
- API tests with the fake worker through `create_app`.
- `@pytest.mark.gpu`: real worker, yolo11n-seg, 1 epoch, `fraction` limited, on a tiny synthetic yolo_seg export;
  asserts `done`, `best.pt`, `predictions.json`.
- vitest: chart scaling helpers, the prediction overlay geometry, the Train page (start form validation, queue
  rendering, table best-value bolding, preview paging) against a mocked API.

## Out of scope

Hyper-parameter search, augmentation settings beyond Ultralytics defaults, multi-GPU DDP for a single run, training
from COCO exports, deploying models back into labeling (auto-labeling with the trained model).
