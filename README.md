# VisionDataPipeline27

[![CI](https://github.com/tinkerfuroc/VisionDataPipeline27/actions/workflows/ci.yml/badge.svg)](https://github.com/tinkerfuroc/VisionDataPipeline27/actions/workflows/ci.yml)

Turn raw video into instance-segmentation datasets and trained YOLO models from one browser tab.

1. **Import** videos, zips of images or frame directories.
2. **Label** any number of objects per clip with [SAM3](https://huggingface.co/facebook/sam3): click, drag a box or
   type a phrase on one frame, then let SAM3 propagate the masks through the clip. Fix any frame with the brush.
3. **Export** YOLO-seg, YOLO-det, COCO and mask-PNG datasets, with leak-free train/val splits (whole clips or clip
   tails, never random frames) and optional synthetic copy-paste augmentation.
4. **Train** YOLO26 / YOLO11 segmentation models on your GPUs, compare runs, browse validation predictions and export
   ONNX.

The server is FastAPI + PyTorch; the GUI is React + Vite. Everything runs locally: a project lives in `data/`.

## Requirements

- Linux x86_64 with an NVIDIA GPU of compute capability 8.0 or newer (Ampere+, for bfloat16). Importing, browsing
  and exporting also work without a GPU; labeling and training need one.
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 and the pinned torch `2.14.0+cu126` itself). Do not
  install torch from PyPI: its default build needs an NVIDIA driver >= 580. Tested with driver 535.
- Node.js 22 to build the web GUI.
- A Hugging Face account with access to the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) model.

## Quickstart

```bash
git clone https://github.com/tinkerfuroc/VisionDataPipeline27.git
cd VisionDataPipeline27
uv sync
uv run vdp doctor               # torch build, GPU selection, bfloat16

cd web && npm ci && npm run build && cd ..   # writes web/dist, which the server serves
```

### Get the SAM3 weights

SAM3 is released under Meta's SAM License; request access on its Hugging Face page first. The labeling engine needs
the *video* model in Hugging Face format, which is converted once from Meta's `sam3.pt`:

```bash
uv run hf auth login
uv run hf download facebook/sam3 --local-dir vision_train/sam3_checkpoint
curl -L -o vision_train/convert_sam3_video_to_hf.py \
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/sam3_video/convert_sam3_video_to_hf.py

uv run vdp convert-weights \
  --checkpoint vision_train/sam3_checkpoint/sam3.pt \
  --converter vision_train/convert_sam3_video_to_hf.py \
  --official-files-from vision_train/sam3_checkpoint \
  --out vision_train/sam3_checkpoint_hf
uv run vdp verify-weights       # checks sam3.model_dir (configs/default.yaml)
```

`vision_train/` and `data/` are git-ignored.

### Run

```bash
HF_HUB_OFFLINE=1 uv run vdp serve      # http://127.0.0.1:28000 — the model loads in the background
uv run python scripts/seed_demo.py     # optional: a demo project with a synthetic clip
```

Open http://127.0.0.1:28000/. `/api/health` reports whether the engine loaded (and why not); `/docs` has the
interactive API docs.

### Using a remote GPU machine

`vdp serve` binds `127.0.0.1` by default and has no authentication, so forward the port rather than binding the
server to the network:

    ssh -N -L 28000:127.0.0.1:28000 user@gpu-host

then open `http://127.0.0.1:28000/` locally; uploads travel through the tunnel. `-o Compression=no` is usually
faster for already-compressed footage, and `-o ServerAliveInterval=30` keeps a NAT'd tunnel alive.

## Configuration

Settings live in [`configs/default.yaml`](configs/default.yaml) (server address, data root, SAM3 model directory
and thresholds, upload limits, export defaults); pass another file with `vdp --config <file> ...`. Relative paths
are resolved against the repository root, so `vdp` can be run from any directory.

## Documentation

- [docs/usage.md](docs/usage.md): the GUI workflow (uploading, labeling several objects, exporting, training,
  augmentation, keyboard shortcuts) and the `vdp train` CLI.
- [docs/api.md](docs/api.md): server API behaviour beyond the OpenAPI schema.
- [docs/manual-qa.md](docs/manual-qa.md): the manual QA checklist to run on real footage.

## Development

```bash
uv run pytest                   # GPU tests skip themselves without CUDA + the converted checkpoint
uv run ruff check server scripts
cd web && npm run dev           # hot-reloading GUI on :5173, proxying /api and /ws to :28000
cd web && npm test && npm run typecheck
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). Parts of the SAM3 engine are ported from
[tinkerfuroc/tk25_vision_train](https://github.com/tinkerfuroc/tk25_vision_train); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The SAM3 weights are not part of this repository and are governed
by Meta's SAM License.
