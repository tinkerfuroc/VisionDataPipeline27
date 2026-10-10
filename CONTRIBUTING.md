# Contributing

Issues and pull requests are welcome. For anything larger than a small fix, open an issue first so we can agree on
the approach.

## Setup

Follow the [README quickstart](README.md#quickstart). The SAM3 weights are only needed for the GPU tests and for
labeling; everything else runs without them.

## Checks

Run these before opening a pull request (CI runs the same on every push and PR):

```bash
uv run ruff check server scripts
uv run pytest                       # tests marked `gpu` skip themselves without CUDA + the checkpoint
cd web && npm run typecheck && npm test
```

If you have a GPU and the converted checkpoint, also run `uv run pytest -m gpu`. For changes to the labeling or
export flow, walk through the relevant items of [docs/manual-qa.md](docs/manual-qa.md) on real footage.

## Layout

- `server/vdp/`: the FastAPI app (`api/`), SAM3 engine (`engine/`), project store (`store/`), import (`ingest/`),
  labeling, export, augmentation and training (`train/`, run as a subprocess per GPU).
- `server/tests/`: pytest suite; tests use a fake engine and a fake training worker, so they run on CPU.
- `web/src/`: the React GUI (`pages/`, `components/`, `hooks/`, `api/` for the REST client).
- `configs/default.yaml`: default settings.

## Conventions

- Python: ruff, line length 110, Python 3.11.
- Keep the README a short front page; reference material belongs in `docs/`.
- Commit messages: an imperative summary line naming the area (e.g. `Train page: ...`, `Export: ...`), with the
  reasoning in the body when it is not obvious.
