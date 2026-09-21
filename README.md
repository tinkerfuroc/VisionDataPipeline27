# vision-sampler-pipeline

SAM3-driven video labeling for one user-chosen component per clip.
Design: `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`.

## Setup

    uv sync
    uv run vsp doctor          # torch build, GPU selection, bfloat16
    uv run vsp verify-weights  # converted SAM3 checkpoint is complete
    uv run pytest              # GPU tests skip themselves without CUDA + checkpoint

Python, torch (`2.14.0+cu126`) and transformers (`5.17.0`) are pinned in `pyproject.toml`.
Do not install torch from PyPI: its default build needs an NVIDIA driver >= 580.

Relative paths in the config (`project.data_root`, `sam3.model_dir`) are resolved against the
repository root, not the current working directory, so `vsp` can be run from any directory.

## Verified on this machine

- Date: 2026-09-20
- GPU: NVIDIA GeForce RTX 3080 Ti, driver 535.183.01, torch 2.14.0+cu126, transformers 5.17.0
- GPU smoke tests: 12 passed
- Peak VRAM, 24-frame propagate at 640x480: 2.47 GiB
- Text prompt "red circle" on the synthetic scene: 1 detection(s)

## Run the server

    HF_HUB_OFFLINE=1 uv run vsp serve        # http://127.0.0.1:28000 ; the model loads in the background
    curl -s --noproxy '*' http://127.0.0.1:28000/api/health

`/api/health` shows `engine.loaded` and, if loading failed, the reason. Interactive API docs: `/docs`.
Importing, browsing and frame cleanup work without a GPU; prompting and propagation return 503 until the
engine is loaded, and 409 (with the running `job_id`) while a propagation job holds the GPU. A propagate
request also returns 409 with `job_id: "interactive"` if a prompt/accept/clear-range call is in flight;
retry. Deleting a clip also returns 409 (with the running `job_id`) while a GPU job runs, so a delete can
never race a propagation's writes. Cancelling an import job removes the partial clip it had started.

## Web GUI

Node 22 is installed with nvm (`source ~/.nvm/nvm.sh` in a fresh shell).

    cd web && npm ci            # once
    npm run build               # writes web/dist, which `vsp serve` serves at http://127.0.0.1:28000/
    npm run dev                 # or: hot-reloading dev server on http://127.0.0.1:5173 (proxies /api and /ws to :28000)
    npm test                    # vitest

### Demo data

With the server running, `uv run python scripts/seed_demo.py` creates a project "demo" with a synthetic 40-frame clip
(a disc moving across the frame), labels frame 0 with one click and propagates forward, then prints the URL to open.
`--no-label` imports only.

### Keyboard

`Enter` accept · `Esc` clear · `Z` undo click · `←/→` step · `Shift+←/→` ±10 · `Space` play/pause · `K` / `Shift+K`
next / previous keyframe · `G` next empty frame · `Del` delete or restore frame · `M` mask overlay · mouse wheel zoom ·
middle-drag pan · right click = negative point.

### Manual QA checklist (run on real footage before trusting a dataset)

1. Import a video and a frame directory (try a stride > 1); cancel one import midway and confirm no clip appears.
2. Label by each method: clicks (add a negative click), a box dragged bottom-right → top-left plus a refining click,
   and a text phrase with several candidates. Check the "no usable mask" message on an empty region.
3. Propagate forward, backward and both from a mid-clip keyframe; press Stop during one and confirm the masks so far stay.
4. Let the object leave the frame: propagation should stop early and seek to the "lost at" frame.
5. Correct a bad frame mid-clip (it becomes a keyframe) and propagate again: the run must stop at the next keyframe.
6. Scrub the timeline and play at clip fps with the overlay on; delete a frame, restore it, prune the tail, clear a range.
7. Reload the page during a propagation: the panel should re-attach to the running job.
8. Stop the server: the banner says it is unreachable; start it without weights: the banner shows the reason and
   importing / cleanup still work.
9. Propagate twice in one page session (let the first run finish, then start a second from a keyframe) and reload
   the page partway through the second run: it must re-attach to the second job, not the first.
10. Open the same clip in two tabs; start a propagation in tab A, then click to prompt in tab B on a different clip:
    B must not move or show A's progress as its own.
11. Delete a keyframe: its marker disappears from the timeline and propagation is blocked from it until it is restored.
12. Read the timeline without relying on colour: masked runs are filled (in the class colour), frames without a mask
    are hollow, deleted frames are hatched and keyframes carry an amber diamond. Hover it for a thumbnail with the
    frame number and state. Zoom into the frame, then press Fit in the tool rail. Tab through the page: every control
    shows a focus ring.
