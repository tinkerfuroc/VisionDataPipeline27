# Augmentation and Prediction Strip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scrollable, filterable strip of validation predictions; augmentation presets for training; a DTD background pool; and a derived "augmented copy" of an export with objects pasted onto textures.

**Architecture:** IoU/filtering lives in `server/vsp/train/gallery.py` behind the existing predictions route. Presets are pydantic models in `server/vsp/train/models.py` turned into Ultralytics arguments by a pure `train_overrides`. A new package `server/vsp/augment/` holds `backgrounds.py` (download + safe extraction), `composite.py` (pure, seeded composite recipe) and `derive.py` (builds an `ExportPlan` from a source export plus composites and writes it with the existing exporters via a small refactor of `export/runner.py`). Routes go in `server/vsp/api/augment.py`. The web gets a `PredictionGallery`, preset controls, an `AugmentPanel` and a `CompositeSamples` strip.

**Tech Stack:** Python 3.11, FastAPI, numpy, OpenCV (headless), Pillow, Ultralytics 8.4.161, albumentations 2.0.8, React 18 + TypeScript + Tailwind + vitest, uv.

**Spec:** `docs/superpowers/specs/2026-09-24-augmentation-and-prediction-strip-design.md`

## Global Constraints

- uv only (`uv sync`, `uv run …`); never pip/conda. `albumentations>=2.0` is already added to `pyproject.toml`/`uv.lock` in the working tree (Task 3 commits it).
- Python lines ≤ 110 chars (check with `awk 'length($0) > 110'`; ruff here doesn't enforce E501); `uv run ruff check server` clean.
- Node commands need `source ~/.nvm/nvm.sh && ` first. Web output pristine (`npm test`, `npm run typecheck`, `npm run build`).
- GUI: every control lives in the right-hand sidebar (`TrainSidebar`), at the existing scale (`.btn`, `.field`, `.heading`, `.note`, `.alert`); charts/overlays use `SERIES_COLORS`, `TEXT_INK`, `TEXT_DIM`, `WELL`, `GRID` from `web/src/lib/chart.ts`; text never wears a series colour.
- Filter names: `all | missed | worst`. Preset names: `default | strong | custom`. Pool name: `dtd`. Derived export id: `<source_id>-aug<per_image>x-<4 hex>`; composite stems `syn_<seed>_<n:06d>`.
- Numbers from the spec, verbatim: composite scale 15–70 % of the canvas short side, rotation U(−180, 180), hflip 0.5, ≥ 80 % inside, overlap ≤ 10 % of the smaller object, 20 placement tries, colour pull U(0, 0.5), contrast U(0.8, 1.2), alpha blur σ 1.5, whole-image blur p 0.3 σ U(0.5, 1.5), noise p 0.3 σ U(2, 8), JPEG quality 90, background crop 50–100 % of the short side, cutout margin 4 px. Augment request bounds: `per_image` 1–10 (3), `max_objects` 1–3 (2), `seed` ≥ 0 (0).
- Never touch `data/` in tests (tests use `tmp_path`). Never run GPU tests during implementation tasks.
- Commit messages end with (after a blank line):
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6`

## File map

| File | Responsibility |
|---|---|
| `server/vsp/train/results.py` | + `mask_iou` |
| `server/vsp/train/gallery.py` (new) | load predictions + GT + cached IoUs; `filter_items` |
| `server/vsp/api/train.py` | predictions route: `filter`, `counts`, `index`, `iou`; exports: `formats`, `derived_from`; image route for train|val |
| `server/vsp/train/models.py` | `AugParams`, `STRONG_AUG`, `TrainParams.augment/aug`, `train_overrides`, `ExportInfo.formats/derived_from` |
| `server/vsp/train/worker.py` | pass overrides + `photometric_transforms()` |
| `server/vsp/cli.py` | `vsp train --augment`, `vsp backgrounds fetch` |
| `server/vsp/augment/{__init__,backgrounds,composite,derive}.py` (new) | pools, composite recipe, derived export |
| `server/vsp/export/runner.py` | extract `write_dataset` + `manifest_for` (shared with derive) |
| `server/vsp/api/augment.py` (new), `server/vsp/api/app.py` | background + augment + samples routes |
| `web/src/api/{types,rest}.ts` | new types and calls |
| `web/src/components/PredictionOverlay.tsx`, `GalleryCard.tsx`, `PredictionGallery.tsx` (new) | strip + enlarge; replaces `PredictionPreview.tsx` |
| `web/src/components/TrainSidebar.tsx`, `RunsTable.tsx` | presets, Aug column, export selection lifted, AugmentPanel slot |
| `web/src/components/AugmentPanel.tsx`, `CompositeSamples.tsx` (new) | augmented copy + DTD download; composite samples |
| `web/src/pages/Train.tsx` | wiring |

---

### Task 1: IoU, filters and the predictions route

**Files:**
- Modify: `server/vsp/train/results.py`, `server/vsp/api/train.py`
- Create: `server/vsp/train/gallery.py`
- Test: `server/tests/test_train_results.py` (append), `server/tests/test_train_gallery.py` (new), `server/tests/test_api_train.py` (append)

**Interfaces:**
- Produces: `results.mask_iou(predictions: list[dict], ground_truth: list[dict], width: int, height: int) -> float | None`; `gallery.FILTERS = ("all", "missed", "worst")`; `gallery.load_gallery(run_dir: Path, labels_dir: Path) -> dict` (`{"conf", "items": [{index, name, width, height, predictions, ground_truth, iou}]}`, raises `NotFound` without predictions.json); `gallery.filter_items(items, name) -> list[dict]`.
- Route: `GET /api/train/runs/{id}/predictions?offset&limit&filter` → `{total, conf, counts: {all, missed, worst}, items}`; `total` is the size of the filtered list.

- [ ] **Step 1: Failing tests**

Append to `server/tests/test_train_results.py`:

```python
from vsp.train.results import mask_iou

SQ = {"class_id": 0, "polygons": [[0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0]]}      # left half
FULL = {"class_id": 0, "polygons": [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]}    # whole image


def test_mask_iou_cases():
    assert mask_iou([], [], 10, 10) is None
    assert mask_iou([SQ], [], 10, 10) == 0.0
    assert mask_iou([], [SQ], 10, 10) == 0.0
    assert mask_iou([SQ], [SQ], 10, 10) == 1.0
    assert mask_iou([SQ], [FULL], 10, 10) == 0.5
```

`server/tests/test_train_gallery.py`:

```python
from __future__ import annotations

import json
import os

import pytest
from vsp.store import NotFound
from vsp.train.gallery import filter_items, load_gallery

FULL = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
LEFT = [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0]


def _setup(tmp_path):
    run_dir, labels = tmp_path / "run", tmp_path / "labels"
    run_dir.mkdir()
    labels.mkdir()
    inst = lambda poly: {"class_id": 0, "score": 0.9, "polygons": [poly]}  # noqa: E731
    images = [  # deliberately not in name order
        {"name": "c.jpg", "width": 10, "height": 10, "instances": [inst(LEFT)]},   # iou 0.5
        {"name": "a.jpg", "width": 10, "height": 10, "instances": [inst(FULL)]},   # iou 1.0
        {"name": "b.jpg", "width": 10, "height": 10, "instances": []},             # missed
        {"name": "d.jpg", "width": 10, "height": 10, "instances": []},             # no GT, no pred
    ]
    (run_dir / "predictions.json").write_text(json.dumps({"conf": 0.25, "images": images}))
    for stem in ("a", "b", "c"):
        (labels / f"{stem}.txt").write_text("0 " + " ".join(str(v) for v in FULL) + "\n")
    return run_dir, labels


def test_items_in_name_order_with_iou(tmp_path):
    run_dir, labels = _setup(tmp_path)
    g = load_gallery(run_dir, labels)
    assert g["conf"] == 0.25
    assert [(i["index"], i["name"], i["iou"]) for i in g["items"]] == [
        (0, "a.jpg", 1.0), (1, "b.jpg", 0.0), (2, "c.jpg", 0.5), (3, "d.jpg", None)]


def test_filters(tmp_path):
    items = load_gallery(*_setup(tmp_path))["items"]
    assert [i["name"] for i in filter_items(items, "all")] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert [i["name"] for i in filter_items(items, "missed")] == ["b.jpg"]
    assert [i["name"] for i in filter_items(items, "worst")] == ["b.jpg", "c.jpg", "a.jpg"]


def test_cache_written_and_rebuilt_when_predictions_change(tmp_path):
    run_dir, labels = _setup(tmp_path)
    load_gallery(run_dir, labels)
    cache = run_dir / "predictions_iou.json"
    assert json.loads(cache.read_text())["c.jpg"] == 0.5
    preds = json.loads((run_dir / "predictions.json").read_text())
    preds["images"][0]["instances"] = []          # c.jpg now missed
    (run_dir / "predictions.json").write_text(json.dumps(preds))
    later = cache.stat().st_mtime + 5
    os.utime(run_dir / "predictions.json", (later, later))
    assert {i["name"]: i["iou"] for i in load_gallery(run_dir, labels)["items"]}["c.jpg"] == 0.0


def test_missing_predictions(tmp_path):
    with pytest.raises(NotFound):
        load_gallery(tmp_path, tmp_path)
```

Append to `server/tests/test_api_train.py` (uses the file's `client`, `start`, `until` helpers; the fake worker predicts one polygon on the first val image, which also has a square label):

```python
def test_predictions_filters_and_counts(client):
    rid = start(client)
    until(client, rid, "done")
    page = client.get(f"/api/train/runs/{rid}/predictions?filter=all").json()
    assert page["counts"]["all"] == 1 and page["counts"]["missed"] == 0 and page["counts"]["worst"] == 1
    item = page["items"][0]
    assert item["index"] == 0 and 0.0 < item["iou"] <= 1.0
    assert client.get(f"/api/train/runs/{rid}/predictions?filter=missed").json()["total"] == 0
    assert client.get(f"/api/train/runs/{rid}/predictions?filter=nope").status_code == 422
```

- [ ] **Step 2: See them fail** — `uv run pytest server/tests/test_train_results.py server/tests/test_train_gallery.py server/tests/test_api_train.py -q -m "not gpu"` → ImportError / KeyError.

- [ ] **Step 3: Implement**

In `results.py` (add `import numpy as np` and `from ..export.polygons import polygons_to_mask`):

```python
def _union_mask(instances: list[dict], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), bool)
    for inst in instances:
        for flat in inst["polygons"]:
            points = [(flat[i] * width, flat[i + 1] * height) for i in range(0, len(flat) - 1, 2)]
            if len(points) >= 3:
                canvas |= polygons_to_mask([points], width, height)
    return canvas


def mask_iou(predictions: list[dict], ground_truth: list[dict], width: int, height: int) -> float | None:
    """IoU of the union of predicted polygons against the union of labelled ones; None when both are empty."""
    pred = _union_mask(predictions, width, height)
    gt = _union_mask(ground_truth, width, height)
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return None
    return round(int(np.logical_and(pred, gt).sum()) / union, 4)
```

`server/vsp/train/gallery.py`:

```python
"""Validation predictions as a gallery: frame order, per-frame mask IoU (cached), and the Missed / Worst filters."""

from __future__ import annotations

from pathlib import Path

from ..store import NotFound
from .results import mask_iou, read_ground_truth
from .runs import atomic_write_json, read_json

FILTERS = ("all", "missed", "worst")


def load_gallery(run_dir: Path, labels_dir: Path) -> dict:
    preds_path = run_dir / "predictions.json"
    preds = read_json(preds_path)
    if preds is None:
        raise NotFound("this run has no predictions yet")
    cache_path = run_dir / "predictions_iou.json"
    cache = None
    if cache_path.is_file() and cache_path.stat().st_mtime >= preds_path.stat().st_mtime:
        cache = read_json(cache_path)
    fresh: dict[str, float | None] = {}
    items = []
    for index, im in enumerate(sorted(preds["images"], key=lambda im: im["name"])):
        gt = read_ground_truth(labels_dir / f"{Path(im['name']).stem}.txt")
        if cache is not None and im["name"] in cache:
            iou = cache[im["name"]]
        else:
            iou = mask_iou(im["instances"], gt, im["width"], im["height"])
        fresh[im["name"]] = iou
        items.append({"index": index, "name": im["name"], "width": im["width"], "height": im["height"],
                      "predictions": im["instances"], "ground_truth": gt, "iou": iou})
    if cache != fresh:
        atomic_write_json(cache_path, fresh)
    return {"conf": preds.get("conf"), "items": items}


def filter_items(items: list[dict], name: str) -> list[dict]:
    if name == "missed":
        return [it for it in items if it["ground_truth"] and not it["predictions"]]
    if name == "worst":
        scored = [it for it in items if it["ground_truth"] and it["iou"] is not None]
        return sorted(scored, key=lambda it: (it["iou"], it["index"]))
    return list(items)
```

Replace the predictions route in `server/vsp/api/train.py` (import `Literal` from typing and `FILTERS, filter_items, load_gallery` from `..train.gallery`):

```python
@router.get("/api/train/runs/{run_id}/predictions")
def predictions(
    run_id: str, state: State, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50),
    filter: Literal["all", "missed", "worst"] = Query("all"),  # noqa: A002 - the public query name
) -> dict:
    run = state.train.runs.read(run_id)
    labels = Path(run["export_dir"]) / "yolo_seg" / "labels" / "val"
    gallery = load_gallery(state.train.runs.dir(run_id), labels)
    lists = {name: filter_items(gallery["items"], name) for name in FILTERS}
    chosen = lists[filter]
    return {"total": len(chosen), "conf": gallery["conf"], "counts": {k: len(v) for k, v in lists.items()},
            "items": chosen[offset:offset + limit]}
```

(If ruff reports the `noqa` as unused, drop it.)

- [ ] **Step 4: Pass** — same command; then `uv run pytest server/tests -q -m "not gpu"` and ruff.
- [ ] **Step 5: Commit** — `git commit -m "Predictions: per-frame IoU (cached) and Missed / Worst filters"`

---

### Task 2: Prediction gallery (strip + enlarge)

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/api/rest.ts`, `web/src/pages/Train.tsx`
- Create: `web/src/components/PredictionOverlay.tsx`, `web/src/components/GalleryCard.tsx`, `web/src/components/PredictionGallery.tsx`
- Delete: `web/src/components/PredictionPreview.tsx`
- Test: `web/src/components/PredictionGallery.test.tsx`

**Interfaces:**
- Types: `PredictionFilter = "all" | "missed" | "worst"`; `PredictionItem` gains `index: number; iou: number | null`; `PredictionPage` gains `counts: { all: number; missed: number; worst: number }`.
- `api.predictions(id, offset, limit, filter: PredictionFilter = "all")` appends `&filter=${filter}`.
- `PredictionOverlay({ width, height, groundTruth, predictions, showGt, showPred, scores })` — an absolutely positioned SVG filling its parent (`absolute inset-0 h-full w-full`), `viewBox="0 0 width height"`.
- `GalleryCard({ src, width, height, groundTruth, predictions, showGt, showPred, caption: ReactNode, onOpen?, active? })` — a `<button>` (or `<div>` without `onOpen`) with the image at its natural aspect ratio (`style={{ aspectRatio: \`${width} / ${height}\` }}`, `object-contain bg-well`), the overlay without scores, and the caption below in `font-cond text-xs`.
- `PredictionGallery({ run: Run })`.

- [ ] **Step 1: Types and client** — apply the type changes above; `predictions` gets the `filter` parameter.

- [ ] **Step 2: Failing test** — `web/src/components/PredictionGallery.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { PredictionItem, PredictionPage, Run } from "../api/types";
import { PredictionGallery } from "./PredictionGallery";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return { ...real, api: { predictions: vi.fn() } };
});
const m = vi.mocked(api);

const item = (i: number, over: Partial<PredictionItem> = {}): PredictionItem => ({
  index: i, name: `f_${String(i).padStart(3, "0")}.jpg`, width: 96, height: 64, iou: 0.9,
  predictions: [{ class_id: 0, score: 0.9, polygons: [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9]] }],
  ground_truth: [{ class_id: 0, polygons: [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9]] }], ...over,
});
const page = (offset: number, limit: number, total: number, filter = "all"): PredictionPage => ({
  total, conf: 0.25, counts: { all: 30, missed: 3, worst: 29 },
  items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, k) =>
    filter === "missed" ? item(offset + k, { predictions: [], iou: 0 }) : item(offset + k)),
});
const run = { id: "r1", export_id: "e1", status: "done" } as Run;

beforeEach(() => {
  m.predictions.mockReset();
  m.predictions.mockImplementation(async (_id: string, offset: number, limit: number, filter = "all") =>
    page(offset, limit, filter === "missed" ? 3 : 30, filter));
});
afterEach(cleanup);

describe("PredictionGallery", () => {
  it("pages by four, loads more near the end, and filters", async () => {
    render(<PredictionGallery run={run} />);
    expect(await screen.findByText("1–4 of 30")).toBeTruthy();
    expect(m.predictions).toHaveBeenLastCalledWith("r1", 0, 20, "all");
    const next = screen.getByRole("button", { name: "Next four" });
    fireEvent.click(next);
    expect(screen.getByText("5–8 of 30")).toBeTruthy();
    fireEvent.click(next);
    fireEvent.click(next);
    await waitFor(() => expect(m.predictions).toHaveBeenLastCalledWith("r1", 20, 20, "all"));
    fireEvent.click(screen.getByRole("button", { name: /Missed 3/ }));
    await waitFor(() => expect(m.predictions).toHaveBeenLastCalledWith("r1", 0, 20, "missed"));
    expect(await screen.findByText("1–3 of 3")).toBeTruthy();
    expect(screen.getAllByText(/missed/).length).toBeGreaterThan(0);
  });

  it("opens a card large, steps, and returns with Escape", async () => {
    render(<PredictionGallery run={run} />);
    fireEvent.click(await screen.findByRole("button", { name: "Open f_002.jpg" }));
    expect(screen.getByText(/3 \/ 30 · f_002.jpg/)).toBeTruthy();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByText(/4 \/ 30 · f_003.jpg/)).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByText("1–4 of 30")).toBeTruthy();
  });
});
```

- [ ] **Step 3: See it fail** — `source ~/.nvm/nvm.sh && cd web && npx vitest run src/components/PredictionGallery.test.tsx`.

- [ ] **Step 4: Implement**

`PredictionOverlay.tsx` — move the three `<polygon>`/`<text>` blocks out of `PredictionPreview.tsx` unchanged (GT dashed ink outline `strokeDasharray="6 4"`, predictions `SERIES_COLORS[0]` fill 0.35, `vectorEffect="non-scaling-stroke"`); score `<text>` only when `scores` is true.

`GalleryCard.tsx`:

```tsx
import type { ReactNode } from "react";
import type { Instance } from "../api/types";
import { PredictionOverlay } from "./PredictionOverlay";

type Props = {
  src: string; name: string; width: number; height: number; groundTruth: Instance[]; predictions: Instance[];
  showGt: boolean; showPred: boolean; caption: ReactNode; onOpen?: () => void;
};

export function GalleryCard({ src, name, width, height, groundTruth, predictions, showGt, showPred, caption, onOpen }: Props) {
  const body = (
    <>
      <div className="relative w-full overflow-hidden rounded bg-well" style={{ aspectRatio: `${width} / ${height}` }}>
        <img src={src} alt={name} loading="lazy" className="absolute inset-0 h-full w-full object-contain" />
        <PredictionOverlay width={width} height={height} groundTruth={groundTruth} predictions={predictions}
          showGt={showGt} showPred={showPred} scores={false} />
      </div>
      <div className="mt-1 flex items-center gap-1.5 font-cond text-xs text-dim">{caption}</div>
    </>
  );
  return onOpen ? (
    <button type="button" aria-label={`Open ${name}`} onClick={onOpen}
      className="snap-start rounded border border-line bg-panel p-1.5 text-left hover:border-dim">
      {body}
    </button>
  ) : (
    <div className="snap-start rounded border border-line bg-panel p-1.5">{body}</div>
  );
}
```

(The overlay SVG must have `preserveAspectRatio="xMidYMid meet"` so it tracks `object-contain`.)

`PredictionGallery.tsx`:

```tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { api, valImageUrl } from "../api/rest";
import type { PredictionFilter, PredictionItem, PredictionPage, Run } from "../api/types";
import { SERIES_COLORS, TEXT_INK } from "../lib/chart";
import { GalleryCard } from "./GalleryCard";
import { PredictionOverlay } from "./PredictionOverlay";

const PAGE = 20;
const VISIBLE = 4;
const FILTERS: { id: PredictionFilter; label: string }[] = [
  { id: "all", label: "All" }, { id: "missed", label: "Missed" }, { id: "worst", label: "Worst IoU" },
];
const inField = (t: EventTarget | null) =>
  t instanceof HTMLElement && ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName);

function caption(it: PredictionItem) {
  if (it.ground_truth.length > 0 && it.predictions.length === 0) {
    return (<><span aria-hidden className="text-danger">✕</span><span className="text-ink">#{it.index + 1} · missed</span></>);
  }
  return <span className="text-ink">#{it.index + 1}{it.iou === null ? "" : ` · IoU ${it.iou.toFixed(2)}`}</span>;
}

export function PredictionGallery({ run }: { run: Run }) {
  const [filter, setFilter] = useState<PredictionFilter>("all");
  const [items, setItems] = useState<PredictionItem[]>([]);
  const [meta, setMeta] = useState<Pick<PredictionPage, "total" | "counts" | "conf"> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [first, setFirst] = useState(0);
  const [open, setOpen] = useState<number | null>(null);
  const [showGt, setShowGt] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const token = useRef(0);
  const loading = useRef(false);
  const strip = useRef<HTMLDivElement>(null);

  const load = useCallback(async (offset: number, reset: boolean) => {
    if (loading.current && !reset) return;
    const my = ++token.current;
    loading.current = true;
    try {
      const p = await api.predictions(run.id, offset, PAGE, filter);
      if (my !== token.current) return;
      setMeta({ total: p.total, counts: p.counts, conf: p.conf });
      setItems((prev) => (reset ? p.items : [...prev, ...p.items]));
      setError(null);
    } catch (e) {
      if (my === token.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (my === token.current) loading.current = false;
    }
  }, [run.id, filter]);

  useEffect(() => {
    setItems([]);
    setFirst(0);
    setOpen(null);
    void load(0, true);
  }, [load]);

  const total = meta?.total ?? 0;
  // near the end of what is loaded: fetch the next page
  useEffect(() => {
    const at = open ?? first;
    if (items.length < total && at + VISIBLE * 2 >= items.length) void load(items.length, false);
  }, [first, open, items.length, total, load]);

  const scrollTo = (index: number) => {
    const card = strip.current?.children[index] as HTMLElement | undefined;
    card?.scrollIntoView?.({ inline: "start", block: "nearest", behavior: "smooth" });
  };
  const page = (d: number) => {
    const next = Math.max(0, Math.min(first + d * VISIBLE, Math.max(0, total - 1)));
    setFirst(next);
    scrollTo(next);
  };
  const step = (d: number) => setOpen((o) => (o === null ? o : Math.max(0, Math.min(o + d, items.length - 1))));
  const close = () => {
    if (open !== null) setFirst(Math.max(0, open - (open % VISIBLE)));
    setOpen(null);
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (inField(e.target)) return;
      if (e.key === "ArrowLeft") (open === null ? page(-1) : step(-1));
      else if (e.key === "ArrowRight") (open === null ? page(1) : step(1));
      else if (e.key === "Escape" && open !== null) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  // keep the counter honest when the user scrolls the strip by hand (browsers only; jsdom has no layout)
  const onScroll = () => {
    const el = strip.current;
    const card = el?.children[0] as HTMLElement | undefined;
    if (!el || !card || card.offsetWidth === 0) return;
    setFirst(Math.round(el.scrollLeft / (card.offsetWidth + 12)));
  };

  if (error !== null) return <p role="alert" className="alert">{error}</p>;
  if (meta === null) return <p className="note">Loading predictions…</p>;

  const toggles = (
    <>
      <label className="flex items-center gap-1.5 text-dim">
        <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
        <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm border-2" style={{ borderColor: TEXT_INK }} />
        Ground truth
      </label>
      <label className="flex items-center gap-1.5 text-dim">
        <input type="checkbox" checked={showPred} onChange={(e) => setShowPred(e.target.checked)} />
        <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: SERIES_COLORS[0] }} />
        Predictions (conf ≥ {meta.conf ?? 0.25})
      </label>
    </>
  );

  if (open !== null && items[open]) {
    const it = items[open];
    return (
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <button type="button" className="btn" onClick={close}>Back to strip</button>
          <button type="button" className="btn" aria-label="Previous image" onClick={() => step(-1)}>‹</button>
          <span className="font-cond text-dim">{open + 1} / {total} · {it.name}</span>
          <button type="button" className="btn" aria-label="Next image" onClick={() => step(1)}>›</button>
          {toggles}
        </div>
        <div className="relative w-full overflow-hidden rounded border border-line bg-well">
          <img src={valImageUrl(run.export_id, it.name)} alt={it.name} className="block w-full" />
          <PredictionOverlay width={it.width} height={it.height} groundTruth={it.ground_truth}
            predictions={it.predictions} showGt={showGt} showPred={showPred} scores />
        </div>
      </div>
    );
  }

  const last = Math.min(first + VISIBLE, total);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {FILTERS.map((f) => (
          <button key={f.id} type="button" className={`btn ${filter === f.id ? "btn-on" : ""}`}
            aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>
            {f.label} {f.id === "worst" ? "" : meta.counts[f.id]}
          </button>
        ))}
        <span className="flex-1" />
        {toggles}
      </div>
      {total === 0 ? (
        <p className="note">No frames match this filter.</p>
      ) : (
        <div className="flex items-center gap-2">
          <button type="button" className="btn self-center" aria-label="Previous four" disabled={first === 0} onClick={() => page(-1)}>‹</button>
          <div ref={strip} onScroll={onScroll}
            className="grid min-w-0 flex-1 snap-x snap-mandatory auto-cols-[calc((100%-2.25rem)/4)] grid-flow-col gap-3 overflow-x-auto pb-1">
            {items.map((it, i) => (
              <GalleryCard key={it.name} src={valImageUrl(run.export_id, it.name)} name={it.name} width={it.width}
                height={it.height} groundTruth={it.ground_truth} predictions={it.predictions} showGt={showGt}
                showPred={showPred} caption={caption(it)} onOpen={() => setOpen(i)} />
            ))}
          </div>
          <button type="button" className="btn self-center" aria-label="Next four" disabled={last >= total} onClick={() => page(1)}>›</button>
        </div>
      )}
      {total > 0 && <p className="note self-center">{first + 1}–{last} of {total}</p>}
    </div>
  );
}
```

In `Train.tsx` replace `PredictionPreview` with `PredictionGallery` (same condition, same heading). Delete `PredictionPreview.tsx`.

- [ ] **Step 5: Pass** — the new test, then `npm test && npm run typecheck && npm run build`. Fix any existing test that referenced the old preview.
- [ ] **Step 6: Commit** — `git commit -m "Validation predictions: a scrollable strip with Missed / Worst filters and an enlarge view"`

---

### Task 3: Augmentation presets (backend)

**Files:**
- Commit: `pyproject.toml`, `uv.lock` (albumentations, already in the working tree)
- Modify: `server/vsp/train/models.py`, `server/vsp/train/worker.py`, `server/vsp/api/train.py` (status gains `aug_presets`), `server/vsp/cli.py` (`--augment`)
- Test: `server/tests/test_train_models.py`, `server/tests/test_train_worker.py`, `server/tests/test_api_train.py`, `server/tests/test_train_cli.py` (append)

**Interfaces:**
- `AugParams` (pydantic, `extra="forbid"`, fields and bounds exactly as the spec table; defaults = the "default preset" column), `STRONG_AUG: AugParams`, `AUG_PRESETS = {"default": AugParams(), "strong": STRONG_AUG}`.
- `TrainParams.augment: Literal["default", "strong", "custom"] = "default"`, `TrainParams.aug: AugParams | None = None`; validator: `custom` requires `aug`.
- `train_overrides(params: dict) -> tuple[dict, bool]` — takes a run.json `params` dict (older runs lack the keys → default) and returns `(ultralytics kwargs, photometric)`; `{}` and `False` for default; otherwise every AugParams field except `photometric`, plus `copy_paste_mode="flip"`.
- `worker.photometric_transforms() -> list` (albumentations objects, exactly the spec list).
- `/api/train/status` gains `aug_presets: {"default": {...}, "strong": {...}}`.

- [ ] **Step 1: Failing tests**

`test_train_models.py`:

```python
from vsp.train.models import STRONG_AUG, AugParams, train_overrides


def test_aug_presets_and_overrides():
    assert train_overrides({"epochs": 5}) == ({}, False)                       # an old run.json
    assert train_overrides(TrainParams().model_dump()) == ({}, False)
    kw, photo = train_overrides(TrainParams(augment="strong").model_dump())
    assert photo is True and kw["degrees"] == 15 and kw["copy_paste"] == 0.3 and kw["copy_paste_mode"] == "flip"
    assert "photometric" not in kw and kw == {**STRONG_AUG.model_dump(exclude={"photometric"}), "copy_paste_mode": "flip"}
    custom = TrainParams(augment="custom", aug=AugParams(degrees=5, photometric=False)).model_dump()
    kw, photo = train_overrides(custom)
    assert kw["degrees"] == 5 and kw["fliplr"] == 0.5 and photo is False


def test_custom_needs_values_and_bounds():
    with pytest.raises(ValidationError):
        TrainParams(augment="custom")
    for bad in ({"degrees": 181}, {"perspective": 0.01}, {"mixup": 1.5}, {"nope": 1}):
        with pytest.raises(ValidationError):
            AugParams(**bad)
```

`test_train_worker.py`:

```python
def test_photometric_transforms():
    import albumentations as A
    from vsp.train.worker import photometric_transforms

    ts = photometric_transforms()
    assert [type(t) for t in ts] == [A.RandomBrightnessContrast, A.MotionBlur, A.GaussianBlur, A.GaussNoise,
                                     A.ImageCompression, A.CLAHE]
    assert [t.p for t in ts] == [0.5, 0.2, 0.1, 0.2, 0.3, 0.1]
```

`test_api_train.py`: `test_status_and_exports` additionally asserts `st["aug_presets"]["strong"]["degrees"] == 15` and `st["defaults"]["augment"] == "default"`; a new test posts `params: {"epochs": 2, "imgsz": 640, "augment": "strong"}` and asserts the run's `params["augment"] == "strong"`, and that `{"augment": "custom"}` without `aug` → 422.

`test_train_cli.py`: `_args()` gains `augment="default"`; a new test runs with `augment="strong"` and asserts the created run.json has `params["augment"] == "strong"`.

- [ ] **Step 2: See them fail.**

- [ ] **Step 3: Implement**

In `models.py` (add `model_validator` and `Literal` imports):

```python
class AugParams(BaseModel):
    """Ultralytics augmentation hyper-parameters we expose; defaults are Ultralytics' own."""

    model_config = ConfigDict(extra="forbid")

    degrees: float = Field(default=0.0, ge=0, le=180)
    shear: float = Field(default=0.0, ge=0, le=45)
    perspective: float = Field(default=0.0, ge=0, le=0.001)
    flipud: float = Field(default=0.0, ge=0, le=1)
    fliplr: float = Field(default=0.5, ge=0, le=1)
    scale: float = Field(default=0.5, ge=0, le=0.9)
    translate: float = Field(default=0.1, ge=0, le=0.9)
    hsv_h: float = Field(default=0.015, ge=0, le=0.1)
    hsv_s: float = Field(default=0.7, ge=0, le=1)
    hsv_v: float = Field(default=0.4, ge=0, le=1)
    mosaic: float = Field(default=1.0, ge=0, le=1)
    mixup: float = Field(default=0.0, ge=0, le=1)
    copy_paste: float = Field(default=0.0, ge=0, le=1)
    photometric: bool = False


STRONG_AUG = AugParams(degrees=15, shear=5, perspective=0.0005, flipud=0.5, scale=0.6, translate=0.2, hsv_h=0.02,
                       hsv_v=0.5, mixup=0.1, copy_paste=0.3, photometric=True)
AUG_PRESETS = {"default": AugParams(), "strong": STRONG_AUG}
```

`TrainParams` gains:

```python
    augment: Literal["default", "strong", "custom"] = "default"
    aug: AugParams | None = None

    @model_validator(mode="after")
    def _custom_needs_values(self) -> TrainParams:
        if self.augment == "custom" and self.aug is None:
            raise ValueError("augment 'custom' needs aug values")
        return self
```

and:

```python
def train_overrides(params: dict) -> tuple[dict, bool]:
    """Ultralytics keyword arguments for a run's params (a run.json dict; older runs have no augment keys)."""
    mode = params.get("augment", "default")
    if mode == "strong":
        aug = STRONG_AUG
    elif mode == "custom":
        aug = AugParams(**params["aug"])
    else:
        return {}, False
    values = aug.model_dump()
    photometric = values.pop("photometric")
    return {**values, "copy_paste_mode": "flip"}, photometric
```

In `worker.py`:

```python
def photometric_transforms() -> list:
    import albumentations as A

    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.MotionBlur(blur_limit=(3, 9), p=0.2),
        A.GaussianBlur(sigma_limit=(0.3, 1.5), p=0.1),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.CLAHE(p=0.1),
    ]
```

and in `train()`'s fresh-start branch: `overrides, photometric = train_overrides(p)`; `if photometric: overrides["augmentations"] = photometric_transforms()`; pass `**overrides` to `model.train(...)`.

`/api/train/status`: add `"aug_presets": {k: v.model_dump() for k, v in AUG_PRESETS.items()}`.

`cli.py`: `p_train.add_argument("--augment", choices=["default", "strong"], default="default")` and pass `augment=args.augment` into `TrainParams`.

- [ ] **Step 4: Pass** — focused tests, full non-GPU suite, ruff.
- [ ] **Step 5: Commit** — `git add pyproject.toml uv.lock …; git commit -m "Training: augmentation presets (default / strong / custom) with photometric Albumentations"`

---

### Task 4: Augmentation presets (web)

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/components/TrainSidebar.tsx`, `web/src/components/RunsTable.tsx`
- Test: `web/src/pages/Train.test.tsx`, `web/src/components/RunsTable.test.tsx` (append)

**Interfaces:**
- `AugParams` type mirroring the backend (13 numbers + `photometric: boolean`); `AugmentMode = "default" | "strong" | "custom"`; `TrainParams` gains `augment?: AugmentMode; aug?: AugParams | null`; `TrainStatus` gains `aug_presets: { default: AugParams; strong: AugParams }`.
- Sidebar: an **Augmentation** `<select aria-label="Augmentation">` (Default / Strong / Custom) under the numeric params. Choosing Custom shows a fieldset of number inputs (label text = field name, e.g. `degrees`, `mixup`), prefilled from `aug_presets.strong`, plus a `photometric` checkbox (label "Blur, noise, JPEG, contrast"). The start body sends `params: { …numbers, augment, aug: augment === "custom" ? values : null }`.
- Runs table: an **Aug** column after GPU showing `r.params.augment ?? "default"`.

- [ ] **Step 1: Failing tests** — in `Train.test.tsx` add `aug_presets` to the mocked status (`default`: Ultralytics defaults, `strong`: degrees 15 etc.), then: (a) select Strong, start → `body.params.augment === "strong"` and `body.params.aug === null`; (b) select Custom, change `degrees` to 7, start → `body.params.aug.degrees === 7` and `body.params.aug.mixup === 0.1` (prefilled from strong). In `RunsTable.test.tsx`: a run with `params.augment = "strong"` shows `strong` in its row; one without shows `default`.
- [ ] **Step 2: See them fail.**
- [ ] **Step 3: Implement** (keep the existing field styling: `label.flex.items-center.justify-between` + `.field w-24`; the custom fieldset lives in the sidebar under the select, in a two-column grid so it stays compact).
- [ ] **Step 4: Pass** — `npm test && npm run typecheck && npm run build`.
- [ ] **Step 5: Commit** — `git commit -m "Train page: augmentation preset select, custom values, Aug column"`

---

### Task 5: Background pools (DTD)

**Files:**
- Create: `server/vsp/augment/__init__.py`, `server/vsp/augment/backgrounds.py`, `server/vsp/api/augment.py`
- Modify: `server/vsp/api/app.py` (include router), `server/vsp/cli.py` (`vsp backgrounds fetch`)
- Test: `server/tests/test_augment_backgrounds.py`, `server/tests/test_api_augment.py`

**Interfaces:**
- `POOLS: dict[str, dict]` = `{"dtd": {"label": "DTD textures", "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz", "prefix": "dtd/images/", "size": 625_239_812}}`.
- `IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}`; `class FetchCancelled(Exception)`.
- `pool_dir(data_root, name) -> Path` (raises `NotFound` for unknown pools); `list_images(data_root, name) -> list[Path]` (sorted, recursive, `[]` when absent); `safe_member_name(member: tarfile.TarInfo, prefix: str) -> str | None`.
- `fetch_pool(data_root, name, *, url=None, force=False, progress=None, cancel=None) -> dict` → `{"pool", "images", "skipped"}`; `progress(phase: str, done: int, total: int)` with phases `download` then `extract`.
- Routes: `GET /api/backgrounds` → `[{name, label, images, size_bytes, fetching}]`; `POST /api/backgrounds/{name}/fetch` body `{force: bool = false}` → `202 {job_id}` (job kind `backgrounds`, `exclusive="backgrounds"`, progress events `{"type": "progress", "phase", "done", "total"}`, result `{pool, images, skipped}` or `{cancelled: true}`).
- CLI: `vsp backgrounds fetch dtd [--force]` prints `download 37%` / `extract 80%` lines (at most one per whole percent) and the final count.

- [ ] **Step 1: Failing tests** — `server/tests/test_augment_backgrounds.py`:

```python
from __future__ import annotations

import io
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from vsp.augment.backgrounds import FetchCancelled, fetch_pool, list_images, safe_member_name

JPEG = b"\xff\xd8\xff\xe0fakejpeg"


def _tarball(path):
    with tarfile.open(path, "w:gz") as tar:
        def add(name, data=JPEG, kind=tarfile.REGTYPE, link=""):
            info = tarfile.TarInfo(name)
            info.type, info.linkname, info.size = kind, link, (len(data) if kind == tarfile.REGTYPE else 0)
            tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
        add("dtd/images/banded/banded_0001.jpg")
        add("dtd/images/woven/woven_0002.png")
        add("dtd/labels/labels_joint_anno.txt", b"text")
        add("dtd/images/../../evil.jpg")
        add("/abs/dtd/images/x.jpg")
        add("dtd/images/link.jpg", kind=tarfile.SYMTYPE, link="/etc/passwd")


@pytest.fixture
def server(tmp_path, monkeypatch):
    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    root = tmp_path / "www"
    root.mkdir()
    _tarball(root / "dtd.tar.gz")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/dtd.tar.gz"
    httpd.shutdown()


def test_only_safe_images_are_extracted(tmp_path, server):
    data_root = tmp_path / "data"
    phases = []
    result = fetch_pool(data_root, "dtd", url=server, progress=lambda ph, d, t: phases.append(ph))
    assert result == {"pool": "dtd", "images": 2, "skipped": False}
    names = sorted(p.relative_to(data_root / "backgrounds" / "dtd").as_posix() for p in list_images(data_root, "dtd"))
    assert names == ["banded/banded_0001.jpg", "woven/woven_0002.png"]
    assert not (tmp_path / "evil.jpg").exists() and "download" in phases and "extract" in phases
    assert not list((data_root / "backgrounds").glob(".*"))  # no partial archive or temp dir left


def test_existing_pool_is_skipped_unless_forced(tmp_path, server):
    data_root = tmp_path / "data"
    fetch_pool(data_root, "dtd", url=server)
    assert fetch_pool(data_root, "dtd", url=server)["skipped"] is True
    assert fetch_pool(data_root, "dtd", url=server, force=True)["images"] == 2


def test_cancel_leaves_nothing(tmp_path, server):
    data_root = tmp_path / "data"
    with pytest.raises(FetchCancelled):
        fetch_pool(data_root, "dtd", url=server, cancel=lambda: True)
    assert list_images(data_root, "dtd") == []
    assert not list((data_root / "backgrounds").glob(".*"))


def test_safe_member_name():
    info = tarfile.TarInfo("dtd/images/a/b.jpg")
    assert safe_member_name(info, "dtd/images/") == "a/b.jpg"
    for bad in ("dtd/images/../x.jpg", "/dtd/images/x.jpg", "dtd/images/x.txt", "other/x.jpg"):
        assert safe_member_name(tarfile.TarInfo(bad), "dtd/images/") is None
```

`server/tests/test_api_augment.py` (build the app like `test_api_train.py`'s fixture; reuse the `server` fixture by importing it from `test_augment_backgrounds` or by moving it to `train_fixtures.py` — moving it is preferred): monkeypatch `vsp.augment.backgrounds.POOLS["dtd"]["url"]` to the local URL (use `monkeypatch.setitem` on a copied dict), `GET /api/backgrounds` → dtd with `images == 0`; `POST /api/backgrounds/dtd/fetch` → 202; wait for the job (`/api/jobs/{id}` done) → `images == 2`; unknown pool → 404.

- [ ] **Step 2: See them fail.**

- [ ] **Step 3: Implement `backgrounds.py`**

```python
"""Background image pools for composites (spec B1). One pool: DTD textures, downloaded on request."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from ..store import NotFound

POOLS: dict[str, dict] = {
    "dtd": {
        "label": "DTD textures",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        "prefix": "dtd/images/",
        "size": 625_239_812,
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_CHUNK = 1 << 20

Progress = Callable[[str, int, int], None]


class FetchCancelled(Exception):
    pass


def backgrounds_root(data_root: Path) -> Path:
    return Path(data_root) / "backgrounds"


def pool_dir(data_root: Path, name: str) -> Path:
    if name not in POOLS:
        raise NotFound(f"background pool {name} not found")
    return backgrounds_root(data_root) / name


def list_images(data_root: Path, name: str) -> list[Path]:
    d = pool_dir(data_root, name)
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def safe_member_name(member: tarfile.TarInfo, prefix: str) -> str | None:
    """The path below `prefix` for a regular image file with a clean relative name; None for anything else."""
    if not member.isfile():
        return None
    name = member.name
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts or not name.startswith(prefix):
        return None
    if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    rel = name[len(prefix):]
    return rel or None


def fetch_pool(
    data_root: Path, name: str, *, url: str | None = None, force: bool = False,
    progress: Progress | None = None, cancel: Callable[[], bool] | None = None,
) -> dict:
    pool = POOLS.get(name)
    dest = pool_dir(data_root, name)
    existing = list_images(data_root, name)
    if existing and not force:
        return {"pool": name, "images": len(existing), "skipped": True}
    root = backgrounds_root(data_root)
    root.mkdir(parents=True, exist_ok=True)
    part = root / f".{name}.tar.gz.part"
    tmp = root / f".{name}.tmp"

    def check() -> None:
        if cancel is not None and cancel():
            raise FetchCancelled

    try:
        check()
        with urllib.request.urlopen(url or pool["url"], timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or pool["size"])
            done = 0
            while chunk := resp.read(_CHUNK):
                check()
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress("download", done, total)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        count = 0
        with tarfile.open(part, "r:gz") as tar:
            members = tar.getmembers()
            for i, member in enumerate(members):
                check()
                rel = safe_member_name(member, pool["prefix"])
                if rel is not None:
                    target = tmp / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tar.extractfile(member)
                    if src is not None:
                        with src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        count += 1
                if progress:
                    progress("extract", i + 1, len(members))
        if count == 0:
            raise ValueError(f"the archive holds no images under {pool['prefix']}")
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return {"pool": name, "images": count, "skipped": False}
```

`server/vsp/augment/__init__.py`: a one-line docstring ("Synthetic training data: background pools, composites, augmented exports.").

`server/vsp/api/augment.py`:

```python
"""Background pools and augmented exports (spec B1, B2)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..augment.backgrounds import POOLS, FetchCancelled, fetch_pool, list_images, pool_dir
from ..jobs import Job
from .deps import State

router = APIRouter(tags=["augment"])
BACKGROUNDS = "backgrounds"  # job kind and exclusive key: one download at a time


class FetchBody(BaseModel):
    force: bool = False


@router.get("/api/backgrounds")
def backgrounds(state: State) -> list[dict]:
    root = state.settings.project.data_root
    fetching = state.jobs.active(BACKGROUNDS)
    return [{"name": n, "label": p["label"], "images": len(list_images(root, n)), "size_bytes": p["size"],
             "fetching": fetching} for n, p in POOLS.items()]


@router.post("/api/backgrounds/{name}/fetch", status_code=202)
def fetch(name: str, state: State, body: FetchBody | None = None) -> dict:
    root = state.settings.project.data_root
    pool_dir(root, name)  # 404 for an unknown pool, before a job exists
    force = body.force if body else False

    def work(job: Job) -> dict:
        try:
            return fetch_pool(root, name, force=force, cancel=job.cancel_requested,
                              progress=lambda ph, d, t: job.emit({"type": "progress", "phase": ph, "done": d, "total": t}))
        except FetchCancelled:
            return {"cancelled": True}

    return {"job_id": state.jobs.start(BACKGROUNDS, work, exclusive=BACKGROUNDS).id}
```

Wire `app.include_router(augment_api.router)` in `app.py` (import `from . import augment as augment_api`).

`cli.py`: subcommand `backgrounds` with `fetch NAME [--force]`; prints progress (one line per whole percent per phase) and `OK: <n> images in <dir>` or `skipped: <n> images already in <dir> (use --force)`; exit 1 with the message on any exception.

- [ ] **Step 4: Pass** — focused tests, full non-GPU suite, ruff.
- [ ] **Step 5: Commit** — `git commit -m "Background pools: download DTD textures (safe extraction), API job and CLI"`

---

### Task 6: Composite recipe

**Files:**
- Create: `server/vsp/augment/composite.py`
- Test: `server/tests/test_augment_composite.py`

**Interfaces:**
- `Cutout(class_id: int, image: np.ndarray (h, w, 3) uint8 RGB, mask: np.ndarray (h, w) bool)`; `Composite(image: np.ndarray (H, W, 3) uint8, objects: list[tuple[int, np.ndarray]])`.
- `cutout_from(image, mask, class_id) -> Cutout | None` (crop to the mask box + 4 px margin, clamped; None for an empty mask).
- `crop_background(rng, bg, width, height) -> np.ndarray` (H, W, 3).
- `transform_cutout(rng, cut, canvas_short) -> tuple[np.ndarray, np.ndarray]` (image, bool mask), cropped to the transformed mask's box.
- `place(rng, mask, canvas_hw, placed: list[np.ndarray], check_overlap: bool) -> tuple[int, int] | None` — top-left `(y, x)`, may be negative.
- `make_composite(rng, width, height, background, cutouts, max_objects, min_area_px) -> Composite`.

- [ ] **Step 1: Failing tests**

```python
from __future__ import annotations

import numpy as np
from vsp.augment.composite import (
    Cutout, crop_background, cutout_from, make_composite, place, transform_cutout,
)


def _cut(class_id=0):
    img = np.zeros((40, 60, 3), np.uint8)
    img[:] = (200, 60, 30)
    mask = np.zeros((40, 60), bool)
    mask[5:35, 10:50] = True
    return cutout_from(img, mask, class_id)


def _bg():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)


def test_cutout_from_crops_with_margin():
    c = _cut()
    assert c.mask.shape == (38, 48) and c.image.shape == (38, 48, 3) and c.mask.sum() == 30 * 40
    assert cutout_from(np.zeros((5, 5, 3), np.uint8), np.zeros((5, 5), bool), 0) is None


def test_crop_background_shape():
    out = crop_background(np.random.default_rng(0), _bg(), 96, 160)
    assert out.shape == (160, 96, 3) and out.dtype == np.uint8


def test_transform_scales_to_range():
    for seed in range(20):
        img, m = transform_cutout(np.random.default_rng(seed), _cut(), 200)
        ys, xs = np.nonzero(m)
        longer = max(np.ptp(ys) + 1, np.ptp(xs) + 1)
        assert img.shape[:2] == m.shape and 0.15 * 200 * 0.9 <= longer <= 0.70 * 200 * 1.5  # rotation grows the box


def test_place_keeps_eighty_percent_inside_and_avoids_overlap():
    rng = np.random.default_rng(3)
    m = np.ones((50, 50), bool)
    placed = []
    for i in range(3):
        pos = place(rng, m, (200, 200), placed, check_overlap=i > 0)
        assert pos is not None
        y, x = pos
        full = np.zeros((200, 200), bool)
        ys, xs = slice(max(0, y), min(200, y + 50)), slice(max(0, x), min(200, x + 50))
        full[ys, xs] = True
        assert full.sum() >= 0.8 * m.sum()
        for other in placed:
            assert (full & other).sum() <= 0.1 * min(full.sum(), other.sum())
        placed.append(full)


def test_make_composite_is_deterministic_and_labels_match():
    args = (160, 96, _bg(), [_cut(0), _cut(1)], 2, 64)
    a = make_composite(np.random.default_rng([7, 0]), *args)
    b = make_composite(np.random.default_rng([7, 0]), *args)
    assert a.image.shape == (96, 160, 3) and np.array_equal(a.image, b.image)
    assert 1 <= len(a.objects) <= 2
    for (ca, ma), (cb, mb) in zip(a.objects, b.objects):
        assert ca == cb and np.array_equal(ma, mb) and ma.shape == (96, 160) and ma.sum() >= 64
    if len(a.objects) == 2:
        assert not (a.objects[0][1] & a.objects[1][1]).any()  # occlusion: each pixel belongs to one object


def test_single_object_when_max_is_one():
    for seed in range(10):
        c = make_composite(np.random.default_rng([seed, 0]), 160, 96, _bg(), [_cut()], 1, 64)
        assert len(c.objects) == 1
```

- [ ] **Step 2: See them fail.**

- [ ] **Step 3: Implement `composite.py`**

```python
"""One synthetic training image: labelled objects cut out by their masks and pasted onto a background (spec B2).
Pure and seeded: the same rng state gives the same composite."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..export.polygons import clean_mask, mask_bbox

MARGIN = 4
SCALE_RANGE = (0.15, 0.70)      # object's longer side as a fraction of the canvas' shorter side
MIN_INSIDE = 0.8                # at most 20 % of an object may fall outside the canvas
MAX_OVERLAP = 0.1               # of the smaller object
TRIES = 20
ALPHA_SIGMA = 1.5


@dataclass(frozen=True)
class Cutout:
    class_id: int
    image: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class Composite:
    image: np.ndarray
    objects: list[tuple[int, np.ndarray]]


def cutout_from(image: np.ndarray, mask: np.ndarray, class_id: int) -> Cutout | None:
    box = mask_bbox(mask)
    if box is None:
        return None
    x, y, w, h = box
    y0, x0 = max(0, y - MARGIN), max(0, x - MARGIN)
    y1, x1 = min(mask.shape[0], y + h + MARGIN), min(mask.shape[1], x + w + MARGIN)
    return Cutout(class_id, image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy())


def crop_background(rng: np.random.Generator, bg: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = bg.shape[:2]
    aspect = width / height
    side = min(h, w) * rng.uniform(0.5, 1.0)           # the crop's shorter side
    ch, cw = (side, side * aspect) if aspect >= 1 else (side / aspect, side)
    shrink = min(1.0, w / cw, h / ch)                  # keep the aspect, fit inside the image
    ch, cw = max(1, int(ch * shrink)), max(1, int(cw * shrink))
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    crop = bg[y0:y0 + ch, x0:x0 + cw]
    interp = cv2.INTER_AREA if cw > width else cv2.INTER_LINEAR
    return cv2.resize(crop, (width, height), interpolation=interp)


def transform_cutout(rng: np.random.Generator, cut: Cutout, canvas_short: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(cut.mask)
    longer = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    s = rng.uniform(*SCALE_RANGE) * canvas_short / longer
    angle = rng.uniform(-180.0, 180.0)
    img, m = cut.image, cut.mask.astype(np.uint8)
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    h, w = m.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    rot[0, 2] += nw / 2 - w / 2
    rot[1, 2] += nh / 2 - h / 2
    img2 = cv2.warpAffine(np.ascontiguousarray(img), rot, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    m2 = cv2.warpAffine(np.ascontiguousarray(m), rot, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    box = mask_bbox(m2)
    if box is None:  # scaled below a pixel: keep a one-pixel object rather than failing
        return img2[:1, :1], np.ones((1, 1), bool)
    x, y, bw, bh = box
    return img2[y:y + bh, x:x + bw], m2[y:y + bh, x:x + bw]


def _paste_mask(mask: np.ndarray, pos: tuple[int, int], canvas_hw: tuple[int, int]) -> np.ndarray:
    H, W = canvas_hw
    y, x = pos
    h, w = mask.shape
    full = np.zeros((H, W), bool)
    ty0, tx0, ty1, tx1 = max(0, y), max(0, x), min(H, y + h), min(W, x + w)
    if ty1 > ty0 and tx1 > tx0:
        full[ty0:ty1, tx0:tx1] = mask[ty0 - y:ty1 - y, tx0 - x:tx1 - x]
    return full


def place(rng: np.random.Generator, mask: np.ndarray, canvas_hw: tuple[int, int], placed: list[np.ndarray],
          check_overlap: bool) -> tuple[int, int] | None:
    H, W = canvas_hw
    h, w = mask.shape
    area = int(mask.sum())
    for _ in range(TRIES):
        y = int(rng.integers(-h // 2, max(-h // 2 + 1, H - h // 2)))
        x = int(rng.integers(-w // 2, max(-w // 2 + 1, W - w // 2)))
        full = _paste_mask(mask, (y, x), canvas_hw)
        visible = int(full.sum())
        if visible < MIN_INSIDE * area:
            continue
        if check_overlap and any(
            (full & other).sum() > MAX_OVERLAP * min(visible, int(other.sum())) for other in placed
        ):
            continue
        return y, x
    return None


def _colour_match(rng: np.random.Generator, obj: np.ndarray, mask: np.ndarray, under: np.ndarray) -> np.ndarray:
    out = obj.astype(np.float32)
    pull, contrast = rng.uniform(0.0, 0.5), rng.uniform(0.8, 1.2)
    for c in range(3):
        mo = out[..., c][mask].mean()
        mb = under[..., c][mask].mean()
        out[..., c] = (out[..., c] - mo) * contrast + mo + pull * (mb - mo)
    return np.clip(out, 0, 255)


def make_composite(rng: np.random.Generator, width: int, height: int, background: np.ndarray,
                   cutouts: list[Cutout], max_objects: int, min_area_px: int) -> Composite:
    canvas = crop_background(rng, background, width, height).astype(np.float32)
    k = int(rng.integers(1, max_objects + 1))
    picks = [cutouts[int(i)] for i in rng.integers(0, len(cutouts), k)]
    owner = np.full((height, width), -1, np.int16)
    placed: list[np.ndarray] = []
    classes: list[int] = []
    for j, cut in enumerate(picks):
        img, m = transform_cutout(rng, cut, min(width, height))
        pos = place(rng, m, (height, width), placed, check_overlap=j > 0)
        if pos is None:
            if j > 0:
                continue
            pos = ((height - m.shape[0]) // 2, (width - m.shape[1]) // 2)   # the first object is always placed
        full = _paste_mask(m, pos, (height, width))
        if not full.any():
            continue
        y, x = pos
        src = np.zeros((height, width, 3), np.float32)
        ty0, tx0 = max(0, y), max(0, x)
        ty1, tx1 = min(height, y + m.shape[0]), min(width, x + m.shape[1])
        src[ty0:ty1, tx0:tx1] = img[ty0 - y:ty1 - y, tx0 - x:tx1 - x]
        matched = _colour_match(rng, src, full, canvas)
        layer = np.where(full[..., None], matched, canvas)   # outside the hard mask the "object" is the background
        alpha = cv2.GaussianBlur(full.astype(np.float32), (0, 0), ALPHA_SIGMA)[..., None]
        alpha = np.where(full[..., None], np.maximum(alpha, 0.5), alpha)   # keep the object's own pixels solid
        canvas = canvas * (1 - alpha) + layer * alpha
        owner[full] = len(placed)
        placed.append(full)
        classes.append(cut.class_id)
    if rng.random() < 0.3:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        canvas = canvas + rng.normal(0.0, rng.uniform(2.0, 8.0), canvas.shape)
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    objects = []
    for idx, cls in enumerate(classes):
        m = clean_mask(owner == idx, min_area_px)
        if m.any():
            objects.append((cls, m))
    return Composite(image, objects)
```

The blend treats pixels outside the hard mask as background-coloured "object" pixels, so the soft alpha fringe
never darkens the background (the layer is zero there otherwise). Add to
`test_make_composite_is_deterministic_and_labels_match`: pixels farther than 4 px from every object mask differ from
the input only by the optional whole-image blur/noise — assert it for a seed where neither fired: loop seeds 0–20 and
require at least one seed for which the far pixels equal `crop_background(np.random.default_rng([seed, 0]), bg, 160, 96)`
exactly (the background crop is the rng's first draw).

- [ ] **Step 4: Pass** — focused tests (run 3×; they are seeded), ruff.
- [ ] **Step 5: Commit** — `git commit -m "Composite recipe: cut objects by mask and paste them onto backgrounds (seeded)"`

---

### Task 7: Derived augmented export

**Files:**
- Modify: `server/vsp/export/runner.py` (extract `write_dataset`, `manifest_for`), `server/vsp/train/models.py` (`ExportInfo.formats`, `derived_from`), `server/vsp/api/train.py` (image route for `train|val`), `server/vsp/api/augment.py` (augment + samples routes)
- Create: `server/vsp/augment/derive.py`
- Test: `server/tests/test_augment_derive.py`, `server/tests/test_api_augment.py` (append), existing export tests must stay green

**Interfaces:**
- `runner.write_dataset(plan, formats, out, tick, *, link_images=False) -> tuple[list[ExportSummary], dict, bool]` (summaries, counts, linked) — copies (or, with `link_images`, hard-links via `place_image`) each frame image to `out/images/<stem>.jpg`, then runs the exporters. `runner.manifest_for(plan, export_id, summaries, counts, linked) -> dict` — the manifest `run_export` writes today. `run_export` uses both; its behaviour and output are unchanged.
- `derive.AugmentOptions(per_image: int = 3, max_objects: int = 2, seed: int = 0, pool: str = "dtd")` (frozen dataclass); `derive.check_augment(data_root, export_id, opts) -> dict` (the source manifest; raises `NotFound` / `BadRequest`); `derive.run_augment(data_root, export_id, opts, progress=None, cancel=None) -> dict` → `{"export_id", "path", "counts", "synthetic"}`; raises `ExportCancelled` on cancel.
- `ExportInfo` gains `formats: tuple[str, ...]`, `derived_from: str | None`; `to_json` includes both.
- Routes: `POST /api/exports/{id}/augment` body `{per_image: 1–10 = 3, max_objects: 1–3 = 2, seed: ≥0 = 0, pool: str = "dtd"}` → 202 `{job_id}` (kind `augment`; progress events `{"type": "progress", "done", "total"}`; result as above or `{cancelled: true}`); `GET /api/exports/{id}/samples?limit=12&seed=0` (limit 1–48) → `[{name, width, height, ground_truth}]` from `yolo_seg/images/train` names starting `syn_` (all train images when there are none), chosen with `random.Random(seed).sample`; `GET /api/exports/{id}/images/{split}/{name}` with `split` in `train|val` (replaces the val-only route; same name validation).

**Algorithm of `run_augment`** (source dir `S = data/exports/<id>`, manifest `M`):

1. `check_augment`: `S/export.json` exists (else NotFound); `"masks"` in `M["formats"]` names (else BadRequest "export … has no masks format; re-export with Mask images ticked"); `list_images(data_root, opts.pool)` non-empty (else BadRequest "background pool … is empty; download it first"); bounds (BadRequest).
2. New id `f"{export_id}-aug{opts.per_image}x-{uuid4().hex[:4]}"`, `out.mkdir()` outside the try (as `run_export`); everything else in `try/except BaseException: rmtree(out); raise`. Work dir `out/.work` (deleted at the end).
3. Real frames: for each `split` in train, val and each `S/masks/masks/<split>/<stem>.png` (sorted): read the indexed mask; for each value `v > 0` present write `out/.work/masks/<stem>_<v-1>.png` (binary, 255 = object) and a `PlanObject(object_id=v-1, class_id=v-1, mask_path=…, min_area_px=M["options"]["min_area_px"])`; `PlanFrame(clip_id=stem.split("_")[0], frame_idx=<running index>, stem=stem, split=split, image_path=S/"images"/f"{stem}.jpg", width, height, objects)`.
4. Cutouts: for each real **train** frame with objects, load the image once (RGB) and `cutout_from(image, read_mask(obj), obj.class_id)` per object.
5. Composites: `total = opts.per_image * len(real train frames)`; for `n` in `range(total)`: `rng = np.random.default_rng([opts.seed, n])`; size from `train_frames[int(rng.integers(len(train_frames)))]`; background `list_images(...)[int(rng.integers(len(pool)))]` loaded RGB (a small `functools.lru_cache(maxsize=64)` loader); `make_composite(rng, w, h, bg, cutouts, opts.max_objects, min_area_px)`; skip when it has no objects; save JPEG q90 to `out/.work/images/syn_<seed>_<n:06d>.jpg` and each object mask to `out/.work/masks/syn_<seed>_<n:06d>_<j>.png`; a train `PlanFrame` (clip_id `"synthetic"`). `tick()` per composite.
6. `plan = ExportPlan(project_id=M["project"]["id"], project_name=M["project"]["name"], classes=tuple(ClassDef(id=c["id"], name=c["name"], color=c["color"]) for c in M["classes"]), options=ExportOptions(**{**M["options"], "val_clip_ids": tuple(M["options"]["val_clip_ids"] or ())}), clip_ids=tuple(c["id"] for c in M["clips"]), clip_names={c["id"]: c["name"] for c in M["clips"]}, val_clip_ids=tuple(M["options"]["val_clip_ids"] or ()), frames=tuple(real + synthetic))`.
7. `write_dataset(plan, formats, out, tick, link_images=True)` with `formats = [f["name"] for f in M["formats"]]`; `manifest = manifest_for(...)`; then overwrite `manifest["clips"] = M["clips"]` and add `derived_from`, `augment: {per_image, max_objects, seed, pool, synthetic_images}`; write `export.json`; `rmtree(out/.work)`.

- [ ] **Step 1: Failing tests** — `server/tests/test_augment_derive.py`. Build a real source export with the store fixtures, as `server/tests/test_export_runner.py::_labelled` does (a 6-frame clip, a 30×40 rectangle mask written with `store.write_mask(clip.id, 1, i, mask)`), exported with `run_export(store, project.id, [clip.id], ["yolo_seg", "coco", "masks"], ExportOptions(), data_root / "exports")`; give the frames texture (e.g. `make_clip` frames are flat grey — that's fine). Create a background pool by writing three random 64×64 JPEGs into `data_root/backgrounds/dtd/x/`. Tests:
  - `test_derive_writes_val_unchanged_and_train_with_composites`: `run_augment(data_root, src_id, AugmentOptions(per_image=2, max_objects=2, seed=1))` → new dir; `sorted(os.listdir(new/"yolo_seg/labels/val")) == sorted(os.listdir(src/"yolo_seg/labels/val"))` and each val label file's bytes equal the source's; train image count == real train × 3 minus skipped composites (assert `== real * 3` — with these sizes none are skipped; if one ever is, the manifest's `synthetic_images` says so: assert train images == real + manifest["augment"]["synthetic_images"]); every `syn_*.txt` label has ≥ 1 line; `json.loads(export.json)` has `derived_from == src_id`, `augment.per_image == 2`, the same `formats` names as the source; `pycocotools.coco.COCO(new/"coco/annotations/instances_train.json")` loads and has annotations for `syn_` images; the source tree is unchanged (compare a `{relpath: (size, mtime_ns)}` snapshot before/after).
  - `test_same_seed_same_composites`: two derivations with the same options → the `syn_*` label files are byte-identical.
  - `test_cancel_removes_output` (`cancel=lambda: True` after the first tick → `ExportCancelled`; no new dir under `exports/` besides the source).
  - `test_check_augment_errors`: missing export → NotFound; an export without masks → BadRequest; empty pool → BadRequest; `per_image=11` → BadRequest.
  - `test_list_exports_reports_formats_and_derived_from` (in the same file): after a derive, `list_exports(data_root)` has both, the derived one first... (ids sort: the derived id starts with the source id, so it sorts after it lexicographically; `list_exports` sorts newest-first by name — assert by id lookup, not position) with `derived_from` set and `formats` containing `"masks"`.
  Append to `test_api_augment.py`: POST augment on a source built the same way → 202 → job done → `GET /api/exports` lists the new id with `derived_from`; `GET /api/exports/{new}/samples?limit=3` → 3 items whose names start with `syn_` and have `ground_truth`; `GET /api/exports/{new}/images/train/<a syn name>.jpg` → 200 image/jpeg; `images/test/x.jpg` → 404 (split not allowed); POST augment on an export without masks → 400.
- [ ] **Step 2: See them fail.**
- [ ] **Step 3: Implement** — the runner refactor first (run the existing export tests: `uv run pytest server/tests/test_export_runner.py server/tests/test_export_formats.py server/tests/test_api_export.py -q` must stay green), then `derive.py` following the algorithm, then `ExportInfo` fields (`formats=tuple(f["name"] for f in manifest["formats"])`, `derived_from=manifest.get("derived_from")`), then routes. The augment route validates with `check_augment` before starting the job (400/404 synchronously), and the job wraps `ExportCancelled` → `{"cancelled": True}`.
- [ ] **Step 4: Pass** — focused tests, full non-GPU suite, ruff, line lengths.
- [ ] **Step 5: Commit** — `git commit -m "Augmented copy of an export: composites on background textures, written by the existing exporters"`

---

### Task 8: Augment panel and composite samples (web)

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/api/rest.ts`, `web/src/components/TrainSidebar.tsx`, `web/src/pages/Train.tsx`, `web/src/components/PredictionGallery.tsx` (use `exportImageUrl`)
- Create: `web/src/components/AugmentPanel.tsx`, `web/src/components/CompositeSamples.tsx`
- Test: `web/src/components/AugmentPanel.test.tsx`, `web/src/pages/Train.test.tsx` (update/append)

**Interfaces:**
- Types: `ExportInfo` gains `formats: string[]; derived_from: string | null`; `BackgroundPool = { name: string; label: string; images: number; size_bytes: number; fetching: string | null }`; `ExportSample = { name: string; width: number; height: number; ground_truth: Instance[] }`; `AugmentRequest = { per_image: number; max_objects: number; seed: number; pool: string }`.
- rest: `listBackgrounds()`, `fetchBackgrounds(name)` → `{job_id}`, `augmentExport(exportId, body)` → `{job_id}`, `exportSamples(exportId, limit, seed)`; `exportImageUrl(exportId, split: "train" | "val", name)`; keep `valImageUrl` as `(e, n) => exportImageUrl(e, "val", n)`.
- `TrainSidebar` props gain `exportId: string | null; onExportChange: (id: string) => void; onExportsChanged: () => void` (the export selection is lifted to `Train.tsx`, which passes the first export as the default when none is chosen). The sidebar renders `<AugmentPanel exportInfo={chosen} onCreated={(id) => { onExportsChanged(); onExportChange(id); }} />` directly under the export select.
- `AugmentPanel({ exportInfo: ExportInfo | null, onCreated: (exportId: string) => void })`:
  - `null` or an export without `"masks"` in `formats` → a `.note`: "Augmented copies need an export with Mask images." (no controls).
  - Polls `api.listBackgrounds()` (on mount and after a fetch job ends). Pool `dtd` with `images === 0` and no job → button **Download DTD textures (625 MB)**; while its job runs (`useJob`) → `<progress>` with "Downloading… 37 %" / "Extracting… 80 %" (from the latest progress event's `phase`, `done`, `total`) and **Cancel** (`api.cancelJob`); `images > 0` → `.note` "DTD textures · 5,640 images".
  - Inputs "Composites per image" (1–10, 3) and "Objects per image" (1–3, 2); **Make augmented copy** (disabled without pool images or while a job runs) → `api.augmentExport(id, {per_image, max_objects, seed: 0, pool: "dtd"})`; progress bar + Cancel while running; on `done` → `onCreated(result.export_id)`; on error → `.alert` with the message; on cancelled → `.note` "Cancelled. Nothing was written."
- `CompositeSamples({ exportInfo: ExportInfo })` — rendered by `Train.tsx` above the Runs section when the selected export has `derived_from`: heading "Composite samples · <id>", a **Shuffle** button (seed + 1), and the same strip layout as the gallery (four visible, scroll-snap, ‹ ›) of `GalleryCard`s with `groundTruth` only (`showGt`, no predictions, caption = name), fetched with `api.exportSamples(id, 12, seed)`.

- [ ] **Step 1: Failing tests** — `AugmentPanel.test.tsx` mocks `../api/rest` (`listBackgrounds`, `fetchBackgrounds`, `augmentExport`, `cancelJob`) and `../hooks/useJob` (return a controllable object: `{ status, done, total, result, error, events? }` — read `web/src/hooks/useJob.ts`/`web/src/lib/jobReducer.ts` for the exact `JobView` shape and mock that). Cases: no masks → the note, no buttons; pool empty → clicking Download calls `fetchBackgrounds("dtd")`; pool ready → setting composites to 5 and clicking Make calls `augmentExport(id, {per_image: 5, max_objects: 2, seed: 0, pool: "dtd"})`; a `done` job with `result.export_id = "new"` calls `onCreated("new")` once. In `Train.test.tsx` update the `listExports` mock with `formats`/`derived_from`, mock `listBackgrounds`/`exportSamples`, and add: selecting an export whose `derived_from` is set shows "Composite samples".
- [ ] **Step 2: See them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Pass** — `npm test && npm run typecheck && npm run build`.
- [ ] **Step 5: Commit** — `git commit -m "Train page: augmented copy panel (DTD download, composites) and composite samples"`

---

### Task 9: Docs

**Files:** `README.md` (Training section), `docs/superpowers/plans/2026-09-20-vsp-roadmap.md` (a row), `docs/superpowers/specs/2026-09-24-yolo-seg-training-design.md` (one line under "Train page" pointing at the new spec for the preview and presets).

- [ ] **Step 1:** README: add "Augmentation" under Training — presets (`--augment strong`, what Strong changes), backgrounds (`uv run vsp backgrounds fetch dtd`, 625 MB, `data/backgrounds/dtd/`), augmented copies (from the Train page; what's in them: val unchanged, train + N composites per frame, `syn_` names, `derived_from`), and the prediction strip (filters, enlarge, keys). Keep it factual against the code (grep the flags).
- [ ] **Step 2:** Roadmap row: `| 7 | 2026-09-24-augmentation-and-prediction-strip.md — augmentation | Prediction strip, presets, DTD pool, augmented exports | For the comparison run | **Done** (… Python / … web tests) |` with real counts from `uv run pytest server/tests -q -m "not gpu"` and `npm test`.
- [ ] **Step 3:** Verify: full Python (non-GPU) + ruff + web test/typecheck/build.
- [ ] **Step 4: Commit** — `git commit -m "Document augmentation, background pools and the prediction strip"`

---

### Task 10: Comparison run (operational)

- [ ] `uv run vsp backgrounds fetch dtd` (≈ 625 MB).
- [ ] Derive a ×3 copy of `20260924-040124-8336a8` (through the API with the server restarted by the user, or directly: `uv run python -c "from vsp.augment.derive import AugmentOptions, run_augment; from vsp.config import Settings; print(run_augment(Settings.load('configs/default.yaml').project.data_root, '20260924-040124-8336a8', AugmentOptions(per_image=3)))"`); look at ~12 composites (Read a few JPEGs) for realism before training.
- [ ] Train in parallel: `cuda:1` — yolo26s-seg `--augment strong` on the augmented copy; `cuda:0` — yolo26s-seg `--augment strong` on the original export (isolates the effect of the composites from the presets). Same other params as before.
- [ ] Report a table against the existing yolo26s-seg run (0.938 / 0.782 / 0.832), plus Missed counts from the gallery filter for each run.
