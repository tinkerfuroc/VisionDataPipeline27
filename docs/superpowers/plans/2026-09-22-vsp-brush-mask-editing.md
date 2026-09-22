# Brush mask editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user paint and erase a frame's mask with a brush; Save stores the painted mask and makes the frame a keyframe of type `paint`.

**Architecture:** The server gains one labeling function (`set_mask`) and one route (`POST /api/clips/{c}/mask`) that take a whole mask as a base64 PNG. The web side gains a pure brush-state module (`brush.ts`: draft reducer, stroke rasteriser and pixel conversions with injected contexts), a thin DOM module (`brushCanvas.ts`: offscreen canvas, base loading, PNG encoding), a `useBrush` hook, a `BrushPanel`, brush input on `FrameCanvas`, and a `mode: "prompt" | "brush"` on the Label page with the locks the spec defines.

**Tech Stack:** Python 3.11 / FastAPI / Pillow / numpy (server); React 18 / TypeScript strict / vitest + @testing-library (web). Python through uv only; web commands through `bash -c 'source ~/.nvm/nvm.sh && cd web && …'`.

**Spec:** `docs/superpowers/specs/2026-09-22-brush-mask-editing-design.md` — the authority for every behaviour below; the main design is `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`.

## Global Constraints

- Python runs through uv only (`uv run pytest server/tests -q`, `uv run ruff check server`). No pip, no conda. Web: `npx vitest run`, `npx tsc --noEmit -p .`, `npm run build`, always after `source ~/.nvm/nvm.sh`.
- Never stage anything under `vision_train/`, `data/`, `web/node_modules/`, `web/dist/`, `.superpowers/`, or `*.pt` / `*.safetensors`. Stage by explicit path. Do not push.
- Every commit message ends with exactly these two lines, whatever model the committer is:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL`.
- A hand-painted frame is a keyframe with `prompt.type == "paint"` and payload `{}`. `PromptSpec` (prompt/accept request model) must NOT accept `"paint"`.
- An empty saved mask removes the frame's mask and its keyframe entry and leaves the object (with its class) in place.
- The mask route runs under `gpu_lease` (409 with `job_id` while a propagation or export holds it).
- Locks while there are unsaved strokes (spec "Locks"): transport, step/keyframe/next-empty/play/delete keys, other tools, propagation and cleanup buttons are disabled with the sentence `Save or discard the brush strokes on frame N first.`; Fit, zoom and pan keep working.
- Keys: `Enter` Save, `Escape` Discard, `Z` Undo stroke, `-` shrink, `=` (or `+`) grow, only while the Brush tool is active; `[` and `]` stay with the cleanup range.
- GUI rules from the design pass still apply: every control in the right sidebar, sentence case, buttons name what they do, `<kbd aria-hidden>` plus `aria-keyshortcuts` on keyed buttons, no hue in chrome, no middle dots, no "→".
- jsdom has no canvas: `HTMLCanvasElement.getContext` returns null there. Every canvas-touching path must tolerate a null context, and anything that must be unit-tested takes an injected context or plain pixel arrays.
- Only `ruff check` is enforced on Python (never run `ruff format` on whole files). TypeScript is `strict`.

---

### Task 1: Server — `set_mask`, the `paint` prompt type and `POST /api/clips/{c}/mask`

**Files:**
- Create: `server/vsp/labeling/paint.py`
- Modify: `server/vsp/store/models.py` (`Prompt.type` literal), `server/vsp/api/labeling.py` (route)
- Test: `server/tests/test_paint.py` (new), `server/tests/test_api_labeling.py`, `server/tests/test_propagate.py`

**Interfaces:**
- Produces: `decode_mask_png(b64: str) -> np.ndarray` (bool HxW, `BadRequest` on junk); `set_mask(store, clip_id, frame, mask, *, class_id=None) -> Clip`; route `POST /api/clips/{clip_id}/mask` with body `{frame: int, mask_png_b64: str, class_id?: int}` returning the clip detail (same shape as accept), 400 on bad input, 409 (`job_id`) while the GPU lease is held.
- Consumes: `create_object` from `vsp.labeling.prompts`, `subtract_range` from `vsp.labeling.ranges`, `Keyframe`, `Prompt`, `now_iso` from `vsp.store`, `encode_mask_png` from `vsp.store.project` (tests).

- [ ] **Step 1: Write the failing tests**

Create `server/tests/test_paint.py`:

```python
import base64

import numpy as np
import pytest
from vsp.engine.fake import FakeEngine
from vsp.labeling.errors import BadRequest
from vsp.labeling.frames import delete_frame
from vsp.labeling.paint import decode_mask_png, set_mask
from vsp.labeling.prompts import OBJECT_ID, PromptSpec, accept
from vsp.labeling.propagate import run_propagation
from vsp.store.project import encode_mask_png

CLICK = PromptSpec(type="clicks", points=[(30, 30, 1)])
KW = {"chunk_size": 8, "min_chunk_size": 4, "lost_after": 3}


def _blob(h, w, rows=slice(10, 30), cols=slice(10, 40)):
    m = np.zeros((h, w), dtype=bool)
    m[rows, cols] = True
    return m


def test_decode_round_trips_the_store_encoding_and_rejects_junk():
    m = _blob(48, 64)
    b64 = base64.b64encode(encode_mask_png(m)).decode("ascii")
    assert np.array_equal(decode_mask_png(b64), m)
    with pytest.raises(BadRequest, match="not a PNG"):
        decode_mask_png(base64.b64encode(b"not a png").decode("ascii"))
    with pytest.raises(BadRequest, match="not a PNG"):
        decode_mask_png("%%%")


def test_first_save_needs_a_class_creates_the_object_and_a_paint_keyframe(store, make_clip):
    _, clip = make_clip(n_frames=6)
    m = _blob(clip.height, clip.width)
    with pytest.raises(BadRequest, match="class_id is required"):
        set_mask(store, clip.id, 2, m)
    with pytest.raises(BadRequest, match="unknown class"):
        set_mask(store, clip.id, 2, m, class_id=7)
    out = set_mask(store, clip.id, 2, m, class_id=0)
    obj = out.objects[0]
    assert obj.id == OBJECT_ID and obj.class_id == 0
    assert [(k.frame_idx, k.prompt.type, k.prompt.payload) for k in obj.keyframes] == [(2, "paint", {})]
    assert np.array_equal(store.read_mask(clip.id, OBJECT_ID, 2), m)
    assert store.frame_states(out)[2] == "keyframe"


def test_later_saves_replace_the_mask_and_keep_one_keyframe_entry(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=12)
    accept(store, eng, clip.id, 0, CLICK, class_id=0)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert store.read_clip(clip.id).objects[0].propagated_ranges == [(1, 11)]
    m = _blob(clip.height, clip.width)
    out = set_mask(store, clip.id, 5, m)  # a propagated frame becomes a painted keyframe
    obj = out.objects[0]
    assert [k.frame_idx for k in obj.keyframes] == [0, 5] and obj.keyframes[1].prompt.type == "paint"
    assert obj.propagated_ranges == [(1, 4), (6, 11)]
    m2 = _blob(clip.height, clip.width, cols=slice(0, 5))
    out = set_mask(store, clip.id, 5, m2)
    assert [k.frame_idx for k in out.objects[0].keyframes] == [0, 5]
    assert np.array_equal(store.read_mask(clip.id, OBJECT_ID, 5), m2)
    out = set_mask(store, clip.id, 0, m)  # painting over a prompt keyframe replaces its prompt record
    assert out.objects[0].keyframes[0].prompt.type == "paint"
    out = set_mask(store, clip.id, 1, m, class_id=0)  # a class on a later save is accepted, like accept()
    assert out.objects[0].class_id == 0


def test_an_empty_mask_removes_the_mask_and_the_keyframe_but_keeps_the_object(store, make_clip):
    eng = FakeEngine()
    _, clip = make_clip(n_frames=6)
    accept(store, eng, clip.id, 0, CLICK, class_id=0)
    run_propagation(store, eng, clip.id, 0, "forward", **KW)
    empty = np.zeros((clip.height, clip.width), dtype=bool)
    out = set_mask(store, clip.id, 0, empty)
    assert out.objects[0].keyframes == [] and out.objects[0].class_id == 0
    assert store.read_mask(clip.id, OBJECT_ID, 0) is None
    assert store.frame_states(out)[0] == "empty"
    out = set_mask(store, clip.id, 3, empty)  # a propagated frame: mask gone, range trimmed
    assert store.read_mask(clip.id, OBJECT_ID, 3) is None
    assert out.objects[0].propagated_ranges == [(1, 2), (4, 5)]
    out = set_mask(store, clip.id, 3, empty)  # clearing an already empty frame is not an error
    assert out.objects[0].propagated_ranges == [(1, 2), (4, 5)]


def test_refused_saves_change_nothing(store, make_clip):
    _, clip = make_clip(n_frames=6)
    empty = np.zeros((clip.height, clip.width), dtype=bool)
    with pytest.raises(BadRequest, match="no object"):
        set_mask(store, clip.id, 1, empty, class_id=0)
    m = _blob(clip.height, clip.width)
    with pytest.raises(BadRequest, match="out of range"):
        set_mask(store, clip.id, 6, m, class_id=0)
    delete_frame(store, clip.id, 1)
    with pytest.raises(BadRequest, match="deleted"):
        set_mask(store, clip.id, 1, m, class_id=0)
    with pytest.raises(BadRequest, match=f"the clip is {clip.width}×{clip.height}"):
        set_mask(store, clip.id, 2, m[:-1, :], class_id=0)
    assert store.read_clip(clip.id).objects == []
    assert store.mask_indices(clip.id, OBJECT_ID) == []
```

Append to `server/tests/test_propagate.py` (add `from vsp.labeling.paint import set_mask` to its imports):

```python
def test_propagation_seeds_from_a_painted_keyframe_and_stops_at_it(store, make_clip):
    eng = FakeEngine()
    clip = _seeded(store, make_clip, eng, n_frames=20)
    painted = np.zeros((clip.height, clip.width), dtype=bool)
    painted[20:40, 60:80] = True  # x0 = 60, well away from the click seed's x0 = 20
    set_mask(store, clip.id, 10, painted)
    res = run_propagation(store, eng, clip.id, 0, "forward", **KW)
    assert res.processed == 9  # frames 1..9: the painted keyframe bounds the run
    assert _x0(store, clip.id, 10) == 60  # and is untouched
    res = run_propagation(store, eng, clip.id, 10, "forward", **KW)
    assert res.processed == 9 and _x0(store, clip.id, 11) == 62  # seeded by the painted mask, shifted once
```

Append to `server/tests/test_api_labeling.py` (it already imports `base64`, `numpy as np`, `SlowEngine`, `_client`, `_clip`, `wait_job`; add `from vsp.store.project import encode_mask_png`):

```python
def test_paint_route_writes_a_keyframe_and_is_refused_while_a_propagation_runs(tmp_path):
    with _client(tmp_path, SlowEngine()) as client:
        cid = _clip(client, tmp_path, n=40)
        info = client.get(f"/api/clips/{cid}").json()["clip"]
        m = np.zeros((info["height"], info["width"]), dtype=bool)
        m[5:15, 5:25] = True
        b64 = base64.b64encode(encode_mask_png(m)).decode("ascii")
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 2, "mask_png_b64": b64}).status_code == 400
        r = client.post(f"/api/clips/{cid}/mask", json={"frame": 2, "mask_png_b64": b64, "class_id": 0})
        assert r.status_code == 200 and r.json()["states"][2] == "keyframe"
        assert r.json()["clip"]["objects"][0]["keyframes"][0]["prompt"] == {"type": "paint", "payload": {}}
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 2, "mask_png_b64": "nope"}).status_code == 400
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 99, "mask_png_b64": b64}).status_code == 400
        job_id = client.post(f"/api/clips/{cid}/propagate", json={"from_frame": 2, "direction": "forward"}).json()["job_id"]
        busy = client.post(f"/api/clips/{cid}/mask", json={"frame": 3, "mask_png_b64": b64})
        assert busy.status_code == 409 and busy.json()["job_id"] == job_id  # it replaces what the runner writes
        wait_job(client, job_id)
```

Mirror the existing 409 test (around `test_api_labeling.py:95-110`) if the lease timing needs the same arrangement it uses.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest server/tests/test_paint.py server/tests/test_api_labeling.py server/tests/test_propagate.py -q`
Expected: `test_paint.py` fails at import (`No module named 'vsp.labeling.paint'`), likewise the two appended tests.

- [ ] **Step 3: Extend the store model**

In `server/vsp/store/models.py` change `class Prompt` to `type: Literal["text", "clicks", "box", "paint"]`.

- [ ] **Step 4: Implement the labeling function**

Create `server/vsp/labeling/paint.py`:

```python
"""Hand-painted masks (brush spec): the whole mask arrives as a PNG and the frame becomes a `paint` keyframe."""

from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..store import Clip, Keyframe, ProjectStore, Prompt, now_iso
from .errors import BadRequest
from .prompts import create_object
from .ranges import subtract_range


def decode_mask_png(b64: str) -> np.ndarray:
    """A base64 PNG -> bool HxW; any pixel brighter than mid-grey is in the mask."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
        img.load()
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as e:
        raise BadRequest(f"mask_png_b64 is not a PNG image: {e}") from e
    return np.array(img.convert("L")) > 127


def set_mask(
    store: ProjectStore, clip_id: str, frame: int, mask: np.ndarray, *, class_id: int | None = None
) -> Clip:
    """Store `mask` as the object's mask on `frame` and make the frame a `paint` keyframe.

    An empty mask removes the frame's mask and keyframe instead. Either way the frame leaves
    `propagated_ranges`. The object (and its class) always stays.
    """
    store.clip_dir(clip_id)  # raises NotFound before the lock table grows an entry for a bogus id
    with store.lock(clip_id):
        clip = store.read_clip(clip_id)
        if not 0 <= frame < clip.frame_count:
            raise BadRequest(f"frame {frame} is out of range 0..{clip.frame_count - 1}")
        if frame in clip.deleted_frames:
            raise BadRequest(f"frame {frame} is deleted; restore it before labeling")
        if mask.shape != (clip.height, clip.width):
            raise BadRequest(f"mask is {mask.shape[1]}×{mask.shape[0]}, the clip is {clip.width}×{clip.height}")
        if not clip.objects and not mask.any():
            raise BadRequest("the mask is empty and the clip has no object yet; paint something first")
        if clip.objects:
            obj = clip.objects[0]
        elif class_id is None:
            raise BadRequest("class_id is required for the first accept on a clip")
        else:
            obj = None
        if class_id is not None:
            known = {c.id for c in store.read_project(clip.project_id).classes}
            if class_id not in known:
                raise BadRequest(f"unknown class id {class_id}")
        if obj is None:
            obj = create_object(clip, class_id)
        elif class_id is not None:
            obj.class_id = class_id
        others = [k for k in obj.keyframes if k.frame_idx != frame]
        if mask.any():
            store.write_mask(clip.id, obj.id, frame, mask)
            painted = Keyframe(frame_idx=frame, prompt=Prompt(type="paint", payload={}), created_at=now_iso())
            obj.keyframes = sorted([*others, painted], key=lambda k: k.frame_idx)
        else:
            store.delete_mask(clip.id, obj.id, frame)
            obj.keyframes = others
        obj.propagated_ranges = subtract_range(obj.propagated_ranges, frame, frame)
        store.write_clip(clip)
        return clip
```

Check `Clip` has `height` and `width` fields (it does: `server/vsp/store/models.py`) and that `Keyframe`, `Prompt`, `now_iso` are exported from `vsp.store` (`server/tests/test_engine_gpu.py` imports them from there).

- [ ] **Step 5: Add the route**

In `server/vsp/api/labeling.py` import `from ..labeling.paint import decode_mask_png, set_mask`, add the body model next to `AcceptBody`:

```python
class MaskBody(BaseModel):
    frame: int
    mask_png_b64: str
    class_id: int | None = None
```

and the route after `prompt_accept`:

```python
@router.post("/mask")
def paint_mask(clip_id: str, body: MaskBody, state: State) -> dict:
    """A hand-painted mask for one frame (brush spec). The frame becomes a `paint` keyframe."""
    state.store.read_clip(clip_id)
    mask = decode_mask_png(body.mask_png_b64)
    with gpu_lease(state):  # a propagation writes the very masks this replaces
        return clip_detail(state, set_mask(state.store, clip_id, body.frame, mask, class_id=body.class_id))
```

Update the module docstring's first line to mention the mask route.

- [ ] **Step 6: Run the tests and lint**

Run: `uv run pytest server/tests -q && uv run ruff check server`
Expected: all pass (the previous count plus 7); `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add server/vsp/labeling/paint.py server/vsp/store/models.py server/vsp/api/labeling.py server/tests/test_paint.py server/tests/test_api_labeling.py server/tests/test_propagate.py
git commit -m "Store hand-painted masks as paint keyframes (POST /api/clips/{c}/mask)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

---

### Task 2: Web — brush state and rasteriser (`brush.ts`), keyboard, API client

**Files:**
- Create: `web/src/lib/brush.ts`, `web/src/lib/brush.test.ts`
- Modify: `web/src/hooks/useKeyboard.ts` (+ test), `web/src/api/types.ts`, `web/src/api/rest.ts`

**Interfaces:**
- Produces (from `brush.ts`): types `BrushMode`, `Stroke`, `BrushDraft`, `BrushAction`; constants `MIN_SIZE = 2`, `MAX_SIZE = 200`, `DEFAULT_SIZE = 24`; `emptyBrush(size?)`, `clampSize(n)`, `brushReducer(d, a)`, `extendStroke(stroke, x, y): boolean` (mutates, dedupes), `BrushContext` (the subset of `CanvasRenderingContext2D` the drawing uses), `drawStroke(ctx, stroke, from, rgb)`, `rasterize(ctx, base: ImageData | null, strokes, rgb, width, height)`, `baseLayerData(mask: {data, width, height}, rgb): Uint8ClampedArray`, `layerToMaskData(layer: Uint8ClampedArray): {data: Uint8ClampedArray; any: boolean}`.
- Produces: `KeyAction` gains `{ type: "brushSize"; dir: 1 | -1 }` (`-` shrinks, `=` / `+` grows); `PromptType` gains `"paint"`; `api.setMask(clipId, frame, maskPngB64, opts: { classId?: number } = {}) -> Promise<ClipDetail>` posting `{frame, mask_png_b64, class_id?}` to `/api/clips/${clipId}/mask`.

- [ ] **Step 1: Write the failing tests**

Create `web/src/lib/brush.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SIZE, MAX_SIZE, MIN_SIZE, baseLayerData, brushReducer, clampSize, drawStroke, emptyBrush, extendStroke,
  layerToMaskData, rasterize, type BrushContext, type Stroke,
} from "./brush";

const stroke = (points: [number, number][], mode: "paint" | "erase" = "paint", size = 10): Stroke => ({ mode, size, points });

function recorder(): BrushContext & { calls: string[] } {
  const calls: string[] = [];
  const ctx = {
    globalCompositeOperation: "source-over", strokeStyle: "", fillStyle: "", lineWidth: 1, lineCap: "butt", lineJoin: "miter",
    beginPath: () => calls.push("beginPath"),
    moveTo: (x: number, y: number) => calls.push(`moveTo ${x},${y}`),
    lineTo: (x: number, y: number) => calls.push(`lineTo ${x},${y}`),
    arc: (x: number, y: number, r: number) => calls.push(`arc ${x},${y},${r}`),
    stroke: () => calls.push(`stroke ${ctx.globalCompositeOperation} ${ctx.strokeStyle} w${ctx.lineWidth} ${ctx.lineCap}/${ctx.lineJoin}`),
    fill: () => calls.push(`fill ${ctx.globalCompositeOperation} ${ctx.fillStyle}`),
    clearRect: (x: number, y: number, w: number, h: number) => calls.push(`clearRect ${x},${y},${w},${h}`),
    putImageData: (_d: ImageData, x: number, y: number) => calls.push(`putImageData ${x},${y}`),
    calls,
  } as unknown as BrushContext & { calls: string[] };
  return ctx;
}

describe("brush draft", () => {
  it("clamps the size and grows or shrinks it by 20 %", () => {
    expect(emptyBrush()).toEqual({ size: DEFAULT_SIZE, strokes: [] });
    expect(clampSize(0)).toBe(MIN_SIZE);
    expect(clampSize(999)).toBe(MAX_SIZE);
    expect(clampSize(7.6)).toBe(8);
    const d = emptyBrush(50);
    expect(brushReducer(d, { type: "grow", dir: 1 }).size).toBe(60);
    expect(brushReducer(d, { type: "grow", dir: -1 }).size).toBe(42);
    const max = emptyBrush(MAX_SIZE);
    expect(brushReducer(max, { type: "grow", dir: 1 })).toBe(max); // unchanged: same object
    expect(brushReducer(d, { type: "size", size: 50 })).toBe(d);
  });
  it("commits, undoes and clears strokes", () => {
    let d = emptyBrush();
    d = brushReducer(d, { type: "commit", stroke: stroke([[1, 1]]) });
    d = brushReducer(d, { type: "commit", stroke: stroke([[2, 2], [3, 3]], "erase") });
    expect(d.strokes).toHaveLength(2);
    d = brushReducer(d, { type: "undo" });
    expect(d.strokes).toEqual([stroke([[1, 1]])]);
    const cleared = brushReducer(d, { type: "clear" });
    expect(cleared.strokes).toEqual([]);
    expect(brushReducer(cleared, { type: "undo" })).toBe(cleared);
    expect(brushReducer(cleared, { type: "clear" })).toBe(cleared);
  });
  it("extendStroke appends distinct points only", () => {
    const s = stroke([[1, 1]]);
    expect(extendStroke(s, 1, 1)).toBe(false);
    expect(extendStroke(s, 2, 3)).toBe(true);
    expect(s.points).toEqual([[1, 1], [2, 3]]);
  });
});

describe("rasteriser", () => {
  it("draws a one-point stroke as a dot and a polyline with round caps, in the class colour", () => {
    const ctx = recorder();
    drawStroke(ctx, stroke([[4, 5]], "paint", 10), 0, [255, 0, 0]);
    expect(ctx.calls).toEqual(["beginPath", "arc 4.5,5.5,5", "fill source-over rgb(255, 0, 0)"]);
    ctx.calls.length = 0;
    drawStroke(ctx, stroke([[0, 0], [10, 0], [10, 10]], "erase", 6), 0, [255, 0, 0]);
    expect(ctx.calls).toEqual([
      "beginPath", "moveTo 0.5,0.5", "lineTo 10.5,0.5", "lineTo 10.5,10.5",
      "stroke destination-out rgb(255, 0, 0) w6 round/round",
    ]);
  });
  it("draws only the new segment when asked to continue from a point", () => {
    const ctx = recorder();
    drawStroke(ctx, stroke([[0, 0], [10, 0], [10, 10]]), 2, [0, 0, 0]);
    expect(ctx.calls).toEqual(["beginPath", "moveTo 10.5,0.5", "lineTo 10.5,10.5", "stroke source-over rgb(0, 0, 0) w10 round/round"]);
  });
  it("rasterize clears, restores the base and replays every stroke", () => {
    const ctx = recorder();
    const base = { data: new Uint8ClampedArray(4), width: 1, height: 1 } as unknown as ImageData;
    rasterize(ctx, base, [stroke([[1, 1]]), stroke([[2, 2]], "erase")], [1, 2, 3], 8, 6);
    expect(ctx.calls[0]).toBe("clearRect 0,0,8,6");
    expect(ctx.calls[1]).toBe("putImageData 0,0");
    expect(ctx.calls.filter((c) => c.startsWith("fill"))).toEqual(["fill source-over rgb(1, 2, 3)", "fill destination-out rgb(1, 2, 3)"]);
    expect(ctx.globalCompositeOperation).toBe("source-over"); // left in the default state
    ctx.calls.length = 0;
    rasterize(ctx, null, [], [1, 2, 3], 8, 6);
    expect(ctx.calls).toEqual(["clearRect 0,0,8,6"]);
  });
  it("converts a stored mask to a colour layer and a layer back to a mask", () => {
    // 2x1 mask image: left pixel white (in the mask), right pixel black
    const mask = { data: new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]), width: 2, height: 1 };
    expect(Array.from(baseLayerData(mask, [10, 20, 30]))).toEqual([10, 20, 30, 255, 0, 0, 0, 0]);
    const layer = new Uint8ClampedArray([10, 20, 30, 200, 10, 20, 30, 100, 0, 0, 0, 0]);
    const out = layerToMaskData(layer);
    expect(Array.from(out.data)).toEqual([255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
    expect(out.any).toBe(true);
    expect(layerToMaskData(new Uint8ClampedArray(8)).any).toBe(false);
  });
});
```

Append to `web/src/hooks/useKeyboard.test.ts`, inside the `keyAction` describe:

```ts
  it("resizes the brush with - and =", () => {
    expect(key("-")).toEqual({ type: "brushSize", dir: -1 });
    expect(key("=")).toEqual({ type: "brushSize", dir: 1 });
    expect(key("+", { shiftKey: true })).toEqual({ type: "brushSize", dir: 1 });
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/lib/brush.test.ts src/hooks/useKeyboard.test.ts'`
Expected: `brush.test.ts` fails to import; the keyboard test fails with `null` for `-`.

- [ ] **Step 3: Implement `brush.ts`**

```ts
import type { Rgb } from "./mask";

/** Brush strokes are recorded in image space so a wheel-zoom mid-stroke cannot distort them. */
export type BrushMode = "paint" | "erase";
export type Stroke = { mode: BrushMode; size: number; points: [number, number][] };
export type BrushDraft = { size: number; strokes: Stroke[] };

export const MIN_SIZE = 2;
export const MAX_SIZE = 200;
export const DEFAULT_SIZE = 24;

export const emptyBrush = (size = DEFAULT_SIZE): BrushDraft => ({ size, strokes: [] });
export const clampSize = (n: number): number => Math.max(MIN_SIZE, Math.min(MAX_SIZE, Math.round(n)));

export type BrushAction =
  | { type: "commit"; stroke: Stroke }
  | { type: "undo" }
  | { type: "clear" }
  | { type: "size"; size: number }
  | { type: "grow"; dir: 1 | -1 };

export function brushReducer(d: BrushDraft, a: BrushAction): BrushDraft {
  switch (a.type) {
    case "commit":
      return { ...d, strokes: [...d.strokes, a.stroke] };
    case "undo":
      return d.strokes.length === 0 ? d : { ...d, strokes: d.strokes.slice(0, -1) };
    case "clear":
      return d.strokes.length === 0 ? d : { ...d, strokes: [] };
    case "size": {
      const size = clampSize(a.size);
      return size === d.size ? d : { ...d, size };
    }
    case "grow":
      return brushReducer(d, { type: "size", size: d.size * (a.dir === 1 ? 1.2 : 1 / 1.2) });
  }
}

/** Append a point to the stroke being drawn; false when it repeats the last point. Mutates: this runs per mouse move. */
export function extendStroke(stroke: Stroke, x: number, y: number): boolean {
  const [px, py] = stroke.points[stroke.points.length - 1];
  if (px === x && py === y) return false;
  stroke.points.push([x, y]);
  return true;
}

/** The part of a 2D context the brush draws with; tests pass a recorder. */
export type BrushContext = Pick<
  CanvasRenderingContext2D,
  | "globalCompositeOperation" | "strokeStyle" | "fillStyle" | "lineWidth" | "lineCap" | "lineJoin"
  | "beginPath" | "moveTo" | "lineTo" | "arc" | "stroke" | "fill" | "clearRect" | "putImageData"
>;

const css = ([r, g, b]: Rgb) => `rgb(${r}, ${g}, ${b})`;

/** Draw `stroke` from its point `from` on (0 = the whole stroke). Paint adds colour, erase cuts alpha. */
export function drawStroke(ctx: BrushContext, stroke: Stroke, from: number, rgb: Rgb): void {
  ctx.globalCompositeOperation = stroke.mode === "paint" ? "source-over" : "destination-out";
  ctx.strokeStyle = css(rgb);
  ctx.fillStyle = css(rgb);
  ctx.lineWidth = stroke.size;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const pts = stroke.points;
  ctx.beginPath();
  if (pts.length === 1) {
    ctx.arc(pts[0][0] + 0.5, pts[0][1] + 0.5, stroke.size / 2, 0, 2 * Math.PI);
    ctx.fill();
    return;
  }
  const start = Math.max(1, from);
  ctx.moveTo(pts[start - 1][0] + 0.5, pts[start - 1][1] + 0.5);
  for (let i = start; i < pts.length; i++) ctx.lineTo(pts[i][0] + 0.5, pts[i][1] + 0.5);
  ctx.stroke();
}

/** Rebuild the layer: the stored mask (already in the class colour) then every stroke in order. */
export function rasterize(
  ctx: BrushContext, base: ImageData | null, strokes: Stroke[], rgb: Rgb, width: number, height: number,
): void {
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, width, height);
  if (base !== null) ctx.putImageData(base, 0, 0);
  for (const s of strokes) drawStroke(ctx, s, 0, rgb);
  ctx.globalCompositeOperation = "source-over";
}

type Pixels = { data: Uint8ClampedArray; width: number; height: number };

/** A stored mask image (white = in the mask) as an opaque class-colour layer on transparent. */
export function baseLayerData(mask: Pixels, [r, g, b]: Rgb): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.data.length);
  for (let i = 0; i < mask.data.length; i += 4) {
    if (mask.data[i] > 127) {
      out[i] = r;
      out[i + 1] = g;
      out[i + 2] = b;
      out[i + 3] = 255;
    }
  }
  return out;
}

/** The layer's alpha as an opaque white-on-black mask image, ready to encode as a PNG for the server. */
export function layerToMaskData(layer: Uint8ClampedArray): { data: Uint8ClampedArray; any: boolean } {
  const data = new Uint8ClampedArray(layer.length);
  let any = false;
  for (let i = 0; i < layer.length; i += 4) {
    const on = layer[i + 3] > 127;
    any ||= on;
    data[i] = data[i + 1] = data[i + 2] = on ? 255 : 0;
    data[i + 3] = 255;
  }
  return { data, any };
}
```

- [ ] **Step 4: Keyboard, types, API client**

`web/src/hooks/useKeyboard.ts`: add to `KeyAction` the member `| { type: "brushSize"; dir: 1 | -1 }` (with a doc comment: `-` shrinks, `=` grows the brush) and, in the first `switch (e.key)`, before the closing brace:

```ts
    case "-":
      return { type: "brushSize", dir: -1 };
    case "=":
    case "+":
      return { type: "brushSize", dir: 1 };
```

`web/src/api/types.ts`: `export type PromptType = "text" | "clicks" | "box" | "paint";`

`web/src/api/rest.ts`, next to `accept`:

```ts
  setMask: (clipId: string, frame: number, maskPngB64: string, opts: { classId?: number } = {}) =>
    request<ClipDetail>(
      `/api/clips/${clipId}/mask`,
      post({ frame, mask_png_b64: maskPngB64, ...(opts.classId === undefined ? {} : { class_id: opts.classId }) }),
    ),
```

Check `web/src/api/rest.test.ts` for how `accept` is tested and add one matching case for `setMask` (body shape with and without `classId`) if that file tests request bodies; otherwise skip.

- [ ] **Step 5: Run the tests, tsc and lint**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npx tsc --noEmit -p .'`
Expected: all pass, tsc silent.

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/brush.ts web/src/lib/brush.test.ts web/src/hooks/useKeyboard.ts web/src/hooks/useKeyboard.test.ts web/src/api/types.ts web/src/api/rest.ts
# add web/src/api/rest.test.ts if it was changed
git commit -m "Brush draft state, stroke rasteriser, brush-size keys and the setMask client

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

---

### Task 3: Web — `brushCanvas.ts`, `useBrush` hook and `BrushPanel`

**Files:**
- Create: `web/src/lib/brushCanvas.ts`, `web/src/hooks/useBrush.ts`, `web/src/components/BrushPanel.tsx`, `web/src/components/BrushPanel.test.tsx`

**Interfaces:**
- Produces (`brushCanvas.ts`): `createLayer(width, height): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D | null }`; `loadBase(store: FrameStore, url: string, rgb, width, height): Promise<ImageData | null>` (fetches the stored mask through `store.image(url)`, draws it on a scratch canvas, returns `baseLayerData` as `ImageData`; null when the image or a context is unavailable); `encodeLayer(canvas): { pngB64: string; any: boolean } | null` (reads the layer's pixels, `layerToMaskData`, writes them to a scratch canvas, `toDataURL("image/png")` minus the `data:image/png;base64,` prefix; null without a context).
- Produces (`useBrush.ts`):

```ts
export type Brush = {
  draft: BrushDraft;           // committed strokes + size
  dirty: boolean;              // strokes.length > 0 or a stroke in progress
  layer: HTMLCanvasElement | null;
  busy: boolean;               // saving
  error: string | null;
  setSize: (size: number) => void;
  grow: (dir: 1 | -1) => void;
  start: (mode: BrushMode, x: number, y: number) => void;
  extend: (x: number, y: number) => void;
  end: () => void;
  undo: () => void;
  discard: () => void;
  save: () => Promise<void>;
};
export function useBrush(o: {
  clipId: string; frame: number; width: number; height: number; rgb: Rgb; store: FrameStore;
  maskUrl: string | null;      // the stored mask to start from (versioned), null when the frame has none
  active: boolean;             // the Brush tool is selected and usable
  classId: number | null; needsClass: boolean;
  apply: (detail: ClipDetail, masksChanged: boolean) => void;
}): Brush;
```

- Produces (`BrushPanel.tsx`): `BrushPanel({ brush, frame, disabledReason }: { brush: Brush; frame: number; disabledReason: string | null })`.

- [ ] **Step 1: Write the failing panel tests**

Create `web/src/components/BrushPanel.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Brush } from "../hooks/useBrush";
import { emptyBrush } from "../lib/brush";
import { BrushPanel } from "./BrushPanel";

const brush = (over: Partial<Brush> = {}): Brush => ({
  draft: emptyBrush(), dirty: false, layer: null, busy: false, error: null,
  setSize: vi.fn(), grow: vi.fn(), start: vi.fn(), extend: vi.fn(), end: vi.fn(), undo: vi.fn(), discard: vi.fn(),
  save: vi.fn(async () => {}), ...over,
});
afterEach(cleanup);

describe("BrushPanel", () => {
  it("shows the size, the hint and disabled buttons while there is nothing to save", () => {
    render(<BrushPanel brush={brush()} frame={12} disabledReason={null} />);
    expect((screen.getByLabelText("Brush size") as HTMLInputElement).value).toBe("24");
    expect(screen.getByText("24 px")).toBeTruthy();
    expect(screen.getByText(/Left-drag to paint, right-drag to erase/)).toBeTruthy();
    expect(screen.getByText("No strokes yet on frame 12.")).toBeTruthy();
    for (const name of [/Save/, /Discard/, /Undo stroke/]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
  });
  it("changes the size from the slider and wires Save, Discard and Undo when dirty", () => {
    const b = brush({ dirty: true, draft: { size: 40, strokes: [{ mode: "paint", size: 40, points: [[1, 1]] }, { mode: "erase", size: 40, points: [[2, 2]] }, { mode: "paint", size: 40, points: [[3, 3]] }] } });
    render(<BrushPanel brush={b} frame={3} disabledReason={null} />);
    fireEvent.change(screen.getByLabelText("Brush size"), { target: { value: "55" } });
    expect(b.setSize).toHaveBeenCalledWith(55);
    expect(screen.getByText("3 strokes on frame 3, not saved.")).toBeTruthy();
    expect(screen.getByText(/Save or discard the brush strokes on frame 3 first/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Save/ }));
    expect(b.save).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Discard/ }));
    expect(b.discard).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Undo stroke/ }));
    expect(b.undo).toHaveBeenCalled();
  });
  it("says why the brush is unavailable, shows saving and the last error", () => {
    const { rerender } = render(<BrushPanel brush={brush()} frame={0} disabledReason="A propagation is running. Stop it to paint." />);
    expect(screen.getByText("A propagation is running. Stop it to paint.")).toBeTruthy();
    expect((screen.getByLabelText("Brush size") as HTMLInputElement).disabled).toBe(true);
    rerender(<BrushPanel brush={brush({ dirty: true, busy: true, error: "mask is 640×480, the clip is 1920×1080" })} frame={0} disabledReason={null} />);
    expect(screen.getByText("Saving…")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toBe("mask is 640×480, the clip is 1920×1080");
    expect((screen.getByRole("button", { name: /Save/ }) as HTMLButtonElement).disabled).toBe(true); // while saving
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/components/BrushPanel.test.tsx'`
Expected: fails to import `./BrushPanel` / `../hooks/useBrush`.

- [ ] **Step 3: Implement `brushCanvas.ts`**

```ts
import { baseLayerData, layerToMaskData } from "./brush";
import type { FrameStore } from "./frameCache";
import type { Rgb } from "./mask";

/** The DOM half of the brush: an offscreen layer canvas, the stored mask as its base, and PNG encoding.
 * Everything here tolerates a missing 2D context (jsdom) by returning null. */

export function createLayer(width: number, height: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D | null } {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return { canvas, ctx: canvas.getContext("2d") };
}

export async function loadBase(store: FrameStore, url: string, rgb: Rgb, width: number, height: number): Promise<ImageData | null> {
  const img = await store.image(url);
  if (img === null) return null;
  const { canvas, ctx } = createLayer(width, height);
  if (ctx === null) return null;
  ctx.drawImage(img, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height);
  const out = ctx.createImageData(width, height);
  out.data.set(baseLayerData(pixels, rgb));
  canvas.width = 0; // release the scratch bitmap
  return out;
}

export function encodeLayer(layer: HTMLCanvasElement): { pngB64: string; any: boolean } | null {
  const ctx = layer.getContext("2d");
  if (ctx === null) return null;
  const { width, height } = layer;
  const { data, any } = layerToMaskData(ctx.getImageData(0, 0, width, height).data);
  const { canvas, ctx: scratch } = createLayer(width, height);
  if (scratch === null) return null;
  const img = scratch.createImageData(width, height);
  img.data.set(data);
  scratch.putImageData(img, 0, 0);
  const url = canvas.toDataURL("image/png");
  canvas.width = 0;
  const prefix = "data:image/png;base64,";
  if (!url.startsWith(prefix)) return null;
  return { pngB64: url.slice(prefix.length), any };
}
```

Check `FrameStore.image(url)` returns `Promise<HTMLImageElement | null>` (`web/src/lib/frameCache.ts:75`).

- [ ] **Step 4: Implement `useBrush.ts`**

```ts
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { brushReducer, drawStroke, emptyBrush, extendStroke, rasterize, type BrushDraft, type BrushMode, type Stroke } from "../lib/brush";
import { createLayer, encodeLayer, loadBase } from "../lib/brushCanvas";
import type { FrameStore } from "../lib/frameCache";
import type { Rgb } from "../lib/mask";

export type Brush = {
  draft: BrushDraft;
  dirty: boolean;
  layer: HTMLCanvasElement | null;
  busy: boolean;
  error: string | null;
  setSize: (size: number) => void;
  grow: (dir: 1 | -1) => void;
  start: (mode: BrushMode, x: number, y: number) => void;
  extend: (x: number, y: number) => void;
  end: () => void;
  undo: () => void;
  discard: () => void;
  save: () => Promise<void>;
};

type Options = {
  clipId: string;
  frame: number;
  width: number;
  height: number;
  rgb: Rgb;
  store: FrameStore;
  /** the stored mask to start from (versioned url), null when the frame has none */
  maskUrl: string | null;
  /** the Brush tool is selected and usable */
  active: boolean;
  classId: number | null;
  needsClass: boolean;
  apply: (detail: ClipDetail, masksChanged: boolean) => void;
};

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** The brush draft for one frame: committed strokes in state, the stroke in progress in a ref (one segment is
 * drawn per mouse move, with no React render), and an offscreen layer canvas the FrameCanvas draws over the frame. */
export function useBrush(o: Options): Brush {
  const { clipId, frame, width, height, rgb, store, maskUrl, active, classId, needsClass, apply } = o;
  const [draft, setDraft] = useState<BrushDraft>(() => emptyBrush());
  const [live, setLive] = useState(false); // a stroke is in progress
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const layer = useMemo(() => createLayer(width, height), [width, height]);
  const base = useRef<ImageData | null>(null);
  const stroke = useRef<Stroke | null>(null);
  const dirty = draft.strokes.length > 0 || live;

  // Start from the frame's stored mask whenever the frame or its mask changes while nothing is painted.
  useEffect(() => {
    if (!active || dirty) return;
    let alive = true;
    const paint = (data: ImageData | null) => {
      if (!alive) return;
      base.current = data;
      if (layer.ctx) rasterize(layer.ctx, data, [], rgb, width, height);
    };
    if (maskUrl === null) paint(null);
    else void loadBase(store, maskUrl, rgb, width, height).then(paint);
    return () => {
      alive = false;
    };
  }, [active, dirty, maskUrl, store, rgb, width, height, layer]);

  const redraw = useCallback(
    (strokes: Stroke[]) => {
      if (layer.ctx) rasterize(layer.ctx, base.current, strokes, rgb, width, height);
    },
    [layer, rgb, width, height],
  );

  const start = useCallback(
    (mode: BrushMode, x: number, y: number) => {
      if (!active || busy) return;
      const s: Stroke = { mode, size: draft.size, points: [[x, y]] };
      stroke.current = s;
      setLive(true);
      setError(null);
      if (layer.ctx) drawStroke(layer.ctx, s, 0, rgb);
    },
    [active, busy, draft.size, layer, rgb],
  );
  const extend = useCallback(
    (x: number, y: number) => {
      const s = stroke.current;
      if (s === null || !extendStroke(s, x, y)) return;
      if (layer.ctx) drawStroke(layer.ctx, s, s.points.length - 1, rgb);
    },
    [layer, rgb],
  );
  const end = useCallback(() => {
    const s = stroke.current;
    stroke.current = null;
    setLive(false);
    if (s !== null) setDraft((d) => brushReducer(d, { type: "commit", stroke: s }));
  }, []);

  const undo = useCallback(() => {
    setDraft((d) => {
      const next = brushReducer(d, { type: "undo" });
      if (next !== d) redraw(next.strokes);
      return next;
    });
  }, [redraw]);
  const discard = useCallback(() => {
    stroke.current = null;
    setLive(false);
    setError(null);
    setDraft((d) => {
      const next = brushReducer(d, { type: "clear" });
      redraw([]);
      return next;
    });
  }, [redraw]);
  const setSize = useCallback((size: number) => setDraft((d) => brushReducer(d, { type: "size", size })), []);
  const grow = useCallback((dir: 1 | -1) => setDraft((d) => brushReducer(d, { type: "grow", dir })), []);

  const save = useCallback(async () => {
    if (!dirty || busy) return;
    const encoded = encodeLayer(layer.canvas);
    if (encoded === null) {
      setError("This browser cannot export the painted mask.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const detail = await api.setMask(clipId, frame, encoded.pngB64, { classId: needsClass && classId !== null ? classId : undefined });
      apply(detail, true);
      setDraft((d) => emptyBrush(d.size));
    } catch (e: unknown) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }, [dirty, busy, layer, clipId, frame, needsClass, classId, apply]);

  // Leaving brush mode with nothing painted drops the base so the next activation re-fetches it.
  useEffect(() => {
    if (!active) {
      stroke.current = null;
      setLive(false);
      setDraft((d) => brushReducer(d, { type: "clear" }));
    }
  }, [active]);

  return { draft, dirty, layer: layer.canvas, busy, error, setSize, grow, start, extend, end, undo, discard, save };
}
```

Notes for the implementer: `redraw` inside `setDraft` updaters is a side effect inside a reducer callback; React may call updaters twice in StrictMode, and `rasterize` is idempotent, so that is acceptable, but if you prefer, compute `next` outside and call `setDraft(next)` (the `draft` value from the closure is current for user-initiated calls). `setError` on a 409 shows the server's `detail` text (`ApiError.message`), which already reads "job … is still running"; the Label page passes that through unchanged.

- [ ] **Step 5: Implement `BrushPanel.tsx`**

```tsx
import type { Brush } from "../hooks/useBrush";
import { MAX_SIZE, MIN_SIZE } from "../lib/brush";

type Props = { brush: Brush; frame: number; disabledReason: string | null };

export function BrushPanel({ brush, frame, disabledReason }: Props) {
  const { draft, dirty, busy, error } = brush;
  const disabled = disabledReason !== null;
  const n = draft.strokes.length;
  const status = busy
    ? "Saving…"
    : !dirty
      ? `No strokes yet on frame ${frame}.`
      : `${n} ${n === 1 ? "stroke" : "strokes"} on frame ${frame}, not saved.`;
  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Brush</h2>
      {disabled ? (
        <p className="note text-ink">{disabledReason}</p>
      ) : (
        <p className="note">Left-drag to paint, right-drag to erase. The brush edits this frame's mask; Save makes the frame a keyframe.</p>
      )}
      <label className="flex items-center gap-2 text-xs">
        <span className="w-16">Brush size</span>
        <input aria-label="Brush size" type="range" min={MIN_SIZE} max={MAX_SIZE} value={draft.size} disabled={disabled}
          className="flex-1" onChange={(e) => brush.setSize(Number(e.target.value))} />
        <span className="w-12 text-right text-dim">{draft.size} px</span>
      </label>
      <p className="note">
        <kbd aria-hidden>-</kbd> <kbd aria-hidden>=</kbd> shrink and grow the brush.
      </p>
      <p className="note text-ink">{status}</p>
      {dirty && !busy && <p className="note">Save or discard the brush strokes on frame {frame} first to do anything else.</p>}
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
      <div className="flex gap-1">
        <button type="button" disabled={!dirty || busy || disabled} aria-keyshortcuts="Enter"
          className={`btn flex-1 ${dirty && !busy ? "btn-on" : ""}`} onClick={() => void brush.save()}>
          Save <kbd aria-hidden>Enter</kbd>
        </button>
        <button type="button" disabled={!dirty || busy} aria-keyshortcuts="Escape" className="btn" onClick={brush.discard}>
          Discard <kbd aria-hidden>Esc</kbd>
        </button>
        <button type="button" disabled={!dirty || busy} aria-keyshortcuts="Z" className="btn" onClick={brush.undo}>
          Undo stroke <kbd aria-hidden>Z</kbd>
        </button>
      </div>
    </section>
  );
}
```

- [ ] **Step 6: Run the tests, tsc and lint**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npx tsc --noEmit -p .'`
Expected: all pass, tsc silent.

- [ ] **Step 7: Commit**

```bash
git add web/src/lib/brushCanvas.ts web/src/hooks/useBrush.ts web/src/components/BrushPanel.tsx web/src/components/BrushPanel.test.tsx
git commit -m "useBrush hook, brush canvas helpers and the Brush panel

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

---

### Task 4: Web — brush input on the canvas, the Brush tool, and the Label page (mode, locks, keys)

**Files:**
- Modify: `web/src/components/FrameCanvas.tsx`, `web/src/components/ToolRail.tsx` (+ test), `web/src/components/Transport.tsx`, `web/src/components/CleanupPanel.tsx`, `web/src/pages/Label.tsx`
- Test: `web/src/pages/Label.test.tsx`, `web/src/components/ToolRail.test.tsx`

**Interfaces:**
- Consumes: `useBrush` / `Brush` (Task 3), `BrushPanel` (Task 3), `KeyAction.brushSize` (Task 2), `MASK_ALPHA` from `frameCache`.
- Produces: `FrameCanvas` prop `brush: { size: number; layer: HTMLCanvasElement | null; onStart: (mode: BrushMode, x: number, y: number) => void; onExtend: (x: number, y: number) => void; onEnd: () => void } | null`; `ToolRail` props `{ tool: RailTool; disabled: boolean; brushDisabled: boolean; locked: boolean; onTool: (tool: RailTool) => void; onFit }` with `export type RailTool = Tool | "brush"`; `Transport` prop `disabled?: boolean` (disables the step and play buttons, not the overlay toggle); `CleanupPanel` prop `lockReason: string | null` (disables Clear, Delete/Restore and Prune and shows the sentence).

- [ ] **Step 1: Write the failing tests**

`web/src/components/ToolRail.test.tsx`: update the existing renders to pass `brushDisabled={false} locked={false}` and add:

```tsx
  it("offers the Brush tool, which can be disabled on its own, and locks the other tools while painting", () => {
    const onTool = vi.fn();
    const { rerender } = render(<ToolRail tool="clicks" disabled={false} brushDisabled onTool={onTool} onFit={() => {}} locked={false} />);
    expect((screen.getByRole("button", { name: "Brush" }) as HTMLButtonElement).disabled).toBe(true);
    rerender(<ToolRail tool="clicks" disabled={false} brushDisabled={false} onTool={onTool} onFit={() => {}} locked={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Brush" }));
    expect(onTool).toHaveBeenCalledWith("brush");
    rerender(<ToolRail tool="brush" disabled={false} brushDisabled={false} onTool={onTool} onFit={() => {}} locked />);
    expect(screen.getByRole("button", { name: "Brush" }).getAttribute("aria-pressed")).toBe("true");
    for (const name of ["Click", "Box", "Text"]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
    expect((screen.getByRole("button", { name: "Fit" }) as HTMLButtonElement).disabled).toBe(false);
  });
```

`web/src/pages/Label.test.tsx`: add `setMask: vi.fn()` to the mocked `api` object, and mock the canvas helpers at the top (after the `vi.mock("../api/rest", …)` block):

```tsx
vi.mock("../lib/brushCanvas", () => ({
  createLayer: () => ({ canvas: document.createElement("canvas"), ctx: null }),
  loadBase: async () => null,
  encodeLayer: () => ({ pngB64: "UE5H", any: true }),
}));
```

Then add a describe block:

```tsx
describe("Label page (brush)", () => {
  const paintOnce = () => {
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { button: 0, clientX: 40, clientY: 40 });
    fireEvent.mouseMove(canvas, { button: 0, clientX: 60, clientY: 60 });
    fireEvent.mouseUp(canvas, { button: 0, clientX: 60, clientY: 60 });
  };

  it("switches to the Brush panel, locks the page while strokes are unsaved, and saves the mask", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.setMask).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Brush" }));
    expect(screen.getByRole("heading", { name: "Brush" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Prompt" })).toBeNull();
    expect(screen.getByText("No strokes yet on frame 0.")).toBeTruthy();

    paintOnce();
    expect(screen.getByText("1 stroke on frame 0, not saved.")).toBeTruthy();
    const lock = /Save or discard the brush strokes on frame 0 first/;
    expect(screen.getAllByText(lock).length).toBeGreaterThan(0);
    expect((screen.getByRole("button", { name: "Next frame" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Clear masks in frame 0" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Click" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy(); // the step key is ignored while locked
    fireEvent.keyDown(window, { key: "=" });
    expect((screen.getByLabelText("Brush size") as HTMLInputElement).value).toBe("29"); // 24 * 1.2

    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(api.setMask).toHaveBeenCalledWith("c1", 0, "UE5H", { classId: undefined }));
    expect(await screen.findByText("No strokes yet on frame 0.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Next frame" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("discards with Escape and undoes with Z; Brush is unavailable on a deleted frame", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Brush" }));
    paintOnce();
    paintOnce();
    expect(screen.getByText("2 strokes on frame 0, not saved.")).toBeTruthy();
    fireEvent.keyDown(window, { key: "z" });
    expect(screen.getByText("1 stroke on frame 0, not saved.")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByText("No strokes yet on frame 0.")).toBeTruthy();
    for (let i = 0; i < 3; i++) fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    expect(await screen.findByText("frame 30 / 39")).toBeTruthy(); // deleted in the fixture
    expect(screen.getByText("This frame is deleted. Restore it to paint on it.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Brush" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
```

The fixture's `detail.states` marks frame 30 deleted; the first Brush click happens on frame 0 (a keyframe with a mask, so `maskUrl` is non-null and `loadBase` is mocked to resolve null). If `needsClass` ends up true for the fixture (its object exists, so it should be false), the `classId` expectation changes accordingly; assert what the page really sends.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/components/ToolRail.test.tsx src/pages/Label.test.tsx'`
Expected: the new ToolRail test fails (no Brush button); the brush describe fails at the Brush button.

- [ ] **Step 3: ToolRail, Transport, CleanupPanel**

`ToolRail.tsx`: `export type RailTool = Tool | "brush";` change `Props` to `{ tool: RailTool; disabled: boolean; brushDisabled: boolean; locked: boolean; onTool: (tool: RailTool) => void; onFit: () => void }`; append to `TOOLS` (typed `[RailTool, string, ReactNode][]`):

```tsx
  ["brush", "Brush", icon(<><path d="M3 14.5c2.5 0 3-1.2 3.4-2.6.4-1.3 1.9-1.6 2.7-.8s.5 2.3-.8 2.7C6.9 14.2 5.7 14.5 3 14.5z" /><path d="M8.6 10.6 14.5 3.5" /></>)],
```

and compute per button: `const off = locked ? id !== tool : id === "brush" ? brushDisabled : disabled;` used as `disabled={off}` and for the pressed styling (`tool === id && !off`). Update the doc comment: the rail holds the prompt tools and the brush.

`Transport.tsx`: add `disabled?: boolean` to `Props`, default false, and `disabled={disabled}` on the first/previous/play/next/last buttons (not the overlay checkbox). Read the file first; keep its markup.

`CleanupPanel.tsx`: add `lockReason: string | null` to `Props`. Clear button disabled when `lockReason !== null` too; the `<p className="note">` under it shows `lockReason ?? clearBlockedReason ?? outcome`; Delete/Restore and Prune buttons get `disabled={lockReason !== null}`.

- [ ] **Step 4: FrameCanvas brush input**

In `FrameCanvas.tsx`:
- import `type BrushMode` from `../lib/brush`; add to `Props`:

```ts
  /** brush mode: left/right drags paint/erase on `brush.layer`, which is drawn in place of the stored mask */
  brush: { size: number; layer: HTMLCanvasElement | null; onStart: (mode: BrushMode, x: number, y: number) => void; onExtend: (x: number, y: number) => void; onEnd: () => void } | null;
```
- a `pointer = useRef<[number, number] | null>(null)` (screen position for the brush circle).
- in `draw()`, where the stored mask is drawn: if `p.brush !== null`, draw `p.brush.layer` (when not null) with `ctx.globalAlpha = MASK_ALPHA` then reset to 1, instead of `assets.current.mask`; the preview candidates branch is unreachable in brush mode (candidates are empty) but leave the order: candidates, then brush layer, then stored mask. After `ctx.restore()`, if `p.brush !== null && pointer.current !== null`, draw the footprint: a circle of radius `p.brush.size / 2 * v.scale` at the pointer with a 1 px white stroke and a 1 px black stroke just outside it (two `arc` + `stroke` calls) so it reads on any background.
- `onMouseDown`: after recording `drag`, if `brush !== null && (e.button === 0 || e.button === 2)`: `brush.onStart(e.button === 2 ? "erase" : "paint", Math.round(ix), Math.round(iy)); draw();` and mark `d.painting = true` (add `painting?: boolean` to `Drag`). Do not clamp: strokes near the edge may run past it; the layer clips.
- `onMouseMove`: always update `pointer.current = [sx, sy]` and call `draw()` when `brush !== null` (the circle follows the pointer even without a drag); if `d?.painting`: `brush.onExtend(Math.round(ix), Math.round(iy))` using the current view for the conversion, then `draw()`; skip the box logic while painting.
- `endDrag(commit)`: if `d.painting`: `brush.onEnd(); draw(); return;` before the box/click logic. On `onMouseLeave` also set `pointer.current = null`.
- cursor style: `brush !== null ? "none" : tool === null ? "default" : "crosshair"`.
- The `useEffect(draw)` on every render already picks up a changed `brush.layer`; nothing else needed.

- [ ] **Step 5: Label page**

In `Label.tsx`:
- state `const [mode, setMode] = useState<"prompt" | "brush">("prompt");`
- reasons: keep `disabledReason` for prompting; add

```ts
  const brushReason = running
    ? "A propagation is running. Stop it to paint."
    : state === "deleted"
      ? "This frame is deleted. Restore it to paint on it."
      : playing
        ? "Pause playback to paint."
        : null;
```
- the hook, after `prompting`:

```ts
  const brush = useBrush({
    clipId, frame, width: detail?.clip.width ?? 1, height: detail?.clip.height ?? 1, rgb: hexToRgb(color), store,
    maskUrl: hasMask(frame) ? maskUrl(clipId, frame, maskVersion(frame)) : null,
    active: mode === "brush" && brushReason === null && detail !== null,
    classId, needsClass: object === null, apply,
  });
  const lockReason = brush.dirty ? `Save or discard the brush strokes on frame ${frame} first.` : null;
```
  (`color`, `hasMask`, `maskVersion` are computed later in the file today; move those three lines up above the hook. `hexToRgb` comes from `../lib/mask`; `apply` is already destructured from `clip`.)
- `usePrompting`'s `enabled` becomes `disabledReason === null && detail !== null && mode === "prompt"`.
- tool switch: `const onTool = (t: RailTool) => { if (t === "brush") { prompting.clear(); setMode("brush"); } else { setMode("prompt"); dispatch({ type: "tool", tool: t }); } };` and `<ToolRail tool={mode === "brush" ? "brush" : draft.tool} disabled={disabledReason !== null} brushDisabled={brushReason !== null || detail === null} locked={lockReason !== null} onTool={onTool} onFit=… />`.
- keyboard: at the top of the handler, `if (lockReason !== null && !["accept", "clear", "undo", "brushSize", "overlay"].includes(action.type)) return;` then in the switch, `accept` / `clear` / `undo` route to `brush.save()` / `brush.discard()` / `brush.undo()` when `mode === "brush"`, otherwise to prompting as today; add `case "brushSize": if (mode === "brush") brush.grow(action.dir); break;`.
- propagate: `blockedReason={lockReason ?? propagateBlocked}`; cleanup: `lockReason={lockReason}`; transport: `disabled={lockReason !== null}` and `onStep`/`onTogglePlay` are no-ops while locked (the button is disabled, the key is filtered; keep `step` itself usable by the timeline? No: also guard `seek` so a timeline click while locked does nothing: `onSeek={lockReason === null ? seek : () => {}}`).
- canvas: `tool={mode === "prompt" && disabledReason === null ? draft.tool : null}` and `brush={mode === "brush" && brushReason === null ? { size: brush.draft.size, layer: brush.layer, onStart: brush.start, onExtend: brush.extend, onEnd: brush.end } : null}`.
- sidebar: `{mode === "brush" ? <BrushPanel brush={brush} frame={frame} disabledReason={brushReason} /> : <PromptPanel prompting={prompting} disabledReason={disabledReason} />}`.
- `propagateBlocked` for the engine-not-loaded case stays; painting works without the engine.

- [ ] **Step 6: Run everything**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npx tsc --noEmit -p . && npm run build'`
Expected: all pass (previous count + 4), tsc silent, build OK.

- [ ] **Step 7: Commit**

```bash
git add web/src/components/FrameCanvas.tsx web/src/components/ToolRail.tsx web/src/components/ToolRail.test.tsx web/src/components/Transport.tsx web/src/components/CleanupPanel.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git commit -m "Brush tool: paint and erase the mask on the canvas, locked until saved or discarded

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```

---

### Task 5: Docs and a live check

**Files:**
- Modify: `README.md` (keyboard map, QA list), `docs/superpowers/specs/2026-09-20-vision-sampler-design.md` (§4 API table, §5.2), `docs/superpowers/plans/2026-09-20-vsp-roadmap.md` (append a section)

- [ ] **Step 1: README**

In the Web GUI keyboard map add `-` / `=`: "shrink / grow the brush (Brush tool)", and note that `Enter`, `Esc` and `Z` act on the brush while it is selected (Save, Discard, Undo stroke). Append QA item 14:

```markdown
14. Brush: pick the Brush tool on a masked frame, paint a missing corner (left-drag) and erase a spill (right-drag);
    the transport, the other tools, Propagate and Cleanup are disabled until you Save (Enter) or Discard (Esc). After
    Save the frame shows a keyframe diamond; propagate from it and check the run stops at the next keyframe. Erase the
    whole mask on another frame and Save: the frame becomes empty. The brush works with the engine unloaded.
```

- [ ] **Step 2: Spec**

In `docs/superpowers/specs/2026-09-20-vision-sampler-design.md`: add to the §4 API table a row `POST /api/clips/{c}/mask {frame, mask_png_b64, class_id?}` — "Store a hand-painted mask; the frame becomes a `paint` keyframe (empty mask: remove mask and keyframe). 409 while the GPU is leased." Add to §5.2 one paragraph: "**Painted masks.** The brush tool edits a frame's mask directly in the browser and uploads the whole mask; such a frame is a keyframe with prompt type `paint`. See `2026-09-22-brush-mask-editing-design.md`." Update §7's tool list to include Brush if it enumerates the tools.

- [ ] **Step 3: Roadmap**

Append:

```markdown

## Brush mask editing (done, 2026-09-22, branch `brush-mask-editing`)

Spec: `docs/superpowers/specs/2026-09-22-brush-mask-editing-design.md`. A Brush tool paints and erases a frame's mask on
the canvas (strokes rasterised incrementally on an offscreen layer in the class colour); Save uploads the whole mask
as a PNG to `POST /api/clips/{c}/mask`, which stores it as a `paint` keyframe (an empty mask removes mask and
keyframe). Unsaved strokes lock navigation, the other tools, propagation and cleanup until saved or discarded.

Deferred: brush hardness/feathering, a lasso or polygon tool, pen pressure and touch, painting across several frames,
and an "unsaved changes" prompt on leaving the page (strokes are simply dropped).
```

- [ ] **Step 4: Live check**

Start the server from the repo root (`nohup uv run vsp serve > <scratchpad>/serve-brush.log 2>&1 &`, wait for `"loaded":true` on `/api/health`; the GPU is not needed for the brush, but the demo project expects the engine). Then with `uv run python` and `httpx.Client(base_url="http://127.0.0.1:28000", trust_env=False)`: fetch the demo clip, build a mask the size of the clip with a rectangle, `encode_mask_png` it, POST it to `/api/clips/{id}/mask` on a frame in the middle, and check the response `states[frame] == "keyframe"` and that `GET /api/clips/{id}/masks/{frame}` returns a PNG whose decoded pixels equal the rectangle. Then POST an all-black PNG for the same frame and check the state is `"empty"`. Put the three results in your report. Leave the server running.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/specs/2026-09-20-vision-sampler-design.md docs/superpowers/plans/2026-09-20-vsp-roadmap.md
git commit -m "Document the brush tool and the mask route

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL"
```
