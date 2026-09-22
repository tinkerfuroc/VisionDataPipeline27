# Brush mask editing — design

Date: 2026-09-22. Status: approved in chat, spec written for the implementation plan.
Amends the main design (`2026-09-20-vision-sampler-design.md`): §5.2 gains a fourth way for a mask to come into
existence, §4's API table gains one route, §7's tool rail gains one tool.

## Goal

Let the user paint and erase a frame's mask directly, for the cases the model gets wrong or cannot be prompted
into: a missing corner, a spill onto a neighbouring part, a thin structure. A hand-painted frame is the strongest
ground truth in the clip, so it becomes a keyframe: propagation never overwrites it and can be seeded from it.

## Decisions taken with the user

- A hand-painted frame **is a keyframe**, recorded with a prompt of type `paint`.
- Brush edits happen on a client-side copy of the mask; **Save uploads the whole finished mask** as a PNG to one new
  route. Strokes are never sent to the server (rejected: rasterising on the server would duplicate the brush geometry;
  turning strokes into SAM prompts would not give the pixels the user painted).
- Erasing everything and saving **removes the mask and the keyframe** (the "deletion" half of the request). The
  frame becomes empty.
- While a frame has unsaved strokes, navigation, playback, the other tools and propagation are **locked** until the
  strokes are saved or discarded. One draft at a time, never lost to a stray arrow key.

## User-facing behaviour

### Tool rail and panel

- A fourth tool, **Brush**, after Text. Disabled under the same conditions as the prompt tools (engine irrelevant: the
  brush does not need the engine, but a running propagation, playback and a deleted frame still disable it; see
  "Locks" for the exact rule).
- With Brush active the sidebar shows a **Brush panel** in place of the Prompt panel:
  - a hint: "Left-drag to paint, right-drag to erase. The brush edits this frame's mask; Save makes the frame a
    keyframe."
  - **Brush size**: a range input, 2–200 image pixels, default 24, with the number shown. Keys `-` and `=` shrink and
    grow it by 20 % (min 2, max 200). `[` and `]` stay with the cleanup range.
  - a status line: "No strokes yet on frame 12." / "3 strokes on frame 12, not saved." / "Saving…"
  - buttons, in the prompt panel's shape: **Save** (`Enter`), **Discard** (`Esc`), **Undo stroke** (`Z`). Save is
    enabled only with unsaved strokes; Discard and Undo likewise.
  - the class select stays where it is (Object panel); the first save on an unlabeled clip uses it, as Accept does.
- On the canvas, with Brush active, a **circle outline** follows the pointer showing the brush footprint at the
  current zoom (radius = size / 2 in image pixels, scaled to screen). Cursor: `none` inside the canvas (the circle is
  the cursor); default outside.

### Painting

- The draft starts from the frame's stored mask, or empty when the frame has none. Painting is allowed on any frame
  that is not deleted.
- Left-drag paints, right-drag erases. A stroke is a round-capped, round-joined polyline of image-space points with
  the stroke's size; a click without movement is a single dot.
- Middle-drag still pans and the wheel still zooms while painting (zooming mid-stroke does not distort it: points
  are recorded in image space).
- **Undo stroke** removes the last stroke. **Discard** drops every stroke and returns to the stored mask.
- Rendering: strokes are rasterised on an offscreen canvas the size of the image, in the class colour on
  transparent: paint = `source-over`, erase = `destination-out`. The offscreen canvas is drawn over the frame at the
  usual overlay alpha (`MASK_ALPHA`) in place of the stored mask layer, so painting costs one segment draw per mouse
  move. Undo and Discard re-rasterise from the base mask.
- On Save the offscreen canvas is read back once; a pixel is in the mask iff its alpha is above 127. The mask is
  encoded as a PNG (white on black is fine; the server binarises with `> 127` after converting to grey) and sent as
  base64.

### Locks

Two flags drive every disabled state around the brush:

- `brushActive`: the Brush tool is selected.
- `dirty`: the draft has at least one stroke.

Rules:
- The Brush tool is available when: the clip is loaded, no job is running on this page (`running` false), playback is
  off, and the current frame is not deleted. The engine may be unloaded.
- While `dirty`: the transport buttons and the step/keyframe/next-empty/play keys are ignored, the tool rail's other
  tools are disabled, propagation's buttons are disabled with the reason "Save or discard the brush strokes on frame
  12 first.", the Cleanup panel's buttons are disabled with the same reason, and the same sentence appears in the
  Brush panel's status line. Fit, zoom and pan keep working.
- Switching away from the Brush tool with no strokes simply clears the draft (like switching prompt tools).
- Leaving the page (route change) with `dirty` strokes drops them; the browser's "unsaved changes" prompt is not used.

### Save

`POST /api/clips/{clip_id}/mask` with body `{frame: int, mask_png_b64: str, class_id?: int}`.

Server (`vsp.labeling.paint.set_mask(store, clip_id, frame, mask, *, class_id=None) -> Clip`):
- 400 if the frame is out of range or deleted ("frame 12 is deleted; restore it before labeling").
- 400 if the PNG cannot be decoded or its size is not `(height, width)` of the clip ("mask is 640×480, the clip is
  1920×1080").
- First save on an unlabeled clip: `class_id` required (same message as accept: "class_id is required for the first
  accept on a clip"), object created with `create_object`; an unknown class is 400. A `class_id` on a later save
  updates the object's class, as accept does.
- Non-empty mask: write it; upsert a `Keyframe(frame_idx=frame, prompt=Prompt(type="paint", payload={}), created_at)`;
  remove `frame` from `propagated_ranges` (`subtract_range(ranges, frame, frame)`).
- Empty mask: delete the mask file (if any), drop the keyframe entry for that frame (if any), remove `frame` from
  `propagated_ranges`. The object stays.
- All under `store.lock(clip_id)`; the route runs under the same `gpu_lease` as clear-range, so it answers 409 with
  the running `job_id` while a propagation or export holds it, and never races a propagation's writes.
- Response: `clip_detail` (the same shape accept returns). The client applies it with `masksChanged` for that frame
  so the overlay re-fetches the stored mask and the timeline shows the diamond.

The store's `Prompt.type` literal and the web `PromptType` gain `"paint"`. `PromptSpec` (the request model for prompt
and accept) does **not** gain `paint`: painting is not a prompt the engine can run.

### Everything else unchanged

- Propagation seeds from a painted keyframe (`check_seed` reads the mask) and stops at it like any keyframe.
- Cleanup removes it like any keyframe (mask and entry).
- Export treats the mask like any other.
- Prompt tools still work on a painted frame; Accept replaces the mask and the keyframe's prompt becomes the new
  prompt (existing upsert behaviour).
- ClipStats counts it as a keyframe. Timeline draws the diamond.

## Components

Server:
- `server/vsp/labeling/paint.py` — `decode_mask_png(b64: str) -> np.ndarray` (bool HxW, raises BadRequest) and
  `set_mask(...)` as above.
- `server/vsp/api/labeling.py` — `POST /api/clips/{clip_id}/mask` (`MaskBody`), registered in the same router.
- `server/vsp/store/models.py` — `Prompt.type` literal gains `"paint"`.

Web:
- `web/src/lib/brush.ts` — pure draft state: `BrushDraft {size, strokes: Stroke[]}`, `Stroke {mode: "paint"|"erase",
  size, points: [number, number][]}`, reducer actions `start(mode, x, y)`, `extend(x, y)`, `undo`, `clear`,
  `size(n)`, `grow` / `shrink`; and `rasterize(ctx, base: HTMLImageElement | null, strokes, rgb, width, height)` plus
  `appendSegment(ctx, stroke, fromIndex, rgb)` that draw with an injected 2D context so tests can record calls.
- `web/src/hooks/useBrush.ts` — owns the draft, the offscreen canvas, the base mask image (fetched through the frame
  cache's image loader from `maskUrl`), `dirty`, `save()` (encode + `api.setMask` + `apply(detail, {masksChanged})`),
  `discard()`, errors. Re-bases when the frame or the stored mask version changes and the draft is clean.
- `web/src/components/BrushPanel.tsx` — the panel above.
- `web/src/components/FrameCanvas.tsx` — new props `brush: {size, layer: HTMLCanvasElement | null, onStart, onExtend,
  onEnd} | null`; when set, left/right drags paint/erase instead of prompting, the brush circle follows the pointer,
  and `brush.layer` is drawn in place of the stored mask layer.
- `web/src/components/ToolRail.tsx` — shows `RailTool = Tool | "brush"`. The prompt draft's `Tool` union in
  `promptState.ts` is unchanged (a prompt draft can never be in brush mode); the page holds `mode: "prompt" | "brush"`
  and derives the rail's selection from it.
- `web/src/pages/Label.tsx` — mode state, locks, keyboard routing (`Enter`/`Esc`/`Z`/`-`/`=` go to the brush while it
  is active), panel switch.
- `web/src/api/rest.ts` — `setMask(clipId, frame, maskPngB64, opts: {classId?})`; `types.ts` — `PromptType` gains
  `"paint"`.
- `web/src/hooks/useKeyboard.ts` — actions `{type: "brushSize", delta: 1 | -1}` for `=` / `-`.

Docs: README keyboard map (`-`/`=`), QA item 14 (paint a correction, save, propagate from it, erase all and save),
spec §4 API table row and §5.2 paragraph, roadmap entry.

## Testing

Server (`server/tests/test_paint.py`, `test_api_labeling.py`): decode round-trip with `encode_mask_png`; wrong size;
undecodable input; first save needs class, creates the object and a `paint` keyframe; later save updates the mask and
keeps one keyframe entry; the frame leaves `propagated_ranges`; empty mask removes mask and keyframe and leaves the
object; deleted frame 400; 409 while a propagation job holds the GPU lease; propagation seeds from a painted frame and
stops at it (one test in `test_propagate.py`).

Web: `brush.test.ts` (reducer: start/extend/undo/clear/size clamping; rasteriser: with a recording context, a paint
stroke sets `source-over` and the class colour, an erase stroke `destination-out`, round caps/joins, line width =
size, a one-point stroke draws a dot; `appendSegment` draws only the new segment); `BrushPanel.test.tsx` (size
input and keys, status line, buttons enabled by `dirty`, Save/Discard/Undo wired); `Label.test.tsx` (Brush tool
switches the panel; painting locks transport, tools, propagate and cleanup with the sentence; Save calls
`api.setMask` with the frame and a data-URL-stripped base64 and applies the detail; Discard unlocks); `useKeyboard`
(`-`/`=`).

## Out of scope

Multi-frame painting, brush hardness or feathering, a lasso/polygon tool, keyboard-driven brush position, touch or
pen pressure, and undo of a *saved* paint (Cleanup already removes a keyframe).
