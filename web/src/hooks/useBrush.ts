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
  /** true while the frame's stored mask is (re)loading into the layer; start() refuses to paint until it clears. */
  loading: boolean;
  /** bumped each time the base finishes loading into the layer, so a consumer (FrameCanvas via Label) can re-render
   * and redraw even though `layer` is the same canvas instance. */
  baseVersion: number;
  error: string | null;
  /** true when save() will actually do something: dirty, not busy, no stroke in progress, and (if this is the
   * clip's first mask) a class is chosen. */
  canSave: boolean;
  /** why canSave is false, when the reason is something the panel should tell the user; null otherwise. */
  saveBlockedReason: string | null;
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
  const [loading, setLoading] = useState(false);
  const [baseVersion, setBaseVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const layer = useMemo(() => createLayer(width, height), [width, height]);
  const base = useRef<ImageData | null>(null);
  const stroke = useRef<Stroke | null>(null);
  const dirty = draft.strokes.length > 0 || live;

  // Start from the frame's stored mask whenever the frame or its mask changes while nothing is painted. The base
  // loads asynchronously (loadBase awaits an image fetch/decode), so between the frame/mask changing and the load
  // resolving the layer must not keep showing the previous frame's mask, and start() must refuse to paint on it:
  // both would either look wrong or corrupt the new frame's mask with strokes rasterised over stale pixels.
  useEffect(() => {
    if (!active || dirty) {
      setLoading(false); // a load abandoned by deactivation must not leave the next activation refusing to paint
      return;
    }
    let alive = true;
    base.current = null;
    setLoading(true);
    if (layer.ctx) rasterize(layer.ctx, null, [], rgb, width, height); // clear synchronously, before the await
    const paint = (data: ImageData | null) => {
      if (!alive) return;
      base.current = data;
      if (layer.ctx) rasterize(layer.ctx, data, [], rgb, width, height);
      setLoading(false);
      setBaseVersion((v) => v + 1); // notifies FrameCanvas (via the brush prop) to redraw even though layer is unchanged
    };
    if (maskUrl === null) paint(null);
    else void loadBase(store, maskUrl, rgb, width, height).then(paint, () => paint(null)); // an unreadable mask: start empty
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
      // While loading, the layer has just been cleared (or holds a stale base mid-swap) and `base.current` is
      // null: painting now would draw over nothing and the stroke would be lost or corrupt the base once it lands.
      if (!active || busy || loading) return;
      const s: Stroke = { mode, size: draft.size, points: [[x, y]] };
      stroke.current = s;
      setLive(true);
      setError(null);
      if (layer.ctx) drawStroke(layer.ctx, s, 0, rgb);
    },
    [active, busy, loading, draft.size, layer, rgb],
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

  // undo/discard are user-initiated (a button click, never a render-in-progress path), so `draft` from the
  // closure is current: computing `next` here keeps the redraw side effect out of the setDraft updater.
  // undo (and setSize/grow, for symmetry) must not act while a stroke is in progress: undo's redraw rasterises
  // `draft.strokes` onto the layer, discarding the stroke a mouse-move already drew there but has not committed.
  const undo = useCallback(() => {
    if (stroke.current !== null) return;
    const next = brushReducer(draft, { type: "undo" });
    if (next !== draft) redraw(next.strokes);
    setDraft(next);
  }, [draft, redraw]);
  const discard = useCallback(() => {
    stroke.current = null;
    setLive(false);
    setError(null);
    const next = brushReducer(draft, { type: "clear" });
    redraw([]);
    setDraft(next);
  }, [draft, redraw]);
  const setSize = useCallback((size: number) => {
    if (stroke.current !== null) return;
    setDraft((d) => brushReducer(d, { type: "size", size }));
  }, []);
  const grow = useCallback((dir: 1 | -1) => {
    if (stroke.current !== null) return;
    setDraft((d) => brushReducer(d, { type: "grow", dir }));
  }, []);

  // Mirrors usePrompting's canAccept: the first save on an unlabeled clip needs a class, same as accept.
  const needsClassNow = needsClass && classId === null;
  const saveBlockedReason = dirty && needsClassNow ? "Choose a class before saving the first mask." : null;
  const canSave = dirty && !busy && !live && !needsClassNow;

  const save = useCallback(async () => {
    // A stroke in progress (stroke.current !== null) must not be saved: encoding the layer now and clearing the
    // draft, only to have the in-flight end() commit the whole stroke into the emptied draft, would leave a stuck
    // "not saved" state. Likewise, the first save on an unlabeled clip needs a class (canSave already reflects
    // both, but save() guards itself too so it is safe to call directly).
    if (!dirty || busy || stroke.current !== null || (needsClass && classId === null)) return;
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

  // Leaving brush mode drops the draft: clear only when there is nothing painted (!dirty, i.e. no committed
  // strokes and no stroke in progress). Strokes must never be silently lost: if `active` falls while strokes
  // exist (the tool became unavailable because a job started or the frame was deleted), keep the strokes; the
  // panel keeps offering Discard (and Save once the tool is available again).
  useEffect(() => {
    if (active || dirty) return;
    stroke.current = null;
    setLive(false);
    setDraft((d) => brushReducer(d, { type: "clear" }));
  }, [active, dirty]);

  return {
    draft, dirty, layer: layer.canvas, busy, loading, baseVersion, error, canSave, saveBlockedReason,
    setSize, grow, start, extend, end, undo, discard, save,
  };
}
