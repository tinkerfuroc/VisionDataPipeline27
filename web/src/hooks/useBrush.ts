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

  // undo/discard are user-initiated (a button click, never a render-in-progress path), so `draft` from the
  // closure is current: computing `next` here keeps the redraw side effect out of the setDraft updater.
  const undo = useCallback(() => {
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
