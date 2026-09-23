import { useCallback, useEffect, useMemo, useRef, type MouseEvent } from "react";
import type { Box, Candidate } from "../api/types";
import type { BrushMode } from "../lib/brush";
import { MASK_ALPHA, PREVIEW_ALPHA, type FrameStore } from "../lib/frameCache";
import {
  clampPoint, fitView, insideImage, normBox, panBy, pickCandidate, toImage, toScreen, zoomAt, type View,
} from "../lib/geometry";
import { hexToRgb } from "../lib/mask";
import type { Draft, Tool } from "../lib/promptState";

type Props = {
  store: FrameStore;
  /** image size in pixels */
  width: number;
  height: number;
  frameUrl: string;
  /** stored mask of this frame, or null when there is none / the overlay is off */
  maskUrl: string | null;
  /** class colour, hex */
  color: string;
  /** null disables prompting input; zoom and pan always work */
  tool: Tool | null;
  draft: Draft;
  /** preview results; while there are any they replace the stored mask */
  candidates: Candidate[];
  onPoint: (x: number, y: number, label: 0 | 1) => void;
  onBox: (box: Box) => void;
  onPickCandidate: (index: number) => void;
  /** bump to fit the frame to the view again (the Fit button lives in the tool rail) */
  fitSignal?: number;
  /** brush mode: left/right drags paint/erase on `brush.layer`, which is drawn in place of the stored mask.
   * `baseVersion` carries no value the draw effect reads; it only exists so a base-mask load (which mutates
   * `layer`'s pixels in place, not its identity) still triggers `useEffect(draw)` below. */
  brush: {
    size: number; layer: HTMLCanvasElement | null; baseVersion: number;
    onStart: (mode: BrushMode, x: number, y: number) => void; onExtend: (x: number, y: number) => void; onEnd: () => void;
  } | null;
};

const DRAG_PX = 4;
const MAX_SCALE = 40;

/** Decoded preview layers together with the exact candidate list they belong to, so a redraw never mixes
 * one request's layers with another's candidates while new ones are still decoding. */
type LayerState = { owner: Candidate[] | null; items: (HTMLCanvasElement | null)[] };

/** `ix`/`iy`: the mousedown point in image space, fixed for the whole drag so a wheel-zoom mid-drag cannot
 * shift the box's first corner or a click's target. */
type Drag = {
  button: number; sx: number; sy: number; ix: number; iy: number; lastX: number; lastY: number; moved: boolean;
  /** a brush stroke is in progress; the box/click logic is skipped for this drag */
  painting?: boolean;
};

export function FrameCanvas(props: Props) {
  const { store, width, height, frameUrl, maskUrl, color, tool, candidates, onPoint, onBox, onPickCandidate, fitSignal, brush } = props;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = useRef({ w: 0, h: 0 });
  const view = useRef<View>({ scale: 1, tx: 0, ty: 0 });
  const fitScale = useRef(1);
  const assets = useRef<{ frame: HTMLImageElement | null; mask: HTMLCanvasElement | null }>({ frame: null, mask: null });
  const layers = useRef<LayerState>({ owner: null, items: [] });
  const drag = useRef<Drag | null>(null);
  const dragBox = useRef<Box | null>(null);
  const latest = useRef(props);
  latest.current = props;
  const rgb = useMemo(() => hexToRgb(color), [color]);
  /** screen position of the pointer, for the brush footprint circle */
  const pointer = useRef<[number, number] | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const p = latest.current;
    const v = view.current;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.current.w, size.current.h);

    ctx.save();
    ctx.translate(v.tx, v.ty);
    ctx.scale(v.scale, v.scale);
    ctx.imageSmoothingEnabled = v.scale < 3;
    if (assets.current.frame) {
      ctx.drawImage(assets.current.frame, 0, 0, p.width, p.height);
      // No mask or preview layer is drawn when the frame itself failed to load: either would be meaningless
      // floating over a black canvas, and a stale stored mask would be actively misleading.
      if (p.candidates.length > 0) {
        const state = layers.current;
        const current = state.owner === p.candidates;
        // Not current: new previews are still decoding. Only bridge the gap when it's the same single
        // candidate being refined (an extra click on the same object) — anything else would flash a mismatched
        // preview (or, previously, the frame's stored mask) against the wrong candidate list.
        const bridgeOld = !current && state.owner !== null && state.owner.length === 1 && p.candidates.length === 1;
        if (current || bridgeOld) {
          state.items.forEach((layer, i) => {
            if (layer === null) return;
            ctx.globalAlpha = i === p.draft.candidate ? 1 : 0.45;
            ctx.drawImage(layer, 0, 0, p.width, p.height);
          });
          ctx.globalAlpha = 1;
        }
      } else if (p.brush !== null) {
        if (p.brush.layer) {
          ctx.globalAlpha = MASK_ALPHA;
          ctx.drawImage(p.brush.layer, 0, 0, p.width, p.height);
          ctx.globalAlpha = 1;
        }
      } else if (assets.current.mask) {
        ctx.drawImage(assets.current.mask, 0, 0, p.width, p.height);
      }
    }
    ctx.restore();

    if (p.brush !== null && pointer.current !== null) {
      const [px, py] = pointer.current;
      const r = (p.brush.size / 2) * v.scale;
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.strokeStyle = "#000000";
      ctx.beginPath();
      ctx.arc(px, py, r + 1, 0, 2 * Math.PI);
      ctx.stroke();
    }

    if (!assets.current.frame) {
      ctx.fillStyle = "#A3A5AA";
      ctx.font = "14px Barlow, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("This frame could not be loaded", size.current.w / 2, size.current.h / 2);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }

    const box = dragBox.current ?? p.draft.box;
    if (box !== null) {
      const [x1, y1] = toScreen(v, box[0], box[1]);
      const [x2, y2] = toScreen(v, box[2], box[3]);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }
    if (p.candidates.length > 1) {
      ctx.font = "bold 13px sans-serif";
      p.candidates.forEach((c, i) => {
        const [x, y] = toScreen(v, c.bbox[0], c.bbox[1]);
        ctx.fillStyle = i === p.draft.candidate ? "#facc15" : "#e5e5e5";
        ctx.fillRect(x, y - 16, 18, 16);
        ctx.fillStyle = "#000000";
        ctx.fillText(String(i + 1), x + 5, y - 4);
      });
    }
    // Each click is numbered in click order, matching the list in the Prompt panel where it can be removed.
    ctx.font = "bold 12px sans-serif";
    ctx.textBaseline = "middle";
    p.draft.points.forEach(([px, py, label], i) => {
      const [x, y] = toScreen(v, px + 0.5, py + 0.5);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = label === 1 ? "#22c55e" : "#ef4444";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
      const tag = String(i + 1);
      const w = (ctx.measureText(tag)?.width ?? 8) + 6; // jsdom's canvas stub has no measureText result
      // Up and to the right of the dot, pulled back inside the canvas for clicks near its top or right edge.
      const tx = Math.min(x + 7, size.current.w - w);
      const ty = Math.max(y - 15, 0);
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(tx, ty, w, 15);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(tag, tx + 3, ty + 7.5);
    });
    ctx.textBaseline = "alphabetic";
  }, []);

  const refit = useCallback(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (wrap === null || canvas === null) return;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    size.current = { w, h };
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    view.current = fitView(width, height, w, h);
    fitScale.current = view.current.scale;
    draw();
  }, [width, height, draw]);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (wrap === null) return;
    const observer = new ResizeObserver(refit);
    observer.observe(wrap);
    refit();
    return () => observer.disconnect();
  }, [refit]);

  const fitted = useRef(fitSignal);
  useEffect(() => {
    if (fitted.current === fitSignal) return;
    fitted.current = fitSignal;
    refit();
  }, [fitSignal, refit]);

  // Swap the frame and its mask in together so the overlay never shows on the wrong frame.
  useEffect(() => {
    let live = true;
    const mask = maskUrl === null ? Promise.resolve(null) : store.layer(maskUrl, rgb, MASK_ALPHA);
    void Promise.all([store.image(frameUrl), mask]).then(([frame, layer]) => {
      if (!live) return;
      assets.current = { frame, mask: layer };
      draw();
    });
    return () => {
      live = false;
    };
  }, [store, frameUrl, maskUrl, rgb, draw]);

  useEffect(() => {
    let live = true;
    if (candidates.length === 0) {
      layers.current = { owner: null, items: [] };
      draw();
      return;
    }
    const pending = candidates.map((c) => store.preview(`data:image/png;base64,${c.mask_png_b64}`, rgb, PREVIEW_ALPHA));
    void Promise.all(pending).then((items) => {
      if (!live) return;
      layers.current = { owner: candidates, items };
      draw();
    });
    return () => {
      live = false;
    };
  }, [store, candidates, rgb, draw]);

  useEffect(draw); // points, box and candidate selection arrive as props

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault(); // needs a non-passive listener, which React's onWheel is not
      const r = canvas.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.0015);
      view.current = zoomAt(view.current, e.clientX - r.left, e.clientY - r.top, factor, fitScale.current * 0.5, MAX_SCALE);
      draw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [draw]);

  const local = (e: MouseEvent): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  const onMouseDown = (e: MouseEvent) => {
    e.preventDefault(); // no text selection, no middle-click autoscroll
    const [sx, sy] = local(e);
    const [ix, iy] = toImage(view.current, sx, sy);
    const d: Drag = { button: e.button, sx, sy, ix, iy, lastX: sx, lastY: sy, moved: false };
    drag.current = d;
    if (brush !== null && (e.button === 0 || e.button === 2)) {
      const bx = Number.isFinite(ix) ? Math.floor(ix) : 0;
      const by = Number.isFinite(iy) ? Math.floor(iy) : 0;
      brush.onStart(e.button === 2 ? "erase" : "paint", bx, by);
      d.painting = true;
      draw();
    }
  };

  const onMouseMove = (e: MouseEvent) => {
    const d = drag.current;
    const [sx, sy] = local(e);
    if (brush !== null) {
      pointer.current = [sx, sy];
      draw();
    }
    if (d === null) return;
    if (Math.hypot(sx - d.sx, sy - d.sy) > DRAG_PX) d.moved = true;
    if (d.painting) {
      // A mouseup outside the window (alt-tab mid-drag) never reaches us, so the drag looks unfinished; when the
      // pointer comes back with no button held, end it here instead of continuing to paint with a phantom drag.
      if (e.buttons === 0) {
        endDrag(false);
        return;
      }
      const [ix, iy] = toImage(view.current, sx, sy);
      const bx = Number.isFinite(ix) ? Math.floor(ix) : 0;
      const by = Number.isFinite(iy) ? Math.floor(iy) : 0;
      brush?.onExtend(bx, by);
      draw();
      return;
    }
    if (d.button === 1) {
      view.current = panBy(view.current, sx - d.lastX, sy - d.lastY);
      d.lastX = sx;
      d.lastY = sy;
      draw();
    } else if (d.button === 0 && tool === "box" && d.moved) {
      dragBox.current = normBox([d.ix, d.iy], toImage(view.current, sx, sy), width, height);
      draw();
    }
  };

  const endDrag = (commit: boolean) => {
    const d = drag.current;
    drag.current = null;
    if (d?.painting) {
      brush?.onEnd();
      draw();
      return;
    }
    const box = dragBox.current;
    dragBox.current = null;
    if (d === null || d.button === 1) return;
    if (box !== null) {
      if (commit && box[2] - box[0] >= 2 && box[3] - box[1] >= 2) onBox(box);
      draw();
      return;
    }
    if (!commit || d.moved || tool === null) return;
    const [ix, iy] = [d.ix, d.iy];
    if (!insideImage(ix, iy, width, height)) return;
    const [x, y] = clampPoint(ix, iy, width, height);
    if (tool === "text") {
      const index = pickCandidate(candidates.map((c) => c.bbox), x, y);
      if (index !== null) onPickCandidate(index);
      return;
    }
    onPoint(x, y, d.button === 2 ? 0 : 1);
  };

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden bg-well">
      <canvas
        ref={canvasRef}
        aria-label="Frame"
        className="block h-full w-full"
        style={{ cursor: brush !== null ? "none" : tool === null ? "default" : "crosshair" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={() => endDrag(true)}
        onMouseLeave={() => {
          pointer.current = null;
          draw(); // without this, the brush footprint circle lingers at its last position when no drag is in progress
          endDrag(false);
        }}
        onContextMenu={(e) => e.preventDefault()}
      />
    </div>
  );
}
