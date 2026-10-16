import { useCallback, useEffect, useMemo, useRef, type MouseEvent } from "react";
import type { Box, Candidate, Rect, RegionRange } from "../api/types";
import type { BrushMode } from "../lib/brush";
import { MASK_ALPHA, PREVIEW_ALPHA, compositeKey, type FrameStore, type MaskPart } from "../lib/frameCache";
import {
  clampPoint, fitView, insideImage, normBox, panBy, pickCandidate, toImage, toScreen, zoomAt, type View,
} from "../lib/geometry";
import { hexToRgb, type Outline } from "../lib/mask";
import type { Draft, Tool } from "../lib/promptState";
import { dragRect, hitTest, isExcluded, normRect, type Hit } from "../lib/regions";

type Props = {
  store: FrameStore;
  /** image size in pixels */
  width: number;
  height: number;
  frameUrl: string;
  /** the stored masks to draw on this frame, one part per visible object, active object last (`maskParts`); empty
   * when the overlay is off. They are composited into one cached layer by `store.composite`. */
  masks: MaskPart[];
  /** class colour of the active (or new) object, hex: candidate previews are tinted in it */
  color: string;
  /** how preview boundaries are drawn; keep its identity stable, it keys the preview cache lookups */
  outline: Outline;
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
  /** the current frame's region: outside the crop darkened, blocked boxes grey and hatched. Drawn always. */
  region?: RegionRange | null;
  /** a crop or blocked box (by index in `region.blocks`) to outline brightly, e.g. the one hovered in the panel */
  regionHighlight?: { kind: "crop" } | { kind: "block"; index: number } | null;
  /** when set, left-button drags edit the region: a press on a handle or inside a block or the crop moves/resizes
   * it, on a block's × deletes it, elsewhere draws a new block (or replaces the crop when `draw` is "crop").
   * `onChange` is called once on mouseup. Prompting input (`tool`) is ignored meanwhile; pan and zoom still work. */
  regionEdit?: { draw: "block" | "crop"; onChange: (next: RegionEdit) => void } | null;
  /** called instead of `onPoint` for a positive click in the region's excluded area */
  onExcluded?: () => void;
};

type RegionEdit = { crop: Rect | null; blocks: Rect[] };

const DRAG_PX = 4;
const MAX_SCALE = 40;
/** How long a frame load may hold back the next one before that one starts anyway. */
export const LOAD_STALL_MS = 3000;

/** One frame with its masks, as asked for by the props. */
type LoadRequest = { store: FrameStore; frameUrl: string; masks: MaskPart[] };
/** `busy`: a load is in flight and not yet stalled; `waiting`: the newest request held back meanwhile; `seq`: the
 * last load started; `drawn`: the latest load drawn, so an older one landing late is not drawn over it. */
type LoadState = { busy: boolean; waiting: LoadRequest | null; seq: number; drawn: number; mounted: boolean };

/** Decoded preview layers together with the exact candidate list they belong to, so a redraw never mixes
 * one request's layers with another's candidates while new ones are still decoding. */
type LayerState = { owner: Candidate[] | null; items: (HTMLCanvasElement | null)[] };

/** `ix`/`iy`: the mousedown point in image space, fixed for the whole drag so a wheel-zoom mid-drag cannot
 * shift the box's first corner or a click's target. */
type Drag = {
  button: number; sx: number; sy: number; ix: number; iy: number; lastX: number; lastY: number; moved: boolean;
  /** a brush stroke is in progress; the box/click logic is skipped for this drag */
  painting?: boolean;
  /** a region edit is in progress (state in `regionDrag`); the box/click logic is skipped for this drag */
  region?: boolean;
};

const HANDLE_PX = 8;
const DELETE_PX = 12;
/** handle radius on screen, the `tol` of hitTest */
const HANDLE_TOL_PX = 6;

/** Outside the crop darkened, blocked boxes grey and hatched: what SAM3 and the export will not see. */
function drawRegion(ctx: CanvasRenderingContext2D, r: RegionEdit, w: number, h: number, scale: number) {
  if (r.crop) {
    const [x0, y0, x1, y1] = r.crop;
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.fill("evenodd");
    ctx.strokeStyle = "#e5e5e5";
    ctx.lineWidth = 1 / scale;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  for (const [x0, y0, x1, y1] of r.blocks) {
    ctx.fillStyle = "rgba(114, 114, 114, 0.85)";
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.clip();
    ctx.strokeStyle = "rgba(30, 31, 33, 0.7)";
    ctx.lineWidth = 1 / scale;
    ctx.beginPath();
    const step = 8 / scale;
    for (let hx = x0 - (y1 - y0); hx < x1; hx += step) {
      ctx.moveTo(hx, y1);
      ctx.lineTo(hx + (y1 - y0), y0);
    }
    ctx.stroke();
    ctx.restore();
  }
}

export function FrameCanvas(props: Props) {
  const { store, width, height, frameUrl, masks, color, outline, tool, candidates, onPoint, onBox, onPickCandidate, fitSignal, brush, regionEdit } = props;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = useRef({ w: 0, h: 0 });
  const view = useRef<View>({ scale: 1, tx: 0, ty: 0 });
  const fitScale = useRef(1);
  const assets = useRef<{ frame: HTMLImageElement | null; mask: HTMLCanvasElement | null }>({ frame: null, mask: null });
  const loader = useRef<LoadState>({ busy: false, waiting: null, seq: 0, drawn: 0, mounted: true });
  const layers = useRef<LayerState>({ owner: null, items: [] });
  const drag = useRef<Drag | null>(null);
  const dragBox = useRef<Box | null>(null);
  /** a live region edit: what was grabbed (null: drawing a new rect), where, and the region as edited so far */
  const regionDrag = useRef<{
    hit: Hit | null; start: [number, number]; draft: RegionEdit;
    /** the region and callback as they were at mousedown: a frame change mid-drag must not redirect the edit */
    base: RegionEdit; draw: "block" | "crop"; onChange: (next: RegionEdit) => void;
  } | null>(null);
  const latest = useRef(props);
  latest.current = props;
  const rgb = useMemo(() => hexToRgb(color), [color]);
  // `masks` is rebuilt on every page render; only its content matters
  const masksKey = compositeKey(masks, MASK_ALPHA);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- masksKey covers every field of `masks`
  const stableMasks = useMemo(() => masks, [masksKey]);
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
      // The stored masks come first: every visible object on this frame, composited into one layer. While the
      // active object is being edited (candidates or brush strokes) the page leaves it out of `masks`, and its
      // preview or brush layer is drawn on top instead.
      if (assets.current.mask) ctx.drawImage(assets.current.mask, 0, 0, p.width, p.height);
      if (p.candidates.length > 0) {
        const state = layers.current;
        const current = state.owner === p.candidates;
        // Not current: new previews are still decoding. Only bridge the gap when it's the same single
        // candidate being refined (an extra click on the same object) — anything else would flash a mismatched
        // preview against the wrong candidate list.
        const bridgeOld = !current && state.owner !== null && state.owner.length === 1 && p.candidates.length === 1;
        if (current || bridgeOld) {
          state.items.forEach((layer, i) => {
            if (layer === null) return;
            ctx.globalAlpha = i === p.draft.candidate ? 1 : 0.45;
            ctx.drawImage(layer, 0, 0, p.width, p.height);
          });
          ctx.globalAlpha = 1;
        }
      } else if (p.brush !== null && p.brush.layer) {
        ctx.globalAlpha = MASK_ALPHA;
        ctx.drawImage(p.brush.layer, 0, 0, p.width, p.height);
        ctx.globalAlpha = 1;
      }
    }
    const live = regionDrag.current?.draft ?? p.region ?? null;
    if (live) drawRegion(ctx, live, p.width, p.height, v.scale);
    const hl = p.regionHighlight;
    const hlRect = !live || !hl ? null : hl.kind === "crop" ? live.crop : live.blocks[hl.index] ?? null;
    if (hlRect) {
      ctx.strokeStyle = "#38BDF8";
      ctx.lineWidth = 2 / v.scale;
      ctx.strokeRect(hlRect[0], hlRect[1], hlRect[2] - hlRect[0], hlRect[3] - hlRect[1]);
    }
    ctx.restore();

    if (p.regionEdit) {
      const region = regionDrag.current?.draft ?? p.region ?? null;
      const rects = region ? [...(region.crop ? [region.crop] : []), ...region.blocks] : [];
      ctx.lineWidth = 1;
      for (const r of rects) {
        for (const [cx, cy] of [[r[0], r[1]], [r[2], r[1]], [r[0], r[3]], [r[2], r[3]]]) {
          const [x, y] = toScreen(v, cx, cy);
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = "#000000";
          ctx.fillRect(x - HANDLE_PX / 2, y - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
          ctx.strokeRect(x - HANDLE_PX / 2, y - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
        }
      }
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const b of region?.blocks ?? []) {
        // outside the NE corner, where hitTest puts the delete target (1.5 handle radii up and right)
        const off = HANDLE_TOL_PX * 1.5;
        const [bx, by] = toScreen(v, b[2], b[1]);
        const [x, y] = [bx + off, by - off];
        ctx.beginPath();
        ctx.arc(x, y, DELETE_PX / 2, 0, 2 * Math.PI);
        ctx.fillStyle = "#ef4444";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
        ctx.fillStyle = "#ffffff";
        ctx.fillText("\u00d7", x, y);
      }
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }

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

  // Swap the frame and its masks in together so an overlay never shows on the wrong frame. One load is in flight at a
  // time; what is asked for meanwhile waits, and only the newest wait survives. A load that lands is drawn unless a
  // later one was drawn first. On a slow link this keeps the view moving during a propagation (frames streamed
  // faster than they load are skipped, not each started and then dropped) and keeps abandoned requests from piling
  // up on the browser's few connections. A load still pending after LOAD_STALL_MS stops holding back the next.
  useEffect(() => {
    const loads = loader.current;
    loads.mounted = true;
    return () => {
      loads.mounted = false;
      loads.waiting = null;
    };
  }, []);

  useEffect(() => {
    const loads = loader.current;
    const start = (want: LoadRequest) => {
      const seq = ++loads.seq;
      loads.busy = true;
      let released = false;
      const release = () => {
        if (released) return;
        released = true;
        clearTimeout(stall);
        loads.busy = false;
        const next = loads.waiting;
        loads.waiting = null;
        if (next !== null && loads.mounted) start(next);
      };
      const stall = setTimeout(release, LOAD_STALL_MS);
      void Promise.all([want.store.image(want.frameUrl), want.store.composite(want.masks)]).then(([frame, layer]) => {
        if (loads.mounted && seq > loads.drawn) {
          loads.drawn = seq;
          assets.current = { frame, mask: layer };
          draw();
        }
        release();
      });
    };
    const want = { store, frameUrl, masks: stableMasks };
    if (loads.busy) loads.waiting = want;
    else start(want);
  }, [store, frameUrl, stableMasks, draw]);

  useEffect(() => {
    let live = true;
    if (candidates.length === 0) {
      layers.current = { owner: null, items: [] };
      draw();
      return;
    }
    const pending = candidates.map((c) => store.preview(`data:image/png;base64,${c.mask_png_b64}`, rgb, PREVIEW_ALPHA, outline));
    void Promise.all(pending).then((items) => {
      if (!live) return;
      layers.current = { owner: candidates, items };
      draw();
    });
    return () => {
      live = false;
    };
  }, [store, candidates, rgb, outline, draw]);

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
    if (regionEdit && e.button === 0) {
      const region = props.region ?? null;
      const base: RegionEdit = { crop: region?.crop ?? null, blocks: region?.blocks ?? [] };
      regionDrag.current = {
        hit: hitTest(region, ix, iy, HANDLE_TOL_PX / view.current.scale), start: [ix, iy], draft: base,
        base, draw: regionEdit.draw, onChange: regionEdit.onChange,
      };
      d.region = true;
      return;
    }
    if (brush !== null && (e.button === 0 || e.button === 2)) {
      const bx = Number.isFinite(ix) ? Math.floor(ix) : 0;
      const by = Number.isFinite(iy) ? Math.floor(iy) : 0;
      brush.onStart(e.button === 2 ? "erase" : "paint", bx, by);
      d.painting = true;
      draw();
    }
  };

  // the Regions tool owns left-drags and clicks, so prompting is off while it is active
  const promptTool = regionEdit ? null : tool;

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
    if (d.region) {
      const rd = regionDrag.current;
      if (e.buttons === 0 || rd === null) {
        endDrag(false);
        return;
      }
      const [ix, iy] = toImage(view.current, sx, sy);
      const { crop, blocks } = rd.base;
      const { hit } = rd;
      if (hit === null) {
        const r = normRect(rd.start, [ix, iy], width, height);
        rd.draft = rd.draw === "crop" ? { crop: r, blocks } : { crop, blocks: [...blocks, r] };
      } else if (hit.part !== "delete") {
        const r = dragRect(hit.kind === "crop" ? crop! : blocks[hit.index], hit.part, ix - rd.start[0], iy - rd.start[1], width, height);
        rd.draft = hit.kind === "crop" ? { crop: r, blocks } : { crop, blocks: blocks.map((b, i) => (i === hit.index ? r : b)) };
      }
      draw();
      return;
    }
    if (d.button === 1) {
      view.current = panBy(view.current, sx - d.lastX, sy - d.lastY);
      d.lastX = sx;
      d.lastY = sy;
      draw();
    } else if (d.button === 0 && promptTool === "box" && d.moved) {
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
    if (d?.region) {
      const rd = regionDrag.current;
      regionDrag.current = null;
      if (commit && rd !== null) commitRegion(rd);
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
    if (!commit || d.moved || promptTool === null) return;
    const [ix, iy] = [d.ix, d.iy];
    if (!insideImage(ix, iy, width, height)) return;
    const [x, y] = clampPoint(ix, iy, width, height);
    if (promptTool === "text") {
      const index = pickCandidate(candidates.map((c) => c.bbox), x, y);
      if (index !== null) onPickCandidate(index);
      return;
    }
    if (d.button === 0 && props.onExcluded && isExcluded(props.region ?? null, x, y)) {
      props.onExcluded();
      return;
    }
    onPoint(x, y, d.button === 2 ? 0 : 1);
  };

  /** report a finished region edit, unless it changed nothing (a click, a too-small new rect) */
  const commitRegion = (rd: NonNullable<typeof regionDrag.current>) => {
    const { base, hit, draft, onChange } = rd;
    const blocks = base.blocks;
    if (hit === null) {
      // untouched arrays mean the pointer never moved: a plain click draws nothing
      const crop = rd.draw === "crop";
      const r = crop ? draft.crop : draft.blocks[draft.blocks.length - 1];
      const drawn = crop ? draft.crop !== base.crop : draft.blocks !== blocks;
      if (drawn && r && r[2] - r[0] >= 2 && r[3] - r[1] >= 2) onChange(draft);
    } else if (hit.part === "delete") {
      onChange({ crop: base.crop, blocks: blocks.filter((_, i) => i !== hit.index) });
    } else {
      const before = hit.kind === "crop" ? base.crop : blocks[hit.index];
      const after = hit.kind === "crop" ? draft.crop : draft.blocks[hit.index];
      if (before && after && before.some((v, i) => v !== after[i])) onChange(draft);
    }
  };

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden bg-well">
      <canvas
        ref={canvasRef}
        aria-label="Frame"
        className="block h-full w-full"
        style={{ cursor: brush !== null ? "none" : regionEdit || tool !== null ? "crosshair" : "default" }}
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
