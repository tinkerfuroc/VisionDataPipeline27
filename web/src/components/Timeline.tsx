import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { FrameState } from "../api/types";
import { GUTTER, clampScroll, laneAt, layoutLanes, scrollToShow, type Lane } from "../lib/lanes";
import { frameAtX, runs } from "../lib/timeline";

type Props = {
  /** the whole clip per frame: sets the frame count, and is drawn as the single lane of a clip without objects */
  states: FrameState[];
  /** one lane per project class, in class-id order (`lanesFor`) */
  lanes: Lane[];
  /** the active object's class, or the class chosen for a "New object" row: its lane is drawn 30 px tall; null
   * before a class is chosen (or a project with no classes), when every lane is short */
  activeClassId: number | null;
  frame: number;
  /** thumbnail shown while hovering; omit to show the frame number only */
  thumb?: { url: (frame: number) => string; aspect: number };
  /** frames marked for clearing (inclusive), drawn as a bracket over the lanes */
  range?: [number, number] | null;
  onSeek: (frame: number) => void;
  /** Shift-drag marks a range instead of seeking; omit to disable marking */
  onRange?: (range: [number, number]) => void;
  /** a click on a lane, or on its name in the gutter, selects that class; omit while the selection is locked */
  onSelect?: (id: number) => void;
};

const TOP = 6; // room above the lanes for the playhead's flag
const RULER = 18; // the frame-number ruler under the lanes
const THUMB_W = 176;
const HIDDEN_ALPHA = 0.4;
// canvas cannot read the Tailwind theme; these mirror tailwind.config.js
const INK = "#E9E7E2";
const DIM = "#A3A5AA";
const LINE = "#4A4D52";
const WELL = "#1E1F21";
const PANEL = "#36383C";
const KEY = "#F59E0B";
const STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
const WORDS: Record<FrameState, string> = { keyframe: "keyframe", masked: "masked", empty: "no mask", deleted: "deleted" };
const NO_CLASSES = "No classes yet";

/**
 * One class's frames in a band `h` px tall at `y`. State is told apart by shape and texture as well as colour: a
 * filled run is masked, a hollow run has no mask, a hatched run is deleted, and a diamond marks a keyframe.
 */
function drawLane(ctx: CanvasRenderingContext2D, states: readonly FrameState[], y: number, h: number, color: string, px: number) {
  for (const r of runs(states)) {
    const x = GUTTER + r.start * px;
    const w = (r.end - r.start + 1) * px;
    if (r.state === "masked" || r.state === "keyframe") {
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, h);
    } else if (r.state === "deleted") {
      ctx.fillStyle = WELL;
      ctx.fillRect(x, y, w, h);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.strokeStyle = DIM;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let hx = x - h; hx < x + w; hx += 5) {
        ctx.moveTo(hx, y + h);
        ctx.lineTo(hx + h, y);
      }
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.strokeStyle = LINE;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, Math.max(w - 1, 1), h - 1);
    }
  }
  if (px >= 5) {
    // frames are discrete: notch every boundary while there is room for it
    ctx.fillStyle = PANEL;
    for (let i = 1; i < states.length; i++) ctx.fillRect(Math.round(GUTTER + i * px), y, 1, h);
  }
  // keyframe diamonds, on top and scaled to the lane, outlined so they hold against any class colour
  const dy = Math.min(7, h / 2);
  const dx = (dy * 6) / 7;
  const mid = y + h / 2;
  states.forEach((s, i) => {
    if (s !== "keyframe") return;
    const cx = GUTTER + i * px + px / 2;
    ctx.beginPath();
    ctx.moveTo(cx, mid - dy);
    ctx.lineTo(cx + dx, mid);
    ctx.lineTo(cx, mid + dy);
    ctx.lineTo(cx - dx, mid);
    ctx.closePath();
    ctx.fillStyle = KEY;
    ctx.fill();
    ctx.strokeStyle = WELL;
    ctx.lineWidth = h >= 20 ? 1.5 : 1;
    ctx.stroke();
  });
}

/** One lane per class (spec §9): the active lane tall, the names in a gutter on the left, the playhead across all. */
export function Timeline({ states, lanes, activeClassId, frame, thumb, range = null, onSeek, onRange, onSelect }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [fontsReady, setFontsReady] = useState(false);
  const [scroll, setScroll] = useState(0);
  const anchor = useRef<number | null>(null); // first frame of a Shift-drag in progress

  const idsKey = lanes.map((l) => l.id).join(",");
  const layout = useMemo(
    () => layoutLanes(lanes.length === 0 ? [null] : lanes.map((l) => l.id), activeClassId),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- idsKey stands for the lane ids, all layoutLanes reads
    [idsKey, activeClassId],
  );
  const height = TOP + layout.view + RULER;

  // keep the active lane in view whenever it (or the set of lanes) changes; the wheel scrolls freely in between
  useEffect(() => setScroll((s) => scrollToShow(s, layout, activeClassId)), [layout, activeClassId]);

  useEffect(() => {
    const el = wrap.current;
    if (el === null) return;
    const observer = new ResizeObserver(() => setWidth(el.clientWidth));
    observer.observe(el);
    setWidth(el.clientWidth);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    // on the window, because a Shift-drag usually ends with the pointer off this narrow strip
    const release = () => {
      anchor.current = null;
    };
    window.addEventListener("mouseup", release);
    return () => window.removeEventListener("mouseup", release);
  }, []);

  useEffect(() => {
    let closed = false;
    void document.fonts?.ready.then(() => {
      if (!closed) setFontsReady(true); // the ruler labels were drawn in the fallback face until now
    });
    return () => {
      closed = true;
    };
  }, []);

  useEffect(() => {
    const c = canvas.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (states.length === 0) return;
    const strip = Math.max(0, width - GUTTER);
    const px = strip / states.length;
    const view = layout.view;

    // the lanes, clipped to the lane viewport (it scrolls once there are more than six)
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, TOP, strip, view);
    ctx.clip();
    for (const box of layout.boxes) {
      const y = TOP + box.top - scroll;
      if (y + box.height <= TOP || y >= TOP + view) continue;
      const lane = box.id === null ? undefined : lanes.find((l) => l.id === box.id);
      ctx.globalAlpha = lane?.hidden ? HIDDEN_ALPHA : 1;
      drawLane(ctx, lane?.states ?? states, y, box.height, lane?.color ?? DIM, px);
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    if (layout.content > view) {
      // a thin scrollbar: where the view sits in the lane stack
      ctx.fillStyle = DIM;
      ctx.fillRect(width - 3, TOP + (scroll * view) / layout.content, 3, (view * view) / layout.content);
    }

    // ruler
    const rulerTop = TOP + view;
    const step = STEPS.find((s) => s * px >= 56) ?? STEPS[STEPS.length - 1];
    ctx.fillStyle = DIM;
    ctx.font = '500 12px "Barlow Semi Condensed", sans-serif';
    ctx.textBaseline = "alphabetic";
    for (let i = 0; i < states.length; i += step) {
      const x = GUTTER + i * px + px / 2;
      ctx.fillRect(Math.round(x), rulerTop + 1, 1, 3);
      ctx.textAlign = i === 0 ? "left" : "center";
      ctx.fillText(String(i), i === 0 ? GUTTER : x, height - 3);
    }

    if (range !== null && range[0] <= range[1]) {
      // the marked range, across every lane: a light wash with a bracket at each end
      const x0 = GUTTER + range[0] * px;
      const x1 = GUTTER + (range[1] + 1) * px;
      ctx.fillStyle = "rgba(233, 231, 226, 0.3)";
      ctx.fillRect(x0, TOP, x1 - x0, view);
      ctx.fillStyle = INK;
      ctx.fillRect(x0, TOP - 4, x1 - x0, 2);
      ctx.fillRect(x0, TOP - 4, 2, view + 8);
      ctx.fillRect(x1 - 2, TOP - 4, 2, view + 8);
      ctx.fillRect(x0, TOP + view + 2, x1 - x0, 2);
    }

    if (hover !== null && hover !== frame) {
      ctx.fillStyle = "rgba(233, 231, 226, 0.25)";
      ctx.fillRect(GUTTER + hover * px, TOP, Math.max(px, 1), view);
    }

    // playhead across every lane: a flag and a light line. Where frames are wide enough the line stops at the lanes
    // and an outline takes over, so it never hides a keyframe diamond on the current frame.
    const cx = GUTTER + frame * px + px / 2;
    ctx.fillStyle = INK;
    if (px >= 14) {
      ctx.fillRect(cx - 1, 0, 2, TOP);
      ctx.fillRect(cx - 1, TOP + view, 2, 4);
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.strokeRect(GUTTER + frame * px + 1, TOP + 1, px - 2, view - 2);
    } else {
      ctx.fillRect(cx - 1, 0, 2, TOP + view + 4);
    }
    ctx.beginPath();
    ctx.moveTo(cx - 5, 0);
    ctx.lineTo(cx + 5, 0);
    ctx.lineTo(cx, 6);
    ctx.closePath();
    ctx.fill();
  }, [states, lanes, layout, scroll, frame, width, height, hover, range, fontsReady]);

  const point = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    return { x, y: e.clientY - rect.top, frame: frameAtX(x - GUTTER, rect.width - GUTTER, states.length) };
  };

  const px = states.length > 0 ? Math.max(0, width - GUTTER) / states.length : 0;
  const tipW = Math.min(THUMB_W, width);
  const tipLeft = hover === null ? 0 : Math.max(0, Math.min(width - tipW, GUTTER + hover * px + px / 2 - tipW / 2));

  return (
    <div
      ref={wrap}
      className="relative w-full select-none"
      onWheel={(e) => {
        if (layout.content > layout.view) setScroll((s) => clampScroll(s + e.deltaY, layout));
      }}
    >
      <canvas
        ref={canvas}
        aria-label="Timeline"
        className="block w-full cursor-pointer"
        style={{ height }}
        onMouseDown={(e) => {
          if (e.button !== 0) return;
          const p = point(e);
          anchor.current = null; // never inherit a drag whose release this strip did not see
          if (p.x < GUTTER) return; // the names in the gutter are buttons of their own
          if (e.shiftKey && onRange !== undefined) {
            anchor.current = p.frame;
            onRange([p.frame, p.frame]);
            return;
          }
          const inLanes = p.y >= TOP && p.y < TOP + layout.view;
          const box = inLanes ? laneAt(layout.boxes, p.y - TOP + scroll) : null;
          if (box !== null && box.id !== null && box.id !== activeClassId) onSelect?.(box.id);
          onSeek(p.frame);
        }}
        onMouseMove={(e) => {
          const p = point(e);
          setHover(p.x < GUTTER ? null : p.frame);
          if (!(e.buttons & 1)) anchor.current = null; // the button was released outside the strip
          else if (anchor.current !== null) onRange?.([Math.min(anchor.current, p.frame), Math.max(anchor.current, p.frame)]);
          else onSeek(p.frame);
        }}
        onMouseLeave={() => setHover(null)}
      />
      <div className="absolute left-0 overflow-hidden" style={{ top: TOP, width: GUTTER, height: layout.view }}>
        {layout.boxes.map((b) => {
          const lane = b.id === null ? undefined : lanes.find((l) => l.id === b.id);
          const style = { top: b.top - scroll, height: b.height, opacity: lane?.hidden ? HIDDEN_ALPHA : 1 };
          if (lane === undefined) {
            return (
              <span key="none" className="absolute left-0 flex w-full items-center truncate pr-1 font-cond text-xs text-dim" style={style}>
                {NO_CLASSES}
              </span>
            );
          }
          const active = lane.id === activeClassId;
          return (
            // not in the tab order: O / Shift+O and the Objects panel select objects from the keyboard
            // every name at text-xs (never shrunk); the active one told apart by colour and aria-pressed
            <button key={lane.id} type="button" tabIndex={-1} aria-pressed={active} title={lane.label}
              disabled={onSelect === undefined}
              className={`absolute left-0 flex w-full items-center truncate pr-1 text-left font-cond text-xs leading-none ${active ? "text-ink" : onSelect === undefined ? "text-dim" : "text-dim hover:text-ink"}`}
              style={style} onClick={() => onSelect?.(lane.id)}>
              {lane.label}
            </button>
          );
        })}
      </div>
      {hover !== null && states.length > 0 && (
        <div className="pointer-events-none absolute z-10 overflow-hidden rounded border border-line bg-well shadow-lg"
          style={{ left: tipLeft, bottom: height + 6, width: tipW }}>
          {thumb !== undefined && (
            <img src={thumb.url(hover)} alt="" className="block w-full bg-well object-contain" style={{ aspectRatio: thumb.aspect }} />
          )}
          <p className="flex justify-between px-1.5 py-0.5 font-cond text-xs">
            <span>{`frame ${hover}`}</span>
            {lanes.length === 0 && <span className="text-dim">{WORDS[states[hover] ?? "empty"]}</span>}
          </p>
          {lanes.length > 0 && (
            <ul className="px-1.5 pb-0.5 font-cond text-xs">
              {lanes.map((l) => (
                <li key={l.id} className="flex justify-between gap-2">
                  <span className="truncate">{l.label}</span>
                  <span className="text-dim">{WORDS[l.states[hover] ?? "empty"]}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
