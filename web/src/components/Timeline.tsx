import { useEffect, useRef, useState, type MouseEvent } from "react";
import type { FrameState } from "../api/types";
import { frameAtX, runs } from "../lib/timeline";

type Props = {
  states: FrameState[];
  frame: number;
  /** class colour of the object, hex: masked frames are drawn in it, so the strip matches the overlay */
  color: string;
  /** thumbnail shown while hovering; omit to show the frame number only */
  thumb?: { url: (frame: number) => string; aspect: number };
  /** frames marked for clearing (inclusive), drawn as a bracket over the strip */
  range?: [number, number] | null;
  onSeek: (frame: number) => void;
  /** Shift-drag marks a range instead of seeking; omit to disable marking */
  onRange?: (range: [number, number]) => void;
};

const HEIGHT = 54;
const BAND_TOP = 6;
const BAND_H = 30;
const BAND_MID = BAND_TOP + BAND_H / 2;
const THUMB_W = 176;
// canvas cannot read the Tailwind theme; these mirror tailwind.config.js
const INK = "#E9E7E2";
const DIM = "#A3A5AA";
const LINE = "#4A4D52";
const WELL = "#1E1F21";
const PANEL = "#36383C";
const KEY = "#F59E0B";
const STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
const WORDS: Record<FrameState, string> = { keyframe: "keyframe", masked: "masked", empty: "no mask", deleted: "deleted" };

/**
 * Frame state is told apart by shape and texture as well as colour: a filled run is masked, a hollow run has no mask,
 * a hatched run is deleted, and a diamond marks a keyframe.
 */
export function Timeline({ states, frame, color, thumb, range = null, onSeek, onRange }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [fontsReady, setFontsReady] = useState(false);
  const anchor = useRef<number | null>(null); // first frame of a Shift-drag in progress

  useEffect(() => {
    const el = wrap.current;
    if (el === null) return;
    const observer = new ResizeObserver(() => setWidth(el.clientWidth));
    observer.observe(el);
    setWidth(el.clientWidth);
    return () => observer.disconnect();
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
    c.height = Math.round(HEIGHT * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, HEIGHT);
    if (states.length === 0) return;
    const px = width / states.length;

    for (const r of runs(states)) {
      const x = r.start * px;
      const w = (r.end - r.start + 1) * px;
      if (r.state === "masked" || r.state === "keyframe") {
        ctx.fillStyle = color;
        ctx.fillRect(x, BAND_TOP, w, BAND_H);
      } else if (r.state === "deleted") {
        ctx.fillStyle = WELL;
        ctx.fillRect(x, BAND_TOP, w, BAND_H);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, BAND_TOP, w, BAND_H);
        ctx.clip();
        ctx.strokeStyle = DIM;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let hx = x - BAND_H; hx < x + w; hx += 5) {
          ctx.moveTo(hx, BAND_TOP + BAND_H);
          ctx.lineTo(hx + BAND_H, BAND_TOP);
        }
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.strokeStyle = LINE;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, BAND_TOP + 0.5, Math.max(w - 1, 1), BAND_H - 1);
      }
    }

    if (px >= 5) {
      // frames are discrete: notch every boundary while there is room for it
      ctx.fillStyle = PANEL;
      for (let i = 1; i < states.length; i++) ctx.fillRect(Math.round(i * px), BAND_TOP, 1, BAND_H);
    }

    // keyframe diamonds, on top and never narrower than 10 px, outlined so they hold against any class colour
    states.forEach((s, i) => {
      if (s !== "keyframe") return;
      const cx = i * px + px / 2;
      ctx.beginPath();
      ctx.moveTo(cx, BAND_MID - 7);
      ctx.lineTo(cx + 6, BAND_MID);
      ctx.lineTo(cx, BAND_MID + 7);
      ctx.lineTo(cx - 6, BAND_MID);
      ctx.closePath();
      ctx.fillStyle = KEY;
      ctx.fill();
      ctx.strokeStyle = WELL;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // ruler
    const step = STEPS.find((s) => s * px >= 56) ?? STEPS[STEPS.length - 1];
    ctx.fillStyle = DIM;
    ctx.font = '500 12px "Barlow Semi Condensed", sans-serif';
    ctx.textBaseline = "alphabetic";
    for (let i = 0; i < states.length; i += step) {
      const x = i * px + px / 2;
      ctx.fillRect(Math.round(x), BAND_TOP + BAND_H + 1, 1, 3);
      ctx.textAlign = i === 0 ? "left" : "center";
      ctx.fillText(String(i), i === 0 ? 0 : x, HEIGHT - 3);
    }

    if (range !== null && range[0] <= range[1]) {
      // the marked range: a light wash with a bracket at each end, clear of the band so the states stay readable
      const x0 = range[0] * px;
      const x1 = (range[1] + 1) * px;
      ctx.fillStyle = "rgba(233, 231, 226, 0.3)";
      ctx.fillRect(x0, BAND_TOP, x1 - x0, BAND_H);
      ctx.fillStyle = INK;
      ctx.fillRect(x0, BAND_TOP - 4, x1 - x0, 2);
      ctx.fillRect(x0, BAND_TOP - 4, 2, BAND_H + 8);
      ctx.fillRect(x1 - 2, BAND_TOP - 4, 2, BAND_H + 8);
      ctx.fillRect(x0, BAND_TOP + BAND_H + 2, x1 - x0, 2);
    }

    if (hover !== null && hover !== frame) {
      ctx.fillStyle = "rgba(233, 231, 226, 0.25)";
      ctx.fillRect(hover * px, BAND_TOP, Math.max(px, 1), BAND_H);
    }

    // playhead: a flag and a light line. Where frames are wide enough the line stops at the band and an outline
    // takes over, so it never hides a keyframe diamond on the current frame.
    const cx = frame * px + px / 2;
    const wide = px >= 14;
    ctx.fillStyle = INK;
    if (wide) {
      ctx.fillRect(cx - 1, 0, 2, BAND_TOP);
      ctx.fillRect(cx - 1, BAND_TOP + BAND_H, 2, 4);
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.strokeRect(frame * px + 1, BAND_TOP + 1, px - 2, BAND_H - 2);
    } else {
      ctx.fillRect(cx - 1, 0, 2, BAND_TOP + BAND_H + 4);
    }
    ctx.beginPath();
    ctx.moveTo(cx - 5, 0);
    ctx.lineTo(cx + 5, 0);
    ctx.lineTo(cx, 6);
    ctx.closePath();
    ctx.fill();
  }, [states, frame, width, color, hover, range, fontsReady]);

  const at = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return frameAtX(e.clientX - rect.left, rect.width, states.length);
  };

  const px = states.length > 0 ? width / states.length : 0;
  const tipW = Math.min(THUMB_W, width);
  const tipLeft = hover === null ? 0 : Math.max(0, Math.min(width - tipW, hover * px + px / 2 - tipW / 2));

  return (
    <div ref={wrap} className="relative w-full select-none">
      <canvas
        ref={canvas}
        aria-label="Timeline"
        className="block w-full cursor-pointer"
        style={{ height: HEIGHT }}
        onMouseDown={(e) => {
          if (e.button !== 0) return;
          const f = at(e);
          if (e.shiftKey && onRange !== undefined) {
            anchor.current = f;
            onRange([f, f]);
          } else {
            onSeek(f);
          }
        }}
        onMouseMove={(e) => {
          const f = at(e);
          setHover(f);
          if (!(e.buttons & 1)) anchor.current = null; // the button was released outside the strip
          else if (anchor.current !== null) onRange?.([Math.min(anchor.current, f), Math.max(anchor.current, f)]);
          else onSeek(f);
        }}
        onMouseUp={() => {
          anchor.current = null;
        }}
        onMouseLeave={() => setHover(null)}
      />
      {hover !== null && states.length > 0 && (
        <div className="pointer-events-none absolute z-10 overflow-hidden rounded border border-line bg-well shadow-lg"
          style={{ left: tipLeft, bottom: HEIGHT + 6, width: tipW }}>
          {thumb !== undefined && (
            <img src={thumb.url(hover)} alt="" className="block w-full bg-well object-contain" style={{ aspectRatio: thumb.aspect }} />
          )}
          <p className="flex justify-between px-1.5 py-0.5 font-cond text-xs">
            <span>{`frame ${hover}`}</span>
            <span className="text-dim">{WORDS[states[hover] ?? "empty"]}</span>
          </p>
        </div>
      )}
    </div>
  );
}
