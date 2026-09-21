import { useEffect, useRef, useState, type MouseEvent } from "react";
import type { FrameState } from "../api/types";
import { frameAtX, runs } from "../lib/timeline";

type Props = { states: FrameState[]; frame: number; onSeek: (frame: number) => void };

const HEIGHT = 28;
const COLORS: Record<FrameState, string> = { empty: "#3f3f46", masked: "#10b981", keyframe: "#f59e0b", deleted: "#7f1d1d" };
const LEGEND: [FrameState, string][] = [["keyframe", "keyframe"], ["masked", "masked"], ["empty", "empty"], ["deleted", "deleted"]];

export function Timeline({ states, frame, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = wrap.current;
    if (el === null) return;
    const observer = new ResizeObserver(() => setWidth(el.clientWidth));
    observer.observe(el);
    setWidth(el.clientWidth);
    return () => observer.disconnect();
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
    const all = runs(states);
    for (const r of all) {
      if (r.state === "keyframe") continue;
      ctx.fillStyle = COLORS[r.state];
      ctx.fillRect(r.start * px, 6, (r.end - r.start + 1) * px, HEIGHT - 12);
    }
    ctx.fillStyle = COLORS.keyframe; // keyframes on top, full height, never thinner than 3 px
    for (const r of all) {
      if (r.state !== "keyframe") continue;
      for (let i = r.start; i <= r.end; i++) {
        const w = Math.max(px, 3);
        ctx.fillRect(i * px + px / 2 - w / 2, 0, w, HEIGHT);
      }
    }
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(frame * px + px / 2 - 1, 0, 2, HEIGHT);
  }, [states, frame, width]);

  const seek = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(frameAtX(e.clientX - rect.left, rect.width, states.length));
  };

  return (
    <div ref={wrap} className="w-full select-none">
      <canvas
        ref={canvas}
        role="slider"
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, states.length - 1)}
        aria-valuenow={frame}
        tabIndex={-1}
        className="block w-full cursor-pointer"
        style={{ height: HEIGHT }}
        onMouseDown={(e) => {
          if (e.button === 0) seek(e);
        }}
        onMouseMove={(e) => {
          if (e.buttons & 1) seek(e);
        }}
      />
      <div className="mt-1 flex gap-4 text-xs text-neutral-400">
        {LEGEND.map(([state, label]) => (
          <span key={state} className="flex items-center gap-1">
            <span className="inline-block h-2 w-3" style={{ background: COLORS[state] }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
