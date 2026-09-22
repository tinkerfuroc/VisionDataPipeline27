import type { ReactNode } from "react";

type Props = {
  frame: number;
  frameCount: number;
  playing: boolean;
  overlay: boolean;
  /** original file name of this frame, for frame-directory clips */
  name: string | null;
  /** disables the step and play buttons (not the overlay toggle); default false */
  disabled?: boolean;
  onStep: (delta: number) => void;
  onTogglePlay: () => void;
  onToggleOverlay: () => void;
};

// Drawn rather than typed: the media-control characters render as colour emoji on some systems.
const glyph = (children: ReactNode) => (
  <svg aria-hidden width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    {children}
  </svg>
);
const GLYPHS = {
  back10: glyph(<path d="M8 3v10L2 8zM14 3v10L8 8z" />),
  prev: glyph(<path d="M11 3v10L5 8zM3.5 3H5v10H3.5z" />),
  play: glyph(<path d="M4 2.5v11L13 8z" />),
  pause: glyph(<path d="M4 3h3v10H4zM9 3h3v10H9z" />),
  next: glyph(<path d="M5 3v10l6-5zM11 3h1.5v10H11z" />),
  forward10: glyph(<path d="M8 3v10l6-5zM2 3v10l6-5z" />),
};
const button = "btn h-7 w-8 px-0";

export function Transport({ frame, frameCount, playing, overlay, name, disabled = false, onStep, onTogglePlay, onToggleOverlay }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-1 text-sm">
      <button type="button" className={button} disabled={disabled} aria-label="Back 10 frames" onClick={() => onStep(-10)}>{GLYPHS.back10}</button>
      <button type="button" className={button} disabled={disabled} aria-label="Previous frame" onClick={() => onStep(-1)}>{GLYPHS.prev}</button>
      <button type="button" className={`${button} ${playing ? "btn-on" : ""}`} disabled={disabled} aria-label={playing ? "Pause" : "Play"} onClick={onTogglePlay}>
        {playing ? GLYPHS.pause : GLYPHS.play}
      </button>
      <button type="button" className={button} disabled={disabled} aria-label="Next frame" onClick={() => onStep(1)}>{GLYPHS.next}</button>
      <button type="button" className={button} disabled={disabled} aria-label="Forward 10 frames" onClick={() => onStep(10)}>{GLYPHS.forward10}</button>
      <span className="ml-3 min-w-[7.5rem] text-base font-medium">{`frame ${frame} / ${Math.max(0, frameCount - 1)}`}</span>
      {name !== null && <span className="font-cond text-dim">{name}</span>}
      <label className="ml-3 flex items-center gap-1.5 font-cond text-dim">
        <input type="checkbox" checked={overlay} aria-keyshortcuts="M" onChange={onToggleOverlay} />
        Mask overlay <kbd aria-hidden>M</kbd>
      </label>
    </div>
  );
}
