type Props = {
  frame: number;
  frameCount: number;
  fps: number;
  playing: boolean;
  overlay: boolean;
  /** original file name of this frame, for frame-directory clips */
  name: string | null;
  onStep: (delta: number) => void;
  onTogglePlay: () => void;
  onToggleOverlay: () => void;
};

const button = "rounded bg-neutral-700 px-2 py-1 text-sm hover:bg-neutral-600";

export function Transport({ frame, frameCount, fps, playing, overlay, name, onStep, onTogglePlay, onToggleOverlay }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <button type="button" className={button} aria-label="Back 10 frames" onClick={() => onStep(-10)}>⏪</button>
      <button type="button" className={button} aria-label="Previous frame" onClick={() => onStep(-1)}>◀</button>
      <button type="button" className={button} aria-label={playing ? "Pause" : "Play"} onClick={onTogglePlay}>
        {playing ? "⏸" : "▶"}
      </button>
      <button type="button" className={button} aria-label="Next frame" onClick={() => onStep(1)}>▶▏</button>
      <button type="button" className={button} aria-label="Forward 10 frames" onClick={() => onStep(10)}>⏩</button>
      <span className="ml-2 tabular-nums">{`frame ${frame} / ${Math.max(0, frameCount - 1)}`}</span>
      {name !== null && <span className="text-xs text-neutral-400">{name}</span>}
      <span className="text-xs text-neutral-500">{fps} fps</span>
      <label className="ml-auto flex items-center gap-1 text-xs">
        <input type="checkbox" checked={overlay} onChange={onToggleOverlay} />
        Mask overlay (M)
      </label>
    </div>
  );
}
