import { useEffect, type Dispatch, type SetStateAction } from "react";

type Options = {
  playing: boolean;
  fps: number;
  frame: number;
  frameCount: number;
  setFrame: Dispatch<SetStateAction<number>>;
  setPlaying: (playing: boolean) => void;
};

/** Advance one frame per 1/fps seconds while playing; stop on the last frame. */
export function usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying }: Options): void {
  useEffect(() => {
    if (!playing || frameCount <= 0) return;
    const id = setInterval(() => setFrame((f) => Math.min(frameCount - 1, f + 1)), 1000 / Math.max(1, fps));
    return () => clearInterval(id);
  }, [playing, fps, frameCount, setFrame]);

  useEffect(() => {
    if (playing && frame >= frameCount - 1) setPlaying(false);
  }, [playing, frame, frameCount, setPlaying]);
}
