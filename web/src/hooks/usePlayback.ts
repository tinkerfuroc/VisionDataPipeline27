import { useEffect, useRef, type Dispatch, type SetStateAction } from "react";

type Options = {
  playing: boolean;
  fps: number;
  frame: number;
  frameCount: number;
  setFrame: Dispatch<SetStateAction<number>>;
  setPlaying: (playing: boolean) => void;
};

/** Advance one frame per 1/fps seconds while playing; stop on the last frame — unless playback was just started
 * while already parked there, in which case it restarts from frame 0 instead of stopping immediately. */
export function usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying }: Options): void {
  const wasPlaying = useRef(playing);

  useEffect(() => {
    if (!playing || frameCount <= 0) return;
    const id = setInterval(() => setFrame((f) => Math.min(frameCount - 1, f + 1)), 1000 / Math.max(1, fps));
    return () => clearInterval(id);
  }, [playing, fps, frameCount, setFrame]);

  useEffect(() => {
    const justStarted = playing && !wasPlaying.current;
    wasPlaying.current = playing;
    if (!playing || frameCount <= 0 || frame < frameCount - 1) return;
    if (justStarted) setFrame(0);
    else setPlaying(false);
  }, [playing, frame, frameCount, setFrame, setPlaying]);
}
