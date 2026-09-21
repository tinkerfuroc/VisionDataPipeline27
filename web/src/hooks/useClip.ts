import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { applyFrameEvent } from "../lib/timeline";

export type ClipHandle = {
  detail: ClipDetail | null;
  error: string | null;
  reload: () => Promise<void>;
  /** adopt the ClipDetail a mutation returned; `masksChanged` re-versions every mask URL */
  apply: (detail: ClipDetail, masksChanged: boolean) => void;
  /** a propagation job wrote (or cleared) this frame's mask */
  patchFrame: (frame: number, hasMask: boolean) => void;
  /** cache-busting version for a frame's mask URL */
  maskVersion: (frame: number) => string;
};

export function useClip(clipId: string): ClipHandle {
  const [detail, setDetail] = useState<ClipDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const versions = useRef({ epoch: 0, frames: new Map<number, number>() });

  const reload = useCallback(async () => {
    try {
      const next = await api.getClip(clipId);
      versions.current.epoch++;
      setDetail(next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [clipId]);

  useEffect(() => {
    setDetail(null);
    versions.current = { epoch: 0, frames: new Map() };
    void reload();
  }, [reload]);

  const apply = useCallback((next: ClipDetail, masksChanged: boolean) => {
    if (masksChanged) versions.current.epoch++;
    setDetail(next);
    setTick((t) => t + 1);
  }, []);

  const patchFrame = useCallback((frame: number, hasMask: boolean) => {
    const frames = versions.current.frames;
    frames.set(frame, (frames.get(frame) ?? 0) + 1);
    setDetail((d) => (d === null ? d : { ...d, states: applyFrameEvent(d.states, frame, hasMask) }));
    setTick((t) => t + 1);
  }, []);

  const maskVersion = useCallback(
    (frame: number) => `${versions.current.epoch}.${versions.current.frames.get(frame) ?? 0}`,
    [],
  );

  return { detail, error, reload, apply, patchFrame, maskVersion };
}
