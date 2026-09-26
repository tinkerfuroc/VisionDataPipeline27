import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { patchObjectFrames, type FrameEvent } from "../lib/objects";

export type ClipHandle = {
  detail: ClipDetail | null;
  error: string | null;
  /** a background refresh failed while a clip was already loaded; the stale detail is still shown */
  refreshError: string | null;
  reload: () => Promise<void>;
  /** adopt the ClipDetail a mutation returned; `masksChanged` re-versions every mask URL */
  apply: (detail: ClipDetail, masksChanged: boolean) => void;
  /** apply a batch of propagation frame events in one update: each listed object's lane follows its event, the
   * clip-level `states` are recomputed from all lanes, and every event's frame gets a new mask version (coalesces a
   * burst of streamed events into one render) */
  patchFrames: (events: FrameEvent[]) => void;
  /** cache-busting version for a frame's mask URL */
  maskVersion: (frame: number) => string;
};

export function useClip(clipId: string): ClipHandle {
  const [detail, setDetail] = useState<ClipDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const versions = useRef({ epoch: 0, frames: new Map<number, number>() });
  // Guards against a slow response (a reload, or a previous clip id's initial load) landing after a newer one.
  const generation = useRef(0);
  // Written synchronously next to the state it describes (a mirror effect would lag one commit behind `detail`).
  const loaded = useRef(false);

  const reload = useCallback(async () => {
    const gen = ++generation.current;
    try {
      const next = await api.getClip(clipId);
      if (gen !== generation.current) return; // superseded while in flight
      versions.current.epoch++;
      loaded.current = true;
      setDetail(next);
      setError(null);
      setRefreshError(null);
    } catch (e) {
      if (gen !== generation.current) return;
      const message = e instanceof Error ? e.message : String(e);
      if (!loaded.current) setError(message);
      else setRefreshError(message);
    }
  }, [clipId]);

  useEffect(() => {
    generation.current++; // drop any response still in flight for the previous clip id
    loaded.current = false;
    setDetail(null);
    setError(null);
    setRefreshError(null);
    versions.current = { epoch: 0, frames: new Map() };
    void reload();
  }, [reload]);

  const apply = useCallback((next: ClipDetail, masksChanged: boolean) => {
    if (masksChanged) versions.current.epoch++;
    setDetail(next);
    setTick((t) => t + 1);
  }, []);

  const patchFrames = useCallback((events: FrameEvent[]) => {
    if (events.length === 0) return;
    const frames = versions.current.frames;
    for (const { frame } of events) frames.set(frame, (frames.get(frame) ?? 0) + 1);
    setDetail((d) => (d === null ? d : patchObjectFrames(d, events)));
    setTick((t) => t + 1);
  }, []);

  const maskVersion = useCallback(
    (frame: number) => `${versions.current.epoch}.${versions.current.frames.get(frame) ?? 0}`,
    [],
  );

  return { detail, error, refreshError, reload, apply, patchFrames, maskVersion };
}
