import { createContext, useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import type { Health } from "../api/types";

/** `refresh` polls right away (e.g. after asking the server to load the engine); absent outside `useHealth`. */
export type HealthState = { health: Health | null; unreachable: boolean; refresh?: () => void };

export const HealthContext = createContext<HealthState>({ health: null, unreachable: false });

export const engineReady = (s: HealthState): boolean => !s.unreachable && s.health !== null && s.health.engine.loaded;

const FAST_MS = 2000; // while the model is still loading
const SLOW_MS = 15000; // loaded, failed, or paused for training (which lasts until the training ends)

export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ health: null, unreachable: false });
  const pollNow = useRef<() => void>(() => {});

  useEffect(() => {
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async (): Promise<void> => {
      let delay = FAST_MS;
      try {
        const health = await api.health();
        if (closed) return;
        setState({ health, unreachable: false });
        if (health.engine.loaded || health.engine.error !== null || health.engine.paused) delay = SLOW_MS;
      } catch {
        if (closed) return;
        setState((s) => ({ health: s.health, unreachable: true }));
      }
      timer = setTimeout(() => void tick(), delay);
    };
    pollNow.current = () => {
      if (timer !== undefined) clearTimeout(timer);
      void tick();
    };
    void tick();
    return () => {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, []);

  const refresh = useCallback(() => pollNow.current(), []);
  return { ...state, refresh };
}
