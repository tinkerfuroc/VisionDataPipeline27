import { createContext, useEffect, useState } from "react";
import { api } from "../api/rest";
import type { Health } from "../api/types";

export type HealthState = { health: Health | null; unreachable: boolean };

export const HealthContext = createContext<HealthState>({ health: null, unreachable: false });

export const engineReady = (s: HealthState): boolean => !s.unreachable && s.health !== null && s.health.engine.loaded;

const FAST_MS = 2000; // while the model is still loading
const SLOW_MS = 15000;

export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ health: null, unreachable: false });

  useEffect(() => {
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async (): Promise<void> => {
      let delay = FAST_MS;
      try {
        const health = await api.health();
        if (closed) return;
        setState({ health, unreachable: false });
        if (health.engine.loaded || health.engine.error !== null) delay = SLOW_MS;
      } catch {
        if (closed) return;
        setState((s) => ({ health: s.health, unreachable: true }));
      }
      timer = setTimeout(() => void tick(), delay);
    };
    void tick();
    return () => {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, []);

  return state;
}
