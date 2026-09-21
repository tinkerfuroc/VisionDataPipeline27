import { useContext, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { HealthContext } from "../hooks/useHealth";
import { EngineBanner } from "./EngineBanner";

/**
 * The one row of page chrome: the wordmark, whatever the page puts beside it, and the engine state on the right.
 * The engine banner hangs under it, so every page that has a header also reports a missing engine.
 */
export function AppHeader({ children }: { children?: ReactNode }) {
  const state = useContext(HealthContext);
  const engine = state.health?.engine ?? null;
  return (
    <>
      <header className="flex h-10 shrink-0 items-center gap-3 border-b border-line bg-panel px-3 text-sm">
        <Link to="/" className="font-cond text-[0.9375rem] font-semibold tracking-wide">
          Vision Sampler
        </Link>
        <div className="flex min-w-0 flex-1 items-center gap-3">{children}</div>
        {engine?.loaded && (
          <span className="flex shrink-0 items-center gap-1.5 font-cond text-dim">
            <span aria-hidden className="inline-block h-1.5 w-1.5 rounded-full bg-ink" />
            SAM3 on {engine.device}
          </span>
        )}
      </header>
      <EngineBanner {...state} />
    </>
  );
}
