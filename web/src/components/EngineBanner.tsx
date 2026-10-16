import { useState } from "react";
import { api } from "../api/rest";
import type { HealthState } from "../hooks/useHealth";

const code = "rounded-sm bg-surround px-1 font-cond";

export function EngineBanner({ health, unreachable, refresh }: HealthState) {
  if (unreachable) {
    return (
      <div role="alert" className="alert">
        Cannot reach the server. Start it with <code className={code}>HF_HUB_OFFLINE=1 uv run vdp serve</code>.
      </div>
    );
  }
  if (health === null || health.engine.loaded) return null;
  if (health.engine.paused) return <PausedBanner device={health.engine.device} refresh={refresh} />;
  if (health.engine.error === null) {
    return (
      <div role="status" className="border-l-2 border-dim bg-well px-3 py-2 text-sm">
        Loading the SAM3 model. You can import and browse clips now; prompting and propagation start working when it is ready.
      </div>
    );
  }
  return (
    <div role="alert" className="alert">
      The SAM3 engine is not loaded: {health.engine.error}. Importing, browsing and frame cleanup still work. Check the
      weights with <code className={code}>uv run vdp verify-weights</code>; rebuild them with{" "}
      <code className={code}>uv run vdp convert-weights</code>. A load that ran out of memory (e.g. while training holds
      the GPU) can simply be retried. <LoadButton label="Retry loading" refresh={refresh} />
    </div>
  );
}

/** POSTs /api/engine/load (a no-op on the server if it is already loaded or loading), then re-polls health. */
function LoadButton({ label, refresh }: { label: string; refresh?: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.loadEngine();
      refresh?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <button type="button" className="btn" disabled={busy} onClick={() => void load()}>{label}</button>
      {error !== null && <span className="text-xs text-danger">{error}</span>}
    </>
  );
}

/** SAM3 was unloaded so training on its GPU gets that memory back; it reloads by itself once that training ends. */
function PausedBanner({ device, refresh }: { device: string | null; refresh?: () => void }) {
  return (
    <div role="status" className="flex flex-col gap-1 border-l-2 border-dim bg-well px-3 py-2 text-sm">
      <span className="flex flex-wrap items-center gap-2">
        SAM3 paused for training ·
        <LoadButton label="Reload now" refresh={refresh} />
      </span>
      <span className="note">
        It reloads by itself when training on {device ?? "its GPU"} ends. Reloading now, while that training runs, brings
        the memory pressure back: labeling or training may run out of VRAM.
      </span>
    </div>
  );
}
