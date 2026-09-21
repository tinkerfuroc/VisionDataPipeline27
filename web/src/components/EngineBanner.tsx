import type { HealthState } from "../hooks/useHealth";

const code = "rounded bg-black/30 px-1 font-mono";

export function EngineBanner({ health, unreachable }: HealthState) {
  if (unreachable) {
    return (
      <div role="alert" className="bg-red-900 px-4 py-2 text-sm">
        Cannot reach the server. Start it with <code className={code}>HF_HUB_OFFLINE=1 uv run vsp serve</code>.
      </div>
    );
  }
  if (health === null || health.engine.loaded) return null;
  if (health.engine.error === null) {
    return (
      <div role="status" className="bg-sky-900 px-4 py-2 text-sm">
        Loading the SAM3 model… prompting and propagation become available when it is ready.
      </div>
    );
  }
  return (
    <div role="alert" className="bg-red-900 px-4 py-2 text-sm">
      The SAM3 engine is not loaded: {health.engine.error}. Importing, browsing and frame cleanup still work. Check the
      weights with <code className={code}>uv run vsp verify-weights</code>; rebuild them with{" "}
      <code className={code}>uv run vsp convert-weights</code>.
    </div>
  );
}
