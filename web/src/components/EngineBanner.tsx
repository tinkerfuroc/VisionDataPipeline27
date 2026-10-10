import type { HealthState } from "../hooks/useHealth";

const code = "rounded-sm bg-surround px-1 font-cond";

export function EngineBanner({ health, unreachable }: HealthState) {
  if (unreachable) {
    return (
      <div role="alert" className="alert">
        Cannot reach the server. Start it with <code className={code}>HF_HUB_OFFLINE=1 uv run vdp serve</code>.
      </div>
    );
  }
  if (health === null || health.engine.loaded) return null;
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
      <code className={code}>uv run vdp convert-weights</code>.
    </div>
  );
}
