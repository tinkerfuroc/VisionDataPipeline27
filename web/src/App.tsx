import { Link, Route, Routes } from "react-router-dom";
import { EngineBanner } from "./components/EngineBanner";
import { HealthContext, useHealth } from "./hooks/useHealth";
import { Clips } from "./pages/Clips";
import { Label } from "./pages/Label";

export function App() {
  const health = useHealth();
  return (
    <HealthContext.Provider value={health}>
      <div className="flex h-full flex-col">
        <header className="flex items-center gap-4 border-b border-neutral-700 px-4 py-2">
          <Link to="/" className="font-semibold">
            Vision Sampler
          </Link>
          {health.health?.engine.loaded && (
            <span className="text-xs text-neutral-400">SAM3 on {health.health.engine.device}</span>
          )}
        </header>
        <EngineBanner {...health} />
        <main className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<Clips />} />
            <Route path="/clips/:clipId" element={<Label />} />
          </Routes>
        </main>
      </div>
    </HealthContext.Provider>
  );
}
