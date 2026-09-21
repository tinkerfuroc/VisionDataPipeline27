import { Link, Route, Routes, useParams } from "react-router-dom";
import { EngineBanner } from "./components/EngineBanner";
import { HealthContext, useHealth } from "./hooks/useHealth";
import { Clips } from "./pages/Clips";
import { Label } from "./pages/Label";

// A fresh `key` per clip id remounts the whole Label subtree on navigation between clips, instead of reusing
// one instance's state (refs, sessionStorage-derived job id, frame, ...) across two different clips.
function LabelRoute() {
  const { clipId } = useParams();
  return <Label key={clipId} />;
}

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
            <Route path="/clips/:clipId" element={<LabelRoute />} />
          </Routes>
        </main>
      </div>
    </HealthContext.Provider>
  );
}
