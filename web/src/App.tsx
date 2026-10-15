import { Route, Routes, useParams } from "react-router-dom";
import { HealthContext, useHealth } from "./hooks/useHealth";
import { Clips } from "./pages/Clips";
import { Export } from "./pages/Export";
import { Label } from "./pages/Label";
import { PhoneUpload } from "./pages/PhoneUpload";
import { Train } from "./pages/Train";

// A fresh `key` per clip id remounts the whole Label subtree on navigation between clips, instead of reusing
// one instance's state (refs, sessionStorage-derived job id, frame, ...) across two different clips.
function LabelRoute() {
  const { clipId } = useParams();
  return <Label key={clipId} />;
}

// Each page renders its own AppHeader (and with it the engine banner), so the Label page can put the clip's name in that row.
export function App() {
  const health = useHealth();
  return (
    <HealthContext.Provider value={health}>
      <div className="flex h-full flex-col">
        <main className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<Clips />} />
            <Route path="/clips/:clipId" element={<LabelRoute />} />
            <Route path="/projects/:projectId/export" element={<Export />} />
            <Route path="/train" element={<Train />} />
            <Route path="/upload" element={<PhoneUpload />} />
          </Routes>
        </main>
      </div>
    </HealthContext.Provider>
  );
}
