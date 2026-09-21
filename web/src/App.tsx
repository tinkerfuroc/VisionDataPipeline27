import { Link, Route, Routes } from "react-router-dom";

export function App() {
  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-4 border-b border-neutral-700 px-4 py-2">
        <Link to="/" className="font-semibold">
          Vision Sampler
        </Link>
      </header>
      <main className="min-h-0 flex-1">
        <Routes>
          <Route path="/" element={<p className="p-4">Clips</p>} />
          <Route path="/clips/:clipId" element={<p className="p-4">Label</p>} />
        </Routes>
      </main>
    </div>
  );
}
