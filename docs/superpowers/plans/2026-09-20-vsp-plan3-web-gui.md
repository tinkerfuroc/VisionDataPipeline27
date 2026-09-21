# Vision Sampler Plan 3 — Web GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser GUI for the `vsp` backend: manage projects/classes, import clips, label one object per clip by click / box / text prompt, propagate with live progress, clean up frames, and review on a timeline.

**Architecture:** A React single-page app in `web/`, built by Vite into `web/dist`, which `vsp serve` already mounts at `/`. All logic that can be pure (timeline derivation, view geometry, mask tinting, prompt drafting, job-event reduction, REST error handling) lives in `web/src/lib` and `web/src/api` and is unit-tested with vitest in a node environment; React components are thin and get jsdom mount tests with a mocked API. The frame view is a plain `<canvas>` (no Konva). Routing uses `HashRouter`, so no server change is needed for deep links. **This plan changes no Python code.**

**Tech Stack:** Node 22 (nvm), React 18, react-router-dom 6, Vite 5, TypeScript 5, Tailwind 3, vitest 2, @testing-library/react + jsdom. Same majors as the template (`tinkerfuroc/tk25_vision_train/web`), from which the `request()` helper shape and the Vite/Tailwind config are ported (MIT — already covered by `THIRD_PARTY_NOTICES.md`).

**Spec:** `docs/superpowers/specs/2026-09-20-vision-sampler-design.md` (§7 GUI, §4 API table, §9 errors, §10 testing). API contract as built: `docs/superpowers/plans/2026-09-20-vsp-roadmap.md` → "Carried forward from Plan 2".

## Global Constraints

- Node is installed with nvm and is NOT on the PATH of non-interactive shells: every command that uses `node`/`npm`/`npx` must run as `bash -c 'source ~/.nvm/nvm.sh && cd web && <command>'`.
- Dependency majors are fixed: `react@^18.3.1`, `react-dom@^18.3.1`, `react-router-dom@^6.30.3`, `vite@^5.4.0`, `@vitejs/plugin-react@^4.3.0`, `typescript@^5.6.0`, `tailwindcss@^3.4.0`, `vitest@^2.1.0`, `jsdom@^25.0.0`, `@testing-library/react@^16.1.0`, `@testing-library/dom@^10.4.0`. `web/package.json` and `web/package-lock.json` change ONLY in Task 1. No other runtime dependency (no Konva, zustand, react-query, axios).
- No Python file, `pyproject.toml` or `uv.lock` changes in this plan. Python is still managed with uv only (`uv run …`); never pip or conda.
- Never commit or stage anything under `vision_train/`, `data/`, `web/node_modules/`, `web/dist/`, or any `*.pt` / `*.safetensors`. Stage files by explicit path.
- This machine exports `http_proxy`: any `curl` to localhost needs `--noproxy '*'`; anything that loads the model needs `HF_HUB_OFFLINE=1`.
- The dev server proxies `/api` and `/ws` to `http://127.0.0.1:28000`; the app only ever uses same-origin relative URLs.
- Single object per clip (v1): the UI never offers a second object; object id is always 1.
- API behaviours the UI must honour (all verified in Plan 2):
  - `POST /propagate` can return 409 with `job_id: "interactive"` for a few ms → retry silently (10 attempts, 150 ms apart). Any other 409 carries the id of the running job.
  - Error bodies are `{"detail": "<string>"}`, except FastAPI validation errors: 422 with `detail` as a list of `{loc, msg}`.
  - Propagation may only start from a keyframe. `prompt`/`accept`/`clear-range`/`DELETE clip` return 409 while a GPU job runs; `delete`/`restore`/`prune` stay allowed.
  - Job events carry a monotonic `seq`; the WebSocket replays from `seq` 0 on every connect, so events with `seq` lower than the next expected one must be ignored. Import `progress` events have no `total`.
  - Drive the timeline from `states` (`empty | masked | keyframe | deleted`), never from `propagated_ranges`.
  - Masks are PNGs where the object is white; `GET …/masks/{i}` is 404 when the object is absent — only request masks for frames whose state is `masked` or `keyframe`. Mask URLs carry a `?v=` version so an overwritten mask is never served from an in-memory cache.
  - Click coordinates must be integers with `0 <= x < width`, `0 <= y < height`; boxes are `[x1, y1, x2, y2]` integers inside the frame.
- Keyboard map (spec §7): `Enter` accept, `Esc` clear, `←/→` step, `Shift+←/→` ±10, `Space` play/pause, `K` / `Shift+K` next / previous keyframe, `G` next empty frame, `Del` delete (or restore) frame, `M` toggle overlay, `Z` undo last click. Shortcuts are ignored while focus is in an input, textarea or select.
- The Export dialog is Plan 4 — do not build it here.
- Commit messages end with a `Co-Authored-By: Claude …` line and `Claude-Session: https://claude.ai/code/session_01YKeRXoSnVkKqPgN7Pd7bSL`.

## File Structure

```
web/
  package.json  package-lock.json  tsconfig.json  vite.config.ts
  tailwind.config.js  postcss.config.js  index.html
  src/
    main.tsx            entry; mounts <App/> under HashRouter
    App.tsx             header + engine banner + routes (/ → Clips, /clips/:clipId → Label)
    styles.css          Tailwind layers + full-height root
    api/types.ts        TypeScript mirrors of the server's JSON shapes
    api/rest.ts         request(), ApiError, detailText(), retryInteractive(), api.*, URL builders
    lib/timeline.ts     runs(), findNext(), labeledPercent(), counts(), applyFrameEvent(), frameAtX(), clampFrame()
    lib/geometry.ts     View transform: fitView, zoomAt, panBy, toImage, toScreen, clampPoint, normBox, pickCandidate
    lib/mask.ts         hexToRgb(), tintMask()
    lib/promptState.ts  Draft reducer + toPrompt()
    lib/jobReducer.ts   JobView reducer over job events / snapshots
    lib/frameCache.ts   LruCache, loadImage(), FrameStore (frames + tinted mask layers)
    hooks/useHealth.ts  polls /api/health
    hooks/useJob.ts     WebSocket job stream with polling fallback
    hooks/useClip.ts    clip detail + mask versions + mutations
    hooks/usePlayback.ts   fps timer
    hooks/usePrompting.ts  draft → preview → accept
    hooks/useKeyboard.ts   keyboard map
    components/EngineBanner.tsx  ImportJobRow.tsx  FrameCanvas.tsx  Timeline.tsx  Transport.tsx
               ObjectPanel.tsx  PromptPanel.tsx  PropagatePanel.tsx  CleanupPanel.tsx
    pages/Clips.tsx  pages/Label.tsx
    test/setup-dom.ts   jsdom stubs (canvas context, ResizeObserver) shared by component tests
scripts/seed_demo.py    creates a demo project with a synthetic moving disc through the HTTP API (manual QA aid)
```

Tests sit next to their subject as `*.test.ts(x)`. Pure-logic tests run in vitest's default `node` environment; component tests start with the docblock `// @vitest-environment jsdom`.

---

### Task 1: Scaffold, typed API client, served by `vsp serve`

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/tailwind.config.js`, `web/postcss.config.js`, `web/index.html`, `web/src/main.tsx`, `web/src/App.tsx`, `web/src/styles.css`, `web/src/vite-env.d.ts`, `web/src/api/types.ts`, `web/src/api/rest.ts`
- Test: `web/src/api/rest.test.ts`
- Modify: `README.md` (add a "Web GUI" section)

**Interfaces:**
- Consumes: the HTTP API on `main` (see the roadmap's "Carried forward from Plan 2").
- Produces (imported by every later task):
  - `web/src/api/types.ts`: `ClassDef`, `Project`, `PromptType`, `Point = [number, number, 0 | 1]`, `Box = [number, number, number, number]`, `PromptSpec`, `Keyframe`, `LabelObject`, `Clip`, `ClipSummary`, `FrameState`, `ClipDetail`, `Candidate`, `JobStatus`, `JobSnapshot`, `JobEvent`, `Health`, `Direction`.
  - `web/src/api/rest.ts`: `class ApiError extends Error { status: number; jobId: string | null }`, `INTERACTIVE`, `detailText(body, fallback)`, `request<T>(path, init?)`, `retryInteractive<T>(fn, attempts?, delayMs?, sleep?)`, `frameUrl(clipId, idx)`, `thumbUrl(clipId, idx)`, `maskUrl(clipId, idx, version)`, `wsUrl(path)`, and the `api` object with: `health()`, `listProjects()`, `createProject(name)`, `getProject(id)`, `addClass(projectId, name)`, `listClips(projectId)`, `importFrames(projectId, {path, fps, stride})`, `importVideo(projectId, file, stride)`, `getClip(id)`, `deleteClip(id)`, `prompt(clipId, frame, prompt)`, `accept(clipId, frame, prompt, {classId?, candidate?})`, `propagate(clipId, fromFrame, direction)`, `deleteFrame(clipId, idx)`, `restoreFrame(clipId, idx)`, `prune(clipId, from)`, `clearRange(clipId, start, end)`, `getJob(id)`, `cancelJob(id)`.

- [ ] **Step 1: Create the package and install pinned majors**

Create `web/package.json`:

```json
{
  "name": "vsp-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "typecheck": "tsc --noEmit",
    "test": "vitest run"
  }
}
```

Run:

```bash
bash -c 'source ~/.nvm/nvm.sh && cd web && npm install --save react@^18.3.1 react-dom@^18.3.1 react-router-dom@^6.30.3'
bash -c 'source ~/.nvm/nvm.sh && cd web && npm install --save-dev vite@^5.4.0 @vitejs/plugin-react@^4.3.0 typescript@^5.6.0 tailwindcss@^3.4.0 postcss@^8.4.0 autoprefixer@^10.4.0 vitest@^2.1.0 jsdom@^25.0.0 @testing-library/react@^16.1.0 @testing-library/dom@^10.4.0 @types/react@^18.3.0 @types/react-dom@^18.3.0'
```

Expected: both finish without `ERR!`; `web/package-lock.json` exists; `web/node_modules/` is git-ignored (`git status --short` shows only `web/package.json` and `web/package-lock.json`).

- [ ] **Step 2: Tooling config**

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "noEmit": true
  },
  "include": ["src", "vite.config.ts"]
}
```

`web/vite.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:28000",
      "/ws": { target: "ws://127.0.0.1:28000", ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
  test: { environment: "node", include: ["src/**/*.test.{ts,tsx}"] },
});
```

`web/tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

`web/postcss.config.js`:

```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

`web/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vision Sampler</title>
  </head>
  <body class="bg-neutral-900 text-neutral-100">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`web/src/vite-env.d.ts`:

```ts
/// <reference types="vite/client" />
```

`web/src/styles.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

html,
body,
#root {
  height: 100%;
}
```

`web/src/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter } from "react-router-dom";
import { App } from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>,
);
```

`web/src/App.tsx` (Task 3 adds the engine banner, Tasks 4–5 replace the two placeholder routes):

```tsx
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
```

- [ ] **Step 3: API types**

`web/src/api/types.ts`:

```ts
export type ClassDef = { id: number; name: string; color: string };
export type Project = { id: string; name: string; created_at: string; classes: ClassDef[] };

export type PromptType = "text" | "clicks" | "box";
/** [x, y, label] — label 1 = positive, 0 = negative */
export type Point = [number, number, 0 | 1];
/** [x1, y1, x2, y2] in image pixels */
export type Box = [number, number, number, number];
export type PromptSpec = { type: PromptType; phrase?: string; points?: Point[]; box?: Box };

export type Keyframe = {
  frame_idx: number;
  prompt: { type: PromptType; payload: Record<string, unknown> };
  created_at: string;
};
export type LabelObject = {
  id: number;
  class_id: number;
  keyframes: Keyframe[];
  propagated_ranges: [number, number][];
};
export type Clip = {
  id: string;
  project_id: string;
  source: { type: "video" | "frame_dir"; name: string };
  fps: number;
  stride: number;
  width: number;
  height: number;
  frame_count: number;
  created_at: string;
  original_names?: Record<string, string>;
  deleted_frames: number[];
  objects: LabelObject[];
};
export type ClipSummary = Clip & { labeled_frames: number };
export type FrameState = "empty" | "masked" | "keyframe" | "deleted";
export type ClipDetail = { clip: Clip; states: FrameState[]; classes: ClassDef[] };

export type Candidate = {
  score: number;
  bbox: Box;
  area: number;
  phrase: string | null;
  mask_png_b64: string;
};

export type Direction = "forward" | "backward" | "both";
export type JobStatus = "pending" | "running" | "done" | "error" | "cancelled";
export type JobSnapshot = {
  id: string;
  kind: string;
  status: JobStatus;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  n_events: number;
};
export type JobEvent = {
  seq: number;
  type: "progress" | "frame" | "log" | "done" | "cancelled" | "error";
  [key: string]: unknown;
};

export type Health = {
  status: string;
  version: string;
  engine: { loaded: boolean; device: string | null; error: string | null };
};
```

- [ ] **Step 4: Write the failing tests for the REST client**

`web/src/api/rest.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, detailText, maskUrl, request, retryInteractive } from "./rest";

function mockFetch(...responses: Response[]) {
  const fn = vi.fn();
  for (const r of responses) fn.mockResolvedValueOnce(r);
  vi.stubGlobal("fetch", fn);
  return fn;
}
const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

describe("request", () => {
  it("returns the parsed JSON body", async () => {
    mockFetch(jsonResponse({ ok: 1 }));
    expect(await request<{ ok: number }>("/api/x")).toEqual({ ok: 1 });
  });

  it("returns undefined for 204", async () => {
    mockFetch(new Response(null, { status: 204 }));
    expect(await request<void>("/api/x", { method: "DELETE" })).toBeUndefined();
  });

  it("raises ApiError with the string detail and the job id of a 409", async () => {
    mockFetch(jsonResponse({ detail: "job abc is still running", job_id: "abc" }, 409));
    const err = await request("/api/x").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.message).toBe("job abc is still running");
    expect(err.jobId).toBe("abc");
  });

  it("flattens a FastAPI 422 detail list", async () => {
    mockFetch(
      jsonResponse(
        { detail: [{ loc: ["body", "prompt", "box"], msg: "Field required" }, { loc: ["body", "frame"], msg: "too small" }] },
        422,
      ),
    );
    const err = await request("/api/x").catch((e) => e);
    expect(err.status).toBe(422);
    expect(err.message).toBe("prompt.box: Field required; frame: too small");
    expect(err.jobId).toBeNull();
  });

  it("falls back to the status line when the error body is not JSON", async () => {
    mockFetch(new Response("<html>bad gateway</html>", { status: 502, statusText: "Bad Gateway" }));
    const err = await request("/api/x").catch((e) => e);
    expect(err.message).toBe("502 Bad Gateway");
  });
});

describe("detailText", () => {
  it("uses the fallback for bodies without a detail", () => {
    expect(detailText({ nope: 1 }, "fallback")).toBe("fallback");
    expect(detailText(null, "fallback")).toBe("fallback");
  });
});

describe("retryInteractive", () => {
  const noSleep = () => Promise.resolve();

  it("retries while the 409 names the interactive lease, then succeeds", async () => {
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new ApiError(409, "busy", "interactive"))
      .mockRejectedValueOnce(new ApiError(409, "busy", "interactive"))
      .mockResolvedValueOnce("ok");
    expect(await retryInteractive(fn, 10, 1, noSleep)).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("does not retry a 409 that names a real job", async () => {
    const fn = vi.fn<() => Promise<string>>().mockRejectedValue(new ApiError(409, "busy", "abc123"));
    await expect(retryInteractive(fn, 10, 1, noSleep)).rejects.toMatchObject({ jobId: "abc123" });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("gives up after the configured number of attempts", async () => {
    const fn = vi.fn<() => Promise<string>>().mockRejectedValue(new ApiError(409, "busy", "interactive"));
    await expect(retryInteractive(fn, 3, 1, noSleep)).rejects.toBeInstanceOf(ApiError);
    expect(fn).toHaveBeenCalledTimes(3);
  });
});

describe("api", () => {
  it("posts a propagate request with the server's field names", async () => {
    const fetchMock = mockFetch(jsonResponse({ job_id: "j1" }, 202));
    expect(await api.propagate("c1", 7, "both")).toEqual({ job_id: "j1" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clips/c1/propagate");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ from_frame: 7, direction: "both" });
  });

  it("sends accept with class_id and candidate only when given", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({}));
    await api.accept("c1", 3, { type: "clicks", points: [[5, 6, 1]] }, { classId: 2, candidate: 1 });
    await api.accept("c1", 3, { type: "clicks", points: [[5, 6, 1]] });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      frame: 3, prompt: { type: "clicks", points: [[5, 6, 1]] }, class_id: 2, candidate: 1,
    });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
      frame: 3, prompt: { type: "clicks", points: [[5, 6, 1]] }, candidate: 0,
    });
  });

  it("uses `from` as the prune field name and uploads videos as multipart", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({ job_id: "j2" }, 202));
    await api.prune("c1", 12);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ from: 12 });
    await api.importVideo("p1", new File(["x"], "a.mp4"), 2);
    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toBe("/api/projects/p1/clips/import-video");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("stride")).toBe("2");
    expect(init.headers).toBeUndefined();
  });

  it("versions mask URLs", () => {
    expect(maskUrl("c1", 4, "2.1")).toBe("/api/clips/c1/masks/4?v=2.1");
  });
});
```

- [ ] **Step 5: Run the tests to see them fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/api/rest.test.ts'`
Expected: FAIL — cannot resolve `./rest`.

- [ ] **Step 6: Implement the REST client**

`web/src/api/rest.ts`:

```ts
import type {
  Candidate, ClassDef, ClipDetail, ClipSummary, Direction, Health, JobSnapshot, Project, PromptSpec,
} from "./types";

/** The `job_id` the server reports while a prompt/accept/clear-range call holds the GPU lease. */
export const INTERACTIVE = "interactive";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly jobId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** `detail` is a string for the server's own errors and a list of {loc, msg} for FastAPI validation errors. */
export function detailText(body: unknown, fallback: string): string {
  if (body === null || typeof body !== "object" || !("detail" in body)) return fallback;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e: { loc?: unknown[]; msg?: unknown }) => {
        const loc = Array.isArray(e?.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
        const msg = String(e?.msg ?? "invalid");
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  return fallback;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (r.status === 204) return undefined as T;
  let body: unknown = null;
  try {
    body = await r.json();
  } catch {
    /* not JSON: keep null */
  }
  if (!r.ok) {
    const jobId =
      body !== null && typeof body === "object" && typeof (body as { job_id?: unknown }).job_id === "string"
        ? (body as { job_id: string }).job_id
        : null;
    throw new ApiError(r.status, detailText(body, `${r.status} ${r.statusText}`), jobId);
  }
  return body as T;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** A propagate request loses to an in-flight preview for a few milliseconds; that 409 is not an error. */
export async function retryInteractive<T>(
  fn: () => Promise<T>,
  attempts = 10,
  delayMs = 150,
  sleep: (ms: number) => Promise<void> = defaultSleep,
): Promise<T> {
  for (let i = 1; ; i++) {
    try {
      return await fn();
    } catch (e) {
      const retry = e instanceof ApiError && e.status === 409 && e.jobId === INTERACTIVE && i < attempts;
      if (!retry) throw e;
      await sleep(delayMs);
    }
  }
}

const post = (body?: unknown): RequestInit =>
  body === undefined
    ? { method: "POST" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };

export const frameUrl = (clipId: string, idx: number) => `/api/clips/${clipId}/frames/${idx}`;
export const thumbUrl = (clipId: string, idx: number) => `/api/clips/${clipId}/thumbs/${idx}`;
export const maskUrl = (clipId: string, idx: number, version: string) => `/api/clips/${clipId}/masks/${idx}?v=${version}`;
export const wsUrl = (path: string) =>
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}${path}`;

export const api = {
  health: () => request<Health>("/api/health"),

  listProjects: () => request<Project[]>("/api/projects"),
  createProject: (name: string) => request<Project>("/api/projects", post({ name })),
  getProject: (id: string) => request<Project>(`/api/projects/${id}`),
  addClass: (projectId: string, name: string) => request<ClassDef>(`/api/projects/${projectId}/classes`, post({ name })),
  listClips: (projectId: string) => request<ClipSummary[]>(`/api/projects/${projectId}/clips`),

  importFrames: (projectId: string, body: { path: string; fps: number; stride: number }) =>
    request<{ job_id: string }>(`/api/projects/${projectId}/clips/import-frames`, post(body)),
  importVideo: (projectId: string, file: File, stride: number) => {
    const form = new FormData();
    form.append("file", file);
    form.append("stride", String(stride));
    // no Content-Type header: the browser must add the multipart boundary itself
    return request<{ job_id: string }>(`/api/projects/${projectId}/clips/import-video`, { method: "POST", body: form });
  },

  getClip: (id: string) => request<ClipDetail>(`/api/clips/${id}`),
  deleteClip: (id: string) => request<void>(`/api/clips/${id}`, { method: "DELETE" }),

  prompt: (clipId: string, frame: number, prompt: PromptSpec) =>
    request<{ candidates: Candidate[] }>(`/api/clips/${clipId}/prompt`, post({ frame, prompt })),
  accept: (clipId: string, frame: number, prompt: PromptSpec, opts: { classId?: number; candidate?: number } = {}) =>
    request<ClipDetail>(
      `/api/clips/${clipId}/accept`,
      post({
        frame,
        prompt,
        ...(opts.classId === undefined ? {} : { class_id: opts.classId }),
        candidate: opts.candidate ?? 0,
      }),
    ),
  propagate: (clipId: string, fromFrame: number, direction: Direction) =>
    request<{ job_id: string }>(`/api/clips/${clipId}/propagate`, post({ from_frame: fromFrame, direction })),

  deleteFrame: (clipId: string, idx: number) => request<ClipDetail>(`/api/clips/${clipId}/frames/${idx}/delete`, post()),
  restoreFrame: (clipId: string, idx: number) => request<ClipDetail>(`/api/clips/${clipId}/frames/${idx}/restore`, post()),
  prune: (clipId: string, from: number) => request<ClipDetail>(`/api/clips/${clipId}/prune`, post({ from })),
  clearRange: (clipId: string, start: number, end: number) =>
    request<ClipDetail>(`/api/clips/${clipId}/clear-range`, post({ start, end })),

  getJob: (id: string) => request<JobSnapshot>(`/api/jobs/${id}`),
  cancelJob: (id: string) => request<JobSnapshot>(`/api/jobs/${id}`, { method: "DELETE" }),
};
```

- [ ] **Step 7: Run tests, typecheck, build**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run build'`
Expected: `13 passed`; the build prints `✓ built in …` and writes `web/dist/index.html`.

- [ ] **Step 8: Verify `vsp serve` serves the build**

```bash
HF_HUB_OFFLINE=1 uv run vsp serve > /tmp/vsp-serve.log 2>&1 &
for i in $(seq 1 40); do curl -s --noproxy '*' -o /dev/null http://127.0.0.1:28000/api/health && break; sleep 0.5; done
curl -s --noproxy '*' http://127.0.0.1:28000/ | grep -c '<div id="root">'
curl -s --noproxy '*' http://127.0.0.1:28000/api/health | head -c 200; echo
kill %1
```

Expected: `1`, then the health JSON (the API still wins over the SPA mount).

- [ ] **Step 9: README section**

Add to `README.md` after the "Run the server" section:

```markdown
## Web GUI

Node 22 is installed with nvm (`source ~/.nvm/nvm.sh` in a fresh shell).

    cd web && npm ci            # once
    npm run build               # writes web/dist, which `vsp serve` serves at http://127.0.0.1:28000/
    npm run dev                 # or: hot-reloading dev server on http://127.0.0.1:5173 (proxies /api and /ws to :28000)
    npm test                    # vitest
```

- [ ] **Step 10: Commit**

```bash
git add web/package.json web/package-lock.json web/tsconfig.json web/vite.config.ts web/tailwind.config.js \
  web/postcss.config.js web/index.html web/src/main.tsx web/src/App.tsx web/src/styles.css web/src/vite-env.d.ts \
  web/src/api/types.ts web/src/api/rest.ts web/src/api/rest.test.ts README.md
git commit -m "feat(web): Vite + React scaffold and typed REST client"
```

---

### Task 2: Pure logic — timeline, geometry, mask tint, prompt draft

**Files:**
- Create: `web/src/lib/timeline.ts`, `web/src/lib/geometry.ts`, `web/src/lib/mask.ts`, `web/src/lib/promptState.ts`
- Test: `web/src/lib/timeline.test.ts`, `web/src/lib/geometry.test.ts`, `web/src/lib/mask.test.ts`, `web/src/lib/promptState.test.ts`

**Interfaces:**
- Consumes: `FrameState`, `Box`, `Point`, `PromptSpec` from `web/src/api/types.ts` (Task 1).
- Produces:
  - `timeline.ts`: `type Run = { state: FrameState; start: number; end: number }` (inclusive), `runs(states)`, `clampFrame(i, count)`, `findNext(states, from, dir: 1 | -1, want)` → `number | null` (strictly beyond `from`, no wrap), `counts(states)`, `labeledPercent(states)`, `applyFrameEvent(states, frame, hasMask)` (returns the SAME array when nothing changes), `frameAtX(x, width, count)`.
  - `geometry.ts`: `type View = { scale; tx; ty }` (screen = image × scale + t), `fitView`, `toImage`, `toScreen`, `zoomAt`, `panBy`, `insideImage`, `clampPoint`, `normBox`, `pickCandidate`.
  - `mask.ts`: `hexToRgb(hex)`, `tintMask(rgba, rgb, alpha)` (in place).
  - `promptState.ts`: `type Tool = "clicks" | "box" | "text"`, `type Draft`, `emptyDraft(tool?)`, `type DraftAction`, `draftReducer(draft, action)`, `toPrompt(draft)` → `PromptSpec | null`.

- [ ] **Step 1: Write the failing tests**

`web/src/lib/timeline.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { FrameState } from "../api/types";
import { applyFrameEvent, clampFrame, counts, findNext, frameAtX, labeledPercent, runs } from "./timeline";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);

describe("runs", () => {
  it("is empty for no frames", () => expect(runs([])).toEqual([]));
  it("run-length encodes with inclusive ends", () => {
    expect(runs(S("eekmmd"))).toEqual([
      { state: "empty", start: 0, end: 1 },
      { state: "keyframe", start: 2, end: 2 },
      { state: "masked", start: 3, end: 4 },
      { state: "deleted", start: 5, end: 5 },
    ]);
  });
});

describe("navigation", () => {
  it("clamps and rounds frame indices", () => {
    expect(clampFrame(-3, 10)).toBe(0);
    expect(clampFrame(12, 10)).toBe(9);
    expect(clampFrame(4.6, 10)).toBe(5);
    expect(clampFrame(5, 0)).toBe(0);
  });
  it("finds the next frame in a state strictly beyond the start, without wrapping", () => {
    const s = S("kmekde");
    expect(findNext(s, 0, 1, "keyframe")).toBe(3);
    expect(findNext(s, 3, 1, "keyframe")).toBeNull();
    expect(findNext(s, 3, -1, "keyframe")).toBe(0);
    expect(findNext(s, 0, 1, "empty")).toBe(2);
    expect(findNext(s, 2, 1, "empty")).toBe(5);
  });
  it("maps a timeline x position to a frame", () => {
    expect(frameAtX(0, 100, 10)).toBe(0);
    expect(frameAtX(55, 100, 10)).toBe(5);
    expect(frameAtX(99.9, 100, 10)).toBe(9);
    expect(frameAtX(100, 100, 10)).toBe(9);
    expect(frameAtX(-5, 100, 10)).toBe(0);
    expect(frameAtX(5, 0, 10)).toBe(0);
  });
});

describe("statistics", () => {
  it("counts every state", () => {
    expect(counts(S("kmeed"))).toEqual({ empty: 2, masked: 1, keyframe: 1, deleted: 1 });
  });
  it("reports labeled frames as a percentage of non-deleted frames", () => {
    expect(labeledPercent(S("kmeed"))).toBe(50);
    expect(labeledPercent(S("dd"))).toBe(0);
    expect(labeledPercent([])).toBe(0);
  });
});

describe("applyFrameEvent", () => {
  it("marks a frame masked or empty and returns a new array", () => {
    const s = S("kee");
    const masked = applyFrameEvent(s, 1, true);
    expect(masked).toEqual(S("kme"));
    expect(masked).not.toBe(s);
    expect(applyFrameEvent(masked, 1, false)).toEqual(S("kee"));
  });
  it("never touches keyframes or deleted frames and returns the same array when nothing changes", () => {
    const s = S("kdm");
    expect(applyFrameEvent(s, 0, false)).toBe(s);
    expect(applyFrameEvent(s, 1, true)).toBe(s);
    expect(applyFrameEvent(s, 2, true)).toBe(s);
    expect(applyFrameEvent(s, 9, true)).toBe(s);
  });
});
```

`web/src/lib/geometry.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { clampPoint, fitView, insideImage, normBox, panBy, pickCandidate, toImage, toScreen, zoomAt } from "./geometry";

describe("view transform", () => {
  it("fits and centres the image inside the box with a margin", () => {
    expect(fitView(640, 480, 1296, 736)).toEqual({ scale: 1.5, tx: 168, ty: 8 });
  });
  it("round-trips between screen and image coordinates", () => {
    const v = { scale: 2, tx: 10, ty: -4 };
    expect(toScreen(v, 5, 7)).toEqual([20, 10]);
    expect(toImage(v, 20, 10)).toEqual([5, 7]);
  });
  it("zooms about the cursor so the image point under it stays put", () => {
    const v = zoomAt({ scale: 1, tx: 0, ty: 0 }, 100, 50, 2, 0.1, 10);
    expect(v).toEqual({ scale: 2, tx: -100, ty: -50 });
    expect(toImage(v, 100, 50)).toEqual([100, 50]);
  });
  it("clamps the zoom to its limits", () => {
    expect(zoomAt({ scale: 1, tx: 0, ty: 0 }, 0, 0, 100, 0.1, 10).scale).toBe(10);
    expect(zoomAt({ scale: 1, tx: 0, ty: 0 }, 0, 0, 0.001, 0.1, 10).scale).toBe(0.1);
  });
  it("pans by a screen delta", () => {
    expect(panBy({ scale: 3, tx: 1, ty: 2 }, 10, -5)).toEqual({ scale: 3, tx: 11, ty: -3 });
  });
});

describe("image-space helpers", () => {
  it("tests whether a point is inside the image", () => {
    expect(insideImage(639.5, 479.9, 640, 480)).toBe(true);
    expect(insideImage(640, 10, 640, 480)).toBe(false);
    expect(insideImage(-0.1, 5, 640, 480)).toBe(false);
  });
  it("clamps a point to integer pixel indices", () => {
    expect(clampPoint(-3.2, 500.7, 640, 480)).toEqual([0, 479]);
    expect(clampPoint(12.9, 7.1, 640, 480)).toEqual([12, 7]);
  });
  it("normalises a drag in any direction to an in-frame integer box", () => {
    expect(normBox([250, 290], [150, 190], 640, 480)).toEqual([150, 190, 250, 290]);
    expect(normBox([-20, -5], [700, 100.4], 640, 480)).toEqual([0, 0, 640, 100]);
  });
  it("picks the smallest candidate box containing the point", () => {
    const boxes: [number, number, number, number][] = [[0, 0, 100, 100], [10, 10, 30, 30], [200, 200, 300, 300]];
    expect(pickCandidate(boxes, 20, 20)).toBe(1);
    expect(pickCandidate(boxes, 50, 50)).toBe(0);
    expect(pickCandidate(boxes, 150, 150)).toBeNull();
  });
});
```

`web/src/lib/mask.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { hexToRgb, tintMask } from "./mask";

describe("mask tinting", () => {
  it("parses class colours and falls back to green", () => {
    expect(hexToRgb("#e6194b")).toEqual([230, 25, 75]);
    expect(hexToRgb("3cb44b")).toEqual([60, 180, 75]);
    expect(hexToRgb("nope")).toEqual([0, 255, 0]);
  });
  it("colours object pixels and makes the background transparent, in place", () => {
    const px = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]);
    const out = tintMask(px, [10, 20, 30], 0.5);
    expect(out).toBe(px);
    expect(Array.from(out)).toEqual([10, 20, 30, 128, 0, 0, 0, 0]);
  });
  it("clamps alpha to 0..1", () => {
    expect(tintMask(new Uint8ClampedArray([255, 255, 255, 255]), [1, 2, 3], 7)[3]).toBe(255);
    expect(tintMask(new Uint8ClampedArray([255, 255, 255, 255]), [1, 2, 3], -1)[3]).toBe(0);
  });
});
```

`web/src/lib/promptState.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { draftReducer, emptyDraft, toPrompt, type Draft, type DraftAction } from "./promptState";

const run = (d: Draft, ...actions: DraftAction[]) => actions.reduce(draftReducer, d);
const pt = (x: number, y: number, label: 0 | 1 = 1): DraftAction => ({ type: "point", x, y, label });

describe("draftReducer", () => {
  it("collects positive and negative clicks", () => {
    const d = run(emptyDraft(), pt(5, 6), pt(7, 8, 0));
    expect(d.points).toEqual([[5, 6, 1], [7, 8, 0]]);
    expect(toPrompt(d)).toEqual({ type: "clicks", points: [[5, 6, 1], [7, 8, 0]] });
  });
  it("sends nothing for an empty draft", () => {
    expect(toPrompt(emptyDraft("clicks"))).toBeNull();
    expect(toPrompt(emptyDraft("box"))).toBeNull();
    expect(toPrompt(emptyDraft("text"))).toBeNull();
  });
  it("ignores clicks in box mode until a box exists, then treats them as refinement", () => {
    const noBox = run(emptyDraft("box"), pt(1, 1));
    expect(noBox.points).toEqual([]);
    const d = run(emptyDraft("box"), { type: "box", box: [10, 10, 50, 50] }, pt(20, 20), pt(40, 40, 0));
    expect(toPrompt(d)).toEqual({ type: "box", box: [10, 10, 50, 50], points: [[20, 20, 1], [40, 40, 0]] });
  });
  it("drawing a new box discards the refinement clicks", () => {
    const d = run(emptyDraft("box"), { type: "box", box: [10, 10, 50, 50] }, pt(20, 20), { type: "box", box: [0, 0, 9, 9] });
    expect(d.box).toEqual([0, 0, 9, 9]);
    expect(d.points).toEqual([]);
  });
  it("ignores a box outside box mode", () => {
    expect(run(emptyDraft("clicks"), { type: "box", box: [1, 2, 3, 4] }).box).toBeNull();
  });
  it("undo removes the last click, then the box, then does nothing", () => {
    let d = run(emptyDraft("box"), { type: "box", box: [10, 10, 50, 50] }, pt(20, 20));
    d = draftReducer(d, { type: "undo" });
    expect(d.points).toEqual([]);
    expect(d.box).not.toBeNull();
    d = draftReducer(d, { type: "undo" });
    expect(d.box).toBeNull();
    expect(draftReducer(d, { type: "undo" })).toBe(d);
  });
  it("text prompts are sent only after submit, trimmed", () => {
    let d = run(emptyDraft("text"), { type: "phrase", phrase: "  red connector " });
    expect(toPrompt(d)).toBeNull();
    d = draftReducer(d, { type: "submit" });
    expect(toPrompt(d)).toEqual({ type: "text", phrase: "red connector" });
    expect(run(d, pt(1, 1)).points).toEqual([]);
  });
  it("remembers the chosen candidate and resets it when the prompt changes", () => {
    let d = run(emptyDraft("text"), { type: "phrase", phrase: "bolt" }, { type: "submit" }, { type: "candidate", index: 2 });
    expect(d.candidate).toBe(2);
    d = draftReducer(d, { type: "submit" });
    expect(d.candidate).toBe(0);
  });
  it("switching tools starts over but keeps the typed phrase; clear keeps the tool", () => {
    const d = run(emptyDraft("clicks"), pt(1, 1), { type: "phrase", phrase: "bolt" }, { type: "tool", tool: "text" });
    expect(d).toEqual({ ...emptyDraft("text"), phrase: "bolt" });
    expect(draftReducer(d, { type: "tool", tool: "text" })).toBe(d);
    expect(run(d, { type: "submit" }, { type: "clear" })).toEqual(emptyDraft("text"));
  });
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/lib'`
Expected: FAIL — the four modules cannot be resolved.

- [ ] **Step 3: Implement**

`web/src/lib/timeline.ts`:

```ts
import type { FrameState } from "../api/types";

/** A maximal run of consecutive frames in one state; `end` is inclusive. */
export type Run = { state: FrameState; start: number; end: number };

export function runs(states: readonly FrameState[]): Run[] {
  const out: Run[] = [];
  for (let i = 0; i < states.length; i++) {
    const last = out[out.length - 1];
    if (last !== undefined && last.state === states[i]) last.end = i;
    else out.push({ state: states[i], start: i, end: i });
  }
  return out;
}

export function clampFrame(i: number, count: number): number {
  if (count <= 0) return 0;
  return Math.max(0, Math.min(count - 1, Math.round(i)));
}

/** Index of the nearest frame in state `want` strictly beyond `from` in direction `dir`; no wrap-around. */
export function findNext(states: readonly FrameState[], from: number, dir: 1 | -1, want: FrameState): number | null {
  for (let i = from + dir; i >= 0 && i < states.length; i += dir) if (states[i] === want) return i;
  return null;
}

export function counts(states: readonly FrameState[]): Record<FrameState, number> {
  const c: Record<FrameState, number> = { empty: 0, masked: 0, keyframe: 0, deleted: 0 };
  for (const s of states) c[s]++;
  return c;
}

/** Labeled (masked or keyframe) frames as a rounded percentage of the frames that are not deleted. */
export function labeledPercent(states: readonly FrameState[]): number {
  const c = counts(states);
  const live = states.length - c.deleted;
  return live === 0 ? 0 : Math.round((100 * (c.masked + c.keyframe)) / live);
}

/** Apply a propagation `frame` event. Keyframes and deleted frames never change; unchanged input is returned as is. */
export function applyFrameEvent(states: FrameState[], frame: number, hasMask: boolean): FrameState[] {
  const current = states[frame];
  if (current === undefined || current === "keyframe" || current === "deleted") return states;
  const next: FrameState = hasMask ? "masked" : "empty";
  if (current === next) return states;
  const copy = states.slice();
  copy[frame] = next;
  return copy;
}

export function frameAtX(x: number, width: number, count: number): number {
  if (width <= 0) return 0;
  return clampFrame(Math.floor((x / width) * count), count);
}
```

`web/src/lib/geometry.ts`:

```ts
import type { Box } from "../api/types";

/** screen = image * scale + (tx, ty) */
export type View = { scale: number; tx: number; ty: number };

export function fitView(imgW: number, imgH: number, boxW: number, boxH: number, margin = 8): View {
  const w = Math.max(1, boxW - 2 * margin);
  const h = Math.max(1, boxH - 2 * margin);
  const scale = Math.min(w / imgW, h / imgH);
  return { scale, tx: (boxW - imgW * scale) / 2, ty: (boxH - imgH * scale) / 2 };
}

export const toImage = (v: View, sx: number, sy: number): [number, number] => [(sx - v.tx) / v.scale, (sy - v.ty) / v.scale];
export const toScreen = (v: View, x: number, y: number): [number, number] => [x * v.scale + v.tx, y * v.scale + v.ty];

export function zoomAt(v: View, sx: number, sy: number, factor: number, minScale: number, maxScale: number): View {
  const scale = Math.max(minScale, Math.min(maxScale, v.scale * factor));
  const [ix, iy] = toImage(v, sx, sy);
  return { scale, tx: sx - ix * scale, ty: sy - iy * scale };
}

export const panBy = (v: View, dx: number, dy: number): View => ({ scale: v.scale, tx: v.tx + dx, ty: v.ty + dy });

export const insideImage = (x: number, y: number, w: number, h: number): boolean => x >= 0 && y >= 0 && x < w && y < h;

/** The server requires integer pixel indices with 0 <= x < w and 0 <= y < h. */
export function clampPoint(x: number, y: number, w: number, h: number): [number, number] {
  return [Math.max(0, Math.min(w - 1, Math.floor(x))), Math.max(0, Math.min(h - 1, Math.floor(y)))];
}

export function normBox(a: [number, number], b: [number, number], w: number, h: number): Box {
  const cx = (v: number) => Math.max(0, Math.min(w, Math.round(v)));
  const cy = (v: number) => Math.max(0, Math.min(h, Math.round(v)));
  const [ax, ay, bx, by] = [cx(a[0]), cy(a[1]), cx(b[0]), cy(b[1])];
  return [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
}

/** Index of the smallest box containing the point (text-prompt candidates overlap; the smallest is the most specific). */
export function pickCandidate(bboxes: readonly Box[], x: number, y: number): number | null {
  let best: number | null = null;
  let bestArea = Infinity;
  bboxes.forEach(([x1, y1, x2, y2], i) => {
    const area = (x2 - x1) * (y2 - y1);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && area < bestArea) {
      best = i;
      bestArea = area;
    }
  });
  return best;
}
```

`web/src/lib/mask.ts`:

```ts
export type Rgb = [number, number, number];

export function hexToRgb(hex: string): Rgb {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (m === null) return [0, 255, 0];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Turn a decoded mask (white = object) into a coloured overlay, in place: object → rgb at `alpha`, rest transparent. */
export function tintMask(rgba: Uint8ClampedArray, rgb: Rgb, alpha: number): Uint8ClampedArray {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255);
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i] > 127) {
      rgba[i] = rgb[0];
      rgba[i + 1] = rgb[1];
      rgba[i + 2] = rgb[2];
      rgba[i + 3] = a;
    } else {
      rgba[i + 3] = 0;
    }
  }
  return rgba;
}
```

`web/src/lib/promptState.ts`:

```ts
import type { Box, Point, PromptSpec } from "../api/types";

export type Tool = "clicks" | "box" | "text";

export type Draft = {
  tool: Tool;
  points: Point[];
  box: Box | null;
  /** what is typed in the text box */
  phrase: string;
  /** the phrase that was submitted; only this is sent to the server */
  query: string;
  /** index of the chosen candidate (text prompts can return several) */
  candidate: number;
};

export const emptyDraft = (tool: Tool = "clicks"): Draft => ({ tool, points: [], box: null, phrase: "", query: "", candidate: 0 });

export type DraftAction =
  | { type: "tool"; tool: Tool }
  | { type: "point"; x: number; y: number; label: 0 | 1 }
  | { type: "box"; box: Box }
  | { type: "undo" }
  | { type: "phrase"; phrase: string }
  | { type: "submit" }
  | { type: "candidate"; index: number }
  | { type: "clear" };

export function draftReducer(d: Draft, a: DraftAction): Draft {
  switch (a.type) {
    case "tool":
      return a.tool === d.tool ? d : { ...emptyDraft(a.tool), phrase: d.phrase };
    case "point":
      if (d.tool === "text" || (d.tool === "box" && d.box === null)) return d;
      return { ...d, points: [...d.points, [a.x, a.y, a.label]], candidate: 0 };
    case "box":
      return d.tool === "box" ? { ...d, box: a.box, points: [], candidate: 0 } : d;
    case "undo":
      if (d.points.length > 0) return { ...d, points: d.points.slice(0, -1), candidate: 0 };
      if (d.box !== null) return { ...d, box: null, candidate: 0 };
      return d;
    case "phrase":
      return { ...d, phrase: a.phrase };
    case "submit":
      return d.tool === "text" ? { ...d, query: d.phrase.trim(), candidate: 0 } : d;
    case "candidate":
      return { ...d, candidate: Math.max(0, a.index) };
    case "clear":
      return emptyDraft(d.tool);
  }
}

/** The prompt to preview/accept for this draft, or null when there is nothing to send yet. */
export function toPrompt(d: Draft): PromptSpec | null {
  if (d.tool === "clicks") return d.points.length > 0 ? { type: "clicks", points: d.points } : null;
  if (d.tool === "box") return d.box !== null ? { type: "box", box: d.box, points: d.points } : null;
  return d.query !== "" ? { type: "text", phrase: d.query } : null;
}
```

- [ ] **Step 4: Run tests and typecheck**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run typecheck'`
Expected: `43 passed` (13 + 9 timeline + 9 geometry + 3 mask + 9 prompt draft); `tsc` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/timeline.ts web/src/lib/timeline.test.ts web/src/lib/geometry.ts web/src/lib/geometry.test.ts \
  web/src/lib/mask.ts web/src/lib/mask.test.ts web/src/lib/promptState.ts web/src/lib/promptState.test.ts
git commit -m "feat(web): pure timeline, geometry, mask and prompt-draft logic"
```

---

### Task 3: Job stream, health polling, engine banner

**Files:**
- Create: `web/src/lib/jobReducer.ts`, `web/src/hooks/useJob.ts`, `web/src/hooks/useHealth.ts`, `web/src/components/EngineBanner.tsx`
- Modify: `web/src/App.tsx`
- Test: `web/src/lib/jobReducer.test.ts`, `web/src/hooks/useJob.test.tsx`, `web/src/components/EngineBanner.test.tsx`

**Interfaces:**
- Consumes: `api.getJob`, `api.health`, `ApiError`, `wsUrl` (Task 1); `JobEvent`, `JobSnapshot`, `JobStatus`, `Health`.
- Produces:
  - `jobReducer.ts`: `type JobView = { status: JobStatus | "idle"; nextSeq: number; done: number; total: number | null; lastFrame: number | null; logs: string[]; result: Record<string, unknown> | null; error: string | null }`, `idleJob`, `isTerminal(status)`, `type JobAction`, `jobReducer(view, action)`.
  - `useJob.ts`: `type JobHandle = JobView & { transport: "ws" | "poll" }`, `useJob(jobId: string | null, onEvent?: (e: JobEvent) => void): JobHandle`. With `transport === "poll"` there are no per-frame events — callers must refresh their own data while the job runs.
  - `useHealth.ts`: `type HealthState = { health: Health | null; unreachable: boolean }`, `useHealth()`, `HealthContext` (React context holding a `HealthState`), `engineReady(state)`.
  - `EngineBanner.tsx`: `EngineBanner(props: HealthState)`.

- [ ] **Step 1: Write the failing tests**

`web/src/lib/jobReducer.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { JobEvent, JobSnapshot } from "../api/types";
import { idleJob, isTerminal, jobReducer, type JobView } from "./jobReducer";

const feed = (...events: JobEvent[]): JobView =>
  events.reduce((v, event) => jobReducer(v, { type: "event", event }), idleJob);
const snapshot = (over: Partial<JobSnapshot>): JobSnapshot => ({
  id: "j", kind: "propagate", status: "running", result: null, error: null, created_at: "", n_events: 0, ...over,
});

describe("jobReducer", () => {
  it("tracks propagation progress and the last streamed frame", () => {
    const v = feed(
      { seq: 0, type: "progress", done: 0, total: 40 },
      { seq: 1, type: "frame", frame: 12, has_mask: true, done: 1, total: 40 },
    );
    expect(v).toMatchObject({ status: "running", done: 1, total: 40, lastFrame: 12, nextSeq: 2 });
  });
  it("keeps total unknown for import progress events", () => {
    expect(feed({ seq: 0, type: "progress", done: 17 })).toMatchObject({ done: 17, total: null });
  });
  it("ignores events it has already seen (the socket replays from seq 0 on reconnect)", () => {
    const v = feed({ seq: 0, type: "progress", done: 5, total: 9 }, { seq: 1, type: "progress", done: 6, total: 9 });
    expect(jobReducer(v, { type: "event", event: { seq: 0, type: "progress", done: 5, total: 9 } })).toBe(v);
  });
  it("keeps the last 50 log lines", () => {
    const events = Array.from({ length: 60 }, (_, i): JobEvent => ({ seq: i, type: "log", message: `m${i}` }));
    const v = feed(...events);
    expect(v.logs).toHaveLength(50);
    expect(v.logs[0]).toBe("m10");
  });
  it("finishes with the result spread into the final event", () => {
    const v = feed({ seq: 0, type: "done", processed: 11, written: 11, lost_at: null });
    expect(v.status).toBe("done");
    expect(v.result).toEqual({ processed: 11, written: 11, lost_at: null });
  });
  it("reports cancellation and errors", () => {
    expect(feed({ seq: 0, type: "cancelled", cancelled: true }).status).toBe("cancelled");
    const failed = feed({ seq: 0, type: "error", message: "PropagationFailed: boom" });
    expect(failed).toMatchObject({ status: "error", error: "PropagationFailed: boom" });
  });
  it("takes a terminal state from a polled snapshot when no final event arrived", () => {
    const running = jobReducer(idleJob, { type: "snapshot", snapshot: snapshot({ status: "running" }) });
    expect(running.status).toBe("running");
    const done = jobReducer(running, { type: "snapshot", snapshot: snapshot({ status: "done", result: { clip_id: "c" } }) });
    expect(done).toMatchObject({ status: "done", result: { clip_id: "c" } });
  });
  it("never leaves a terminal state except through reset", () => {
    const done = feed({ seq: 0, type: "done" });
    expect(jobReducer(done, { type: "snapshot", snapshot: snapshot({ status: "running" }) })).toBe(done);
    expect(jobReducer(done, { type: "event", event: { seq: 1, type: "progress", done: 1 } }).status).toBe("done");
    expect(jobReducer(done, { type: "reset" })).toBe(idleJob);
  });
  it("knows which states are terminal", () => {
    expect(["done", "error", "cancelled"].every((s) => isTerminal(s as JobView["status"]))).toBe(true);
    expect(["idle", "pending", "running"].some((s) => isTerminal(s as JobView["status"]))).toBe(false);
  });
});
```

`web/src/hooks/useJob.test.tsx`:

```tsx
// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import { useJob } from "./useJob";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { getJob: vi.fn() },
}));

class FakeSocket {
  static instances: FakeSocket[] = [];
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
  emit(event: object) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
}
const socket = () => FakeSocket.instances[FakeSocket.instances.length - 1];

beforeEach(() => {
  FakeSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.mocked(api.getJob).mockReset();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("useJob", () => {
  it("is idle without a job id and opens no socket", () => {
    const { result } = renderHook(() => useJob(null));
    expect(result.current.status).toBe("idle");
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it("streams events, drops replays and forwards each new event once", () => {
    const seen: number[] = [];
    const { result } = renderHook(() => useJob("j1", (e) => seen.push(e.seq)));
    expect(socket().url).toMatch(/\/ws\/jobs\/j1$/);
    act(() => {
      socket().emit({ seq: 0, type: "progress", done: 0, total: 3 });
      socket().emit({ seq: 1, type: "frame", frame: 4, has_mask: true, done: 1, total: 3 });
      socket().emit({ seq: 0, type: "progress", done: 0, total: 3 });
    });
    expect(seen).toEqual([0, 1]);
    expect(result.current).toMatchObject({ status: "running", done: 1, total: 3, lastFrame: 4, transport: "ws" });
  });

  it("falls back to polling when the socket closes before the job ends", async () => {
    vi.mocked(api.getJob).mockResolvedValue({
      id: "j1", kind: "import", status: "done", result: { clip_id: "c9" }, error: null, created_at: "", n_events: 3,
    });
    const { result } = renderHook(() => useJob("j1"));
    act(() => socket().onclose?.());
    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.transport).toBe("poll");
    expect(result.current.result).toEqual({ clip_id: "c9" });
  });

  it("does not poll when the socket closes after the final event", () => {
    const { result } = renderHook(() => useJob("j1"));
    act(() => {
      socket().emit({ seq: 0, type: "done", clip_id: "c1" });
      socket().onclose?.();
    });
    expect(result.current.status).toBe("done");
    expect(api.getJob).not.toHaveBeenCalled();
  });

  it("closes the socket on unmount and reconnects for a new job id", () => {
    const { rerender, unmount } = renderHook(({ id }) => useJob(id), { initialProps: { id: "j1" } });
    const first = socket();
    rerender({ id: "j2" });
    expect(first.closed).toBe(true);
    expect(socket().url).toMatch(/j2$/);
    unmount();
    expect(socket().closed).toBe(true);
  });
});
```

`web/src/components/EngineBanner.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { Health } from "../api/types";
import { EngineBanner } from "./EngineBanner";

const health = (engine: Health["engine"]): Health => ({ status: "ok", version: "0", engine });
afterEach(cleanup);

describe("EngineBanner", () => {
  it("shows nothing when the engine is loaded or health is still unknown", () => {
    const { container, rerender } = render(
      <EngineBanner health={health({ loaded: true, device: "cuda:0", error: null })} unreachable={false} />,
    );
    expect(container.textContent).toBe("");
    rerender(<EngineBanner health={null} unreachable={false} />);
    expect(container.textContent).toBe("");
  });
  it("says the model is loading", () => {
    render(<EngineBanner health={health({ loaded: false, device: null, error: null })} unreachable={false} />);
    expect(screen.getByRole("status").textContent).toContain("Loading the SAM3 model");
  });
  it("shows the load error with the two repair commands", () => {
    render(<EngineBanner health={health({ loaded: false, device: null, error: "no CUDA device" })} unreachable={false} />);
    const text = screen.getByRole("alert").textContent ?? "";
    expect(text).toContain("no CUDA device");
    expect(text).toContain("uv run vsp verify-weights");
    expect(text).toContain("uv run vsp convert-weights");
  });
  it("says when the server cannot be reached", () => {
    render(<EngineBanner health={null} unreachable={true} />);
    expect(screen.getByRole("alert").textContent).toContain("Cannot reach the server");
  });
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/lib/jobReducer.test.ts src/hooks src/components'`
Expected: FAIL — modules cannot be resolved.

- [ ] **Step 3: Implement**

`web/src/lib/jobReducer.ts`:

```ts
import type { JobEvent, JobSnapshot, JobStatus } from "../api/types";

export type JobView = {
  status: JobStatus | "idle";
  /** seq of the next event we have not seen; lower seqs are replays */
  nextSeq: number;
  done: number;
  /** unknown for imports: a video's frame count is not known until decoding ends */
  total: number | null;
  lastFrame: number | null;
  logs: string[];
  result: Record<string, unknown> | null;
  error: string | null;
};

export const idleJob: JobView = {
  status: "idle", nextSeq: 0, done: 0, total: null, lastFrame: null, logs: [], result: null, error: null,
};

export const isTerminal = (s: JobView["status"]): boolean => s === "done" || s === "error" || s === "cancelled";

export type JobAction =
  | { type: "reset" }
  | { type: "event"; event: JobEvent }
  | { type: "snapshot"; snapshot: JobSnapshot };

function resultOf(event: JobEvent): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(event)) if (k !== "seq" && k !== "type") out[k] = v;
  return out;
}

export function jobReducer(view: JobView, action: JobAction): JobView {
  if (action.type === "reset") return idleJob;

  if (action.type === "snapshot") {
    if (isTerminal(view.status)) return view;
    const s = action.snapshot;
    if (!isTerminal(s.status)) return view.status === "running" ? view : { ...view, status: "running" };
    return { ...view, status: s.status, result: s.result, error: s.error };
  }

  const e = action.event;
  if (e.seq < view.nextSeq) return view;
  const base: JobView = { ...view, nextSeq: e.seq + 1 };
  if (isTerminal(view.status)) return base;
  base.status = "running";
  const done = typeof e.done === "number" ? e.done : base.done;
  const total = typeof e.total === "number" ? e.total : base.total;
  switch (e.type) {
    case "progress":
      return { ...base, done, total };
    case "frame":
      return { ...base, done, total, lastFrame: typeof e.frame === "number" ? e.frame : base.lastFrame };
    case "log":
      return { ...base, logs: [...base.logs, String(e.message ?? "")].slice(-50) };
    case "done":
    case "cancelled":
      return { ...base, status: e.type, result: resultOf(e) };
    case "error":
      return { ...base, status: "error", error: String(e.message ?? "the job failed") };
  }
}
```

`web/src/hooks/useJob.ts`:

```ts
import { useEffect, useReducer, useRef, useState } from "react";
import { ApiError, api, wsUrl } from "../api/rest";
import type { JobEvent } from "../api/types";
import { idleJob, isTerminal, jobReducer, type JobView } from "../lib/jobReducer";

export type JobHandle = JobView & { transport: "ws" | "poll" };

const POLL_MS = 1000;

/** Follow one job: WebSocket events while the socket lives, status polling if it drops before the job ends. */
export function useJob(jobId: string | null, onEvent?: (event: JobEvent) => void): JobHandle {
  const [view, dispatch] = useReducer(jobReducer, idleJob);
  const [transport, setTransport] = useState<"ws" | "poll">("ws");
  const callback = useRef(onEvent);
  callback.current = onEvent;

  useEffect(() => {
    dispatch({ type: "reset" });
    setTransport("ws");
    if (jobId === null) return;

    let closed = false;
    let finished = false;
    let nextSeq = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async (): Promise<void> => {
      if (closed || finished) return;
      try {
        const snapshot = await api.getJob(jobId);
        if (closed) return;
        dispatch({ type: "snapshot", snapshot });
        if (isTerminal(snapshot.status)) {
          finished = true;
          return;
        }
      } catch (e) {
        if (closed) return;
        if (e instanceof ApiError && e.status === 404) {
          finished = true;
          const message = "the job no longer exists (was the server restarted?)";
          dispatch({ type: "event", event: { seq: nextSeq, type: "error", message } });
          return;
        }
      }
      timer = setTimeout(() => void poll(), POLL_MS);
    };

    const ws = new WebSocket(wsUrl(`/ws/jobs/${jobId}`));
    ws.onmessage = (m: MessageEvent) => {
      const event = JSON.parse(String(m.data)) as JobEvent;
      if (event.seq < nextSeq) return;
      nextSeq = event.seq + 1;
      if (event.type === "done" || event.type === "cancelled" || event.type === "error") finished = true;
      dispatch({ type: "event", event });
      callback.current?.(event);
    };
    ws.onclose = () => {
      if (closed || finished) return;
      setTransport("poll");
      void poll();
    };

    return () => {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
      ws.close();
    };
  }, [jobId]);

  return { ...view, transport };
}
```

`web/src/hooks/useHealth.ts`:

```ts
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
```

`web/src/components/EngineBanner.tsx`:

```tsx
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
```

Replace `web/src/App.tsx`:

```tsx
import { Link, Route, Routes } from "react-router-dom";
import { EngineBanner } from "./components/EngineBanner";
import { HealthContext, useHealth } from "./hooks/useHealth";

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
            <Route path="/" element={<p className="p-4">Clips</p>} />
            <Route path="/clips/:clipId" element={<p className="p-4">Label</p>} />
          </Routes>
        </main>
      </div>
    </HealthContext.Provider>
  );
}
```

- [ ] **Step 4: Run tests, typecheck, build**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run build'`
Expected: `61 passed` (43 + 9 reducer + 5 hook + 4 banner), no `act(...)` warnings in the output; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/jobReducer.ts web/src/lib/jobReducer.test.ts web/src/hooks/useJob.ts web/src/hooks/useJob.test.tsx \
  web/src/hooks/useHealth.ts web/src/components/EngineBanner.tsx web/src/components/EngineBanner.test.tsx web/src/App.tsx
git commit -m "feat(web): job event stream with polling fallback, health polling and engine banner"
```

---

### Task 4: Clips page — projects, classes, imports, clip cards

**Files:**
- Create: `web/src/components/ImportJobRow.tsx`, `web/src/pages/Clips.tsx`
- Modify: `web/src/App.tsx` (route `/` renders `<Clips />`)
- Test: `web/src/pages/Clips.test.tsx`

**Interfaces:**
- Consumes: `api.*`, `thumbUrl`, `ApiError` (Task 1); `useJob`, `isTerminal` (Task 3).
- Produces: `Clips()` page component; `ImportJobRow({ jobId, label, onFinished, onDismiss })`. Nothing later imports from this task except `App.tsx`.

Behaviour:
- The selected project id is remembered in `localStorage` under `vsp.project`; if it no longer exists the first project is selected.
- A finished import (`done`) reloads the clip list and removes its row; a failed or cancelled import keeps its row (with the error) until dismissed. Cancel calls `DELETE /api/jobs/{id}`; the server removes the partial clip.
- Percent labeled on a card = `labeled_frames / (frame_count − deleted_frames.length)`.
- Deleting a clip asks `window.confirm` first; a 409 (GPU job running) is shown as an error line, not thrown.

- [ ] **Step 1: Write the failing test**

`web/src/pages/Clips.test.tsx`:

```tsx
// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { Clips } from "./Clips";

vi.mock("../api/rest", async (original) => {
  const real = await original<typeof import("../api/rest")>();
  return {
    ...real,
    api: {
      listProjects: vi.fn(), createProject: vi.fn(), addClass: vi.fn(), listClips: vi.fn(),
      importFrames: vi.fn(), importVideo: vi.fn(), deleteClip: vi.fn(), getJob: vi.fn(), cancelJob: vi.fn(),
    },
  };
});

class FakeSocket {
  static last: FakeSocket | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(readonly url: string) {
    FakeSocket.last = this;
  }
  close() {}
}

const project: Project = {
  id: "demo-abc123", name: "demo", created_at: "2026-09-20T00:00:00+00:00",
  classes: [{ id: 0, name: "connector", color: "#e6194b" }],
};
const clip: ClipSummary = {
  id: "c1", project_id: project.id, source: { type: "video", name: "bench.mp4" }, fps: 30, stride: 1,
  width: 640, height: 480, frame_count: 40, created_at: "", deleted_frames: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  objects: [], labeled_frames: 15,
};
const m = vi.mocked(api);
const renderPage = () => render(<MemoryRouter><Clips /></MemoryRouter>);

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("WebSocket", FakeSocket);
  Object.values(m).forEach((fn) => fn.mockReset());
  m.listProjects.mockResolvedValue([project]);
  m.listClips.mockResolvedValue([clip]);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Clips page", () => {
  it("selects the first project and lists its classes and clips", async () => {
    renderPage();
    expect(await screen.findByText("bench.mp4")).toBeTruthy();
    expect(screen.getByText("connector")).toBeTruthy();
    expect(screen.getByText(/40 frames/).textContent).toContain("50% labeled");
    expect(screen.getByRole("link", { name: /open/i }).getAttribute("href")).toBe("/clips/c1");
    expect(m.listClips).toHaveBeenCalledWith(project.id);
    expect(localStorage.getItem("vsp.project")).toBe(project.id);
  });

  it("creates a project and selects it", async () => {
    const created: Project = { id: "new-1", name: "new", created_at: "", classes: [] };
    m.createProject.mockResolvedValue(created);
    m.listProjects.mockResolvedValueOnce([project]).mockResolvedValue([project, created]);
    m.listClips.mockResolvedValueOnce([clip]).mockResolvedValue([]);
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("New project name"), { target: { value: " new " } });
    fireEvent.click(screen.getByRole("button", { name: "Create project" }));
    await waitFor(() => expect(m.createProject).toHaveBeenCalledWith("new"));
    await waitFor(() => expect(m.listClips).toHaveBeenLastCalledWith("new-1"));
    expect((screen.getByLabelText("Project") as HTMLSelectElement).value).toBe("new-1");
  });

  it("adds a class to the selected project", async () => {
    m.addClass.mockResolvedValue({ id: 1, name: "bolt", color: "#3cb44b" });
    m.listProjects
      .mockResolvedValueOnce([project])
      .mockResolvedValue([{ ...project, classes: [...project.classes, { id: 1, name: "bolt", color: "#3cb44b" }] }]);
    renderPage();
    await screen.findByText("connector");
    fireEvent.change(screen.getByLabelText("New class name"), { target: { value: "bolt" } });
    fireEvent.click(screen.getByRole("button", { name: "Add class" }));
    expect(await screen.findByText("bolt")).toBeTruthy();
    expect(m.addClass).toHaveBeenCalledWith(project.id, "bolt");
  });

  it("starts a frame-directory import, shows its progress and reloads the clips when it is done", async () => {
    m.importFrames.mockResolvedValue({ job_id: "j7" });
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Frame directory"), { target: { value: "~/footage/run1" } });
    fireEvent.change(screen.getByLabelText("Frame directory fps"), { target: { value: "15" } });
    fireEvent.click(screen.getByRole("button", { name: "Import frames" }));
    await waitFor(() => expect(m.importFrames).toHaveBeenCalledWith(project.id, { path: "~/footage/run1", fps: 15, stride: 1 }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j7$/));
    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ seq: 0, type: "progress", done: 12 }) }));
    expect(screen.getByText(/12 frames/)).toBeTruthy();
    const before = m.listClips.mock.calls.length;
    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ seq: 1, type: "done", clip_id: "c2", frame_count: 12 }) }));
    await waitFor(() => expect(m.listClips.mock.calls.length).toBe(before + 1));
    await waitFor(() => expect(screen.queryByText(/12 frames/)).toBeNull());
  });

  it("keeps a failed import visible until it is dismissed", async () => {
    m.importFrames.mockResolvedValue({ job_id: "j8" });
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Frame directory"), { target: { value: "~/nope" } });
    fireEvent.click(screen.getByRole("button", { name: "Import frames" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j8$/));
    act(() => FakeSocket.last!.onmessage!({ data: JSON.stringify({ seq: 0, type: "error", message: "IngestError: no frames" }) }));
    expect(await screen.findByText(/IngestError: no frames/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(/IngestError: no frames/)).toBeNull();
  });

  it("confirms before deleting a clip and shows a 409 instead of throwing", async () => {
    const confirm = vi.fn().mockReturnValueOnce(false).mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    m.deleteClip.mockRejectedValue(new ApiError(409, "job abc is still running", "abc"));
    renderPage();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Delete clip" }));
    expect(m.deleteClip).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete clip" }));
    expect(await screen.findByText(/job abc is still running/)).toBeTruthy();
    expect(screen.getByText("bench.mp4")).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run it to see it fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/pages/Clips.test.tsx'`
Expected: FAIL — `./Clips` cannot be resolved.

- [ ] **Step 3: Implement**

`web/src/components/ImportJobRow.tsx`:

```tsx
import { useEffect, useRef } from "react";
import { api } from "../api/rest";
import { useJob } from "../hooks/useJob";
import { isTerminal } from "../lib/jobReducer";

type Props = {
  jobId: string;
  label: string;
  /** called once when the job reaches a terminal state */
  onFinished: (jobId: string, ok: boolean) => void;
  onDismiss: (jobId: string) => void;
};

export function ImportJobRow({ jobId, label, onFinished, onDismiss }: Props) {
  const job = useJob(jobId);
  const reported = useRef(false);

  useEffect(() => {
    if (isTerminal(job.status) && !reported.current) {
      reported.current = true;
      onFinished(jobId, job.status === "done");
    }
  }, [job.status, jobId, onFinished]);

  const running = !isTerminal(job.status);
  return (
    <li className="flex items-center gap-3 rounded bg-neutral-800 px-3 py-2 text-sm">
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {running && <span className="text-neutral-300">{job.done} frames…</span>}
      {job.status === "cancelled" && <span className="text-amber-400">cancelled</span>}
      {job.status === "error" && <span className="text-red-400">{job.error}</span>}
      {running ? (
        <button type="button" className="rounded bg-neutral-700 px-2 py-1" onClick={() => void api.cancelJob(jobId).catch(() => {})}>
          Cancel
        </button>
      ) : (
        <button type="button" className="rounded bg-neutral-700 px-2 py-1" onClick={() => onDismiss(jobId)}>
          Dismiss
        </button>
      )}
    </li>
  );
}
```

`web/src/pages/Clips.tsx`:

```tsx
import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, thumbUrl } from "../api/rest";
import type { ClipSummary, Project } from "../api/types";
import { ImportJobRow } from "../components/ImportJobRow";

const PROJECT_KEY = "vsp.project";
const input = "rounded border border-neutral-600 bg-neutral-800 px-2 py-1 text-sm";
const button = "rounded bg-sky-700 px-3 py-1 text-sm hover:bg-sky-600 disabled:opacity-40";

function percentLabeled(c: ClipSummary): number {
  const live = c.frame_count - c.deleted_frames.length;
  return live <= 0 ? 0 : Math.round((100 * c.labeled_frames) / live);
}

export function Clips() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<string | null>(() => localStorage.getItem(PROJECT_KEY));
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [imports, setImports] = useState<{ id: string; label: string }[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [projectName, setProjectName] = useState("");
  const [className, setClassName] = useState("");
  const [video, setVideo] = useState<File | null>(null);
  const [videoStride, setVideoStride] = useState(1);
  const [dirPath, setDirPath] = useState("");
  const [dirFps, setDirFps] = useState(30);
  const [dirStride, setDirStride] = useState(1);

  const fail = useCallback((e: unknown) => setError(e instanceof Error ? e.message : String(e)), []);

  const loadProjects = useCallback(async () => {
    const list = await api.listProjects();
    setProjects(list);
    setProjectId((current) => (list.some((p) => p.id === current) ? current : (list[0]?.id ?? null)));
  }, []);

  const loadClips = useCallback(async (pid: string) => setClips(await api.listClips(pid)), []);

  useEffect(() => {
    loadProjects().catch(fail);
  }, [loadProjects, fail]);

  useEffect(() => {
    if (projectId === null) {
      setClips([]);
      return;
    }
    localStorage.setItem(PROJECT_KEY, projectId);
    loadClips(projectId).catch(fail);
  }, [projectId, loadClips, fail]);

  const project = projects.find((p) => p.id === projectId) ?? null;

  const guarded = (fn: () => Promise<void>) => (e?: FormEvent) => {
    e?.preventDefault();
    setError(null);
    fn().catch(fail);
  };

  const createProject = guarded(async () => {
    const name = projectName.trim();
    if (name === "") return;
    const created = await api.createProject(name);
    setProjectName("");
    await loadProjects();
    setProjectId(created.id);
  });

  const addClass = guarded(async () => {
    const name = className.trim();
    if (name === "" || project === null) return;
    await api.addClass(project.id, name);
    setClassName("");
    await loadProjects();
  });

  const importVideo = guarded(async () => {
    if (video === null || project === null) return;
    const { job_id } = await api.importVideo(project.id, video, videoStride);
    setImports((list) => [...list, { id: job_id, label: video.name }]);
    setVideo(null);
  });

  const importFrames = guarded(async () => {
    const path = dirPath.trim();
    if (path === "" || project === null) return;
    const { job_id } = await api.importFrames(project.id, { path, fps: dirFps, stride: dirStride });
    setImports((list) => [...list, { id: job_id, label: path }]);
  });

  const dismiss = useCallback((jobId: string) => setImports((list) => list.filter((j) => j.id !== jobId)), []);
  const finished = useCallback(
    (jobId: string, ok: boolean) => {
      if (!ok) return;
      dismiss(jobId);
      if (projectId !== null) loadClips(projectId).catch(fail);
    },
    [dismiss, projectId, loadClips, fail],
  );

  const deleteClip = (clip: ClipSummary) =>
    guarded(async () => {
      if (!window.confirm(`Delete "${clip.source.name}" and all of its masks?`)) return;
      await api.deleteClip(clip.id);
      if (projectId !== null) await loadClips(projectId);
    })();

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-6 overflow-y-auto p-4">
      {error !== null && (
        <p role="alert" className="rounded bg-red-900 px-3 py-2 text-sm">
          {error}
        </p>
      )}

      <section className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col text-xs text-neutral-400">
          Project
          <select aria-label="Project" className={input} value={projectId ?? ""} onChange={(e) => setProjectId(e.target.value || null)}>
            {projects.length === 0 && <option value="">(none yet)</option>}
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <form className="flex items-end gap-2" onSubmit={createProject}>
          <input aria-label="New project name" className={input} placeholder="New project name" value={projectName}
            onChange={(e) => setProjectName(e.target.value)} />
          <button type="submit" className={button}>Create project</button>
        </form>
      </section>

      {project !== null && (
        <>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-300">Classes</h2>
            <div className="flex flex-wrap items-center gap-2">
              {project.classes.map((c) => (
                <span key={c.id} className="flex items-center gap-1 rounded bg-neutral-800 px-2 py-1 text-sm">
                  <span className="inline-block h-3 w-3 rounded-full" style={{ background: c.color }} />
                  {c.name}
                </span>
              ))}
              <form className="flex items-center gap-2" onSubmit={addClass}>
                <input aria-label="New class name" className={input} placeholder="New class name" value={className}
                  onChange={(e) => setClassName(e.target.value)} />
                <button type="submit" className={button}>Add class</button>
              </form>
            </div>
          </section>

          <section className="grid gap-4 md:grid-cols-2">
            <form className="flex flex-col gap-2 rounded border border-neutral-700 p-3" onSubmit={importVideo}>
              <h2 className="text-sm font-semibold text-neutral-300">Import video</h2>
              <input aria-label="Video file" type="file" accept="video/*,.mp4,.mov,.avi,.mkv" className="text-sm"
                onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
              <label className="flex items-center gap-2 text-xs text-neutral-400">
                Keep every Nth frame
                <input aria-label="Video stride" type="number" min={1} className={`${input} w-20`} value={videoStride}
                  onChange={(e) => setVideoStride(Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <button type="submit" className={button} disabled={video === null}>Import video</button>
            </form>

            <form className="flex flex-col gap-2 rounded border border-neutral-700 p-3" onSubmit={importFrames}>
              <h2 className="text-sm font-semibold text-neutral-300">Import frame directory</h2>
              <input aria-label="Frame directory" className={input} placeholder="~/footage/run1 (a path on the server)"
                value={dirPath} onChange={(e) => setDirPath(e.target.value)} />
              <div className="flex gap-4 text-xs text-neutral-400">
                <label className="flex items-center gap-2">
                  fps
                  <input aria-label="Frame directory fps" type="number" min={1} step="any" className={`${input} w-20`}
                    value={dirFps} onChange={(e) => setDirFps(Math.max(1, Number(e.target.value) || 30))} />
                </label>
                <label className="flex items-center gap-2">
                  Keep every Nth frame
                  <input aria-label="Frame directory stride" type="number" min={1} className={`${input} w-20`}
                    value={dirStride} onChange={(e) => setDirStride(Math.max(1, Number(e.target.value) || 1))} />
                </label>
              </div>
              <button type="submit" className={button} disabled={dirPath.trim() === ""}>Import frames</button>
            </form>
          </section>

          {imports.length > 0 && (
            <ul className="flex flex-col gap-2">
              {imports.map((j) => (
                <ImportJobRow key={j.id} jobId={j.id} label={j.label} onFinished={finished} onDismiss={dismiss} />
              ))}
            </ul>
          )}

          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-300">Clips</h2>
            {clips.length === 0 && <p className="text-sm text-neutral-400">No clips yet — import a video or a frame directory.</p>}
            <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {clips.map((c) => (
                <li key={c.id} className="overflow-hidden rounded border border-neutral-700 bg-neutral-800">
                  <img src={thumbUrl(c.id, 0)} alt="" className="aspect-video w-full bg-black object-contain" />
                  <div className="flex flex-col gap-1 p-3 text-sm">
                    <span className="truncate font-medium">{c.source.name}</span>
                    <span className="text-xs text-neutral-400">
                      {c.frame_count} frames · {c.width}×{c.height} · {c.fps} fps · {percentLabeled(c)}% labeled
                    </span>
                    <div className="mt-1 flex gap-2">
                      <Link to={`/clips/${c.id}`} className={button}>Open</Link>
                      <button type="button" aria-label="Delete clip" className="rounded bg-neutral-700 px-3 py-1 text-sm hover:bg-red-800"
                        onClick={() => deleteClip(c)}>
                        Delete
                      </button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
```

In `web/src/App.tsx`, import the page and use it for the `/` route:

```tsx
import { Clips } from "./pages/Clips";
// …
            <Route path="/" element={<Clips />} />
```

- [ ] **Step 4: Run tests, typecheck, build**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run build'`
Expected: `67 passed` (61 + 6); no React `act(...)` or "not wrapped" warnings; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/ImportJobRow.tsx web/src/pages/Clips.tsx web/src/pages/Clips.test.tsx web/src/App.tsx
git commit -m "feat(web): Clips page with projects, classes, imports and clip cards"
```

---

### Task 5: Label page as a viewer — frame canvas, mask overlay, timeline, transport, playback

**Files:**
- Create: `web/src/lib/frameCache.ts`, `web/src/hooks/useClip.ts`, `web/src/hooks/usePlayback.ts`, `web/src/components/FrameCanvas.tsx`, `web/src/components/Timeline.tsx`, `web/src/components/Transport.tsx`, `web/src/components/ClipInfo.tsx`, `web/src/pages/Label.tsx`, `web/src/test/setup-dom.ts`
- Modify: `web/src/App.tsx` (route `/clips/:clipId` renders `<Label />`)
- Test: `web/src/lib/frameCache.test.ts`, `web/src/hooks/useClip.test.tsx`, `web/src/hooks/usePlayback.test.tsx`, `web/src/components/Timeline.test.tsx`, `web/src/pages/Label.test.tsx`

**Interfaces:**
- Consumes: `api.getClip`, `frameUrl`, `maskUrl` (Task 1); `runs`, `frameAtX`, `clampFrame`, `counts`, `labeledPercent`, `applyFrameEvent`, geometry helpers, `hexToRgb`, `tintMask`, `Draft`, `Tool`, `emptyDraft` (Task 2).
- Produces:
  - `frameCache.ts`: `class LruCache<V>` (`get`, `set`, `delete`, `size`), `loadImage(url)`, `tintToCanvas(img, rgb, alpha)`, `class FrameStore` with `image(url): Promise<HTMLImageElement | null>` and `layer(url, rgb, alpha): Promise<HTMLCanvasElement | null>`; constructor `(load = loadImage, tint = tintToCanvas)` for tests.
  - `useClip.ts`: `useClip(clipId): ClipHandle` where `ClipHandle = { detail: ClipDetail | null; error: string | null; reload(): Promise<void>; apply(detail: ClipDetail, masksChanged: boolean): void; patchFrame(frame: number, hasMask: boolean): void; maskVersion(frame: number): string }`.
  - `usePlayback.ts`: `usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying })`.
  - `FrameCanvas.tsx`: props `{ store, width, height, frameUrl, maskUrl, color, tool: Tool | null, draft: Draft, candidates: Candidate[], onPoint(x, y, label), onBox(box), onPickCandidate(index) }`. `tool === null` disables prompting input (zoom and pan always work). Written once here; Tasks 6–7 only pass different props.
  - `Timeline.tsx`: props `{ states, frame, onSeek(frame) }`. `Transport.tsx`: props `{ frame, frameCount, fps, playing, overlay, name, onStep(delta), onTogglePlay(), onToggleOverlay() }`. `ClipInfo.tsx`: props `{ detail }`.
  - `test/setup-dom.ts`: `stubCanvasAndObservers()` for every jsdom test that mounts a canvas component.

Design notes for the implementer:
- Mouse events (not pointer events) are used on purpose: jsdom's `MouseEvent` support is solid, so the components can be driven from tests.
- The canvas redraws imperatively (`draw()`); view, drag state and decoded assets live in refs so a wheel or drag never re-renders React.
- A frame and its mask are swapped in together (one `Promise.all`), so the overlay never shows on the wrong frame.
- Cache sizes are deliberately small (64 frames, 48 tinted layers): a decoded 1080p layer is ~8 MB.
- Playback prefetches 30 frames ahead (4 when paused). Masks are requested only for frames whose state is `masked` or `keyframe`.

- [ ] **Step 1: Write the failing tests**

`web/src/test/setup-dom.ts`:

```ts
import { vi } from "vitest";

/** jsdom has no canvas implementation and no ResizeObserver; components only need them not to throw. */
export function stubCanvasAndObservers(): void {
  const ctx = new Proxy({}, { get: () => () => undefined, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => ctx) as unknown as HTMLCanvasElement["getContext"];
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
}
```

`web/src/lib/frameCache.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { FrameStore, LruCache } from "./frameCache";

const img = (name: string) => ({ name }) as unknown as HTMLImageElement;
const canvas = (name: string) => ({ name }) as unknown as HTMLCanvasElement;

describe("LruCache", () => {
  it("evicts the least recently used entry; get() refreshes", () => {
    const c = new LruCache<number>(2);
    c.set("a", 1);
    c.set("b", 2);
    expect(c.get("a")).toBe(1);
    c.set("c", 3);
    expect(c.get("b")).toBeUndefined();
    expect(c.get("a")).toBe(1);
    expect(c.get("c")).toBe(3);
    expect(c.size).toBe(2);
  });
});

describe("FrameStore", () => {
  it("loads each URL once, even for concurrent requests", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const store = new FrameStore(load, vi.fn());
    const [a, b] = await Promise.all([store.image("/f/1"), store.image("/f/1")]);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(1);
  });
  it("does not cache a failed load, so the next request retries", async () => {
    const load = vi.fn<(url: string) => Promise<HTMLImageElement | null>>().mockResolvedValueOnce(null).mockResolvedValueOnce(img("ok"));
    const store = new FrameStore(load, vi.fn());
    expect(await store.image("/f/1")).toBeNull();
    expect(await store.image("/f/1")).not.toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
  });
  it("tints a mask once per colour and alpha", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const tint = vi.fn(() => canvas("layer"));
    const store = new FrameStore(load, tint);
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45);
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45);
    await store.layer("/m/1?v=0.0", [0, 255, 0], 0.45);
    expect(tint).toHaveBeenCalledTimes(2);
  });
  it("returns null for a mask that cannot be loaded", async () => {
    const store = new FrameStore(vi.fn(async () => null), vi.fn());
    expect(await store.layer("/m/9?v=0.0", [1, 2, 3], 0.5)).toBeNull();
  });
});
```

`web/src/hooks/useClip.test.tsx`:

```tsx
// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { useClip } from "./useClip";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { getClip: vi.fn() },
}));

const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 640, height: 480,
    frame_count: 3, created_at: "", deleted_frames: [], objects: [],
  },
  states: ["keyframe", "empty", "masked"],
  classes: [],
};

beforeEach(() => vi.mocked(api.getClip).mockReset());
afterEach(cleanup);

describe("useClip", () => {
  it("loads the clip detail", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).toEqual(detail));
    expect(api.getClip).toHaveBeenCalledWith("c1");
  });
  it("reports a load failure", async () => {
    vi.mocked(api.getClip).mockRejectedValue(new ApiError(404, "clip c1 not found"));
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.error).toBe("clip c1 not found"));
  });
  it("patches one frame from a job event and gives that frame's mask a new version", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = [result.current.maskVersion(1), result.current.maskVersion(2)];
    act(() => result.current.patchFrame(1, true));
    expect(result.current.detail?.states).toEqual(["keyframe", "masked", "masked"]);
    expect(result.current.maskVersion(1)).not.toBe(before[0]);
    expect(result.current.maskVersion(2)).toBe(before[1]);
  });
  it("re-versions a frame whose mask was overwritten even if its state did not change", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = result.current.maskVersion(2);
    act(() => result.current.patchFrame(2, true));
    expect(result.current.maskVersion(2)).not.toBe(before);
  });
  it("apply() replaces the detail and re-versions every mask only when masks changed", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = result.current.maskVersion(0);
    const next: ClipDetail = { ...detail, states: ["keyframe", "deleted", "masked"] };
    act(() => result.current.apply(next, false));
    expect(result.current.detail).toBe(next);
    expect(result.current.maskVersion(0)).toBe(before);
    act(() => result.current.apply(next, true));
    expect(result.current.maskVersion(0)).not.toBe(before);
  });
});
```

`web/src/hooks/usePlayback.test.tsx`:

```tsx
// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePlayback } from "./usePlayback";

function usePlayer(frameCount: number, fps: number) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });
  return { frame, playing };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("usePlayback", () => {
  it("advances one frame per 1/fps seconds", () => {
    const { result } = renderHook(() => usePlayer(100, 10));
    act(() => vi.advanceTimersByTime(350));
    expect(result.current.frame).toBe(3);
  });
  it("stops on the last frame", () => {
    const { result } = renderHook(() => usePlayer(3, 10));
    act(() => vi.advanceTimersByTime(1000));
    expect(result.current.frame).toBe(2);
    expect(result.current.playing).toBe(false);
  });
});
```

`web/src/components/Timeline.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FrameState } from "../api/types";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Timeline } from "./Timeline";

const states: FrameState[] = ["keyframe", "masked", "masked", "empty", "empty", "empty", "deleted", "empty", "empty", "empty"];

beforeEach(() => {
  stubCanvasAndObservers();
  HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ left: 10, top: 0, width: 100, height: 28 }) as DOMRect;
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Timeline", () => {
  it("seeks on click and while dragging with the left button held", () => {
    const onSeek = vi.fn();
    render(<Timeline states={states} frame={0} onSeek={onSeek} />);
    const strip = screen.getByRole("slider");
    fireEvent.mouseDown(strip, { clientX: 65, button: 0, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(5);
    fireEvent.mouseMove(strip, { clientX: 95, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(8);
    fireEvent.mouseMove(strip, { clientX: 20, buttons: 0 });
    expect(onSeek).toHaveBeenCalledTimes(2);
  });
  it("exposes the position to assistive technology", () => {
    render(<Timeline states={states} frame={4} onSeek={() => {}} />);
    const strip = screen.getByRole("slider");
    expect(strip.getAttribute("aria-valuenow")).toBe("4");
    expect(strip.getAttribute("aria-valuemax")).toBe("9");
  });
});
```

`web/src/pages/Label.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Label } from "./Label";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { getClip: vi.fn() },
}));

const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "bench.mp4" }, fps: 10, stride: 1, width: 640, height: 480,
    frame_count: 40, created_at: "", deleted_frames: [30], original_names: { "3": "img_0007.png" },
    objects: [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states: Array.from({ length: 40 }, (_, i) => (i === 0 ? "keyframe" : i < 20 ? "masked" : i === 30 ? "deleted" : "empty")),
  classes: [{ id: 0, name: "connector", color: "#e6194b" }],
};

const renderLabel = () =>
  render(
    <MemoryRouter initialEntries={["/clips/c1"]}>
      <Routes>
        <Route path="/clips/:clipId" element={<Label />} />
      </Routes>
    </MemoryRouter>,
  );

beforeEach(() => {
  stubCanvasAndObservers();
  vi.mocked(api.getClip).mockReset();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Label page (viewer)", () => {
  it("shows the clip, its statistics and the current frame", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    expect(await screen.findByText("bench.mp4")).toBeTruthy();
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    expect(screen.getByText(/1 keyframe/)).toBeTruthy();
    expect(screen.getByText(/51% labeled/)).toBeTruthy();
  });
  it("steps with the transport buttons, clamps at the ends and shows original file names", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Back 10 frames" }));
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    expect(screen.getByText("frame 3 / 39")).toBeTruthy();
    expect(screen.getByText("img_0007.png")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Forward 10 frames" }));
    expect(screen.getByText("frame 13 / 39")).toBeTruthy();
  });
  it("toggles playback and the overlay", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    const overlay = screen.getByRole("checkbox", { name: /overlay/i }) as HTMLInputElement;
    expect(overlay.checked).toBe(true);
    fireEvent.click(overlay);
    expect(overlay.checked).toBe(false);
  });
  it("shows a load error with a way back", async () => {
    vi.mocked(api.getClip).mockRejectedValue(new ApiError(404, "clip c1 not found"));
    renderLabel();
    expect(await screen.findByText(/clip c1 not found/)).toBeTruthy();
    expect(screen.getByRole("link", { name: /back to clips/i })).toBeTruthy();
  });
});
```

(39 live frames, 20 labeled → `Math.round(100 × 20 / 39) = 51`.)

- [ ] **Step 2: Run them to see them fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/lib/frameCache.test.ts src/hooks/useClip.test.tsx src/hooks/usePlayback.test.tsx src/components/Timeline.test.tsx src/pages/Label.test.tsx'`
Expected: FAIL — modules cannot be resolved.

- [ ] **Step 3: Implement the cache and the hooks**

`web/src/lib/frameCache.ts`:

```ts
import { tintMask, type Rgb } from "./mask";

export class LruCache<V> {
  private readonly map = new Map<string, V>();
  constructor(private readonly max: number) {}

  get(key: string): V | undefined {
    const value = this.map.get(key);
    if (value !== undefined) {
      this.map.delete(key);
      this.map.set(key, value);
    }
    return value;
  }

  set(key: string, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.max) this.map.delete(this.map.keys().next().value as string);
  }

  delete(key: string): void {
    this.map.delete(key);
  }

  get size(): number {
    return this.map.size;
  }
}

export function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

/** Decode a mask image (white = object) into a canvas holding the coloured, semi-transparent overlay. */
export function tintToCanvas(img: HTMLImageElement, rgb: Rgb, alpha: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  tintMask(data.data, rgb, alpha);
  ctx.putImageData(data, 0, 0);
  return canvas;
}

/** Small on purpose: a decoded 1080p frame or layer is about 8 MB. */
const MAX_IMAGES = 64;
const MAX_LAYERS = 48;

export class FrameStore {
  private readonly images = new LruCache<Promise<HTMLImageElement | null>>(MAX_IMAGES);
  private readonly layers = new LruCache<Promise<HTMLCanvasElement | null>>(MAX_LAYERS);

  constructor(
    private readonly load: (url: string) => Promise<HTMLImageElement | null> = loadImage,
    private readonly tint: (img: HTMLImageElement, rgb: Rgb, alpha: number) => HTMLCanvasElement = tintToCanvas,
  ) {}

  image(url: string): Promise<HTMLImageElement | null> {
    let pending = this.images.get(url);
    if (pending === undefined) {
      pending = this.load(url).then((img) => {
        if (img === null) this.images.delete(url); // a failed load must not stick
        return img;
      });
      this.images.set(url, pending);
    }
    return pending;
  }

  layer(url: string, rgb: Rgb, alpha: number): Promise<HTMLCanvasElement | null> {
    const key = `${url}|${rgb.join(",")}|${alpha}`;
    let pending = this.layers.get(key);
    if (pending === undefined) {
      pending = this.load(url).then((img) => {
        if (img === null) {
          this.layers.delete(key);
          return null;
        }
        return this.tint(img, rgb, alpha);
      });
      this.layers.set(key, pending);
    }
    return pending;
  }
}
```

`web/src/hooks/useClip.ts`:

```ts
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import type { ClipDetail } from "../api/types";
import { applyFrameEvent } from "../lib/timeline";

export type ClipHandle = {
  detail: ClipDetail | null;
  error: string | null;
  reload: () => Promise<void>;
  /** adopt the ClipDetail a mutation returned; `masksChanged` re-versions every mask URL */
  apply: (detail: ClipDetail, masksChanged: boolean) => void;
  /** a propagation job wrote (or cleared) this frame's mask */
  patchFrame: (frame: number, hasMask: boolean) => void;
  /** cache-busting version for a frame's mask URL */
  maskVersion: (frame: number) => string;
};

export function useClip(clipId: string): ClipHandle {
  const [detail, setDetail] = useState<ClipDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const versions = useRef({ epoch: 0, frames: new Map<number, number>() });

  const reload = useCallback(async () => {
    try {
      const next = await api.getClip(clipId);
      versions.current.epoch++;
      setDetail(next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [clipId]);

  useEffect(() => {
    setDetail(null);
    versions.current = { epoch: 0, frames: new Map() };
    void reload();
  }, [reload]);

  const apply = useCallback((next: ClipDetail, masksChanged: boolean) => {
    if (masksChanged) versions.current.epoch++;
    setDetail(next);
    setTick((t) => t + 1);
  }, []);

  const patchFrame = useCallback((frame: number, hasMask: boolean) => {
    const frames = versions.current.frames;
    frames.set(frame, (frames.get(frame) ?? 0) + 1);
    setDetail((d) => (d === null ? d : { ...d, states: applyFrameEvent(d.states, frame, hasMask) }));
    setTick((t) => t + 1);
  }, []);

  const maskVersion = useCallback(
    (frame: number) => `${versions.current.epoch}.${versions.current.frames.get(frame) ?? 0}`,
    [],
  );

  return { detail, error, reload, apply, patchFrame, maskVersion };
}
```

`web/src/hooks/usePlayback.ts`:

```ts
import { useEffect, type Dispatch, type SetStateAction } from "react";

type Options = {
  playing: boolean;
  fps: number;
  frame: number;
  frameCount: number;
  setFrame: Dispatch<SetStateAction<number>>;
  setPlaying: (playing: boolean) => void;
};

/** Advance one frame per 1/fps seconds while playing; stop on the last frame. */
export function usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying }: Options): void {
  useEffect(() => {
    if (!playing || frameCount <= 0) return;
    const id = setInterval(() => setFrame((f) => Math.min(frameCount - 1, f + 1)), 1000 / Math.max(1, fps));
    return () => clearInterval(id);
  }, [playing, fps, frameCount, setFrame]);

  useEffect(() => {
    if (playing && frame >= frameCount - 1) setPlaying(false);
  }, [playing, frame, frameCount, setPlaying]);
}
```

- [ ] **Step 4: Implement the components**

`web/src/components/Timeline.tsx`:

```tsx
import { useEffect, useRef, useState, type MouseEvent } from "react";
import type { FrameState } from "../api/types";
import { frameAtX, runs } from "../lib/timeline";

type Props = { states: FrameState[]; frame: number; onSeek: (frame: number) => void };

const HEIGHT = 28;
const COLORS: Record<FrameState, string> = { empty: "#3f3f46", masked: "#10b981", keyframe: "#f59e0b", deleted: "#7f1d1d" };
const LEGEND: [FrameState, string][] = [["keyframe", "keyframe"], ["masked", "masked"], ["empty", "empty"], ["deleted", "deleted"]];

export function Timeline({ states, frame, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = wrap.current;
    if (el === null) return;
    const observer = new ResizeObserver(() => setWidth(el.clientWidth));
    observer.observe(el);
    setWidth(el.clientWidth);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width * dpr);
    c.height = Math.round(HEIGHT * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, HEIGHT);
    if (states.length === 0) return;
    const px = width / states.length;
    const all = runs(states);
    for (const r of all) {
      if (r.state === "keyframe") continue;
      ctx.fillStyle = COLORS[r.state];
      ctx.fillRect(r.start * px, 6, (r.end - r.start + 1) * px, HEIGHT - 12);
    }
    ctx.fillStyle = COLORS.keyframe; // keyframes on top, full height, never thinner than 3 px
    for (const r of all) {
      if (r.state !== "keyframe") continue;
      for (let i = r.start; i <= r.end; i++) {
        const w = Math.max(px, 3);
        ctx.fillRect(i * px + px / 2 - w / 2, 0, w, HEIGHT);
      }
    }
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(frame * px + px / 2 - 1, 0, 2, HEIGHT);
  }, [states, frame, width]);

  const seek = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(frameAtX(e.clientX - rect.left, rect.width, states.length));
  };

  return (
    <div ref={wrap} className="w-full select-none">
      <canvas
        ref={canvas}
        role="slider"
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, states.length - 1)}
        aria-valuenow={frame}
        tabIndex={-1}
        className="block w-full cursor-pointer"
        style={{ height: HEIGHT }}
        onMouseDown={(e) => {
          if (e.button === 0) seek(e);
        }}
        onMouseMove={(e) => {
          if (e.buttons & 1) seek(e);
        }}
      />
      <div className="mt-1 flex gap-4 text-xs text-neutral-400">
        {LEGEND.map(([state, label]) => (
          <span key={state} className="flex items-center gap-1">
            <span className="inline-block h-2 w-3" style={{ background: COLORS[state] }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
```

`web/src/components/Transport.tsx`:

```tsx
type Props = {
  frame: number;
  frameCount: number;
  fps: number;
  playing: boolean;
  overlay: boolean;
  /** original file name of this frame, for frame-directory clips */
  name: string | null;
  onStep: (delta: number) => void;
  onTogglePlay: () => void;
  onToggleOverlay: () => void;
};

const button = "rounded bg-neutral-700 px-2 py-1 text-sm hover:bg-neutral-600";

export function Transport({ frame, frameCount, fps, playing, overlay, name, onStep, onTogglePlay, onToggleOverlay }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <button type="button" className={button} aria-label="Back 10 frames" onClick={() => onStep(-10)}>⏪</button>
      <button type="button" className={button} aria-label="Previous frame" onClick={() => onStep(-1)}>◀</button>
      <button type="button" className={button} aria-label={playing ? "Pause" : "Play"} onClick={onTogglePlay}>
        {playing ? "⏸" : "▶"}
      </button>
      <button type="button" className={button} aria-label="Next frame" onClick={() => onStep(1)}>▶▏</button>
      <button type="button" className={button} aria-label="Forward 10 frames" onClick={() => onStep(10)}>⏩</button>
      <span className="ml-2 tabular-nums">{`frame ${frame} / ${Math.max(0, frameCount - 1)}`}</span>
      {name !== null && <span className="text-xs text-neutral-400">{name}</span>}
      <span className="text-xs text-neutral-500">{fps} fps</span>
      <label className="ml-auto flex items-center gap-1 text-xs">
        <input type="checkbox" checked={overlay} onChange={onToggleOverlay} />
        Mask overlay (M)
      </label>
    </div>
  );
}
```

`web/src/components/ClipInfo.tsx`:

```tsx
import type { ClipDetail } from "../api/types";
import { counts, labeledPercent } from "../lib/timeline";

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function ClipInfo({ detail }: { detail: ClipDetail }) {
  const c = counts(detail.states);
  return (
    <section className="text-sm">
      <h2 className="truncate font-semibold">{detail.clip.source.name}</h2>
      <p className="text-xs text-neutral-400">
        {detail.clip.width}×{detail.clip.height} · {detail.clip.fps} fps · stride {detail.clip.stride}
      </p>
      <p className="mt-1 text-xs text-neutral-300">
        {plural(c.keyframe, "keyframe")} · {c.masked} masked · {c.empty} empty · {c.deleted} deleted ·{" "}
        {labeledPercent(detail.states)}% labeled
      </p>
    </section>
  );
}
```

`web/src/components/FrameCanvas.tsx`:

```tsx
import { useCallback, useEffect, useMemo, useRef, type MouseEvent } from "react";
import type { Box, Candidate } from "../api/types";
import type { FrameStore } from "../lib/frameCache";
import {
  clampPoint, fitView, insideImage, normBox, panBy, pickCandidate, toImage, toScreen, zoomAt, type View,
} from "../lib/geometry";
import { hexToRgb } from "../lib/mask";
import type { Draft, Tool } from "../lib/promptState";

type Props = {
  store: FrameStore;
  /** image size in pixels */
  width: number;
  height: number;
  frameUrl: string;
  /** stored mask of this frame, or null when there is none / the overlay is off */
  maskUrl: string | null;
  /** class colour, hex */
  color: string;
  /** null disables prompting input; zoom and pan always work */
  tool: Tool | null;
  draft: Draft;
  /** preview results; while there are any they replace the stored mask */
  candidates: Candidate[];
  onPoint: (x: number, y: number, label: 0 | 1) => void;
  onBox: (box: Box) => void;
  onPickCandidate: (index: number) => void;
};

const MASK_ALPHA = 0.45;
const PREVIEW_ALPHA = 0.55;
const DRAG_PX = 4;
const MAX_SCALE = 40;

type Drag = { button: number; sx: number; sy: number; lastX: number; lastY: number; moved: boolean };

export function FrameCanvas(props: Props) {
  const { store, width, height, frameUrl, maskUrl, color, tool, candidates, onPoint, onBox, onPickCandidate } = props;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = useRef({ w: 0, h: 0 });
  const view = useRef<View>({ scale: 1, tx: 0, ty: 0 });
  const fitScale = useRef(1);
  const assets = useRef<{ frame: HTMLImageElement | null; mask: HTMLCanvasElement | null }>({ frame: null, mask: null });
  const layers = useRef<(HTMLCanvasElement | null)[]>([]);
  const drag = useRef<Drag | null>(null);
  const dragBox = useRef<Box | null>(null);
  const latest = useRef(props);
  latest.current = props;
  const rgb = useMemo(() => hexToRgb(color), [color]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const p = latest.current;
    const v = view.current;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.current.w, size.current.h);

    ctx.save();
    ctx.translate(v.tx, v.ty);
    ctx.scale(v.scale, v.scale);
    ctx.imageSmoothingEnabled = v.scale < 3;
    if (assets.current.frame) ctx.drawImage(assets.current.frame, 0, 0, p.width, p.height);
    if (p.candidates.length > 0) {
      layers.current.forEach((layer, i) => {
        if (layer === null) return;
        ctx.globalAlpha = i === p.draft.candidate ? 1 : 0.45;
        ctx.drawImage(layer, 0, 0, p.width, p.height);
      });
      ctx.globalAlpha = 1;
    } else if (assets.current.mask) {
      ctx.drawImage(assets.current.mask, 0, 0, p.width, p.height);
    }
    ctx.restore();

    const box = dragBox.current ?? p.draft.box;
    if (box !== null) {
      const [x1, y1] = toScreen(v, box[0], box[1]);
      const [x2, y2] = toScreen(v, box[2], box[3]);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }
    if (p.candidates.length > 1) {
      ctx.font = "bold 13px sans-serif";
      p.candidates.forEach((c, i) => {
        const [x, y] = toScreen(v, c.bbox[0], c.bbox[1]);
        ctx.fillStyle = i === p.draft.candidate ? "#facc15" : "#e5e5e5";
        ctx.fillRect(x, y - 16, 18, 16);
        ctx.fillStyle = "#000000";
        ctx.fillText(String(i + 1), x + 5, y - 4);
      });
    }
    for (const [px, py, label] of p.draft.points) {
      const [x, y] = toScreen(v, px + 0.5, py + 0.5);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = label === 1 ? "#22c55e" : "#ef4444";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }
  }, []);

  const refit = useCallback(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (wrap === null || canvas === null) return;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    size.current = { w, h };
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    view.current = fitView(width, height, w, h);
    fitScale.current = view.current.scale;
    draw();
  }, [width, height, draw]);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (wrap === null) return;
    const observer = new ResizeObserver(refit);
    observer.observe(wrap);
    refit();
    return () => observer.disconnect();
  }, [refit]);

  // Swap the frame and its mask in together so the overlay never shows on the wrong frame.
  useEffect(() => {
    let live = true;
    const mask = maskUrl === null ? Promise.resolve(null) : store.layer(maskUrl, rgb, MASK_ALPHA);
    void Promise.all([store.image(frameUrl), mask]).then(([frame, layer]) => {
      if (!live) return;
      assets.current = { frame, mask: layer };
      draw();
    });
    return () => {
      live = false;
    };
  }, [store, frameUrl, maskUrl, rgb, draw]);

  useEffect(() => {
    let live = true;
    if (candidates.length === 0) {
      layers.current = [];
      draw();
      return;
    }
    const pending = candidates.map((c) => store.layer(`data:image/png;base64,${c.mask_png_b64}`, rgb, PREVIEW_ALPHA));
    void Promise.all(pending).then((ready) => {
      if (!live) return;
      layers.current = ready;
      draw();
    });
    return () => {
      live = false;
    };
  }, [store, candidates, rgb, draw]);

  useEffect(draw); // points, box and candidate selection arrive as props

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault(); // needs a non-passive listener, which React's onWheel is not
      const r = canvas.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.0015);
      view.current = zoomAt(view.current, e.clientX - r.left, e.clientY - r.top, factor, fitScale.current * 0.5, MAX_SCALE);
      draw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [draw]);

  const local = (e: MouseEvent): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  const onMouseDown = (e: MouseEvent) => {
    e.preventDefault(); // no text selection, no middle-click autoscroll
    const [sx, sy] = local(e);
    drag.current = { button: e.button, sx, sy, lastX: sx, lastY: sy, moved: false };
  };

  const onMouseMove = (e: MouseEvent) => {
    const d = drag.current;
    if (d === null) return;
    const [sx, sy] = local(e);
    if (Math.hypot(sx - d.sx, sy - d.sy) > DRAG_PX) d.moved = true;
    if (d.button === 1) {
      view.current = panBy(view.current, sx - d.lastX, sy - d.lastY);
      d.lastX = sx;
      d.lastY = sy;
      draw();
    } else if (d.button === 0 && tool === "box" && d.moved) {
      dragBox.current = normBox(toImage(view.current, d.sx, d.sy), toImage(view.current, sx, sy), width, height);
      draw();
    }
  };

  const endDrag = (commit: boolean) => {
    const d = drag.current;
    drag.current = null;
    const box = dragBox.current;
    dragBox.current = null;
    if (d === null || d.button === 1) return;
    if (box !== null) {
      if (commit && box[2] - box[0] >= 2 && box[3] - box[1] >= 2) onBox(box);
      draw();
      return;
    }
    if (!commit || d.moved || tool === null) return;
    const [ix, iy] = toImage(view.current, d.sx, d.sy);
    if (!insideImage(ix, iy, width, height)) return;
    const [x, y] = clampPoint(ix, iy, width, height);
    if (tool === "text") {
      const index = pickCandidate(candidates.map((c) => c.bbox), x, y);
      if (index !== null) onPickCandidate(index);
      return;
    }
    onPoint(x, y, d.button === 2 ? 0 : 1);
  };

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden bg-black">
      <canvas
        ref={canvasRef}
        aria-label="Frame"
        className="block h-full w-full"
        style={{ cursor: tool === null ? "default" : "crosshair" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={() => endDrag(true)}
        onMouseLeave={() => endDrag(false)}
        onContextMenu={(e) => e.preventDefault()}
      />
      <button type="button" className="absolute right-2 top-2 rounded bg-neutral-800/80 px-2 py-1 text-xs" onClick={refit}>
        Fit
      </button>
    </div>
  );
}
```

`web/src/pages/Label.tsx` (viewer version; Tasks 6 and 7 replace it):

```tsx
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { frameUrl, maskUrl } from "../api/rest";
import { ClipInfo } from "../components/ClipInfo";
import { FrameCanvas } from "../components/FrameCanvas";
import { Timeline } from "../components/Timeline";
import { Transport } from "../components/Transport";
import { useClip } from "../hooks/useClip";
import { usePlayback } from "../hooks/usePlayback";
import { FrameStore } from "../lib/frameCache";
import { hexToRgb } from "../lib/mask";
import { emptyDraft } from "../lib/promptState";
import { clampFrame } from "../lib/timeline";

const MASK_ALPHA = 0.45; // must match FrameCanvas so prefetched layers are cache hits
const NO_DRAFT = emptyDraft();
const noop = () => {};

export function Label() {
  const { clipId = "" } = useParams();
  const clip = useClip(clipId);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- a new clip gets a fresh cache
  const store = useMemo(() => new FrameStore(), [clipId]);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState(true);

  const detail = clip.detail;
  const frameCount = detail?.clip.frame_count ?? 0;
  const fps = detail?.clip.fps ?? 30;
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });

  const step = useCallback((delta: number) => setFrame((f) => clampFrame(f + delta, frameCount)), [frameCount]);
  const seek = useCallback((f: number) => setFrame(clampFrame(f, frameCount)), [frameCount]);

  const object = detail?.clip.objects[0] ?? null;
  const color = detail?.classes.find((c) => c.id === object?.class_id)?.color ?? "#22c55e";
  const hasMask = (i: number) => detail !== null && (detail.states[i] === "masked" || detail.states[i] === "keyframe");
  const maskVersion = clip.maskVersion;

  useEffect(() => {
    if (detail === null) return;
    const rgb = hexToRgb(color);
    const ahead = playing ? 30 : 4;
    for (let i = frame + 1; i <= Math.min(frame + ahead, frameCount - 1); i++) {
      void store.image(frameUrl(clipId, i));
      if (overlay && hasMask(i)) void store.layer(maskUrl(clipId, i, maskVersion(i)), rgb, MASK_ALPHA);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- hasMask is derived from detail
  }, [store, clipId, frame, playing, overlay, detail, color, frameCount, maskVersion]);

  if (clip.error !== null) {
    return (
      <div className="p-4 text-sm">
        <p role="alert" className="mb-2 rounded bg-red-900 px-3 py-2">{clip.error}</p>
        <Link to="/" className="underline">Back to clips</Link>
      </div>
    );
  }
  if (detail === null) return <p className="p-4 text-sm text-neutral-400">Loading…</p>;

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1">
          <FrameCanvas
            store={store}
            width={detail.clip.width}
            height={detail.clip.height}
            frameUrl={frameUrl(clipId, frame)}
            maskUrl={overlay && hasMask(frame) ? maskUrl(clipId, frame, maskVersion(frame)) : null}
            color={color}
            tool={null}
            draft={NO_DRAFT}
            candidates={[]}
            onPoint={noop}
            onBox={noop}
            onPickCandidate={noop}
          />
        </div>
        <aside className="flex w-80 flex-col gap-4 overflow-y-auto border-l border-neutral-700 p-3">
          <Link to="/" className="text-xs text-neutral-400 underline">← Back to clips</Link>
          <ClipInfo detail={detail} />
        </aside>
      </div>
      <footer className="flex flex-col gap-2 border-t border-neutral-700 p-3">
        <Transport
          frame={frame}
          frameCount={frameCount}
          fps={fps}
          playing={playing}
          overlay={overlay}
          name={detail.clip.original_names?.[String(frame)] ?? null}
          onStep={step}
          onTogglePlay={() => setPlaying((p) => !p)}
          onToggleOverlay={() => setOverlay((o) => !o)}
        />
        <Timeline states={detail.states} frame={frame} onSeek={seek} />
      </footer>
    </div>
  );
}
```

There is no ESLint in this project; drop the two `eslint-disable` comments if you prefer — they are documentation only.

In `web/src/App.tsx`, import the page and use it for the label route:

```tsx
import { Label } from "./pages/Label";
// …
            <Route path="/clips/:clipId" element={<Label />} />
```

- [ ] **Step 5: Run tests, typecheck, build**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run build'`
Expected: `85 passed` (67 + 5 cache + 5 useClip + 2 playback + 2 timeline + 4 label); no warnings; build succeeds.

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/frameCache.ts web/src/lib/frameCache.test.ts web/src/hooks/useClip.ts web/src/hooks/useClip.test.tsx \
  web/src/hooks/usePlayback.ts web/src/hooks/usePlayback.test.tsx web/src/components/FrameCanvas.tsx \
  web/src/components/Timeline.tsx web/src/components/Timeline.test.tsx web/src/components/Transport.tsx \
  web/src/components/ClipInfo.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx web/src/test/setup-dom.ts web/src/App.tsx
git commit -m "feat(web): Label page viewer with frame canvas, mask overlay, timeline and playback"
```

---

### Task 6: Prompting — click / box / text preview, class picker, accept

**Files:**
- Create: `web/src/hooks/usePrompting.ts`, `web/src/components/PromptPanel.tsx`, `web/src/components/ObjectPanel.tsx`
- Modify: `web/src/pages/Label.tsx` (full replacement below), `web/src/pages/Label.test.tsx` (add the prompting tests below; keep the four viewer tests)
- Test: `web/src/hooks/usePrompting.test.tsx`, `web/src/components/PromptPanel.test.tsx`

**Interfaces:**
- Consumes: `api.prompt`, `api.accept`, `api.addClass`, `ApiError` (Task 1); `Draft`, `DraftAction`, `draftReducer`, `emptyDraft`, `toPrompt`, `Tool` (Task 2); `HealthContext`, `engineReady` (Task 3); `ClipHandle`, `FrameCanvas` (Task 5).
- Produces:
  - `usePrompting.ts`: `usePrompting({ clipId, frame, enabled, clip, classId }): Prompting` where `Prompting = { draft: Draft; dispatch: Dispatch<DraftAction>; candidates: Candidate[]; busy: boolean; error: string | null; busyJobId: string | null; canAccept: boolean; accept(): Promise<void>; clear(): void }`. `busyJobId` is the id of a running GPU job the server named in a 409 (Task 7 adopts it).
  - `PromptPanel.tsx`: props `{ prompting: Prompting; disabledReason: string | null }`.
  - `ObjectPanel.tsx`: props `{ detail: ClipDetail; classId: number | null; onClassId(id: number): void; onAddClass(name: string): Promise<void> }`.

Behaviour:
- A draft belongs to one frame: changing the frame clears it, and the old draft is never sent for the new frame. Every change to the prompt (not to the chosen candidate) fires one `POST /prompt`; responses that arrive out of order are dropped (request counter).
- An empty candidate list is shown as "No usable mask …" (the server filters masks under 64 px or over 45 % of the frame).
- `accept()` sends `class_id` only when the clip has no object yet or the picked class differs from the object's class; it then adopts the returned `ClipDetail` with `masksChanged = true` and clears the draft.
- Prompting is disabled (FrameCanvas `tool = null`) when the engine is not ready, the frame is deleted, or playback is running; the reason is shown in the panel.

- [ ] **Step 1: Write the failing tests**

`web/src/hooks/usePrompting.test.tsx`:

```tsx
// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { Candidate, ClipDetail } from "../api/types";
import type { ClipHandle } from "./useClip";
import { usePrompting } from "./usePrompting";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { prompt: vi.fn(), accept: vi.fn() },
}));

const cand = (area: number): Candidate => ({ score: 0.9, bbox: [10, 10, 50, 50], area, phrase: null, mask_png_b64: "AAAA" });
const baseDetail = (withObject: boolean): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 640, height: 480,
    frame_count: 10, created_at: "", deleted_frames: [],
    objects: withObject ? [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }] : [],
  },
  states: Array(10).fill("empty"),
  classes: [{ id: 0, name: "a", color: "#ff0000" }, { id: 1, name: "b", color: "#00ff00" }],
});
const handle = (detail: ClipDetail): ClipHandle => ({
  detail, error: null, refreshError: null, reload: vi.fn(), apply: vi.fn(), patchFrame: vi.fn(), maskVersion: () => "0.0",
});
type Args = { frame: number; classId: number | null; clip: ClipHandle; enabled: boolean };
const setup = (over: Partial<Args> = {}) =>
  renderHook((a: Args) => usePrompting({ clipId: "c1", ...a }), {
    initialProps: { frame: 2, classId: 0, clip: handle(baseDetail(true)), enabled: true, ...over },
  });

beforeEach(() => {
  vi.mocked(api.prompt).mockReset();
  vi.mocked(api.accept).mockReset();
});
afterEach(cleanup);

describe("usePrompting", () => {
  it("previews after every click", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    expect(api.prompt).toHaveBeenLastCalledWith("c1", 2, { type: "clicks", points: [[5, 6, 1]] });
    expect(result.current.canAccept).toBe(true);
    act(() => result.current.dispatch({ type: "point", x: 7, y: 8, label: 0 }));
    await waitFor(() => expect(api.prompt).toHaveBeenCalledTimes(2));
  });

  it("drops a response that arrives after a newer request", async () => {
    let resolveFirst: (v: { candidates: Candidate[] }) => void = () => {};
    vi.mocked(api.prompt)
      .mockReturnValueOnce(new Promise((r) => (resolveFirst = r)))
      .mockResolvedValueOnce({ candidates: [cand(222)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "point", x: 1, y: 1, label: 1 }));
    act(() => result.current.dispatch({ type: "point", x: 2, y: 2, label: 1 }));
    await waitFor(() => expect(result.current.candidates[0]?.area).toBe(222));
    await act(async () => resolveFirst({ candidates: [cand(111)] }));
    expect(result.current.candidates[0].area).toBe(222);
  });

  it("does not re-request when only the chosen candidate changes", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(1), cand(2)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "tool", tool: "text" }));
    act(() => result.current.dispatch({ type: "phrase", phrase: "bolt" }));
    act(() => result.current.dispatch({ type: "submit" }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(2));
    act(() => result.current.dispatch({ type: "candidate", index: 1 }));
    expect(api.prompt).toHaveBeenCalledTimes(1);
  });

  it("clears the draft when the frame changes or prompting is disabled", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result, rerender } = setup();
    const clip = handle(baseDetail(true));
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    rerender({ frame: 3, classId: 0, clip, enabled: true });
    expect(result.current.draft.points).toEqual([]);
    expect(result.current.candidates).toEqual([]);
    expect(api.prompt).toHaveBeenCalledTimes(1); // the old frame's draft is never sent for the new frame
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    rerender({ frame: 3, classId: 0, clip, enabled: false });
    expect(result.current.draft.points).toEqual([]);
  });

  it("explains an empty result and surfaces server errors with the blocking job id", async () => {
    vi.mocked(api.prompt).mockResolvedValueOnce({ candidates: [] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.error).toMatch(/No usable mask/));
    expect(result.current.canAccept).toBe(false);
    vi.mocked(api.prompt).mockRejectedValueOnce(new ApiError(409, "job j9 is still running", "j9"));
    act(() => result.current.dispatch({ type: "point", x: 7, y: 8, label: 1 }));
    await waitFor(() => expect(result.current.error).toBe("job j9 is still running"));
    expect(result.current.busyJobId).toBe("j9");
  });

  it("accepts with the chosen candidate, sends the class only when needed and adopts the result", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const accepted = baseDetail(true);
    vi.mocked(api.accept).mockResolvedValue(accepted);
    const clip = handle(baseDetail(true));
    const { result, rerender } = setup({ clip });
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.canAccept).toBe(true));
    await act(() => result.current.accept());
    expect(api.accept).toHaveBeenLastCalledWith("c1", 2, { type: "clicks", points: [[5, 6, 1]] }, { classId: undefined, candidate: 0 });
    expect(clip.apply).toHaveBeenCalledWith(accepted, true);
    expect(result.current.draft.points).toEqual([]);

    rerender({ frame: 2, classId: 1, clip, enabled: true }); // the user picked another class
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.canAccept).toBe(true));
    await act(() => result.current.accept());
    expect(vi.mocked(api.accept).mock.calls[1][3]).toEqual({ classId: 1, candidate: 0 });
  });

  it("requires a class for the first accept on a clip", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result } = setup({ clip: handle(baseDetail(false)), classId: null });
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    expect(result.current.canAccept).toBe(false);
    await act(() => result.current.accept());
    expect(api.accept).not.toHaveBeenCalled();
    expect(result.current.error).toMatch(/class/i);
  });
});
```

`web/src/components/PromptPanel.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Prompting } from "../hooks/usePrompting";
import { emptyDraft } from "../lib/promptState";
import { PromptPanel } from "./PromptPanel";

const prompting = (over: Partial<Prompting> = {}): Prompting => ({
  draft: emptyDraft("clicks"), dispatch: vi.fn(), candidates: [], busy: false, error: null, busyJobId: null,
  canAccept: false, accept: vi.fn(async () => {}), clear: vi.fn(), ...over,
});
afterEach(cleanup);

describe("PromptPanel", () => {
  it("switches tools", () => {
    const p = prompting();
    render(<PromptPanel prompting={p} disabledReason={null} />);
    expect(screen.getByRole("button", { name: "Click" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    expect(p.dispatch).toHaveBeenCalledWith({ type: "tool", tool: "box" });
  });
  it("submits a text phrase with the form", () => {
    const p = prompting({ draft: { ...emptyDraft("text"), phrase: "bolt" } });
    render(<PromptPanel prompting={p} disabledReason={null} />);
    fireEvent.change(screen.getByLabelText("Text prompt"), { target: { value: "red bolt" } });
    expect(p.dispatch).toHaveBeenCalledWith({ type: "phrase", phrase: "red bolt" });
    fireEvent.submit(screen.getByLabelText("Text prompt").closest("form")!);
    expect(p.dispatch).toHaveBeenCalledWith({ type: "submit" });
  });
  it("lists numbered candidates and lets the user choose one", () => {
    const c = { score: 0.91, bbox: [0, 0, 9, 9] as [number, number, number, number], area: 1234, phrase: "bolt", mask_png_b64: "" };
    const p = prompting({ draft: { ...emptyDraft("text"), query: "bolt" }, candidates: [c, { ...c, score: 0.5 }], canAccept: true });
    render(<PromptPanel prompting={p} disabledReason={null} />);
    fireEvent.click(screen.getByRole("button", { name: /^2 ·/ }));
    expect(p.dispatch).toHaveBeenCalledWith({ type: "candidate", index: 1 });
  });
  it("enables Accept only when there is something to accept, and wires Accept / Clear / Undo", () => {
    const idle = prompting();
    const { rerender } = render(<PromptPanel prompting={idle} disabledReason={null} />);
    expect((screen.getByRole("button", { name: /Accept/ }) as HTMLButtonElement).disabled).toBe(true);
    const ready = prompting({ canAccept: true });
    rerender(<PromptPanel prompting={ready} disabledReason={null} />);
    fireEvent.click(screen.getByRole("button", { name: /Accept/ }));
    expect(ready.accept).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Clear/ }));
    expect(ready.clear).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Undo/ }));
    expect(ready.dispatch).toHaveBeenCalledWith({ type: "undo" });
  });
  it("shows why prompting is disabled, and the last error", () => {
    render(<PromptPanel prompting={prompting({ error: "No usable mask" })} disabledReason="This frame is deleted." />);
    expect(screen.getByText("This frame is deleted.")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toBe("No usable mask");
    expect((screen.getByRole("button", { name: "Click" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
```

Add to `web/src/pages/Label.test.tsx` — extend the `vi.mock` factory's `api` object to `{ getClip: vi.fn(), prompt: vi.fn(), accept: vi.fn(), addClass: vi.fn() }`, wrap the rendered tree in a ready health context, give the canvas a real size, and add the two tests:

```tsx
// new imports
import { waitFor } from "@testing-library/react";
import { HealthContext, type HealthState } from "../hooks/useHealth";

const ready: HealthState = {
  health: { status: "ok", version: "0", engine: { loaded: true, device: "cuda:0", error: null } },
  unreachable: false,
};

// renderLabel now takes the health state:
const renderLabel = (health: HealthState = ready) =>
  render(
    <HealthContext.Provider value={health}>
      <MemoryRouter initialEntries={["/clips/c1"]}>
        <Routes>
          <Route path="/clips/:clipId" element={<Label />} />
        </Routes>
      </MemoryRouter>
    </HealthContext.Provider>,
  );

// in beforeEach, after stubCanvasAndObservers(): a 656×496 box fits the 640×480 frame at scale 1 with an 8 px margin,
// so screen (108, 58) is image pixel (100, 50)
Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 656 });
Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => 496 });
HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ left: 0, top: 0, width: 656, height: 496 }) as DOMRect;
Object.values(vi.mocked(api)).forEach((fn) => fn.mockReset());

describe("Label page (prompting)", () => {
  it("previews a click on the canvas and accepts it", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockResolvedValue({
      candidates: [{ score: 0.9, bbox: [80, 30, 120, 70], area: 1200, phrase: null, mask_png_b64: "AAAA" }],
    });
    const accepted = { ...detail, states: detail.states.map((s, i) => (i === 25 ? "keyframe" : s)) } as ClipDetail;
    vi.mocked(api.accept).mockResolvedValue(accepted);
    renderLabel();
    await screen.findByText("bench.mp4");
    for (let i = 0; i < 25; i++) fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 0 });
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 25, { type: "clicks", points: [[100, 50, 1]] }));
    const accept = screen.getByRole("button", { name: /Accept/ }) as HTMLButtonElement;
    await waitFor(() => expect(accept.disabled).toBe(false));
    fireEvent.click(accept);
    await waitFor(() => expect(api.accept).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/2 keyframes/)).toBeTruthy();
  });

  it("right-click adds a negative point; prompting is off while the engine is not loaded", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [] });
    const { unmount } = renderLabel();
    await screen.findByText("bench.mp4");
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 2 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 2 });
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 0]] }));
    unmount();

    vi.mocked(api.prompt).mockClear();
    renderLabel({ health: { status: "ok", version: "0", engine: { loaded: false, device: null, error: null } }, unreachable: false });
    await screen.findByText("bench.mp4");
    const again = screen.getByLabelText("Frame");
    fireEvent.mouseDown(again, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(again, { clientX: 108, clientY: 58, button: 0 });
    expect(screen.getByText(/engine is not loaded/i)).toBeTruthy();
    expect(api.prompt).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/hooks/usePrompting.test.tsx src/components/PromptPanel.test.tsx src/pages/Label.test.tsx'`
Expected: FAIL — `./usePrompting` and `./PromptPanel` cannot be resolved; the two new Label tests fail because the page has no prompt panel.

- [ ] **Step 3: Implement**

`web/src/hooks/usePrompting.ts`:

```ts
import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type Dispatch } from "react";
import { ApiError, api } from "../api/rest";
import type { Candidate } from "../api/types";
import { draftReducer, emptyDraft, toPrompt, type Draft, type DraftAction } from "../lib/promptState";
import type { ClipHandle } from "./useClip";

export type Prompting = {
  draft: Draft;
  dispatch: Dispatch<DraftAction>;
  candidates: Candidate[];
  busy: boolean;
  error: string | null;
  /** id of the running GPU job a 409 named, so the page can attach to it */
  busyJobId: string | null;
  canAccept: boolean;
  accept: () => Promise<void>;
  clear: () => void;
};

type Options = { clipId: string; frame: number; enabled: boolean; clip: ClipHandle; classId: number | null };

const NO_MASK = "No usable mask: nothing was found, or the mask is too small or too large. Add a click or try another prompt.";

export function usePrompting({ clipId, frame, enabled, clip, classId }: Options): Prompting {
  const [draft, dispatch] = useReducer(draftReducer, undefined, () => emptyDraft());
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyJobId, setBusyJobId] = useState<string | null>(null);
  const requestId = useRef(0);
  const draftIsStale = useRef(false);

  const promptKey = JSON.stringify([draft.tool, draft.points, draft.box, draft.query]);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- promptKey covers every field toPrompt reads
  const prompt = useMemo(() => toPrompt(draft), [promptKey]);

  // a draft belongs to one frame, and only exists while prompting is possible
  useEffect(() => {
    draftIsStale.current = true; // the effect below runs in this same commit and still sees the old frame's draft
    dispatch({ type: "clear" });
    setError(null);
  }, [clipId, frame, enabled]);

  useEffect(() => {
    const id = ++requestId.current; // also invalidates any request still in flight
    const stale = draftIsStale.current;
    draftIsStale.current = false;
    if (prompt === null || !enabled || stale) {
      setCandidates([]);
      setBusy(false);
      return;
    }
    setBusy(true);
    api
      .prompt(clipId, frame, prompt)
      .then((r) => {
        if (id !== requestId.current) return;
        setCandidates(r.candidates);
        setError(r.candidates.length === 0 ? NO_MASK : null);
        setBusyJobId(null);
        setBusy(false);
      })
      .catch((e: unknown) => {
        if (id !== requestId.current) return;
        setCandidates([]);
        setError(e instanceof Error ? e.message : String(e));
        setBusyJobId(e instanceof ApiError && e.status === 409 ? e.jobId : null);
        setBusy(false);
      });
  }, [clipId, frame, enabled, prompt]);

  const object = clip.detail?.clip.objects[0] ?? null;
  const needsClass = object === null;
  const canAccept = enabled && prompt !== null && !busy && draft.candidate < candidates.length && !(needsClass && classId === null);

  const apply = clip.apply;
  const accept = useCallback(async () => {
    if (prompt === null || candidates.length === 0) return;
    if (needsClass && classId === null) {
      setError("Choose a class before accepting the first mask.");
      return;
    }
    const sendClass = classId !== null && (object === null || object.class_id !== classId);
    try {
      const next = await api.accept(clipId, frame, prompt, { classId: sendClass ? classId : undefined, candidate: draft.candidate });
      apply(next, true);
      dispatch({ type: "clear" });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusyJobId(e instanceof ApiError && e.status === 409 ? e.jobId : null);
    }
  }, [prompt, candidates.length, needsClass, classId, object, clipId, frame, draft.candidate, apply]);

  const clear = useCallback(() => {
    dispatch({ type: "clear" });
    setError(null);
  }, []);

  return { draft, dispatch, candidates, busy, error, busyJobId, canAccept, accept, clear };
}
```

`web/src/components/PromptPanel.tsx`:

```tsx
import type { FormEvent } from "react";
import type { Prompting } from "../hooks/usePrompting";
import type { Tool } from "../lib/promptState";

type Props = { prompting: Prompting; disabledReason: string | null };

const TOOLS: [Tool, string, string][] = [
  ["clicks", "Click", "Left click = include, right click = exclude. Each click refreshes the preview."],
  ["box", "Box", "Drag a box around the object, then refine with clicks."],
  ["text", "Text", "Describe the object, then pick one of the numbered candidates."],
];
const base = "rounded px-3 py-1 text-sm disabled:opacity-40";

export function PromptPanel({ prompting, disabledReason }: Props) {
  const { draft, dispatch, candidates, busy, error, canAccept, accept, clear } = prompting;
  const disabled = disabledReason !== null;
  const hint = TOOLS.find(([tool]) => tool === draft.tool)?.[2];

  const submit = (e: FormEvent) => {
    e.preventDefault();
    dispatch({ type: "submit" });
  };

  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Prompt</h2>
      <div className="flex gap-1">
        {TOOLS.map(([tool, label]) => (
          <button key={tool} type="button" disabled={disabled} aria-pressed={draft.tool === tool}
            className={`${base} ${draft.tool === tool ? "bg-sky-700" : "bg-neutral-700"}`}
            onClick={() => dispatch({ type: "tool", tool })}>
            {label}
          </button>
        ))}
      </div>
      {disabled ? <p className="text-xs text-amber-400">{disabledReason}</p> : <p className="text-xs text-neutral-400">{hint}</p>}

      {draft.tool === "text" && (
        <form className="flex gap-1" onSubmit={submit}>
          <input aria-label="Text prompt" disabled={disabled} value={draft.phrase} placeholder="e.g. red connector"
            className="min-w-0 flex-1 rounded border border-neutral-600 bg-neutral-800 px-2 py-1"
            onChange={(e) => dispatch({ type: "phrase", phrase: e.target.value })} />
          <button type="submit" disabled={disabled || draft.phrase.trim() === ""} className={`${base} bg-neutral-700`}>Find</button>
        </form>
      )}

      {candidates.length > 1 && (
        <ul className="flex flex-col gap-1">
          {candidates.map((c, i) => (
            <li key={i}>
              <button type="button" aria-pressed={draft.candidate === i}
                className={`${base} w-full text-left ${draft.candidate === i ? "bg-amber-600" : "bg-neutral-700"}`}
                onClick={() => dispatch({ type: "candidate", index: i })}>
                {i + 1} · score {c.score.toFixed(2)} · {c.area} px
              </button>
            </li>
          ))}
        </ul>
      )}
      {candidates.length === 1 && (
        <p className="text-xs text-neutral-400">Preview: {candidates[0].area} px, score {candidates[0].score.toFixed(2)}</p>
      )}
      {busy && <p className="text-xs text-neutral-400">Predicting…</p>}
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}

      <div className="flex gap-1">
        <button type="button" disabled={!canAccept} className={`${base} bg-emerald-700`} onClick={() => void accept()}>Accept (Enter)</button>
        <button type="button" disabled={disabled} className={`${base} bg-neutral-700`} onClick={clear}>Clear (Esc)</button>
        <button type="button" disabled={disabled} className={`${base} bg-neutral-700`} onClick={() => dispatch({ type: "undo" })}>Undo (Z)</button>
      </div>
    </section>
  );
}
```

`web/src/components/ObjectPanel.tsx`:

```tsx
import { useState, type FormEvent } from "react";
import type { ClipDetail } from "../api/types";

type Props = {
  detail: ClipDetail;
  classId: number | null;
  onClassId: (id: number) => void;
  onAddClass: (name: string) => Promise<void>;
};

const field = "min-w-0 flex-1 rounded border border-neutral-600 bg-neutral-800 px-2 py-1";

export function ObjectPanel({ detail, classId, onClassId, onAddClass }: Props) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const object = detail.clip.objects[0] ?? null;
  const pending = object !== null && classId !== null && classId !== object.class_id;

  const add = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (trimmed === "") return;
    setError(null);
    onAddClass(trimmed).then(() => setName(""), (err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  };

  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Object</h2>
      <select aria-label="Class" className={field} value={classId ?? ""} onChange={(e) => onClassId(Number(e.target.value))}>
        {classId === null && <option value="">(choose a class)</option>}
        {detail.classes.map((c) => (
          <option key={c.id} value={c.id}>{c.name}</option>
        ))}
      </select>
      <form className="flex gap-1" onSubmit={add}>
        <input aria-label="New class" className={field} placeholder="New class" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="submit" className="rounded bg-neutral-700 px-3 py-1">Add</button>
      </form>
      {pending && <p className="text-xs text-amber-400">The new class is applied with the next Accept.</p>}
      {object === null && <p className="text-xs text-neutral-400">No object yet — prompt on any frame and accept the mask.</p>}
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}
    </section>
  );
}
```

Replace `web/src/pages/Label.tsx`:

```tsx
import { useCallback, useContext, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, frameUrl, maskUrl } from "../api/rest";
import { ClipInfo } from "../components/ClipInfo";
import { FrameCanvas } from "../components/FrameCanvas";
import { ObjectPanel } from "../components/ObjectPanel";
import { PromptPanel } from "../components/PromptPanel";
import { Timeline } from "../components/Timeline";
import { Transport } from "../components/Transport";
import { useClip } from "../hooks/useClip";
import { HealthContext, engineReady } from "../hooks/useHealth";
import { usePlayback } from "../hooks/usePlayback";
import { usePrompting } from "../hooks/usePrompting";
import { FrameStore, MASK_ALPHA } from "../lib/frameCache";
import { hexToRgb } from "../lib/mask";
import { clampFrame } from "../lib/timeline";


export function Label() {
  const { clipId = "" } = useParams();
  const clip = useClip(clipId);
  const health = useContext(HealthContext);
  const store = useMemo(() => new FrameStore(), [clipId]); // a new clip gets a fresh cache
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [classId, setClassId] = useState<number | null>(null);

  const detail = clip.detail;
  const frameCount = detail?.clip.frame_count ?? 0;
  const fps = detail?.clip.fps ?? 30;
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });

  const object = detail?.clip.objects[0] ?? null;
  useEffect(() => {
    if (classId === null && detail !== null) setClassId(object?.class_id ?? detail.classes[0]?.id ?? null);
  }, [classId, detail, object]);

  const disabledReason = !engineReady(health)
    ? "The SAM3 engine is not loaded."
    : detail?.states[frame] === "deleted"
      ? "This frame is deleted — restore it to label it."
      : playing
        ? "Pause playback to prompt."
        : null;
  const prompting = usePrompting({ clipId, frame, enabled: disabledReason === null && detail !== null, clip, classId });

  const step = useCallback((delta: number) => setFrame((f) => clampFrame(f + delta, frameCount)), [frameCount]);
  const seek = useCallback((f: number) => setFrame(clampFrame(f, frameCount)), [frameCount]);

  const shownClass = classId ?? object?.class_id ?? null;
  const color = detail?.classes.find((c) => c.id === shownClass)?.color ?? "#22c55e";
  const hasMask = (i: number) => detail !== null && (detail.states[i] === "masked" || detail.states[i] === "keyframe");
  const maskVersion = clip.maskVersion;

  useEffect(() => {
    if (detail === null) return;
    const rgb = hexToRgb(color);
    const ahead = playing ? 30 : 4;
    for (let i = frame + 1; i <= Math.min(frame + ahead, frameCount - 1); i++) {
      void store.image(frameUrl(clipId, i));
      if (overlay && hasMask(i)) void store.layer(maskUrl(clipId, i, maskVersion(i)), rgb, MASK_ALPHA);
    }
    // hasMask is derived from detail
  }, [store, clipId, frame, playing, overlay, detail, color, frameCount, maskVersion]);

  const addClass = useCallback(
    async (name: string) => {
      if (detail === null) return;
      const created = await api.addClass(detail.clip.project_id, name);
      clip.apply({ ...detail, classes: [...detail.classes, created] }, false);
      setClassId(created.id);
    },
    [detail, clip],
  );

  if (clip.error !== null) {
    return (
      <div className="p-4 text-sm">
        <p role="alert" className="mb-2 rounded bg-red-900 px-3 py-2">{clip.error}</p>
        <Link to="/" className="underline">Back to clips</Link>
      </div>
    );
  }
  if (detail === null) return <p className="p-4 text-sm text-neutral-400">Loading…</p>;

  const { draft, dispatch } = prompting;
  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1">
          <FrameCanvas
            store={store}
            width={detail.clip.width}
            height={detail.clip.height}
            frameUrl={frameUrl(clipId, frame)}
            maskUrl={overlay && hasMask(frame) ? maskUrl(clipId, frame, maskVersion(frame)) : null}
            color={color}
            tool={disabledReason === null ? draft.tool : null}
            draft={draft}
            candidates={prompting.candidates}
            onPoint={(x, y, label) => dispatch({ type: "point", x, y, label })}
            onBox={(box) => dispatch({ type: "box", box })}
            onPickCandidate={(index) => dispatch({ type: "candidate", index })}
          />
        </div>
        <aside className="flex w-80 flex-col gap-4 overflow-y-auto border-l border-neutral-700 p-3">
          <Link to="/" className="text-xs text-neutral-400 underline">← Back to clips</Link>
          <ClipInfo detail={detail} />
          <ObjectPanel detail={detail} classId={classId} onClassId={setClassId} onAddClass={addClass} />
          <PromptPanel prompting={prompting} disabledReason={disabledReason} />
        </aside>
      </div>
      <footer className="flex flex-col gap-2 border-t border-neutral-700 p-3">
        <Transport
          frame={frame}
          frameCount={frameCount}
          fps={fps}
          playing={playing}
          overlay={overlay}
          name={detail.clip.original_names?.[String(frame)] ?? null}
          onStep={step}
          onTogglePlay={() => setPlaying((p) => !p)}
          onToggleOverlay={() => setOverlay((o) => !o)}
        />
        <Timeline states={detail.states} frame={frame} onSeek={seek} />
      </footer>
    </div>
  );
}
```

- [ ] **Step 4: Run tests, typecheck, build**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run build'`
Expected: `108 passed` (94 after the review-driven fixes to Tasks 3–5, + 7 prompting hook + 5 prompt panel + 2 label prompting); no warnings; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/hooks/usePrompting.ts web/src/hooks/usePrompting.test.tsx web/src/components/PromptPanel.tsx \
  web/src/components/PromptPanel.test.tsx web/src/components/ObjectPanel.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git commit -m "feat(web): click, box and text prompting with live preview, class picker and accept"
```

---

### Task 7: Propagation, cleanup, keyboard shortcuts, demo seed and QA checklist

**Files:**
- Create: `web/src/hooks/useKeyboard.ts`, `web/src/components/PropagatePanel.tsx`, `web/src/components/CleanupPanel.tsx`, `scripts/seed_demo.py`
- Modify: `web/src/pages/Label.tsx` (full replacement below), `web/src/pages/Label.test.tsx` (extend the mock and add the tests below), `README.md`
- Test: `web/src/hooks/useKeyboard.test.ts`

**Interfaces:**
- Consumes: `api.propagate`, `api.cancelJob`, `api.deleteFrame`, `api.restoreFrame`, `api.prune`, `api.clearRange`, `retryInteractive`, `ApiError` (Task 1); `findNext` (Task 2); `useJob`, `JobHandle`, `isTerminal` (Task 3); `useClip`, `usePrompting` and the components of Tasks 5–6. After the Task 5 review, `ClipHandle` also has `refreshError: string | null` (a failed background `reload()` keeps the loaded clip) and `frameCache.ts` exports `MASK_ALPHA` / `PREVIEW_ALPHA`.
- Produces: `useKeyboard.ts`: `type KeyAction`, `keyAction(e): KeyAction | null` (pure), `useKeyboard(onAction)`. `PropagatePanel.tsx`: props `{ job: JobHandle; running: boolean; blockedReason: string | null; follow: boolean; error: string | null; onStart(direction): void; onStop(): void; onFollow(follow: boolean): void }`. `CleanupPanel.tsx`: props `{ frame: number; frameCount: number; state: FrameState; clearBlockedReason: string | null; error: string | null; onToggleDelete(): void; onPrune(): void; onClearRange(start: number, end: number): void }`.

Behaviour:
- Propagation starts from the current frame and only from a keyframe; otherwise the buttons are disabled with the hint "Go to a keyframe (K) to propagate from it." `POST /propagate` goes through `retryInteractive`. A 409 that names a real job id means a job is already running (for example after a page reload): attach to it instead of showing an error. The job id is also kept in `sessionStorage` under `vsp.job.<clipId>` so a reload re-attaches; it is removed when the job ends.
- While a job runs: prompting is disabled, each `frame` event patches that frame's state and mask version, and with "Follow" on (default) the view jumps to the streamed frame. If the socket dropped (`transport === "poll"`), the clip is reloaded once a second instead.
- When the job ends the clip is reloaded (keyframes, ranges, every mask version). If the result has a numeric `lost_at`, the view seeks there.
- Stop = `DELETE /api/jobs/{id}`; masks that landed are kept (the server guarantees it).
- `Del` deletes the current frame, or restores it when it is already deleted. Prune asks for confirmation and deletes every frame from the current one to the end. Clear range removes propagated masks in `[start, end]` (keyframes are kept by the server) and is disabled while a job runs.
- Keyboard handling blurs a focused button before acting, so `Enter` / `Space` never also "click" the last-used button.

- [ ] **Step 1: Write the failing tests**

`web/src/hooks/useKeyboard.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { keyAction } from "./useKeyboard";

const key = (k: string, mods: Partial<{ shiftKey: boolean; ctrlKey: boolean; metaKey: boolean; altKey: boolean }> = {}) =>
  keyAction({ key: k, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false, ...mods });

describe("keyAction", () => {
  it("maps accept, clear and undo", () => {
    expect(key("Enter")).toEqual({ type: "accept" });
    expect(key("Escape")).toEqual({ type: "clear" });
    expect(key("z")).toEqual({ type: "undo" });
    expect(key("Z")).toEqual({ type: "undo" });
  });
  it("steps by 1, or by 10 with Shift", () => {
    expect(key("ArrowRight")).toEqual({ type: "step", delta: 1 });
    expect(key("ArrowLeft")).toEqual({ type: "step", delta: -1 });
    expect(key("ArrowRight", { shiftKey: true })).toEqual({ type: "step", delta: 10 });
    expect(key("ArrowLeft", { shiftKey: true })).toEqual({ type: "step", delta: -10 });
  });
  it("jumps between keyframes and to the next empty frame", () => {
    expect(key("k")).toEqual({ type: "keyframe", dir: 1 });
    expect(key("K", { shiftKey: true })).toEqual({ type: "keyframe", dir: -1 });
    expect(key("g")).toEqual({ type: "nextEmpty" });
  });
  it("maps play, delete and overlay", () => {
    expect(key(" ")).toEqual({ type: "play" });
    expect(key("Delete")).toEqual({ type: "delete" });
    expect(key("m")).toEqual({ type: "overlay" });
  });
  it("ignores unknown keys and browser shortcuts", () => {
    expect(key("q")).toBeNull();
    expect(key("z", { ctrlKey: true })).toBeNull();
    expect(key("k", { metaKey: true })).toBeNull();
    expect(key("ArrowRight", { altKey: true })).toBeNull();
  });
});
```

In `web/src/pages/Label.test.tsx`: extend the mocked `api` object with `propagate: vi.fn(), cancelJob: vi.fn(), deleteFrame: vi.fn(), restoreFrame: vi.fn(), prune: vi.fn(), clearRange: vi.fn(), getJob: vi.fn()`; add a fake socket (same class as in `useJob.test.tsx`) stubbed as the global `WebSocket` in `beforeEach`, together with `sessionStorage.clear()`; then add:

```tsx
class FakeSocket {
  static last: FakeSocket | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(readonly url: string) {
    FakeSocket.last = this;
  }
  close() {}
  emit(event: object) {
    act(() => this.onmessage?.({ data: JSON.stringify(event) }));
  }
}
// beforeEach: FakeSocket.last = null; vi.stubGlobal("WebSocket", FakeSocket); sessionStorage.clear();
// (import `act` from "@testing-library/react")

describe("Label page (propagation, cleanup, keyboard)", () => {
  it("propagates from a keyframe, follows the streamed frames and reloads when the job ends", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j1" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(api.propagate).toHaveBeenCalledWith("c1", 0, "forward"));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));
    expect(sessionStorage.getItem("vsp.job.c1")).toBe("j1");
    FakeSocket.last!.emit({ seq: 0, type: "progress", done: 0, total: 39 });
    FakeSocket.last!.emit({ seq: 1, type: "frame", frame: 20, has_mask: true, done: 1, total: 39 });
    expect(screen.getByText("frame 20 / 39")).toBeTruthy();
    expect(screen.getByText(/20 masked/)).toBeTruthy();
    expect(screen.getByText(/propagation is running/i)).toBeTruthy();
    const loads = vi.mocked(api.getClip).mock.calls.length;
    FakeSocket.last!.emit({ seq: 2, type: "done", processed: 39, written: 38, lost_at: 33, cancelled: false });
    await waitFor(() => expect(vi.mocked(api.getClip).mock.calls.length).toBe(loads + 1));
    expect(await screen.findByText("frame 33 / 39")).toBeTruthy();
    expect(screen.getByText(/lost the object at frame 33/i)).toBeTruthy();
    expect(sessionStorage.getItem("vsp.job.c1")).toBeNull();
  });

  it("only offers propagation on a keyframe, and Stop cancels the job", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j2" });
    vi.mocked(api.cancelJob).mockResolvedValue({ id: "j2", kind: "propagate", status: "running", result: null, error: null, created_at: "", n_events: 0 });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/Go to a keyframe/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Previous frame" }));
    fireEvent.click(screen.getByRole("button", { name: "Both" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j2$/));
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(api.cancelJob).toHaveBeenCalledWith("j2"));
  });

  it("attaches to the job a 409 names instead of reporting an error", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockRejectedValue(new ApiError(409, "job j7 is still running", "j7"));
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "◀ Backward" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j7$/));
  });

  it("handles the keyboard map", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const deleted = { ...detail, states: detail.states.map((s, i) => (i === 11 ? "deleted" : s)) } as ClipDetail;
    vi.mocked(api.deleteFrame).mockResolvedValue(deleted);
    vi.mocked(api.restoreFrame).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    expect(screen.getByText("frame 11 / 39")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Delete" });
    await waitFor(() => expect(api.deleteFrame).toHaveBeenCalledWith("c1", 11));
    await screen.findByText(/2 deleted/);
    fireEvent.keyDown(window, { key: "Delete" });
    await waitFor(() => expect(api.restoreFrame).toHaveBeenCalledWith("c1", 11));
    fireEvent.keyDown(window, { key: "K", shiftKey: true });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    fireEvent.keyDown(window, { key: "g" });
    expect(screen.getByText("frame 20 / 39")).toBeTruthy();
    fireEvent.keyDown(window, { key: "m" });
    expect((screen.getByRole("checkbox", { name: /overlay/i }) as HTMLInputElement).checked).toBe(false);
    fireEvent.keyDown(window, { key: " " });
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
  });

  it("ignores shortcuts typed into a field", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(screen.getByLabelText("New class"), { key: "ArrowRight" });
    fireEvent.keyDown(screen.getByLabelText("New class"), { key: "m" });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy();
    expect((screen.getByRole("checkbox", { name: /overlay/i }) as HTMLInputElement).checked).toBe(true);
  });

  it("prunes after confirmation and clears a range", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prune).mockResolvedValue(detail);
    vi.mocked(api.clearRange).mockResolvedValue(detail);
    const confirm = vi.fn().mockReturnValueOnce(false).mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    fireEvent.click(screen.getByRole("button", { name: /Prune/ }));
    expect(api.prune).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Prune/ }));
    await waitFor(() => expect(api.prune).toHaveBeenCalledWith("c1", 10));
    fireEvent.change(screen.getByLabelText("Range start"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("Range end"), { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: /Clear masks/ }));
    await waitFor(() => expect(api.clearRange).toHaveBeenCalledWith("c1", 5, 9));
  });
});
```

(The Transport also has a "Forward 10 frames" button, which is why the propagate buttons are matched by their exact names. `detail` has frame 30 deleted already, so after deleting frame 11 the info line reads "2 deleted". Frame 20 is the first `empty` frame after frame 0.)

- [ ] **Step 2: Run them to see them fail**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run src/hooks/useKeyboard.test.ts src/pages/Label.test.tsx'`
Expected: FAIL — `./useKeyboard` cannot be resolved; the six new Label tests fail (no Propagate / Cleanup panels, no shortcuts).

- [ ] **Step 3: Implement**

`web/src/hooks/useKeyboard.ts`:

```ts
import { useEffect, useRef } from "react";

export type KeyAction =
  | { type: "accept" }
  | { type: "clear" }
  | { type: "undo" }
  | { type: "step"; delta: number }
  | { type: "play" }
  | { type: "keyframe"; dir: 1 | -1 }
  | { type: "nextEmpty" }
  | { type: "delete" }
  | { type: "overlay" };

type KeyLike = { key: string; shiftKey: boolean; ctrlKey: boolean; metaKey: boolean; altKey: boolean };

/** Spec §7 keyboard map. Ctrl / Meta / Alt combinations belong to the browser. */
export function keyAction(e: KeyLike): KeyAction | null {
  if (e.ctrlKey || e.metaKey || e.altKey) return null;
  switch (e.key) {
    case "Enter":
      return { type: "accept" };
    case "Escape":
      return { type: "clear" };
    case "ArrowRight":
      return { type: "step", delta: e.shiftKey ? 10 : 1 };
    case "ArrowLeft":
      return { type: "step", delta: e.shiftKey ? -10 : -1 };
    case " ":
      return { type: "play" };
    case "Delete":
      return { type: "delete" };
  }
  switch (e.key.toLowerCase()) {
    case "z":
      return { type: "undo" };
    case "k":
      return { type: "keyframe", dir: e.shiftKey ? -1 : 1 };
    case "g":
      return { type: "nextEmpty" };
    case "m":
      return { type: "overlay" };
  }
  return null;
}

const isField = (t: EventTarget | null): boolean =>
  t instanceof HTMLElement && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);

export function useKeyboard(onAction: (action: KeyAction) => void): void {
  const handler = useRef(onAction);
  handler.current = onAction;

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isField(e.target)) return;
      const action = keyAction(e);
      if (action === null) return;
      e.preventDefault();
      // Enter / Space would otherwise also activate whichever button was clicked last
      if (document.activeElement instanceof HTMLElement && document.activeElement.tagName === "BUTTON") document.activeElement.blur();
      handler.current(action);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
```

`web/src/components/PropagatePanel.tsx`:

```tsx
import type { Direction } from "../api/types";
import type { JobHandle } from "../hooks/useJob";

type Props = {
  job: JobHandle;
  running: boolean;
  /** why propagation cannot start from here, or null */
  blockedReason: string | null;
  follow: boolean;
  error: string | null;
  onStart: (direction: Direction) => void;
  onStop: () => void;
  onFollow: (follow: boolean) => void;
};

const button = "rounded bg-neutral-700 px-3 py-1 text-sm hover:bg-neutral-600 disabled:opacity-40";
const num = (v: unknown): number | null => (typeof v === "number" ? v : null);

export function PropagatePanel({ job, running, blockedReason, follow, error, onStart, onStop, onFollow }: Props) {
  const blocked = running || blockedReason !== null;
  const lostAt = num(job.result?.lost_at);
  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Propagate</h2>
      <div className="flex gap-1">
        <button type="button" className={button} disabled={blocked} onClick={() => onStart("backward")}>◀ Backward</button>
        <button type="button" className={button} disabled={blocked} onClick={() => onStart("both")}>Both</button>
        <button type="button" className={button} disabled={blocked} onClick={() => onStart("forward")}>Forward ▶</button>
        <button type="button" className={`${button} ml-auto bg-red-800 hover:bg-red-700`} disabled={!running} onClick={onStop}>Stop</button>
      </div>
      {!running && blockedReason !== null && <p className="text-xs text-neutral-400">{blockedReason}</p>}
      <label className="flex items-center gap-1 text-xs">
        <input type="checkbox" checked={follow} onChange={(e) => onFollow(e.target.checked)} />
        Follow the propagation
      </label>

      {running && (
        <>
          <p className="text-xs text-amber-400">Propagating — prompting is paused. Stop keeps the masks written so far.</p>
          <progress className="w-full" value={job.total === null ? undefined : job.done} max={job.total ?? undefined} />
          <p className="text-xs tabular-nums text-neutral-300">
            {job.done} / {job.total ?? "?"} frames{job.transport === "poll" ? " (live updates lost — polling)" : ""}
          </p>
        </>
      )}
      {job.logs.length > 0 && <p className="text-xs text-neutral-400">{job.logs[job.logs.length - 1]}</p>}
      {job.status === "done" && (
        <p className="text-xs text-emerald-400">
          Done: {num(job.result?.processed) ?? 0} frames processed, {num(job.result?.written) ?? 0} masks written.
        </p>
      )}
      {job.status === "cancelled" && <p className="text-xs text-amber-400">Stopped. The masks written so far were kept.</p>}
      {lostAt !== null && (
        <p className="text-xs text-amber-400">Lost the object at frame {lostAt} — correct it there and propagate again.</p>
      )}
      {job.status === "error" && <p role="alert" className="text-xs text-red-400">{job.error}</p>}
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}
    </section>
  );
}
```

`web/src/components/CleanupPanel.tsx`:

```tsx
import { useState } from "react";
import type { FrameState } from "../api/types";

type Props = {
  frame: number;
  frameCount: number;
  state: FrameState;
  /** why clear-range is unavailable (a job is running), or null */
  clearBlockedReason: string | null;
  error: string | null;
  onToggleDelete: () => void;
  onPrune: () => void;
  onClearRange: (start: number, end: number) => void;
};

const button = "rounded bg-neutral-700 px-3 py-1 text-sm hover:bg-neutral-600 disabled:opacity-40";
const field = "w-20 rounded border border-neutral-600 bg-neutral-800 px-2 py-1";

export function CleanupPanel({ frame, frameCount, state, clearBlockedReason, error, onToggleDelete, onPrune, onClearRange }: Props) {
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const last = Math.max(0, frameCount - 1);
  const clampIndex = (v: string) => Math.max(0, Math.min(last, Math.floor(Number(v)) || 0));
  const valid = start <= end;

  return (
    <section className="flex flex-col gap-2 text-sm">
      <h2 className="font-semibold">Cleanup</h2>
      <div className="flex gap-1">
        <button type="button" className={button} onClick={onToggleDelete}>
          {state === "deleted" ? "Restore frame (Del)" : "Delete frame (Del)"}
        </button>
        <button type="button" className={button} onClick={onPrune}>Prune from here</button>
      </div>
      <p className="text-xs text-neutral-400">Deleted frames are left out of the export. Prune deletes frame {frame} and everything after it.</p>

      <div className="flex flex-wrap items-center gap-1">
        <input aria-label="Range start" type="number" min={0} max={last} className={field} value={start}
          onChange={(e) => setStart(clampIndex(e.target.value))} />
        <button type="button" className={button} onClick={() => setStart(frame)}>[ here</button>
        <span>–</span>
        <input aria-label="Range end" type="number" min={0} max={last} className={field} value={end}
          onChange={(e) => setEnd(clampIndex(e.target.value))} />
        <button type="button" className={button} onClick={() => setEnd(frame)}>here ]</button>
      </div>
      <button type="button" className={button} disabled={!valid || clearBlockedReason !== null} onClick={() => onClearRange(start, end)}>
        Clear masks in range
      </button>
      <p className="text-xs text-neutral-400">{clearBlockedReason ?? "Removes propagated masks in the range; keyframes are kept."}</p>
      {error !== null && <p role="alert" className="text-xs text-red-400">{error}</p>}
    </section>
  );
}
```

Replace `web/src/pages/Label.tsx` (final version):

```tsx
import { useCallback, useContext, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, INTERACTIVE, api, frameUrl, maskUrl, retryInteractive } from "../api/rest";
import type { ClipDetail, Direction, JobEvent } from "../api/types";
import { CleanupPanel } from "../components/CleanupPanel";
import { ClipInfo } from "../components/ClipInfo";
import { FrameCanvas } from "../components/FrameCanvas";
import { ObjectPanel } from "../components/ObjectPanel";
import { PromptPanel } from "../components/PromptPanel";
import { PropagatePanel } from "../components/PropagatePanel";
import { Timeline } from "../components/Timeline";
import { Transport } from "../components/Transport";
import { useClip } from "../hooks/useClip";
import { HealthContext, engineReady } from "../hooks/useHealth";
import { useJob } from "../hooks/useJob";
import { useKeyboard, type KeyAction } from "../hooks/useKeyboard";
import { usePlayback } from "../hooks/usePlayback";
import { usePrompting } from "../hooks/usePrompting";
import { FrameStore, MASK_ALPHA } from "../lib/frameCache";
import { isTerminal } from "../lib/jobReducer";
import { hexToRgb } from "../lib/mask";
import { clampFrame, findNext } from "../lib/timeline";

const jobKey = (clipId: string) => `vsp.job.${clipId}`;
const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function Label() {
  const { clipId = "" } = useParams();
  const clip = useClip(clipId);
  const health = useContext(HealthContext);
  const store = useMemo(() => new FrameStore(), [clipId]); // a new clip gets a fresh cache
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [classId, setClassId] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => sessionStorage.getItem(jobKey(clipId)));
  const [follow, setFollow] = useState(true);
  const [propagateError, setPropagateError] = useState<string | null>(null);
  const [cleanupError, setCleanupError] = useState<string | null>(null);

  const detail = clip.detail;
  const frameCount = detail?.clip.frame_count ?? 0;
  const fps = detail?.clip.fps ?? 30;
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });

  const object = detail?.clip.objects[0] ?? null;
  useEffect(() => {
    if (classId === null && detail !== null) setClassId(object?.class_id ?? detail.classes[0]?.id ?? null);
  }, [classId, detail, object]);

  // ---- propagation job -------------------------------------------------------------------------------------------
  const { patchFrame, reload, apply } = clip;
  const onJobEvent = useCallback(
    (event: JobEvent) => {
      if (event.type !== "frame" || typeof event.frame !== "number") return;
      patchFrame(event.frame, event.has_mask === true);
      if (follow) setFrame(event.frame);
    },
    [patchFrame, follow],
  );
  const job = useJob(jobId, onJobEvent);
  const running = jobId !== null && !isTerminal(job.status);

  useEffect(() => {
    if (jobId === null || !isTerminal(job.status)) return;
    sessionStorage.removeItem(jobKey(clipId));
    const lostAt = job.result?.lost_at;
    void reload().then(() => {
      if (typeof lostAt === "number") setFrame(lostAt);
    });
  }, [jobId, job.status, job.result, clipId, reload]);

  useEffect(() => {
    if (!running || job.transport !== "poll") return;
    const id = setInterval(() => void reload(), 1000); // no frame events without the socket
    return () => clearInterval(id);
  }, [running, job.transport, reload]);

  const attach = useCallback(
    (id: string) => {
      sessionStorage.setItem(jobKey(clipId), id);
      setJobId(id);
      setPlaying(false);
    },
    [clipId],
  );

  const startPropagation = useCallback(
    (direction: Direction) => {
      setPropagateError(null);
      retryInteractive(() => api.propagate(clipId, frame, direction)).then(
        (r) => attach(r.job_id),
        (e: unknown) => {
          if (e instanceof ApiError && e.status === 409 && e.jobId !== null && e.jobId !== INTERACTIVE) attach(e.jobId);
          else setPropagateError(message(e));
        },
      );
    },
    [clipId, frame, attach],
  );

  const stopPropagation = useCallback(() => {
    if (jobId !== null) api.cancelJob(jobId).catch((e: unknown) => setPropagateError(message(e)));
  }, [jobId]);

  // ---- prompting -------------------------------------------------------------------------------------------------
  const state = detail?.states[frame] ?? "empty";
  const disabledReason = !engineReady(health)
    ? "The SAM3 engine is not loaded."
    : running
      ? "A propagation is running — Stop it to prompt."
      : state === "deleted"
        ? "This frame is deleted — restore it to label it."
        : playing
          ? "Pause playback to prompt."
          : null;
  const prompting = usePrompting({ clipId, frame, enabled: disabledReason === null && detail !== null, clip, classId });

  useEffect(() => {
    if (prompting.busyJobId !== null && prompting.busyJobId !== INTERACTIVE && jobId === null) attach(prompting.busyJobId);
  }, [prompting.busyJobId, jobId, attach]);

  // ---- cleanup ---------------------------------------------------------------------------------------------------
  const cleanup = useCallback(
    (call: Promise<ClipDetail>, masksChanged: boolean) => {
      setCleanupError(null);
      call.then((next) => apply(next, masksChanged), (e: unknown) => setCleanupError(message(e)));
    },
    [apply],
  );
  const toggleDelete = useCallback(
    () => cleanup(state === "deleted" ? api.restoreFrame(clipId, frame) : api.deleteFrame(clipId, frame), false),
    [cleanup, state, clipId, frame],
  );
  const prune = useCallback(() => {
    if (window.confirm(`Delete frame ${frame} and every frame after it?`)) cleanup(api.prune(clipId, frame), false);
  }, [cleanup, clipId, frame]);
  const clearRange = useCallback(
    (start: number, end: number) => cleanup(api.clearRange(clipId, start, end), true),
    [cleanup, clipId],
  );

  // ---- navigation and keyboard -----------------------------------------------------------------------------------
  const step = useCallback((delta: number) => setFrame((f) => clampFrame(f + delta, frameCount)), [frameCount]);
  const seek = useCallback((f: number) => setFrame(clampFrame(f, frameCount)), [frameCount]);
  const jump = (target: number | null) => {
    if (target !== null) setFrame(target);
  };

  useKeyboard((action: KeyAction) => {
    if (detail === null) return;
    switch (action.type) {
      case "accept":
        if (prompting.canAccept) void prompting.accept();
        break;
      case "clear":
        prompting.clear();
        break;
      case "undo":
        prompting.dispatch({ type: "undo" });
        break;
      case "step":
        step(action.delta);
        break;
      case "play":
        if (!running) setPlaying((p) => !p);
        break;
      case "keyframe":
        jump(findNext(detail.states, frame, action.dir, "keyframe"));
        break;
      case "nextEmpty":
        jump(findNext(detail.states, frame, 1, "empty"));
        break;
      case "delete":
        toggleDelete();
        break;
      case "overlay":
        setOverlay((o) => !o);
        break;
    }
  });

  // ---- prefetch --------------------------------------------------------------------------------------------------
  const shownClass = classId ?? object?.class_id ?? null;
  const color = detail?.classes.find((c) => c.id === shownClass)?.color ?? "#22c55e";
  const hasMask = (i: number) => detail !== null && (detail.states[i] === "masked" || detail.states[i] === "keyframe");
  const maskVersion = clip.maskVersion;

  useEffect(() => {
    if (detail === null || running) return; // during a job the view follows the stream; nothing to look ahead for
    const rgb = hexToRgb(color);
    const ahead = playing ? 30 : 4;
    for (let i = frame + 1; i <= Math.min(frame + ahead, frameCount - 1); i++) {
      void store.image(frameUrl(clipId, i));
      if (overlay && hasMask(i)) void store.layer(maskUrl(clipId, i, maskVersion(i)), rgb, MASK_ALPHA);
    }
    // hasMask is derived from detail
  }, [store, clipId, frame, playing, overlay, detail, color, frameCount, maskVersion, running]);

  const addClass = useCallback(
    async (name: string) => {
      if (detail === null) return;
      const created = await api.addClass(detail.clip.project_id, name);
      apply({ ...detail, classes: [...detail.classes, created] }, false);
      setClassId(created.id);
    },
    [detail, apply],
  );

  if (clip.error !== null) {
    return (
      <div className="p-4 text-sm">
        <p role="alert" className="mb-2 rounded bg-red-900 px-3 py-2">{clip.error}</p>
        <Link to="/" className="underline">Back to clips</Link>
      </div>
    );
  }
  if (detail === null) return <p className="p-4 text-sm text-neutral-400">Loading…</p>;

  const propagateBlocked = !engineReady(health)
    ? "The SAM3 engine is not loaded."
    : state !== "keyframe"
      ? "Go to a keyframe (K) to propagate from it."
      : null;
  const { draft, dispatch } = prompting;

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1">
          <FrameCanvas
            store={store}
            width={detail.clip.width}
            height={detail.clip.height}
            frameUrl={frameUrl(clipId, frame)}
            maskUrl={overlay && hasMask(frame) ? maskUrl(clipId, frame, maskVersion(frame)) : null}
            color={color}
            tool={disabledReason === null ? draft.tool : null}
            draft={draft}
            candidates={prompting.candidates}
            onPoint={(x, y, label) => dispatch({ type: "point", x, y, label })}
            onBox={(box) => dispatch({ type: "box", box })}
            onPickCandidate={(index) => dispatch({ type: "candidate", index })}
          />
        </div>
        <aside className="flex w-80 flex-col gap-4 overflow-y-auto border-l border-neutral-700 p-3">
          <Link to="/" className="text-xs text-neutral-400 underline">← Back to clips</Link>
          <ClipInfo detail={detail} />
          {clip.refreshError !== null && (
            <p role="alert" className="rounded bg-red-900 px-2 py-1 text-xs">Could not refresh the clip: {clip.refreshError}</p>
          )}
          <ObjectPanel detail={detail} classId={classId} onClassId={setClassId} onAddClass={addClass} />
          <PromptPanel prompting={prompting} disabledReason={disabledReason} />
          <PropagatePanel
            job={job}
            running={running}
            blockedReason={propagateBlocked}
            follow={follow}
            error={propagateError}
            onStart={startPropagation}
            onStop={stopPropagation}
            onFollow={setFollow}
          />
          <CleanupPanel
            frame={frame}
            frameCount={frameCount}
            state={state}
            clearBlockedReason={running ? "Clearing a range is unavailable while a propagation runs." : null}
            error={cleanupError}
            onToggleDelete={toggleDelete}
            onPrune={prune}
            onClearRange={clearRange}
          />
        </aside>
      </div>
      <footer className="flex flex-col gap-2 border-t border-neutral-700 p-3">
        <Transport
          frame={frame}
          frameCount={frameCount}
          fps={fps}
          playing={playing}
          overlay={overlay}
          name={detail.clip.original_names?.[String(frame)] ?? null}
          onStep={step}
          onTogglePlay={() => {
            if (!running) setPlaying((p) => !p);
          }}
          onToggleOverlay={() => setOverlay((o) => !o)}
        />
        <Timeline states={detail.states} frame={frame} onSeek={seek} />
      </footer>
    </div>
  );
}
```

- [ ] **Step 4: Demo seed script**

`scripts/seed_demo.py`:

```python
"""Create a demo project with a synthetic moving disc through the HTTP API (manual QA aid for the web GUI).

Usage (server running):  uv run python scripts/seed_demo.py [--base http://127.0.0.1:28000] [--frames 40] [--no-label]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "data" / "demo_frames"  # under the repository, hence inside the default allowed root "~"
W, H, R = 640, 480, 45


def write_frames(n: int) -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("*.png"):
        old.unlink()
    for i in range(n):
        img = np.full((H, W, 3), 40, np.uint8)
        cv2.rectangle(img, (420, 60), (560, 160), (90, 90, 90), -1)  # a static distractor
        cv2.circle(img, (120 + 8 * i, 240), R, (30, 200, 230), -1)
        cv2.imwrite(str(FRAMES_DIR / f"{i:04d}.png"), img)


def wait_job(client: httpx.Client, job_id: str) -> dict:
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:28000")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--no-label", action="store_true", help="import only; do not accept a prompt or propagate")
    args = ap.parse_args()

    write_frames(args.frames)
    # trust_env=False: this machine exports http_proxy, which must not be used for localhost
    with httpx.Client(base_url=args.base, trust_env=False, timeout=120) as client:
        health = client.get("/api/health").json()["engine"]
        project = client.post("/api/projects", json={"name": "demo"}).json()
        cls = client.post(f"/api/projects/{project['id']}/classes", json={"name": "disc"}).json()
        started = client.post(
            f"/api/projects/{project['id']}/clips/import-frames", json={"path": str(FRAMES_DIR), "fps": 10, "stride": 1}
        )
        started.raise_for_status()
        job = wait_job(client, started.json()["job_id"])
        if job["status"] != "done":
            print("import failed:", job["error"], file=sys.stderr)
            return 1
        clip_id = job["result"]["clip_id"]
        print(f"clip {clip_id}: {job['result']['frame_count']} frames")

        if not args.no_label:
            if not health["loaded"]:
                print("engine not loaded (yet) — imported only:", health["error"] or "still loading", file=sys.stderr)
            else:
                prompt = {"type": "clicks", "points": [[120, 240, 1]]}
                client.post(f"/api/clips/{clip_id}/accept", json={"frame": 0, "prompt": prompt, "class_id": cls["id"]}).raise_for_status()
                started = client.post(f"/api/clips/{clip_id}/propagate", json={"from_frame": 0, "direction": "forward"})
                started.raise_for_status()
                result = wait_job(client, started.json()["job_id"])
                print("propagate:", result["status"], result["result"])
        print(f"open {args.base}/#/clips/{clip_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: README — seed script and the manual QA checklist**

Append to the "Web GUI" section of `README.md`:

```markdown
### Demo data

With the server running, `uv run python scripts/seed_demo.py` creates a project "demo" with a synthetic 40-frame clip
(a disc moving across the frame), labels frame 0 with one click and propagates forward, then prints the URL to open.
`--no-label` imports only.

### Keyboard

`Enter` accept · `Esc` clear · `Z` undo click · `←/→` step · `Shift+←/→` ±10 · `Space` play/pause · `K` / `Shift+K`
next / previous keyframe · `G` next empty frame · `Del` delete or restore frame · `M` mask overlay · mouse wheel zoom ·
middle-drag pan · right click = negative point.

### Manual QA checklist (run on real footage before trusting a dataset)

1. Import a video and a frame directory (try a stride > 1); cancel one import midway and confirm no clip appears.
2. Label by each method: clicks (add a negative click), a box dragged bottom-right → top-left plus a refining click,
   and a text phrase with several candidates. Check the "no usable mask" message on an empty region.
3. Propagate forward, backward and both from a mid-clip keyframe; press Stop during one and confirm the masks so far stay.
4. Let the object leave the frame: propagation should stop early and seek to the "lost at" frame.
5. Correct a bad frame mid-clip (it becomes a keyframe) and propagate again: the run must stop at the next keyframe.
6. Scrub the timeline and play at clip fps with the overlay on; delete a frame, restore it, prune the tail, clear a range.
7. Reload the page during a propagation: the panel should re-attach to the running job.
8. Stop the server: the banner says it is unreachable; start it without weights: the banner shows the reason and
   importing / cleanup still work.
```

- [ ] **Step 6: Run everything**

Run: `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run build'`
Expected: `119 passed` (108 + 5 keyboard + 6 label); no warnings; build succeeds.

Run: `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q`
Expected: clean; `167 passed` (no Python change in this plan).

- [ ] **Step 7: End-to-end smoke against the real server**

```bash
HF_HUB_OFFLINE=1 uv run vsp serve > /tmp/vsp-serve.log 2>&1 &
for i in $(seq 1 120); do curl -s --noproxy '*' http://127.0.0.1:28000/api/health | grep -q '"loaded":true' && break; sleep 0.5; done
uv run python scripts/seed_demo.py
curl -s --noproxy '*' http://127.0.0.1:28000/ | grep -c '<div id="root">'
kill %1
```

Expected: `clip <id>: 40 frames`, `propagate: done {... 'written': 39 ...}` (the disc stays in frame for all 40 frames), an `open http://127.0.0.1:28000/#/clips/<id>` line, then `1`. `data/` is git-ignored, so `git status --short` stays clean apart from this task's files.

- [ ] **Step 8: Commit**

```bash
git add web/src/hooks/useKeyboard.ts web/src/hooks/useKeyboard.test.ts web/src/components/PropagatePanel.tsx \
  web/src/components/CleanupPanel.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx scripts/seed_demo.py README.md
git commit -m "feat(web): propagation with live progress, frame cleanup, keyboard shortcuts and a demo seed script"
```

---

## Final verification (after all tasks)

- `bash -c 'source ~/.nvm/nvm.sh && cd web && npx vitest run && npm run build'` → 119 passed, build succeeds, no warnings.
- `uv run ruff check server && HF_HUB_OFFLINE=1 uv run pytest -q` → clean, 167 passed.
- `git status --short` is empty; `git ls-files | grep -c -E "vision_train|node_modules|web/dist"` prints `0`.
- Best-effort visual check (Firefox 111 is installed; it has no "wait for network idle", so a blank or "Loading…" capture is not a failure): with the server running and seeded, `firefox --headless --window-size 1500,950 --screenshot /tmp/vsp-label.png 'http://127.0.0.1:28000/#/clips/<id>'`.
- The user runs the README's manual QA checklist on real footage — this is where click-mask quality on real components is judged.
