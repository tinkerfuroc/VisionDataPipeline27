# Plan B: Label Page with Several Objects per Clip — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the Label page shows, edits, propagates and clears any number of objects per clip, one "active" object at a
time, and the server's three plan-A legacy shims are removed once the page no longer needs them.

**Architecture:** the clip detail's new `object_states` lanes become the page's source of truth: `useClip` patches
one lane per object from each propagation `frame` event and recomputes the aggregate `states` from the lanes. A pure
selection model (`lib/activeObject.ts`, `hooks/useObjects.ts`) holds the active row ("an object" or "New object") and
the hidden set; every write (prompt, accept, brush save, propagate "Selected", clear "Active object") is addressed
by that selection. The canvas draws all visible objects through one composite layer per frame (`FrameStore.composite`),
and the Timeline draws one lane per object with a name gutter. Tasks 1–6 are web only; Task 7 removes the server
shims; Task 8 documents and runs the manual QA.

**Tech Stack:** React 18 + TypeScript 5 (`strict`, `noUnusedLocals`, `noUnusedParameters`), Vite 5, Vitest 2.1 +
Testing Library (jsdom opted in per test file), Tailwind 3; server: Python 3.11, FastAPI, pydantic v2, pytest (uv).

**Spec:** `docs/superpowers/specs/2026-09-26-multi-object-labeling-and-export-selection-design.md` — §7 (API), §8
(Label page), §9 (Timeline), §13, §14 plan B row. Plan A (implemented):
`docs/superpowers/plans/2026-09-26-multi-object-server.md`. The implemented server code is the source of truth for
request and response shapes.

## Global Constraints

- **Tools.** Web, without npm: `cd web && node node_modules/.bin/vitest run [files]`,
  `node node_modules/.bin/tsc --noEmit`, `node node_modules/.bin/vite build`. Python only through uv:
  `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server`. Never pip, conda or npm scripts.
- **Green at every commit.** Before each commit run the full web suite and `tsc --noEmit` (tests are type-checked:
  `tsconfig.json` includes `src`), and the server suite + ruff. Tasks 1–6 do not touch the server, Task 7 does not
  touch the web app.
- **Commits.** Stage only the files the task lists (`git add <paths>`, `git rm` for deletions), never `git add -A`
  (another agent may be committing in this worktree). End every commit message with the two attribution lines your
  session's system reminder gives: `Co-Authored-By: <your model> <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6`.
- **Controls live in the right sidebar** (`<aside>` in `Label.tsx`). The Timeline stays in the footer; its lane names
  and lanes are click targets for selection, nothing else is added outside the sidebar.
- **Exact values.**
  - Timeline: inactive lane `LANE_H = 16` px, active lane `ACTIVE_H = 30` px, `LANE_GAP = 2` px, name gutter
    `GUTTER = 72` px, `MAX_LANES = 6`; the lane viewport is at most `MAX_VIEW = 30 + 5 × (16 + 2) = 120` px and
    scrolls beyond that; 6 px above the lanes for the playhead flag (`TOP`), an 18 px ruler below (`RULER`).
    Every lane name in the gutter is `text-xs`, active or not (the user's standing rule: every control at a
    comfortably large scale, never shrunk — which is why inactive lanes are 16 px, not 10); the active lane's name is
    `text-ink` with `aria-pressed="true"`, the others `text-dim`. Hidden objects' lanes (and their gutter names) at
    opacity `0.4`. A clip with no objects: one 30 px lane named "No objects yet". With a "New object" row active,
    every lane is 16 px.
  - Canvas: every visible object is filled in its class colour at `MASK_ALPHA = 0.45`; only the active object gets
    the Mask outline (`outlineFor(style, classRgb)`), the others are fill only (outline edge 0). Draw order: the
    inactive objects in id order, then the active object last (on top).
  - `FrameStore` keeps the 48-entry `layers` LRU; each entry is one frame's composite, keyed by
    `compositeKey(parts, alpha)` = alpha + for each part in draw order: URL (object id, frame, `?v=` mask version),
    class RGB, outline RGB/edge or `-`.
  - Keys: `O` next object, `Shift+O` previous object (both wrap), `N` new object (class of the active object).
  - Defaults: Propagate scope "All objects"; Cleanup scope "Active object".
- **Legacy shims** (plan A) are removed in Task 7 only, after Tasks 1–6 made the web app send `object_id` on every
  `/prompt`, `/accept` and `/mask`, use `/objects/{id}/masks/{idx}` and read per-object `frame` events. Until Task 4,
  prompt/accept/brush bodies still omit `object_id` (legacy: they write to the clip's first object); that interim is
  deliberate and short.
- Do not hand out or manually QA a build between Tasks 3 and 4: the brush reads the active object's mask while
  `/mask` still writes the first object until Task 4.
- `web/src/pages/Clips.tsx` does not read `clip.objects`; it is unchanged.

## File map

| File | Change | Task |
|---|---|---|
| `web/src/api/types.ts` | `object_states`, `next_object_id`, `ObjectSelection`, `WriteTarget`, `AcceptResult`, `MaskResult` | 1 |
| `web/src/api/rest.ts` | `objectMaskUrl`, `propagate`/`clearRange` `objects`, `relabelObject`, `deleteObject` (1); `maskUrl` removed (2); `prompt`/`accept`/`setMask` take the object (4) | 1, 2, 4 |
| `web/src/lib/objects.ts` (new) | frame events, lanes, aggregate (1); colours, labels, `maskParts` (2); `lanesFor` (5) | 1, 2, 5 |
| `web/src/hooks/useClip.ts` | `patchFrames(FrameEvent[])`, `patchFrame` removed | 1 |
| `web/src/lib/frameCache.ts` | `MaskPart`, `compositeKey`, `composeLayers`, `FrameStore.composite`; `layer()` removed | 2 |
| `web/src/components/FrameCanvas.tsx` | `masks: MaskPart[]` replaces `maskUrl` | 2 |
| `web/src/lib/activeObject.ts` (new) | `Selection`, `initialSelection`, `reconcile`, `cycle`, `writeTarget` | 3 |
| `web/src/hooks/useObjects.ts` (new) | active row + hidden set | 3 |
| `web/src/components/ObjectsPanel.tsx` (new), `ObjectPanel.tsx` (deleted) | Objects panel | 3 |
| `web/src/hooks/useKeyboard.ts` | `O` / `Shift+O` / `N` | 3 |
| `web/src/hooks/usePrompting.ts`, `useBrush.ts`, `components/PromptPanel.tsx` | object ids, accept-all | 4 |
| `web/src/lib/lanes.ts` (new), `components/Timeline.tsx` | one lane per object | 5 |
| `web/src/components/PropagatePanel.tsx`, `CleanupPanel.tsx` | scope choices, lost/skipped notes | 6 |
| `web/src/pages/Label.tsx` | wiring | 1–6 |
| tests next to each file | see each task | 1–6 |
| `server/vsp/api/labeling.py`, `api/clips.py`, `labeling/propagate.py`, tests, `scripts/seed_demo.py`, README API section | shims removed | 7 |
| `README.md` | Label page, keys, manual QA | 8 |

---

### Task 1: Types, API client, per-object lanes and frame events

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/api/rest.ts`, `web/src/hooks/useClip.ts`, `web/src/pages/Label.tsx`
- Create: `web/src/lib/objects.ts`, `web/src/lib/objects.test.ts`
- Test: `web/src/api/rest.test.ts`, `web/src/hooks/useClip.test.tsx`, `web/src/hooks/usePrompting.test.tsx`
  (fixture only), `web/src/pages/Label.test.tsx` (fixtures and events)

**Interfaces:**
- Consumes (server, implemented): `GET /api/clips/{id}` and every mutation return
  `{clip, states, classes, object_states: {"<id>": FrameState[]}}`; propagation job events
  `{type: "frame", frame, objects: [{id, has_mask}], has_mask, done, total}` (the top-level `has_mask` is a legacy
  shim the page must not read; events may repeat earlier frames out of order).
- Produces:
  - `types.ts`: `ClipDetail.object_states: Record<string, FrameState[]>`; `Clip.next_object_id?: number`;
    `type ObjectSelection = "all" | number[]`;
    `type WriteTarget = { objectId: number } | { objectId: null; classId: number }`;
    `type AcceptResult = ClipDetail & { object_ids: number[] }`; `type MaskResult = ClipDetail & { object_id: number }`.
  - `rest.ts`: `objectMaskUrl(clipId: string, objectId: number, idx: number, version: string): string`;
    `api.propagate(clipId, fromFrame, direction, objects: ObjectSelection = "all")`;
    `api.clearRange(clipId, start, end, objects: ObjectSelection = "all")`;
    `api.relabelObject(clipId: string, objectId: number, classId: number): Promise<ClipDetail>`;
    `api.deleteObject(clipId: string, objectId: number): Promise<ClipDetail>`.
  - `lib/objects.ts`: `type ObjectFrame = { id: number; hasMask: boolean }`;
    `type FrameEvent = { frame: number; objects: ObjectFrame[]; done?: number }` (`done`: the job's progress count,
    when the event carried one);
    `parseFrameEvent(event: JobEvent): FrameEvent | null`; `isLabeled(s: FrameState | undefined): boolean`;
    `emptyLane(states: readonly FrameState[]): FrameState[]`;
    `laneOf(detail: ClipDetail, objectId: number | null): FrameState[]`;
    `labeledCount(lane: readonly FrameState[]): number`;
    `aggregateAt(lanes: readonly (readonly FrameState[])[], frame: number, fallback: FrameState): FrameState`;
    `patchObjectFrames(detail: ClipDetail, events: readonly FrameEvent[]): ClipDetail`.
  - `useClip`: `ClipHandle.patchFrames(events: FrameEvent[]): void`; `patchFrame` is removed.

- [ ] **Step 1: Write the failing tests**

Create `web/src/lib/objects.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { ClipDetail, FrameState, JobEvent } from "../api/types";
import { aggregateAt, emptyLane, labeledCount, laneOf, parseFrameEvent, patchObjectFrames } from "./objects";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);
const obj = (id: number, class_id = 0) => ({ id, class_id, keyframes: [], propagated_ranges: [] });

/** #1 cup: keyframe on 0; #2 bowl: masked on 1; frame 3 deleted */
const detail = (): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 4, created_at: "", deleted_frames: [3], objects: [obj(1), obj(2, 1)],
  },
  states: S("kmed"),
  classes: [{ id: 0, name: "cup", color: "#ff0000" }, { id: 1, name: "bowl", color: "#00ff00" }],
  object_states: { "1": S("keed"), "2": S("emed") },
});

describe("parseFrameEvent", () => {
  it("reads the per-object outcomes of a frame event", () => {
    const e: JobEvent = {
      seq: 3, type: "frame", frame: 7, objects: [{ id: 1, has_mask: true }, { id: 2, has_mask: false }], done: 1, total: 9,
    };
    expect(parseFrameEvent(e)).toEqual({ frame: 7, objects: [{ id: 1, hasMask: true }, { id: 2, hasMask: false }], done: 1 });
  });
  it("ignores other events and malformed entries", () => {
    expect(parseFrameEvent({ seq: 0, type: "progress", done: 0, total: 9 })).toBeNull();
    expect(parseFrameEvent({ seq: 1, type: "frame", frame: 2, has_mask: true })).toBeNull(); // no objects list
    expect(parseFrameEvent({ seq: 1, type: "frame", frame: 2, objects: [null, { has_mask: true }, { id: 4 }] })).toEqual({
      frame: 2, objects: [{ id: 4, hasMask: false }],
    });
  });
});

describe("lanes", () => {
  it("gives each object's lane, and an empty lane keeping the deleted frames for a new object or an unknown id", () => {
    const d = detail();
    expect(laneOf(d, 2)).toBe(d.object_states["2"]);
    expect(laneOf(d, null)).toEqual(S("eeed"));
    expect(laneOf(d, 9)).toEqual(S("eeed"));
    expect(emptyLane(S("kmdm"))).toEqual(S("eede"));
  });
  it("counts an object's labelled frames", () => expect(labeledCount(S("kmmed"))).toBe(3));
  it("aggregates one frame over the lanes: deleted, then keyframe, then masked", () => {
    expect(aggregateAt([S("m"), S("k")], 0, "empty")).toBe("keyframe");
    expect(aggregateAt([S("e"), S("m")], 0, "empty")).toBe("masked");
    expect(aggregateAt([S("e"), S("e")], 0, "masked")).toBe("empty");
    expect(aggregateAt([S("d"), S("d")], 0, "empty")).toBe("deleted");
    expect(aggregateAt([], 0, "deleted")).toBe("deleted");
  });
});

describe("patchObjectFrames", () => {
  it("updates each listed object's lane and recomputes the aggregate for that frame", () => {
    const d = detail();
    const next = patchObjectFrames(d, [{ frame: 2, objects: [{ id: 1, hasMask: true }] }]);
    expect(next.object_states["1"]).toEqual(S("kemd"));
    expect(next.object_states["2"]).toBe(d.object_states["2"]); // an untouched lane keeps its identity
    expect(next.states).toEqual(S("kmmd"));
    expect(d.states).toEqual(S("kmed")); // the input is not mutated
  });
  it("keeps a frame masked while an object that did not take part still has a mask there", () => {
    const withBoth = patchObjectFrames(detail(), [{ frame: 1, objects: [{ id: 1, hasMask: true }] }]);
    const next = patchObjectFrames(withBoth, [{ frame: 1, objects: [{ id: 1, hasMask: false }] }]);
    expect(next.object_states["1"][1]).toBe("empty");
    expect(next.states[1]).toBe("masked"); // #2 still has its mask on frame 1
    const cleared = patchObjectFrames(detail(), [{ frame: 1, objects: [{ id: 2, hasMask: false }] }]);
    expect(cleared.states[1]).toBe("empty");
  });
  it("applies events out of frame order, as a lagging group re-emits earlier frames", () => {
    const next = patchObjectFrames(detail(), [
      { frame: 2, objects: [{ id: 1, hasMask: true }] },
      { frame: 1, objects: [{ id: 1, hasMask: true }] },
      { frame: 2, objects: [{ id: 2, hasMask: true }] },
    ]);
    expect(next.object_states["1"]).toEqual(S("kmmd"));
    expect(next.object_states["2"]).toEqual(S("emmd"));
    expect(next.states).toEqual(S("kmmd"));
  });
  it("never changes keyframes or deleted frames, ignores unknown objects, and returns the same detail when nothing changed", () => {
    const d = detail();
    expect(
      patchObjectFrames(d, [
        { frame: 0, objects: [{ id: 1, hasMask: false }] },
        { frame: 3, objects: [{ id: 2, hasMask: true }] },
        { frame: 2, objects: [{ id: 9, hasMask: true }] },
        { frame: 1, objects: [{ id: 2, hasMask: true }] },
      ]),
    ).toBe(d);
  });
});
```

In `web/src/api/rest.test.ts`: change the import line to
`import { ApiError, UploadCancelled, api, detailText, maskUrl, objectMaskUrl, request, retryInteractive } from "./rest";`,
replace the test `"posts a propagate request with the server's field names"` with the first test below, and add the
other three inside `describe("api", ...)`:

```ts
  it("posts a propagate request with the server's field names", async () => {
    const fetchMock = mockFetch(jsonResponse({ job_id: "j1" }, 202), jsonResponse({ job_id: "j2" }, 202));
    expect(await api.propagate("c1", 7, "both")).toEqual({ job_id: "j1" });
    await api.propagate("c1", 7, "forward", [2]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clips/c1/propagate");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ from_frame: 7, direction: "both", objects: "all" });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ from_frame: 7, direction: "forward", objects: [2] });
  });

  it("clears a range of every object, or of the listed ones", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({}));
    await api.clearRange("c1", 2, 5);
    await api.clearRange("c1", 2, 5, [3]);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/clips/c1/clear-range");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ start: 2, end: 5, objects: "all" });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ start: 2, end: 5, objects: [3] });
  });

  it("relabels and deletes an object", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({}));
    await api.relabelObject("c1", 2, 5);
    await api.deleteObject("c1", 2);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/clips/c1/objects/2");
    expect(fetchMock.mock.calls[0][1].method).toBe("PATCH");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ class_id: 5 });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/clips/c1/objects/2");
    expect(fetchMock.mock.calls[1][1].method).toBe("DELETE");
  });

  it("versions per-object mask URLs", () => {
    expect(objectMaskUrl("c1", 2, 4, "2.1")).toBe("/api/clips/c1/objects/2/masks/4?v=2.1");
  });
```

In `web/src/hooks/useClip.test.tsx`: replace the `detail` fixture and the three patch tests (`"patches one frame from a
job event..."`, `"patchFrames applies a batch..."`, `"re-versions a frame whose mask was overwritten..."`) with:

```ts
const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 640, height: 480,
    frame_count: 3, created_at: "", deleted_frames: [], objects: [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states: ["keyframe", "empty", "masked"],
  classes: [],
  object_states: { "1": ["keyframe", "empty", "masked"] },
};
```

```ts
  it("patches an object's lane from a job event, recomputes the clip state and re-versions that frame only", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = [result.current.maskVersion(1), result.current.maskVersion(2)];
    act(() => result.current.patchFrames([{ frame: 1, objects: [{ id: 1, hasMask: true }] }]));
    expect(result.current.detail?.object_states["1"]).toEqual(["keyframe", "masked", "masked"]);
    expect(result.current.detail?.states).toEqual(["keyframe", "masked", "masked"]);
    expect(result.current.maskVersion(1)).not.toBe(before[0]);
    expect(result.current.maskVersion(2)).toBe(before[1]);
  });
  it("patchFrames applies a batch in one update and re-versions every touched frame", async () => {
    const lane: ClipDetail["states"] = ["keyframe", "empty", "empty", "empty"];
    vi.mocked(api.getClip).mockResolvedValue({ ...detail, states: lane, object_states: { "1": lane } });
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const before = [0, 1, 2, 3].map((f) => result.current.maskVersion(f));
    act(() =>
      result.current.patchFrames([
        { frame: 1, objects: [{ id: 1, hasMask: true }] },
        { frame: 2, objects: [{ id: 1, hasMask: true }] },
        { frame: 3, objects: [{ id: 1, hasMask: false }] },
      ]),
    );
    expect(result.current.detail?.states).toEqual(["keyframe", "masked", "masked", "empty"]);
    expect(result.current.maskVersion(0)).toBe(before[0]); // not in the batch: unchanged
    expect(result.current.maskVersion(1)).not.toBe(before[1]);
    expect(result.current.maskVersion(2)).not.toBe(before[2]);
    expect(result.current.maskVersion(3)).not.toBe(before[3]); // stayed "empty", but its mask was still touched
  });
  it("re-versions a frame whose mask was overwritten even if no state changed", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    const { result } = renderHook(() => useClip("c1"));
    await waitFor(() => expect(result.current.detail).not.toBeNull());
    const shown = result.current.detail;
    const before = result.current.maskVersion(2);
    act(() => result.current.patchFrames([{ frame: 2, objects: [{ id: 1, hasMask: true }] }]));
    expect(result.current.maskVersion(2)).not.toBe(before);
    expect(result.current.detail).toBe(shown); // nothing changed: the same detail object
  });
```

In `web/src/hooks/usePrompting.test.tsx` (fixture only, the hook is unchanged until Task 4): in `baseDetail` add after
`classes: [...]`:

```ts
  object_states: withObject ? { "1": Array(10).fill("empty") } : {},
```

and in `handle` remove `patchFrame: vi.fn(), ` (keep `patchFrames: vi.fn()`).

In `web/src/pages/Label.test.tsx`:
- add `FrameState` to the type import: `import type { ClipDetail, FrameState } from "../api/types";`
- replace the `detail` fixture with:

```ts
const states: FrameState[] = Array.from({ length: 40 }, (_, i) =>
  i === 0 ? "keyframe" : i < 20 ? "masked" : i === 30 ? "deleted" : "empty",
);
const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "bench.mp4" }, fps: 10, stride: 1, width: 640, height: 480,
    frame_count: 40, created_at: "", deleted_frames: [30], original_names: { "3": "img_0007.png" },
    objects: [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states,
  classes: [{ id: 0, name: "connector", color: "#e6194b" }],
  object_states: { "1": states },
};
/** a one-object detail with these states; its object's lane equals the clip's, as the server reports for one object */
const oneObject = (s: FrameState[]): ClipDetail => ({ ...detail, states: s, object_states: { "1": s } });
```

- replace `const accepted = { ...detail, states: detail.states.map((s, i) => (i === 25 ? "keyframe" : s)) } as ClipDetail;`
  with `const accepted = oneObject(detail.states.map((s, i) => (i === 25 ? "keyframe" : s)));`
- replace `const deleted = { ...detail, states: detail.states.map((s, i) => (i === 11 ? "deleted" : s)) } as ClipDetail;`
  with `const deleted = oneObject(detail.states.map((s, i) => (i === 11 ? "deleted" : s)));`
- replace the `emptyDetail` declaration (the three lines `const emptyDetail: ClipDetail = { ...detail, states: ... };`)
  with `const emptyDetail = oneObject(Array.from({ length: 40 }, (_, i) => (i === 0 ? "keyframe" : "empty")));`
- in every emitted `frame` event replace `has_mask: true` with `objects: [{ id: 1, has_mask: true }]` (four places:
  the "propagates from a keyframe" test, the foreign-job test, the re-attach test and the coalescing loop).
- add after the "propagates from a keyframe, follows the streamed frames and reloads when the job ends" test:

```ts
  it("follows the furthest frame of a run, not an earlier frame a lagging object re-emits", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j1" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(FakeSocket.last?.url).toMatch(/j1$/));
    // one batch: the leading group reaches frame 22, then the lagging group re-emits frame 21 at the same `done`
    FakeSocket.last!.emit({ seq: 0, type: "frame", frame: 22, objects: [{ id: 1, has_mask: true }], done: 3, total: 39 });
    FakeSocket.last!.emit({ seq: 1, type: "frame", frame: 21, objects: [{ id: 1, has_mask: true }], done: 3, total: 39 });
    expect(await screen.findByText("frame 22 / 39")).toBeTruthy();
    // a later batch holding only a re-emitted earlier frame does not pull the playhead back either
    FakeSocket.last!.emit({ seq: 2, type: "frame", frame: 20, objects: [{ id: 1, has_mask: true }], done: 3, total: 39 });
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    expect(screen.getByText("frame 22 / 39")).toBeTruthy();
    FakeSocket.last!.emit({ seq: 3, type: "frame", frame: 23, objects: [{ id: 1, has_mask: true }], done: 4, total: 39 });
    expect(await screen.findByText("frame 23 / 39")).toBeTruthy();
  });
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && node node_modules/.bin/vitest run src/lib/objects.test.ts src/api/rest.test.ts src/hooks/useClip.test.tsx src/pages/Label.test.tsx`
Expected: FAIL — `./objects` does not exist, `objectMaskUrl`/`relabelObject`/`deleteObject` missing, `patchFrames`
does not understand `objects`, and the Label page ignores the new frame events ("20 masked" not found).

- [ ] **Step 3: Implement**

`web/src/api/types.ts` — in `Clip`, after `objects: LabelObject[];` add:

```ts
  /** the next object id the server hands out (ids are never reused); absent or 0 in older clip files */
  next_object_id?: number;
```

Replace `export type ClipDetail = { clip: Clip; states: FrameState[]; classes: ClassDef[] };` with:

```ts
export type ClipDetail = {
  clip: Clip;
  /** the whole clip per frame (any object): deleted > keyframe > masked > empty */
  states: FrameState[];
  classes: ClassDef[];
  /** each object's own per-frame states, keyed by the object id as a string (JSON keys); a deleted frame is
   * "deleted" in every lane */
  object_states: Record<string, FrameState[]>;
};
/** every object ("all"), or the listed object ids (never empty) */
export type ObjectSelection = "all" | number[];
/** where a mask is written: an existing object, or a new object of `classId` created by that very write */
export type WriteTarget = { objectId: number } | { objectId: null; classId: number };
/** `/accept`: the ids written, one per accepted candidate when several became new objects */
export type AcceptResult = ClipDetail & { object_ids: number[] };
/** `/mask`: the id of the object written */
export type MaskResult = ClipDetail & { object_id: number };
```

`web/src/api/rest.ts`:
- add `ObjectSelection` to the type import list.
- after `export const thumbUrl = ...` add:

```ts
/** One object's stored mask on one frame; `version` busts the cache when the mask is rewritten in place. */
export const objectMaskUrl = (clipId: string, objectId: number, idx: number, version: string) =>
  `/api/clips/${clipId}/objects/${objectId}/masks/${idx}?v=${version}`;
```

- replace the `propagate` and `clearRange` entries of `api` and add the two object routes after `clearRange`:

```ts
  propagate: (clipId: string, fromFrame: number, direction: Direction, objects: ObjectSelection = "all") =>
    request<{ job_id: string }>(`/api/clips/${clipId}/propagate`, post({ from_frame: fromFrame, direction, objects })),
```

```ts
  clearRange: (clipId: string, start: number, end: number, objects: ObjectSelection = "all") =>
    request<ClipDetail>(`/api/clips/${clipId}/clear-range`, post({ start, end, objects })),
  /** relabel at once; allowed during a propagation (it touches no mask) */
  relabelObject: (clipId: string, objectId: number, classId: number) =>
    request<ClipDetail>(`/api/clips/${clipId}/objects/${objectId}`, patch({ class_id: classId })),
  /** 409 while a propagation or an export runs */
  deleteObject: (clipId: string, objectId: number) =>
    request<ClipDetail>(`/api/clips/${clipId}/objects/${objectId}`, { method: "DELETE" }),
```

Create `web/src/lib/objects.ts`:

```ts
import type { ClipDetail, FrameState, JobEvent } from "../api/types";
import { applyFrameEvent } from "./timeline";

/** One object's outcome on one frame of a propagation: the server's `{id, has_mask}`. */
export type ObjectFrame = { id: number; hasMask: boolean };
/** A propagation `frame` event, listing only the objects still active on that frame. Events are not in frame
 * order when objects' anchors diverge (the lagging group re-emits earlier frames), so each is applied on its own.
 * `done` is the job's progress (the furthest any object has got), which the page follows rather than `frame`. */
export type FrameEvent = { frame: number; objects: ObjectFrame[]; done?: number };

/** The `frame` event of a propagation job, or null for any other (or malformed) event. The server's top-level
 * `has_mask` is deliberately not read: a frame's clip-level state comes from all the lanes. */
export function parseFrameEvent(event: JobEvent): FrameEvent | null {
  if (event.type !== "frame" || typeof event.frame !== "number" || !Array.isArray(event.objects)) return null;
  const objects: ObjectFrame[] = [];
  for (const item of event.objects as unknown[]) {
    if (item === null || typeof item !== "object") continue;
    const { id, has_mask } = item as { id?: unknown; has_mask?: unknown };
    if (typeof id === "number") objects.push({ id, hasMask: has_mask === true });
  }
  return typeof event.done === "number" ? { frame: event.frame, objects, done: event.done } : { frame: event.frame, objects };
}

export const isLabeled = (s: FrameState | undefined): boolean => s === "masked" || s === "keyframe";

/** A lane with no masks: every frame empty except the clip's deleted ones. */
export function emptyLane(states: readonly FrameState[]): FrameState[] {
  return states.map((s) => (s === "deleted" ? "deleted" : "empty"));
}

/** One object's per-frame states. A "New object" row (`null`) or an id the detail does not know has no masks. */
export function laneOf(detail: ClipDetail, objectId: number | null): FrameState[] {
  const lane = objectId === null ? undefined : detail.object_states[String(objectId)];
  return lane ?? emptyLane(detail.states);
}

/** Frames an object has a mask on (propagated or keyframe). */
export function labeledCount(lane: readonly FrameState[]): number {
  let n = 0;
  for (const s of lane) if (isLabeled(s)) n++;
  return n;
}

/** The clip-level state of one frame from every object's lane, by the server's rule (deleted > keyframe > masked >
 * empty); `fallback` when the clip has no objects. */
export function aggregateAt(lanes: readonly (readonly FrameState[])[], frame: number, fallback: FrameState): FrameState {
  if (lanes.length === 0) return fallback;
  let best: FrameState = "empty";
  for (const lane of lanes) {
    const s = lane[frame];
    if (s === "deleted") return "deleted";
    if (s === "keyframe") best = "keyframe";
    else if (s === "masked" && best === "empty") best = "masked";
  }
  return best;
}

/**
 * Apply propagation frame events: each listed object's lane follows `applyFrameEvent` (keyframes and deleted frames
 * never change; an object not in the clip is ignored), then every frame whose lanes changed gets its clip-level
 * state recomputed from all lanes — an object that did not take part keeps a frame "masked". Returns `detail`
 * itself when nothing changed.
 */
export function patchObjectFrames(detail: ClipDetail, events: readonly FrameEvent[]): ClipDetail {
  let lanes = detail.object_states;
  const touched = new Set<number>();
  for (const e of events) {
    for (const o of e.objects) {
      const key = String(o.id);
      const lane = lanes[key];
      if (lane === undefined) continue;
      const next = applyFrameEvent(lane, e.frame, o.hasMask);
      if (next === lane) continue;
      if (lanes === detail.object_states) lanes = { ...lanes };
      lanes[key] = next;
      touched.add(e.frame);
    }
  }
  if (lanes === detail.object_states) return detail;
  const all = Object.values(lanes);
  let states = detail.states;
  for (const f of touched) {
    const s = aggregateAt(all, f, states[f]);
    if (s === states[f]) continue;
    if (states === detail.states) states = states.slice();
    states[f] = s;
  }
  return { ...detail, object_states: lanes, states };
}
```

`web/src/hooks/useClip.ts`:
- replace `import { applyFrameEvent } from "../lib/timeline";` with `import { patchObjectFrames, type FrameEvent } from "../lib/objects";`
- in `ClipHandle` replace the `patchFrame` and `patchFrames` members with:

```ts
  /** apply a batch of propagation frame events in one update: each listed object's lane follows its event, the
   * clip-level `states` are recomputed from all lanes, and every event's frame gets a new mask version (coalesces a
   * burst of streamed events into one render) */
  patchFrames: (events: FrameEvent[]) => void;
```

- replace the `patchFrames` and `patchFrame` callbacks with:

```ts
  const patchFrames = useCallback((events: FrameEvent[]) => {
    if (events.length === 0) return;
    const frames = versions.current.frames;
    for (const { frame } of events) frames.set(frame, (frames.get(frame) ?? 0) + 1);
    setDetail((d) => (d === null ? d : patchObjectFrames(d, events)));
    setTick((t) => t + 1);
  }, []);
```

- return `{ detail, error, refreshError, reload, apply, patchFrames, maskVersion }`.

`web/src/pages/Label.tsx`:
- add `import { parseFrameEvent, type FrameEvent } from "../lib/objects";` after the `../lib/mask` import.
- replace `const frameEventBuffer = useRef<{ frame: number; hasMask: boolean }[]>([]);` with

```ts
  const frameEventBuffer = useRef<FrameEvent[]>([]);
  /** the highest `done` the playhead has followed in this job: a lagging object's re-emitted earlier frames (same or
   * lower `done`) must not pull the view back */
  const followedDone = useRef(-1);
```

- in `flushFrameEvents` replace
  `if (followRef.current) setFrame(clampFrame(events[events.length - 1].frame, frameCountRef.current));` with:

```ts
    if (followRef.current) {
      // follow the event with the highest `done` (the first one to reach it), not the last event of the batch
      let lead: FrameEvent | null = null;
      for (const e of events) if ((e.done ?? 0) > (lead?.done ?? followedDone.current)) lead = e;
      if (lead !== null) {
        followedDone.current = lead.done ?? 0;
        setFrame(clampFrame(lead.frame, frameCountRef.current));
      }
    }
```

- in the cleanup of the effect keyed on `[jobId]` (the one that cancels the pending flush and empties
  `frameEventBuffer`), also reset `followedDone.current = -1;`.
- replace the body of `onJobEvent` with:

```ts
    (event: JobEvent) => {
      const parsed = parseFrameEvent(event);
      if (parsed === null) return;
      if (!ownJob.current) return; // a job this page did not start must not drive this clip
      frameEventBuffer.current.push(parsed);
      if (flushHandle.current === null) flushHandle.current = requestAnimationFrame(flushFrameEvents);
    },
```

- [ ] **Step 4: Run the tests**

Run: `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit`
Expected: all pass, no type errors. Then `uv run pytest server/tests -q -m "not gpu"` (untouched, green).

- [ ] **Step 5: Commit**

```bash
git add web/src/api/types.ts web/src/api/rest.ts web/src/api/rest.test.ts web/src/lib/objects.ts web/src/lib/objects.test.ts web/src/hooks/useClip.ts web/src/hooks/useClip.test.tsx web/src/hooks/usePrompting.test.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git commit -m "Label page: per-object lanes from clip detail and propagation frame events" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 2: One composite mask layer per frame; the canvas draws every visible object

**Files:**
- Modify: `web/src/lib/frameCache.ts`, `web/src/lib/objects.ts`, `web/src/components/FrameCanvas.tsx`,
  `web/src/api/rest.ts` (remove `maskUrl`), `web/src/pages/Label.tsx`
- Test: `web/src/lib/frameCache.test.ts` (rewrite), `web/src/lib/objects.test.ts`,
  `web/src/components/FrameCanvas.test.tsx` (rewrite), `web/src/api/rest.test.ts`, `web/src/pages/Label.test.tsx`

**Interfaces:**
- Consumes: `objectMaskUrl`, `laneOf`/`isLabeled` (Task 1); `tintToCanvas`, `Outline`, `outlineFor`, `OverlayStyle`
  (existing).
- Produces:
  - `frameCache.ts`: `type MaskPart = { url: string; rgb: Rgb; outline: Outline | null }`;
    `compositeKey(parts: readonly MaskPart[], alpha: number): string`;
    `composeLayers(layers: HTMLCanvasElement[]): HTMLCanvasElement`;
    `new FrameStore(load?, tint?, compose?: (layers: HTMLCanvasElement[]) => HTMLCanvasElement)`;
    `FrameStore.composite(parts: readonly MaskPart[], alpha?: number /* MASK_ALPHA */): Promise<HTMLCanvasElement | null>`.
    `FrameStore.layer()` is removed; `image()` and `preview()` are unchanged.
  - `lib/objects.ts`: `FALLBACK_COLOR = "#22c55e"`; `sortedObjects(detail): LabelObject[]` (id order);
    `className(detail, classId: number | null): string | null`;
    `classColor(detail, classId: number | null): string`; `objectLabel(detail, obj: LabelObject): string` ("#2 cup");
    `maskParts(detail, o: { clipId: string; frame: number; version: string; hidden: ReadonlySet<number>; activeId: number | null; excludeActive: boolean; style: OverlayStyle }): MaskPart[]`.
  - `FrameCanvas` props: `masks: MaskPart[]` replaces `maskUrl`; `color`/`outline` now only tint candidate previews.
  - `rest.ts`: `maskUrl` removed.

- [ ] **Step 1: Write the failing tests**

Replace `web/src/lib/frameCache.test.ts` with:

```ts
import { describe, expect, it, vi } from "vitest";
import { FrameStore, LruCache, compositeKey, type MaskPart } from "./frameCache";
import type { Outline } from "./mask";

const OUT: Outline = { rgb: [255, 255, 255], edge: 5 };
type Load = (url: string) => Promise<HTMLImageElement | null>;

const img = (name: string) => ({ name }) as unknown as HTMLImageElement;
const canvas = (name: string) => ({ name }) as unknown as HTMLCanvasElement;
const part = (id: number, over: Partial<MaskPart> = {}): MaskPart => ({
  url: `/api/clips/c1/objects/${id}/masks/4?v=0.0`, rgb: [id, 0, 0], outline: null, ...over,
});
/** a store whose tint names each layer after its image, and whose compose records what it was given */
const setup = (load: Load = vi.fn<Load>(async (url) => img(url))) => {
  const tint = vi.fn((i: HTMLImageElement) => canvas(`tinted ${(i as unknown as { name: string }).name}`));
  const compose = vi.fn((layers: HTMLCanvasElement[]) => canvas(`composite of ${layers.length}`));
  return { store: new FrameStore(load, tint, compose), load, tint, compose };
};

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

describe("FrameStore images", () => {
  it("loads each URL once, even for concurrent requests", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const store = new FrameStore(load, vi.fn());
    const [a, b] = await Promise.all([store.image("/f/1"), store.image("/f/1")]);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(1);
  });
  it("does not cache a failed load, so the next request retries", async () => {
    const load = vi.fn<Load>().mockResolvedValueOnce(null).mockResolvedValueOnce(img("ok"));
    const store = new FrameStore(load, vi.fn());
    expect(await store.image("/f/1")).toBeNull();
    expect(await store.image("/f/1")).not.toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
  });
});

describe("FrameStore composite layers", () => {
  it("composites a frame's visible objects into one cached layer", async () => {
    const { store, load, tint, compose } = setup();
    const parts = [part(1), part(2, { outline: OUT })];
    const a = await store.composite(parts, 0.45);
    const b = await store.composite(parts.map((p) => ({ ...p })), 0.45);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(2);
    expect(tint).toHaveBeenCalledTimes(2);
    expect(compose).toHaveBeenCalledTimes(1);
    expect(compose.mock.calls[0][0]).toEqual([canvas(`tinted ${parts[0].url}`), canvas(`tinted ${parts[1].url}`)]);
  });
  it("outlines the active object only; the others are fill only", async () => {
    const { store, tint } = setup();
    await store.composite([part(1), part(2, { outline: OUT })], 0.45);
    expect(tint).toHaveBeenNthCalledWith(1, img(part(1).url), [1, 0, 0], 0.45, { rgb: [1, 0, 0], edge: 0 });
    expect(tint).toHaveBeenNthCalledWith(2, img(part(2).url), [2, 0, 0], 0.45, OUT);
  });
  it("keys the layer by mask version, object order, active object, class colours and outline style", async () => {
    const { store, compose } = setup();
    const base = [part(1), part(2, { outline: OUT })];
    const variants: MaskPart[][] = [
      base,
      [part(1), part(2, { outline: OUT, url: "/api/clips/c1/objects/2/masks/4?v=0.1" })], // a new mask version
      [part(2, { outline: OUT }), part(1)], // another order
      [part(1, { outline: OUT }), part(2)], // another active object
      [part(1, { rgb: [9, 9, 9] }), part(2, { outline: OUT })], // a relabelled object
      [part(1), part(2, { outline: { ...OUT, edge: 2 } })], // another outline thickness
      [part(1), part(2, { outline: { ...OUT, rgb: [0, 0, 0] } })], // another outline colour
    ];
    for (const v of variants) await store.composite(v, 0.45);
    await store.composite(base.map((p) => ({ ...p })), 0.45); // equal content: a cache hit
    expect(compose).toHaveBeenCalledTimes(variants.length);
    expect(new Set(variants.map((v) => compositeKey(v, 0.45))).size).toBe(variants.length);
    expect(compositeKey(base, 0.45)).not.toBe(compositeKey(base, 0.55));
  });
  it("holds 48 frames however many objects each has", async () => {
    const { store, compose } = setup();
    const frame = (i: number) => [1, 2, 3].map((id) => part(id, { url: `/api/clips/c1/objects/${id}/masks/${i}?v=0.0` }));
    for (let i = 0; i < 49; i++) await store.composite(frame(i), 0.45);
    expect(compose).toHaveBeenCalledTimes(49);
    await store.composite(frame(48), 0.45);
    await store.composite(frame(1), 0.45);
    expect(compose).toHaveBeenCalledTimes(49); // frames 1..48 are all still cached
    await store.composite(frame(0), 0.45);
    expect(compose).toHaveBeenCalledTimes(50); // frame 0 was the one evicted
  });
  it("skips a mask that fails to load and does not keep the partial layer", async () => {
    const load = vi.fn<Load>(async (url) => img(url));
    load.mockImplementationOnce(async (url) => img(url)).mockImplementationOnce(async () => null);
    const { store, compose } = setup(load);
    const parts = [part(1), part(2)];
    await store.composite(parts, 0.45);
    expect(compose).toHaveBeenLastCalledWith([canvas(`tinted ${parts[0].url}`)]);
    await store.composite(parts, 0.45);
    expect(load).toHaveBeenCalledTimes(4); // not cached: both masks are fetched again
    expect(compose).toHaveBeenLastCalledWith([canvas(`tinted ${parts[0].url}`), canvas(`tinted ${parts[1].url}`)]);
  });
  it("returns null without loading anything for no parts, and null when no mask loads", async () => {
    const { store, load, compose } = setup(vi.fn<Load>(async () => null));
    expect(await store.composite([], 0.45)).toBeNull();
    expect(load).not.toHaveBeenCalled();
    expect(await store.composite([part(1)], 0.45)).toBeNull();
    expect(compose).not.toHaveBeenCalled();
  });
  it("previews never evict composite layers", async () => {
    const { store, compose, tint } = setup();
    await store.composite([part(1)], 0.45);
    for (let i = 0; i < 20; i++) await store.preview(`data:image/png;base64,p${i}`, [0, 255, 0], 0.55, OUT);
    await store.composite([part(1)], 0.45);
    expect(compose).toHaveBeenCalledTimes(1);
    expect(tint).toHaveBeenCalledTimes(21);
  });
  it("tints a preview once per colour, alpha and outline", async () => {
    const { store, tint } = setup();
    await store.preview("data:p", [255, 0, 0], 0.55, OUT);
    await store.preview("data:p", [255, 0, 0], 0.55, OUT);
    await store.preview("data:p", [255, 0, 0], 0.55, { ...OUT, edge: 0 });
    await store.preview("data:p", [0, 255, 0], 0.55, OUT);
    expect(tint).toHaveBeenCalledTimes(3);
  });
});
```

Append to `web/src/lib/objects.test.ts` (and extend its import line to
`import { aggregateAt, classColor, className, emptyLane, FALLBACK_COLOR, labeledCount, laneOf, maskParts, objectLabel, parseFrameEvent, patchObjectFrames, sortedObjects } from "./objects";`):

```ts
describe("names and colours", () => {
  it("names objects by id and class, in id order", () => {
    const d = detail();
    d.clip.objects.reverse();
    expect(sortedObjects(d).map((o) => o.id)).toEqual([1, 2]);
    expect(objectLabel(d, d.clip.objects[0])).toBe("#2 bowl");
    expect(objectLabel(d, { ...obj(3), class_id: 7 })).toBe("#3 class 7");
    expect(className(d, 7)).toBeNull();
    expect(classColor(d, 1)).toBe("#00ff00");
    expect(classColor(d, null)).toBe(FALLBACK_COLOR);
  });
});

describe("maskParts", () => {
  const style = { color: null, thickness: 5 };
  /** frame 1: both objects masked */
  const both = () => patchObjectFrames(detail(), [{ frame: 1, objects: [{ id: 1, hasMask: true }] }]);
  type Opts = Parameters<typeof maskParts>[1];
  const ids = (over: Partial<Opts>) =>
    maskParts(both(), {
      clipId: "c1", frame: 1, version: "0.0", hidden: new Set(), activeId: 1, excludeActive: false, style, ...over,
    }).map((p) => p.url.split("/")[5]);

  it("lists the objects with a mask on the frame, the active one last and outlined, the others fill only", () => {
    const parts = maskParts(both(), {
      clipId: "c1", frame: 1, version: "2.3", hidden: new Set(), activeId: 1, excludeActive: false, style,
    });
    expect(parts).toEqual([
      { url: "/api/clips/c1/objects/2/masks/1?v=2.3", rgb: [0, 255, 0], outline: null },
      { url: "/api/clips/c1/objects/1/masks/1?v=2.3", rgb: [255, 0, 0], outline: { rgb: [255, 0, 0], edge: 5 } },
    ]);
  });
  it("leaves out hidden objects, objects without a mask there, and the active object while it is edited", () => {
    expect(ids({ hidden: new Set([2]) })).toEqual(["1"]);
    expect(ids({ excludeActive: true })).toEqual(["2"]);
    expect(ids({ frame: 2 })).toEqual([]); // neither has a mask
    expect(ids({ frame: 0 })).toEqual(["1"]); // a keyframe counts
    expect(ids({ frame: 3 })).toEqual([]); // deleted
    expect(ids({ activeId: null })).toEqual(["1", "2"]); // a New object row: id order, nothing outlined
  });
  it("outlines the active object in the Mask outline colour when one is set", () => {
    const [, active] = maskParts(both(), {
      clipId: "c1", frame: 1, version: "0.0", hidden: new Set(), activeId: 1, excludeActive: false,
      style: { color: "#0000ff", thickness: 2 },
    });
    expect(active.outline).toEqual({ rgb: [0, 0, 255], edge: 2 });
  });
});
```

Replace `web/src/components/FrameCanvas.test.tsx` with:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FrameStore, MaskPart } from "../lib/frameCache";
import { fitView, toImage } from "../lib/geometry";
import { emptyDraft } from "../lib/promptState";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { FrameCanvas } from "./FrameCanvas";

// jsdom does no layout, so wrap.clientWidth/clientHeight are always 0: fitView(width, height, 0, 0) is exactly the
// view FrameCanvas computes on mount, with no extra stubbing needed.
const store = { image: async () => null, composite: async () => null, preview: async () => null } as unknown as FrameStore;

type Props = ComponentProps<typeof FrameCanvas>;
const baseProps = (over: Partial<Props> = {}): Props => ({
  store, width: 100, height: 100, frameUrl: "f", masks: [], color: "#22c55e", outline: { rgb: [34, 197, 94], edge: 5 },
  tool: null, draft: emptyDraft(), candidates: [], onPoint: () => {}, onBox: () => {}, onPickCandidate: () => {},
  brush: null, ...over,
});
const brushProp = () => ({ size: 24, layer: null, baseVersion: 0, onStart: vi.fn(), onExtend: vi.fn(), onEnd: vi.fn() });

beforeEach(() => {
  stubCanvasAndObservers();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("FrameCanvas brush pointer handling", () => {
  it("floors the pointer to an image pixel index when starting a brush stroke (F3)", () => {
    const brush = brushProp();
    render(<FrameCanvas {...baseProps({ brush })} />);
    const canvas = screen.getByLabelText("Frame");
    const [sx, sy] = [37, 52];
    fireEvent.mouseDown(canvas, { clientX: sx, clientY: sy, button: 0 });
    const [ix, iy] = toImage(fitView(100, 100, 0, 0), sx, sy);
    expect(brush.onStart).toHaveBeenCalledWith("paint", Math.floor(ix), Math.floor(iy));
  });

  it("ends a painting drag when a mousemove arrives with no button held (F6)", () => {
    const brush = brushProp();
    render(<FrameCanvas {...baseProps({ brush })} />);
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 10, clientY: 10, button: 0 });
    expect(brush.onStart).toHaveBeenCalledTimes(1);
    fireEvent.mouseMove(canvas, { clientX: 20, clientY: 20, buttons: 0 });
    expect(brush.onEnd).toHaveBeenCalledTimes(1);
    expect(brush.onExtend).not.toHaveBeenCalled();
  });
});

describe("FrameCanvas stored masks", () => {
  it("asks the store for one composite of the given masks, and again only when they change", () => {
    const composite = vi.fn(async () => null);
    const s = { image: async () => null, composite, preview: async () => null } as unknown as FrameStore;
    const parts: MaskPart[] = [
      { url: "/api/clips/c1/objects/1/masks/0?v=0.0", rgb: [1, 2, 3], outline: null },
      { url: "/api/clips/c1/objects/2/masks/0?v=0.0", rgb: [4, 5, 6], outline: { rgb: [9, 9, 9], edge: 3 } },
    ];
    const { rerender } = render(<FrameCanvas {...baseProps({ store: s, masks: parts })} />);
    expect(composite).toHaveBeenCalledWith(parts);
    rerender(<FrameCanvas {...baseProps({ store: s, masks: parts.map((p) => ({ ...p })) })} />); // equal content
    expect(composite).toHaveBeenCalledTimes(1);
    rerender(<FrameCanvas {...baseProps({ store: s, masks: [parts[0]] })} />);
    expect(composite).toHaveBeenLastCalledWith([parts[0]]);
  });
});
```

In `web/src/api/rest.test.ts` delete the test `"versions mask URLs"` and remove `maskUrl, ` from the import line.

In `web/src/pages/Label.test.tsx` add `import { FrameStore } from "../lib/frameCache";`, add after `oneObject`:

```ts
/** two objects: #1 connector (as in `detail`) and #2 cup, keyframed on 5 and masked on 6..9 */
const cupLane: FrameState[] = Array.from({ length: 40 }, (_, i) =>
  i === 5 ? "keyframe" : i > 5 && i < 10 ? "masked" : i === 30 ? "deleted" : "empty",
);
const two: ClipDetail = {
  ...detail,
  clip: { ...detail.clip, objects: [...detail.clip.objects, { id: 2, class_id: 1, keyframes: [], propagated_ranges: [] }] },
  states: states.map((s, i) => (i === 5 ? "keyframe" : s)),
  classes: [...detail.classes, { id: 1, name: "cup", color: "#3cb44b" }],
  object_states: { "1": states, "2": cupLane },
};
```

and add to `describe("Label page (viewer)", ...)`:

```ts
  it("draws every visible object's stored mask on the frame, the active object last and outlined", async () => {
    const composite = vi.spyOn(FrameStore.prototype, "composite").mockResolvedValue(null);
    try {
      vi.mocked(api.getClip).mockResolvedValue(two);
      renderLabel();
      await screen.findByText("bench.mp4");
      for (let i = 0; i < 6; i++) fireEvent.keyDown(window, { key: "ArrowRight" });
      expect(screen.getByText("frame 6 / 39")).toBeTruthy();
      const onFrame6 = composite.mock.calls.map(([parts]) => parts).filter((parts) => parts.some((p) => p.url.includes("/masks/6?")));
      const last = onFrame6[onFrame6.length - 1];
      expect(last.map((p) => p.url.split("?")[0])).toEqual(["/api/clips/c1/objects/2/masks/6", "/api/clips/c1/objects/1/masks/6"]);
      expect(last[0].outline).toBeNull();
      expect(last[1].outline).toEqual({ rgb: [230, 25, 75], edge: 5 });
    } finally {
      composite.mockRestore();
    }
  });
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && node node_modules/.bin/vitest run src/lib/frameCache.test.ts src/lib/objects.test.ts src/components/FrameCanvas.test.tsx src/pages/Label.test.tsx`
Expected: FAIL — `composite`, `compositeKey`, `maskParts`, `objectLabel` ... do not exist; FrameCanvas has no `masks`.

- [ ] **Step 3: Implement**

`web/src/lib/frameCache.ts` — keep `LruCache`, `loadImage`, `tintToCanvas`, `MASK_ALPHA`, `PREVIEW_ALPHA` and the
size constants as they are (update the `MAX_LAYERS` comment to "one composite per frame, however many objects"); add
after `PREVIEW_ALPHA`:

```ts
/** One object's stored mask on one frame, as drawn: filled in `rgb`, plus the configurable outline for the active
 * object only (`null` = fill only). */
export type MaskPart = { url: string; rgb: Rgb; outline: Outline | null };

/** Everything a composite layer depends on: each part's URL (object id, frame, mask version) in draw order, its class
 * colour and its outline, and the fill alpha. Equal keys mean an identical layer. */
export function compositeKey(parts: readonly MaskPart[], alpha: number): string {
  const one = (p: MaskPart) =>
    `${p.url}|${p.rgb.join(",")}|${p.outline === null ? "-" : `${p.outline.rgb.join(",")}/${p.outline.edge}`}`;
  return `${alpha};${parts.map(one).join(";")}`;
}

/** Draw tinted layers into one canvas, in order (later on top); a single layer is returned as is. The inputs are
 * released: only the composite is cached. */
export function composeLayers(layers: HTMLCanvasElement[]): HTMLCanvasElement {
  if (layers.length === 1) return layers[0];
  const out = document.createElement("canvas");
  out.width = layers[0].width;
  out.height = layers[0].height;
  const ctx = out.getContext("2d")!;
  for (const layer of layers) {
    ctx.drawImage(layer, 0, 0, out.width, out.height);
    layer.width = 0;
  }
  return out;
}
```

and replace the `FrameStore` class with:

```ts
export class FrameStore {
  private readonly images = new LruCache<Promise<HTMLImageElement | null>>(MAX_IMAGES);
  private readonly layers = new LruCache<Promise<HTMLCanvasElement | null>>(MAX_LAYERS);
  private readonly previews = new LruCache<Promise<HTMLCanvasElement | null>>(MAX_PREVIEWS);

  constructor(
    private readonly load: (url: string) => Promise<HTMLImageElement | null> = loadImage,
    private readonly tint: (img: HTMLImageElement, rgb: Rgb, alpha: number, outline: Outline) => HTMLCanvasElement = tintToCanvas,
    private readonly compose: (layers: HTMLCanvasElement[]) => HTMLCanvasElement = composeLayers,
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

  /**
   * A frame's visible objects, tinted and drawn in order into ONE cached layer, so the 48-entry cache holds 48 frames
   * whatever the object count. A mask that fails to load is left out and the partial layer is not cached (the next
   * request fetches again); with no parts, or nothing loaded, the result is null.
   */
  composite(parts: readonly MaskPart[], alpha: number = MASK_ALPHA): Promise<HTMLCanvasElement | null> {
    if (parts.length === 0) return Promise.resolve(null);
    const key = compositeKey(parts, alpha);
    let pending = this.layers.get(key);
    if (pending === undefined) {
      pending = Promise.all(parts.map((p) => this.load(p.url))).then((imgs) => {
        const tinted: HTMLCanvasElement[] = [];
        imgs.forEach((img, i) => {
          const p = parts[i];
          if (img !== null) tinted.push(this.tint(img, p.rgb, alpha, p.outline ?? { rgb: p.rgb, edge: 0 }));
        });
        if (tinted.length < parts.length) this.layers.delete(key);
        return tinted.length === 0 ? null : this.compose(tinted);
      });
      this.layers.set(key, pending);
    }
    return pending;
  }

  /** A text-prompt candidate, cached separately so previews cannot evict stored mask layers. */
  preview(dataUrl: string, rgb: Rgb, alpha: number, outline: Outline): Promise<HTMLCanvasElement | null> {
    const key = `${dataUrl}|${rgb.join(",")}|${alpha}|${outline.rgb.join(",")}|${outline.edge}`;
    let pending = this.previews.get(key);
    if (pending === undefined) {
      pending = this.load(dataUrl).then((img) => {
        if (img === null) {
          this.previews.delete(key);
          return null;
        }
        return this.tint(img, rgb, alpha, outline);
      });
      this.previews.set(key, pending);
    }
    return pending;
  }
}
```

`web/src/lib/objects.ts` — replace the import block with:

```ts
import { objectMaskUrl } from "../api/rest";
import type { ClipDetail, FrameState, JobEvent, LabelObject } from "../api/types";
import type { MaskPart } from "./frameCache";
import { hexToRgb } from "./mask";
import { outlineFor, type OverlayStyle } from "./overlayStyle";
import { applyFrameEvent } from "./timeline";
```

and append:

```ts
/** Colour of an object whose class is unknown (and of the page before a clip loads). */
export const FALLBACK_COLOR = "#22c55e";

export function sortedObjects(detail: ClipDetail): LabelObject[] {
  return [...detail.clip.objects].sort((a, b) => a.id - b.id);
}

export function className(detail: ClipDetail, classId: number | null): string | null {
  return detail.classes.find((c) => c.id === classId)?.name ?? null;
}

export function classColor(detail: ClipDetail, classId: number | null): string {
  return detail.classes.find((c) => c.id === classId)?.color ?? FALLBACK_COLOR;
}

/** "#2 cup": how every panel, lane and message names an object. */
export function objectLabel(detail: ClipDetail, obj: LabelObject): string {
  return `#${obj.id} ${className(detail, obj.class_id) ?? `class ${obj.class_id}`}`;
}

/**
 * The stored masks to draw on one frame (spec §8): every visible object with a mask there, in its class colour; the
 * inactive ones in id order, fill only, and the active one last with the Mask outline. `excludeActive` leaves the
 * active object out while its candidate preview or brush layer is drawn in its place.
 */
export function maskParts(
  detail: ClipDetail,
  o: {
    clipId: string; frame: number; version: string; hidden: ReadonlySet<number>; activeId: number | null;
    excludeActive: boolean; style: OverlayStyle;
  },
): MaskPart[] {
  const parts: MaskPart[] = [];
  let active: MaskPart | null = null;
  for (const obj of sortedObjects(detail)) {
    if (o.hidden.has(obj.id) || !isLabeled(laneOf(detail, obj.id)[o.frame])) continue;
    const isActive = obj.id === o.activeId;
    if (isActive && o.excludeActive) continue;
    const rgb = hexToRgb(classColor(detail, obj.class_id));
    const part: MaskPart = {
      url: objectMaskUrl(o.clipId, obj.id, o.frame, o.version),
      rgb,
      outline: isActive ? outlineFor(o.style, rgb) : null,
    };
    if (isActive) active = part;
    else parts.push(part);
  }
  if (active !== null) parts.push(active);
  return parts;
}
```

`web/src/api/rest.ts` — delete the `maskUrl` export.

`web/src/components/FrameCanvas.tsx`:
- import line: `import { MASK_ALPHA, PREVIEW_ALPHA, compositeKey, type FrameStore, type MaskPart } from "../lib/frameCache";`
- in `Props` replace the `maskUrl` and `color`/`outline` docs with:

```ts
  /** the stored masks to draw on this frame, one part per visible object, active object last (`maskParts`); empty
   * when the overlay is off. They are composited into one cached layer by `store.composite`. */
  masks: MaskPart[];
  /** class colour of the active (or new) object, hex: candidate previews are tinted in it */
  color: string;
  /** how preview boundaries are drawn; keep its identity stable, it keys the preview cache lookups */
  outline: Outline;
```

- destructure `masks` instead of `maskUrl` in the `const { store, ... } = props;` line.
- after `const rgb = useMemo(() => hexToRgb(color), [color]);` add:

```ts
  // `masks` is rebuilt on every page render; only its content matters
  const masksKey = compositeKey(masks, MASK_ALPHA);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- masksKey covers every field of `masks`
  const stableMasks = useMemo(() => masks, [masksKey]);
```

- in `draw`, replace the three-way `if (p.candidates.length > 0) {...} else if (p.brush !== null) {...} else if
  (assets.current.mask) {...}` with:

```ts
      // The stored masks come first: every visible object on this frame, composited into one layer. While the
      // active object is being edited (candidates or brush strokes) the page leaves it out of `masks`, and its
      // preview or brush layer is drawn on top instead.
      if (assets.current.mask) ctx.drawImage(assets.current.mask, 0, 0, p.width, p.height);
      if (p.candidates.length > 0) {
        const state = layers.current;
        const current = state.owner === p.candidates;
        // Not current: new previews are still decoding. Only bridge the gap when it's the same single
        // candidate being refined (an extra click on the same object) — anything else would flash a mismatched
        // preview against the wrong candidate list.
        const bridgeOld = !current && state.owner !== null && state.owner.length === 1 && p.candidates.length === 1;
        if (current || bridgeOld) {
          state.items.forEach((layer, i) => {
            if (layer === null) return;
            ctx.globalAlpha = i === p.draft.candidate ? 1 : 0.45;
            ctx.drawImage(layer, 0, 0, p.width, p.height);
          });
          ctx.globalAlpha = 1;
        }
      } else if (p.brush !== null && p.brush.layer) {
        ctx.globalAlpha = MASK_ALPHA;
        ctx.drawImage(p.brush.layer, 0, 0, p.width, p.height);
        ctx.globalAlpha = 1;
      }
```

- replace the "Swap the frame and its mask in together" effect with:

```ts
  // Swap the frame and its masks in together so an overlay never shows on the wrong frame.
  useEffect(() => {
    let live = true;
    void Promise.all([store.image(frameUrl), store.composite(stableMasks)]).then(([frame, layer]) => {
      if (!live) return;
      assets.current = { frame, mask: layer };
      draw();
    });
    return () => {
      live = false;
    };
  }, [store, frameUrl, stableMasks, draw]);
```

`web/src/pages/Label.tsx`:
- rest import: replace `maskUrl` with `objectMaskUrl`.
- replace `import { FrameStore, MASK_ALPHA } from "../lib/frameCache";` with `import { FrameStore } from "../lib/frameCache";`
- replace the objects import with `import { maskParts, parseFrameEvent, type FrameEvent } from "../lib/objects";`
- after `const message = ...` add:

```ts
/** Interim, until Task 3 gives the page an Objects panel: no object is hidden. */
const NO_HIDDEN: ReadonlySet<number> = new Set();
```

- after `const object = detail?.clip.objects[0] ?? null;` add
  `const activeId = object?.id ?? null; // interim: Task 3 makes this the Objects panel's active object`
- in the `useBrush({...})` call replace
  `maskUrl: hasMask(frame) ? maskUrl(clipId, frame, maskVersion(frame)) : null,` with
  `maskUrl: activeId !== null && hasMask(frame) ? objectMaskUrl(clipId, activeId, frame, maskVersion(frame)) : null,`
- after the `useBrush({...})` call add (one definition, used by both the canvas and the prefetch, so the prefetched
  composites are the ones the canvas asks for):

```ts
  // while the active object is being edited, its preview or brush layer is drawn in its place
  const editing = (mode === "brush" && brushReason === null) || prompting.candidates.length > 0;
```

- replace the prefetch effect with:

```ts
  useEffect(() => {
    if (detail === null || running) return; // during a job the view follows the stream; nothing to look ahead for
    const ahead = playing ? 30 : 4;
    for (let i = frame + 1; i <= Math.min(frame + ahead, frameCount - 1); i++) {
      void store.image(frameUrl(clipId, i));
      if (!overlay) continue;
      // the parts FrameCanvas will ask for on that frame, so playback hits the composite cache
      const parts = maskParts(detail, {
        clipId, frame: i, version: maskVersion(i), hidden: NO_HIDDEN, activeId, excludeActive: editing, style: overlayStyle,
      });
      if (parts.length > 0) void store.composite(parts);
    }
  }, [store, clipId, frame, playing, overlay, detail, overlayStyle, activeId, editing, frameCount, maskVersion, running]);
```

- in `<FrameCanvas ...>` replace the `maskUrl={...}` prop with:

```tsx
            masks={
              overlay
                ? maskParts(detail, {
                    clipId, frame, version: maskVersion(frame), hidden: NO_HIDDEN, activeId, excludeActive: editing,
                    style: overlayStyle,
                  })
                : []
            }
```

- [ ] **Step 4: Run the tests**

Run: `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit`
Expected: all pass. `grep -rn "maskUrl(\|\.layer(" web/src` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/frameCache.ts web/src/lib/frameCache.test.ts web/src/lib/objects.ts web/src/lib/objects.test.ts web/src/components/FrameCanvas.tsx web/src/components/FrameCanvas.test.tsx web/src/api/rest.ts web/src/api/rest.test.ts web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git commit -m "Label page: one composite mask layer per frame for every visible object" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 3: Active object, Objects panel and object keys

**Files:**
- Create: `web/src/lib/activeObject.ts`, `web/src/lib/activeObject.test.ts`, `web/src/hooks/useObjects.ts`,
  `web/src/hooks/useObjects.test.tsx`, `web/src/components/ObjectsPanel.tsx`, `web/src/components/ObjectsPanel.test.tsx`
- Delete: `web/src/components/ObjectPanel.tsx` (`git rm`)
- Modify: `web/src/hooks/useKeyboard.ts`, `web/src/pages/Label.tsx`
- Test: `web/src/hooks/useKeyboard.test.ts`, `web/src/pages/Label.test.tsx`

**Interfaces:**
- Consumes: `laneOf`, `labeledCount`, `isLabeled`, `sortedObjects`, `classColor`, `className`, `objectLabel`,
  `FALLBACK_COLOR` (Tasks 1–2); `api.relabelObject`, `api.deleteObject` (Task 1).
- Produces:
  - `lib/activeObject.ts`: `type Selection = { kind: "object"; id: number } | { kind: "new"; classId: number | null }`;
    `sortedIds(detail): number[]`; `initialSelection(detail): Selection`;
    `reconcile(sel: Selection, detail: ClipDetail): Selection`;
    `cycle(sel: Selection, detail: ClipDetail, dir: 1 | -1): Selection`;
    `writeTarget(selection: Selection | null): WriteTarget | null`.
  - `hooks/useObjects.ts`: `useObjects(clipId: string, detail: ClipDetail | null): ObjectsHandle` with
    `ObjectsHandle = { selection: Selection | null; activeId: number | null; active: LabelObject | null; classId: number | null; hidden: ReadonlySet<number>; select(id: number): void; step(dir: 1 | -1): void; startNew(): void; setNewClass(classId: number): void; toggleHidden(id: number): void }`.
  - `components/ObjectsPanel.tsx`: `ObjectsPanel(props: { detail; selection: Selection; hidden: ReadonlySet<number>; lockReason: string | null; deleteBlockedReason: string | null; error: string | null; onSelect(id); onToggleHidden(id); onClass(classId); onNew(); onDelete(); onAddClass(name): Promise<void> })`.
  - `useKeyboard`: `KeyAction` gains `{ type: "object"; dir: 1 | -1 }` (`o` / `Shift+O`) and `{ type: "newObject" }` (`n`).
  - `Label.tsx`: `activeLane` (active object's lane) drives `hasMask`, `K`, `G`.

- [ ] **Step 1: Write the failing tests**

Create `web/src/lib/activeObject.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { ClipDetail } from "../api/types";
import { cycle, initialSelection, reconcile, writeTarget } from "./activeObject";

const CLASSES = [{ id: 0, name: "cup", color: "#e6194b" }, { id: 1, name: "bowl", color: "#3cb44b" }];
const make = (ids: number[], classes = CLASSES): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 1, created_at: "", deleted_frames: [],
    objects: ids.map((id) => ({ id, class_id: 0, keyframes: [], propagated_ranges: [] })),
  },
  states: ["empty"],
  classes,
  object_states: Object.fromEntries(ids.map((id) => [String(id), ["empty"]])),
});

describe("initialSelection", () => {
  it("starts on the lowest object id, or on a new object of the first class", () => {
    expect(initialSelection(make([3, 1]))).toEqual({ kind: "object", id: 1 });
    expect(initialSelection(make([]))).toEqual({ kind: "new", classId: 0 });
    expect(initialSelection(make([], []))).toEqual({ kind: "new", classId: null });
  });
});

describe("reconcile", () => {
  it("keeps a selection that is still valid", () => {
    const sel = { kind: "object" as const, id: 2 };
    expect(reconcile(sel, make([1, 2]))).toBe(sel);
    const fresh = { kind: "new" as const, classId: 1 };
    expect(reconcile(fresh, make([1]))).toBe(fresh);
  });
  it("moves off a deleted object to the next id, else the previous one, else a new object", () => {
    expect(reconcile({ kind: "object", id: 2 }, make([1, 3]))).toEqual({ kind: "object", id: 3 });
    expect(reconcile({ kind: "object", id: 3 }, make([1, 2]))).toEqual({ kind: "object", id: 2 });
    expect(reconcile({ kind: "object", id: 1 }, make([]))).toEqual({ kind: "new", classId: 0 });
  });
  it("gives a new object the first class when its class is unknown", () => {
    expect(reconcile({ kind: "new", classId: 7 }, make([]))).toEqual({ kind: "new", classId: 0 });
    expect(reconcile({ kind: "new", classId: null }, make([]))).toEqual({ kind: "new", classId: 0 });
  });
});

describe("cycle", () => {
  const d = make([1, 2, 3]);
  it("steps through the objects in id order, wrapping around", () => {
    expect(cycle({ kind: "object", id: 3 }, d, 1)).toEqual({ kind: "object", id: 1 });
    expect(cycle({ kind: "object", id: 1 }, d, -1)).toEqual({ kind: "object", id: 3 });
    expect(cycle({ kind: "object", id: 1 }, d, 1)).toEqual({ kind: "object", id: 2 });
  });
  it("goes from a new object to the first or the last, and stays put without objects", () => {
    expect(cycle({ kind: "new", classId: 0 }, d, 1)).toEqual({ kind: "object", id: 1 });
    expect(cycle({ kind: "new", classId: 0 }, d, -1)).toEqual({ kind: "object", id: 3 });
    const none = { kind: "new" as const, classId: 0 };
    expect(cycle(none, make([]), 1)).toBe(none);
  });
});

describe("writeTarget", () => {
  it("writes to the active object, or creates the new one with its class", () => {
    expect(writeTarget({ kind: "object", id: 2 })).toEqual({ objectId: 2 });
    expect(writeTarget({ kind: "new", classId: 1 })).toEqual({ objectId: null, classId: 1 });
    expect(writeTarget({ kind: "new", classId: null })).toBeNull();
    expect(writeTarget(null)).toBeNull();
  });
});
```

Create `web/src/hooks/useObjects.test.tsx`:

```tsx
// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ClipDetail } from "../api/types";
import { useObjects } from "./useObjects";

/** objects as [id, class] pairs; classes 0 = cup, 1 = bowl */
const make = (objects: [number, number][]): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 1, created_at: "", deleted_frames: [],
    objects: objects.map(([id, class_id]) => ({ id, class_id, keyframes: [], propagated_ranges: [] })),
  },
  states: ["empty"],
  classes: [{ id: 0, name: "cup", color: "#e6194b" }, { id: 1, name: "bowl", color: "#3cb44b" }],
  object_states: Object.fromEntries(objects.map(([id]) => [String(id), ["empty"]])),
});
type P = { id: string; d: ClipDetail | null };
const setup = (d: ClipDetail | null) => renderHook(({ id, d: detail }: P) => useObjects(id, detail), { initialProps: { id: "c1", d } });

afterEach(cleanup);

describe("useObjects", () => {
  it("has no selection until the clip loads, then starts on the lowest object id", () => {
    const { result, rerender } = setup(null);
    expect(result.current.selection).toBeNull();
    rerender({ id: "c1", d: make([[2, 1], [1, 0]]) });
    expect(result.current.activeId).toBe(1);
    expect(result.current.active?.class_id).toBe(0);
    expect(result.current.classId).toBe(0);
  });
  it("selects, steps with wrap-around, and starts a new object of the active object's class", () => {
    const { result } = setup(make([[1, 0], [2, 1]]));
    act(() => result.current.select(2));
    expect([result.current.activeId, result.current.classId]).toEqual([2, 1]);
    act(() => result.current.step(1));
    expect(result.current.activeId).toBe(1);
    act(() => result.current.startNew());
    expect(result.current.selection).toEqual({ kind: "new", classId: 0 });
    expect([result.current.activeId, result.current.active]).toEqual([null, null]);
    act(() => result.current.setNewClass(1));
    expect(result.current.classId).toBe(1);
    act(() => result.current.step(-1));
    expect(result.current.activeId).toBe(2);
  });
  it("moves to the next object when the active one disappears, and to a new object when none is left", () => {
    const { result, rerender } = setup(make([[1, 0], [2, 0], [3, 0]]));
    act(() => result.current.select(2));
    rerender({ id: "c1", d: make([[1, 0], [3, 0]]) });
    expect(result.current.activeId).toBe(3);
    rerender({ id: "c1", d: make([]) });
    expect(result.current.selection).toEqual({ kind: "new", classId: 0 });
  });
  it("hides objects for this clip only, and forgets the selection on another clip", () => {
    const { result, rerender } = setup(make([[1, 0], [2, 0]]));
    act(() => result.current.toggleHidden(2));
    expect([...result.current.hidden]).toEqual([2]);
    act(() => result.current.toggleHidden(2));
    expect(result.current.hidden.size).toBe(0);
    act(() => result.current.toggleHidden(1));
    act(() => result.current.select(2));
    rerender({ id: "c2", d: make([[1, 0], [2, 0]]) });
    expect(result.current.hidden.size).toBe(0);
    expect(result.current.activeId).toBe(1);
  });
});
```

Create `web/src/components/ObjectsPanel.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ClipDetail, FrameState } from "../api/types";
import { ObjectsPanel } from "./ObjectsPanel";

const lane = (labelled: number): FrameState[] =>
  Array.from({ length: 10 }, (_, i) => (i === 0 ? "keyframe" : i < labelled ? "masked" : "empty"));
const detail: ClipDetail = {
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 64, height: 48,
    frame_count: 10, created_at: "", deleted_frames: [],
    // stored out of id order on purpose: the panel lists them by id
    objects: [{ id: 2, class_id: 1, keyframes: [], propagated_ranges: [] }, { id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states: lane(4),
  classes: [{ id: 0, name: "cup", color: "#e6194b" }, { id: 1, name: "bowl", color: "#3cb44b" }],
  object_states: { "1": lane(4), "2": lane(1) },
};
type Props = ComponentProps<typeof ObjectsPanel>;
const props = (over: Partial<Props> = {}): Props => ({
  detail, selection: { kind: "object", id: 1 }, hidden: new Set<number>(), lockReason: null, deleteBlockedReason: null,
  error: null, onSelect: vi.fn(), onToggleHidden: vi.fn(), onClass: vi.fn(), onNew: vi.fn(), onDelete: vi.fn(),
  onAddClass: vi.fn(async () => {}), ...over,
});
const rows = () => within(screen.getByRole("list", { name: "Objects" }));

afterEach(cleanup);

describe("ObjectsPanel", () => {
  it("lists the objects in id order with class and labelled-frame count; a click selects, the eye hides", () => {
    const p = props({ hidden: new Set([2]) });
    render(<ObjectsPanel {...p} />);
    const names = rows().getAllByRole("button").map((b) => b.getAttribute("aria-label") ?? b.textContent);
    expect(names).toEqual(["#1 cup4 frames", "Hide #1 cup", "#2 bowl1 frame", "Show #2 bowl"]);
    expect(rows().getByRole("button", { name: /^#1 cup/ }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(rows().getByRole("button", { name: /^#2 bowl/ }));
    expect(p.onSelect).toHaveBeenCalledWith(2);
    fireEvent.click(rows().getByRole("button", { name: "Show #2 bowl" }));
    expect(p.onToggleHidden).toHaveBeenCalledWith(2);
  });
  it("changes the active object's class, or the class of the new object", () => {
    const p = props();
    const { rerender } = render(<ObjectsPanel {...p} />);
    expect(screen.getByText("Class of #1")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Class"), { target: { value: "1" } });
    expect(p.onClass).toHaveBeenCalledWith(1);
    rerender(<ObjectsPanel {...p} selection={{ kind: "new", classId: 1 }} />);
    expect(rows().getByRole("button", { name: "New object (bowl)" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText("Class of the new object")).toBeTruthy();
    expect((screen.getByRole("button", { name: /^\+ New object/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Delete object" }) as HTMLButtonElement).disabled).toBe(true);
  });
  it("starts a new object and deletes the active one on request", () => {
    const p = props();
    render(<ObjectsPanel {...p} />);
    fireEvent.click(screen.getByRole("button", { name: /^\+ New object/ }));
    expect(p.onNew).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete object" }));
    expect(p.onDelete).toHaveBeenCalled();
  });
  it("keeps the selection while brush strokes are unsaved, and says why Delete is unavailable", () => {
    render(
      <ObjectsPanel {...props({ lockReason: "Save or discard the brush strokes on frame 0 first.", deleteBlockedReason: "A propagation is running. Stop it to delete an object." })} />,
    );
    expect((rows().getByRole("button", { name: /^#2 bowl/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((rows().getByRole("button", { name: /^#1 cup/ }) as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByRole("button", { name: /^\+ New object/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Delete object" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("A propagation is running. Stop it to delete an object.")).toBeTruthy();
  });
  it("tells an empty clip how to start, and adds a class", async () => {
    const p = props({ detail: { ...detail, clip: { ...detail.clip, objects: [] }, object_states: {} }, selection: { kind: "new", classId: 0 } });
    render(<ObjectsPanel {...p} />);
    expect(screen.getByText(/Nothing is labelled yet/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("New class"), { target: { value: " plate " } });
    fireEvent.submit(screen.getByLabelText("New class").closest("form")!);
    await waitFor(() => expect(p.onAddClass).toHaveBeenCalledWith("plate"));
  });
});
```

Append to `describe("keyAction", ...)` in `web/src/hooks/useKeyboard.test.ts`:

```ts
  it("steps through the objects with O / Shift+O and starts a new one with N", () => {
    expect(key("o")).toEqual({ type: "object", dir: 1 });
    expect(key("O", { shiftKey: true })).toEqual({ type: "object", dir: -1 });
    expect(key("n")).toEqual({ type: "newObject" });
    expect(key("N", { shiftKey: true })).toEqual({ type: "newObject" });
    expect(key("o", { ctrlKey: true })).toBeNull();
  });
```

In `web/src/pages/Label.test.tsx`: change the testing-library import to include `within`; add
`relabelObject: vi.fn(),` and `deleteObject: vi.fn(),` to the mocked `api`; in "propagates from a keyframe, follows
the streamed frames and reloads when the job ends" (~line 192) replace
`expect(screen.getByText(/propagation is running/i)).toBeTruthy();` with
`expect(screen.getByText("A propagation is running. Stop it to prompt.")).toBeTruthy();` (the ObjectsPanel adds a
second note matching the regex, "A propagation is running. Stop it to delete an object."); add:

```ts
describe("Label page (objects)", () => {
  const panel = () => within(screen.getByRole("list", { name: "Objects" }));
  const row = (name: RegExp) => panel().getByRole("button", { name });

  it("lists the objects; one is selected by click or with O / Shift+O, and N starts a new object", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    expect(row(/^#1 connector/).getAttribute("aria-pressed")).toBe("true");
    expect(row(/^#2 cup/).textContent).toContain("5 frames");
    fireEvent.click(row(/^#2 cup/));
    expect(row(/^#2 cup/).getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(window, { key: "o" });
    expect(row(/^#1 connector/).getAttribute("aria-pressed")).toBe("true"); // wraps around
    fireEvent.keyDown(window, { key: "O", shiftKey: true });
    expect(row(/^#2 cup/).getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(window, { key: "n" });
    expect(row(/^New object \(cup\)/).getAttribute("aria-pressed")).toBe("true");
  });

  it("K and G follow the active object's lane", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "k" });
    expect(screen.getByText("frame 0 / 39")).toBeTruthy(); // #1 has no keyframe after 0 (the clip does: 5)
    fireEvent.keyDown(window, { key: "g" });
    expect(screen.getByText("frame 20 / 39")).toBeTruthy();
    fireEvent.click(row(/^#2 cup/));
    fireEvent.keyDown(window, { key: "K", shiftKey: true });
    expect(screen.getByText("frame 5 / 39")).toBeTruthy(); // #2's keyframe
    fireEvent.keyDown(window, { key: "g" });
    expect(screen.getByText("frame 10 / 39")).toBeTruthy(); // #2 is masked on 6..9
  });

  it("relabels the active object at once", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    const relabelled: ClipDetail = {
      ...two, clip: { ...two.clip, objects: [{ ...two.clip.objects[0], class_id: 1 }, two.clip.objects[1]] },
    };
    vi.mocked(api.relabelObject).mockResolvedValue(relabelled);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.change(screen.getByLabelText("Class"), { target: { value: "1" } });
    await waitFor(() => expect(api.relabelObject).toHaveBeenCalledWith("c1", 1, 1));
    expect(await panel().findByRole("button", { name: /^#1 cup/ })).toBeTruthy();
  });

  it("deletes the active object after a confirmation naming its frame count, then selects the next one", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    const onlyTwo: ClipDetail = {
      ...two, clip: { ...two.clip, objects: [two.clip.objects[1]] }, states: cupLane, object_states: { "2": cupLane },
    };
    vi.mocked(api.deleteObject).mockResolvedValue(onlyTwo);
    const confirm = vi.fn().mockReturnValueOnce(false).mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Delete object" }));
    expect(confirm).toHaveBeenCalledWith("Delete #1 connector and its masks on 20 frames?");
    expect(api.deleteObject).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete object" }));
    await waitFor(() => expect(api.deleteObject).toHaveBeenCalledWith("c1", 1));
    expect((await panel().findByRole("button", { name: /^#2 cup/ })).getAttribute("aria-pressed")).toBe("true");
  });
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && node node_modules/.bin/vitest run src/lib/activeObject.test.ts src/hooks/useObjects.test.tsx src/components/ObjectsPanel.test.tsx src/hooks/useKeyboard.test.ts src/pages/Label.test.tsx`
Expected: FAIL — the modules do not exist, `o`/`n` map to nothing, the page has no Objects list.

- [ ] **Step 3: Implement**

Create `web/src/lib/activeObject.ts`:

```ts
import type { ClipDetail, WriteTarget } from "../api/types";

/** The active row of the Objects panel: an existing object, or "New object" — which only comes into existence when
 * its first mask is accepted or painted, with `classId`. */
export type Selection = { kind: "object"; id: number } | { kind: "new"; classId: number | null };

export function sortedIds(detail: ClipDetail): number[] {
  return detail.clip.objects.map((o) => o.id).sort((a, b) => a - b);
}

/** A freshly opened clip: its lowest object id, or a new object of the first class when it has none. */
export function initialSelection(detail: ClipDetail): Selection {
  const ids = sortedIds(detail);
  return ids.length > 0 ? { kind: "object", id: ids[0] } : { kind: "new", classId: detail.classes[0]?.id ?? null };
}

/**
 * Keep `sel` valid for `detail`: a deleted object hands over to the next id (else the previous one, else "New
 * object"), and a new object whose class is unknown gets the first class. A valid selection is returned as is.
 * Ids are never reused, so a stale id can only ever mean "deleted".
 */
export function reconcile(sel: Selection, detail: ClipDetail): Selection {
  const ids = sortedIds(detail);
  if (sel.kind === "object") {
    if (ids.includes(sel.id)) return sel;
    const next = ids.find((id) => id > sel.id) ?? ids[ids.length - 1];
    return next !== undefined ? { kind: "object", id: next } : { kind: "new", classId: detail.classes[0]?.id ?? null };
  }
  if (sel.classId !== null && detail.classes.some((c) => c.id === sel.classId)) return sel;
  const classId = detail.classes[0]?.id ?? null;
  return classId === sel.classId ? sel : { kind: "new", classId };
}

/** `O` / `Shift+O`: the next / previous object in id order, wrapping; from "New object", the first / last. */
export function cycle(sel: Selection, detail: ClipDetail, dir: 1 | -1): Selection {
  const ids = sortedIds(detail);
  if (ids.length === 0) return sel;
  if (sel.kind === "new") return { kind: "object", id: dir === 1 ? ids[0] : ids[ids.length - 1] };
  const i = ids.indexOf(sel.id);
  const j = i === -1 ? 0 : (i + dir + ids.length) % ids.length;
  return { kind: "object", id: ids[j] };
}

/** Where an accept or a brush save writes; null while a new object has no class (or before the clip loads). */
export function writeTarget(selection: Selection | null): WriteTarget | null {
  if (selection === null) return null;
  if (selection.kind === "object") return { objectId: selection.id };
  return selection.classId === null ? null : { objectId: null, classId: selection.classId };
}
```

Create `web/src/hooks/useObjects.ts`:

```ts
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ClipDetail, LabelObject } from "../api/types";
import { cycle, initialSelection, reconcile, type Selection } from "../lib/activeObject";

export type ObjectsHandle = {
  /** the active row; null until the clip loads */
  selection: Selection | null;
  /** the active object's id; null on a "New object" row */
  activeId: number | null;
  active: LabelObject | null;
  /** the active object's class, or the class the new object will get */
  classId: number | null;
  /** objects hidden from the canvas (display only, for this page session and clip) */
  hidden: ReadonlySet<number>;
  select: (id: number) => void;
  step: (dir: 1 | -1) => void;
  startNew: () => void;
  setNewClass: (classId: number) => void;
  toggleHidden: (id: number) => void;
};

export function useObjects(clipId: string, detail: ClipDetail | null): ObjectsHandle {
  const [chosen, setChosen] = useState<Selection | null>(null);
  const [hidden, setHidden] = useState<ReadonlySet<number>>(() => new Set());

  useEffect(() => {
    setChosen(null);
    setHidden(new Set());
  }, [clipId]);

  // Derived, not stored: the moment a detail without the chosen object (or class) arrives, the selection moves on.
  const selection = useMemo(
    () => (detail === null ? null : reconcile(chosen ?? initialSelection(detail), detail)),
    [chosen, detail],
  );
  const active =
    selection?.kind === "object" ? (detail?.clip.objects.find((o) => o.id === selection.id) ?? null) : null;
  const classId = selection === null ? null : selection.kind === "object" ? (active?.class_id ?? null) : selection.classId;

  const select = useCallback((id: number) => setChosen({ kind: "object", id }), []);
  const step = useCallback(
    (dir: 1 | -1) => {
      if (detail !== null && selection !== null) setChosen(cycle(selection, detail, dir));
    },
    [detail, selection],
  );
  const startNew = useCallback(() => setChosen({ kind: "new", classId }), [classId]);
  const setNewClass = useCallback((id: number) => setChosen({ kind: "new", classId: id }), []);
  const toggleHidden = useCallback(
    (id: number) =>
      setHidden((h) => {
        const next = new Set(h);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }),
    [],
  );

  return { selection, activeId: active?.id ?? null, active, classId, hidden, select, step, startNew, setNewClass, toggleHidden };
}
```

Create `web/src/components/ObjectsPanel.tsx`:

```tsx
import { useState, type FormEvent } from "react";
import type { ClipDetail } from "../api/types";
import type { Selection } from "../lib/activeObject";
import { classColor, className, labeledCount, laneOf, objectLabel, sortedObjects } from "../lib/objects";

type Props = {
  detail: ClipDetail;
  selection: Selection;
  hidden: ReadonlySet<number>;
  /** unsaved brush strokes: the active row cannot change (the strokes belong to it) */
  lockReason: string | null;
  /** why the active object cannot be deleted now (a propagation is running), or null */
  deleteBlockedReason: string | null;
  error: string | null;
  onSelect: (id: number) => void;
  onToggleHidden: (id: number) => void;
  /** relabel the active object at once, or set the new object's class */
  onClass: (classId: number) => void;
  onNew: () => void;
  onDelete: () => void;
  onAddClass: (name: string) => Promise<void>;
};

const glyph = (d: string) => (
  <svg aria-hidden width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" />
    <path d={d} />
  </svg>
);
const EYE = glyph("M9.5 8a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z");
const EYE_OFF = glyph("M2.5 13.5 13.5 2.5");

/** One row per object (spec §8): swatch, "#id class", labelled-frame count, and an eye that hides it from the frame. */
export function ObjectsPanel(p: Props) {
  const { detail, selection } = p;
  const [name, setName] = useState("");
  const [addError, setAddError] = useState<string | null>(null);
  const locked = p.lockReason !== null;
  const objects = sortedObjects(detail);
  const active = selection.kind === "object" ? (objects.find((o) => o.id === selection.id) ?? null) : null;
  const classId = active?.class_id ?? (selection.kind === "new" ? selection.classId : null);

  const add = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (trimmed === "") return;
    setAddError(null);
    p.onAddClass(trimmed).then(() => setName(""), (err: unknown) => setAddError(err instanceof Error ? err.message : String(err)));
  };

  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Objects</h2>
      <ul aria-label="Objects" className="flex flex-col gap-1">
        {objects.map((o) => {
          const isActive = active?.id === o.id;
          const label = objectLabel(detail, o);
          const n = labeledCount(laneOf(detail, o.id));
          const isHidden = p.hidden.has(o.id);
          return (
            <li key={o.id} className="flex items-center gap-1">
              <button type="button" aria-pressed={isActive} disabled={locked && !isActive}
                className={`btn flex-1 justify-start gap-2 ${isActive ? "btn-on" : ""}`} onClick={() => p.onSelect(o.id)}>
                <span aria-hidden className="h-3.5 w-3.5 shrink-0 rounded-sm border border-line" style={{ background: classColor(detail, o.class_id) }} />
                <span className="truncate">{label}</span>
                <span className="ml-auto text-xs opacity-70">{n} {n === 1 ? "frame" : "frames"}</span>
              </button>
              <button type="button" className="btn px-2" aria-pressed={isHidden} aria-label={`${isHidden ? "Show" : "Hide"} ${label}`}
                title={isHidden ? "Hidden from the frame (display only)" : "Hide from the frame (display only)"}
                onClick={() => p.onToggleHidden(o.id)}>
                {isHidden ? EYE_OFF : EYE}
              </button>
            </li>
          );
        })}
        {selection.kind === "new" && (
          <li>
            <button type="button" aria-pressed className="btn btn-on w-full justify-start gap-2">
              <span aria-hidden className="h-3.5 w-3.5 shrink-0 rounded-sm border border-dashed border-line" style={{ background: classColor(detail, classId) }} />
              New object ({className(detail, classId) ?? "choose a class"})
            </button>
          </li>
        )}
      </ul>
      <label className="flex items-center gap-2 text-xs">
        <span className="w-28 shrink-0">{active !== null ? `Class of #${active.id}` : "Class of the new object"}</span>
        <select aria-label="Class" className="field flex-1" value={classId ?? ""} onChange={(e) => p.onClass(Number(e.target.value))}>
          {classId === null && <option value="">Choose a class</option>}
          {detail.classes.map((c) => (
            <option key={c.id} value={c.id}>{c.name}</option>
          ))}
        </select>
      </label>
      <div className="flex gap-1">
        <button type="button" className="btn flex-1" aria-keyshortcuts="N" disabled={locked || selection.kind === "new"} onClick={p.onNew}>
          + New object <kbd aria-hidden>N</kbd>
        </button>
        <button type="button" className="btn btn-danger" disabled={active === null || locked || p.deleteBlockedReason !== null}
          onClick={p.onDelete}>
          Delete object
        </button>
      </div>
      {active !== null && p.deleteBlockedReason !== null && <p className="note">{p.deleteBlockedReason}</p>}
      <p className="note">
        <kbd>O</kbd> / <kbd>Shift+O</kbd> next / previous object · <kbd>N</kbd> new object
      </p>
      {objects.length === 0 && <p className="note">Nothing is labelled yet. Prompt on any frame, then accept the mask.</p>}
      {objects.length > 0 && selection.kind === "new" && (
        <p className="note">The new object is created when you accept or paint its first mask.</p>
      )}
      <form className="flex gap-1" onSubmit={add}>
        <input aria-label="New class" className="field flex-1" placeholder="New class" value={name} onChange={(e) => setName(e.target.value)} />
        <button type="submit" className="btn">Add</button>
      </form>
      {p.error !== null && <p role="alert" className="text-xs text-danger">{p.error}</p>}
      {addError !== null && <p role="alert" className="text-xs text-danger">{addError}</p>}
    </section>
  );
}
```

`web/src/hooks/useKeyboard.ts`:
- add to the `KeyAction` union:

```ts
  /** `O` / `Shift+O`: next / previous object */
  | { type: "object"; dir: 1 | -1 }
  /** `N`: start a new object */
  | { type: "newObject" }
```

- in the second `switch (e.key.toLowerCase())` add:

```ts
    case "o":
      return { type: "object", dir: e.shiftKey ? -1 : 1 };
    case "n":
      return { type: "newObject" };
```

`web/src/pages/Label.tsx`:
- replace `import { ObjectPanel } from "../components/ObjectPanel";` with `import { ObjectsPanel } from "../components/ObjectsPanel";`
- add `import { useObjects } from "../hooks/useObjects";` after the `useKeyboard` import and
  `import { initialSelection } from "../lib/activeObject";` before the `frameCache` import.
- replace the objects import with
  `import { FALLBACK_COLOR, classColor, isLabeled, labeledCount, laneOf, maskParts, objectLabel, parseFrameEvent, type FrameEvent } from "../lib/objects";`
- delete the `NO_HIDDEN` constant (and replace its two uses by `hidden`, below).
- delete `const [classId, setClassId] = useState<number | null>(null);`
- replace

```ts
  const object = detail?.clip.objects[0] ?? null;
  const activeId = object?.id ?? null; // interim: Task 3 makes this the Objects panel's active object
  useEffect(() => {
    if (classId === null && detail !== null) setClassId(object?.class_id ?? detail.classes[0]?.id ?? null);
  }, [classId, detail, object]);
```

with

```ts
  const objects = useObjects(clipId, detail);
  const { hidden, select, step: stepObject, startNew, setNewClass, toggleHidden } = objects;
  const activeId = objects.activeId;
  // K, G and every "has a mask here" check follow the active object's lane, not the whole clip's states
  const activeLane = useMemo(() => (detail === null ? [] : laneOf(detail, activeId)), [detail, activeId]);
  const [objectsError, setObjectsError] = useState<string | null>(null);
```

- replace

```ts
  const shownClass = classId ?? object?.class_id ?? null;
  const color = detail?.classes.find((c) => c.id === shownClass)?.color ?? "#22c55e";
```

with `const color = detail === null ? FALLBACK_COLOR : classColor(detail, objects.classId);`
- replace the `hasMask` line with `const hasMask = (i: number) => isLabeled(activeLane[i]);`
- in the `usePrompting({...})` call replace `classId` with
  `classId: detail?.clip.objects[0]?.class_id ?? objects.classId` and add the comment
  `// interim until Task 4: requests still omit object_id, so the server writes to the clip's first object`
- in the `useBrush({...})` call replace `classId, needsClass: object === null, apply,` with
  `classId: objects.classId, needsClass: (detail?.clip.objects.length ?? 0) === 0, apply, // interim until Task 4`
- in the keyboard handler replace the `keyframe` and `nextEmpty` cases and add two cases:

```ts
      case "keyframe":
        jump(findNext(activeLane, frame, action.dir, "keyframe"));
        break;
      case "nextEmpty":
        jump(findNext(activeLane, frame, 1, "empty"));
        break;
      case "object":
        stepObject(action.dir);
        break;
      case "newObject":
        startNew();
        break;
```

- in the prefetch effect and in the `masks={...}` prop replace `hidden: NO_HIDDEN` with `hidden`, and add `hidden` to
  the prefetch effect's dependency list.
- replace `addClass` with the following, and add `changeClass` and `deleteObject` after it:

```ts
  const addClass = useCallback(
    async (name: string) => {
      if (detail === null) return;
      const created = await api.addClass(detail.clip.project_id, name);
      apply({ ...detail, classes: [...detail.classes, created] }, false);
      if (objects.selection?.kind === "new") setNewClass(created.id); // an existing object is relabelled only on request
    },
    [detail, apply, objects.selection, setNewClass],
  );
  const changeClass = useCallback(
    (classId: number) => {
      if (activeId === null) {
        setNewClass(classId);
        return;
      }
      setObjectsError(null);
      // relabelling touches no mask, so the server allows it during a propagation
      api.relabelObject(clipId, activeId, classId).then((next) => apply(next, false), (e: unknown) => setObjectsError(message(e)));
    },
    [activeId, clipId, apply, setNewClass],
  );
  const deleteObject = useCallback(() => {
    const target = objects.active;
    if (detail === null || target === null) return;
    const n = labeledCount(laneOf(detail, target.id));
    if (!window.confirm(`Delete ${objectLabel(detail, target)} and its masks on ${n} ${n === 1 ? "frame" : "frames"}?`)) return;
    setObjectsError(null);
    api.deleteObject(clipId, target.id).then((next) => apply(next, true), (e: unknown) => setObjectsError(message(e)));
  }, [detail, objects.active, clipId, apply]);
```

- replace the `<ObjectPanel ... />` element with:

```tsx
          <ObjectsPanel
            detail={detail}
            selection={objects.selection ?? initialSelection(detail)}
            hidden={hidden}
            lockReason={lockReason}
            deleteBlockedReason={running ? "A propagation is running. Stop it to delete an object." : null}
            error={objectsError}
            onSelect={select}
            onToggleHidden={toggleHidden}
            onClass={changeClass}
            onNew={startNew}
            onDelete={deleteObject}
            onAddClass={addClass}
          />
```

Delete the old panel: `git rm web/src/components/ObjectPanel.tsx`.

- [ ] **Step 4: Run the tests**

Run: `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit`
Expected: all pass (the existing K/G/propagation tests use the one-object fixtures, whose lane equals the clip's).

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/activeObject.ts web/src/lib/activeObject.test.ts web/src/hooks/useObjects.ts web/src/hooks/useObjects.test.tsx web/src/components/ObjectsPanel.tsx web/src/components/ObjectsPanel.test.tsx web/src/hooks/useKeyboard.ts web/src/hooks/useKeyboard.test.ts web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git rm web/src/components/ObjectPanel.tsx
git commit -m "Label page: Objects panel with an active object, relabel, delete, hide, O / Shift+O / N" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 4: Prompting and the brush write to the active object; accept-all

**Files:**
- Modify: `web/src/api/rest.ts`, `web/src/hooks/usePrompting.ts`, `web/src/hooks/useBrush.ts`,
  `web/src/components/PromptPanel.tsx`, `web/src/pages/Label.tsx`
- Test: `web/src/api/rest.test.ts`, `web/src/hooks/usePrompting.test.tsx` (rewrite),
  `web/src/hooks/useBrush.test.tsx` (rewrite), `web/src/components/PromptPanel.test.tsx`,
  `web/src/pages/Label.test.tsx`

**Interfaces:**
- Consumes: `Selection`, `writeTarget` (Task 3); `WriteTarget`, `AcceptResult`, `MaskResult` (Task 1).
- Produces:
  - `api.prompt(clipId, frame, prompt, objectId: number | null)` — body `{frame, prompt, object_id}`.
  - `api.accept(clipId, frame, prompt, target: WriteTarget, opts: { candidate?: number; candidates?: number[] } = {}): Promise<AcceptResult>`
    — body `{frame, prompt, object_id, class_id (new object only), candidate, candidates (when given)}`.
  - `api.setMask(clipId, frame, maskPngB64, target: WriteTarget): Promise<MaskResult>`.
  - `usePrompting({ clipId, frame, enabled, clip, selection: Selection | null, onCreated: (ids: number[]) => void })`;
    `Prompting` gains `offersAcceptAll: boolean`, `canAcceptAll: boolean`, `acceptAll(): Promise<void>`.
  - `useBrush` options `classId`/`needsClass` are replaced by `selection: Selection | null` and
    `onCreated: (id: number) => void`.
  - `PromptPanel` gains the prop `target: string` (a sentence naming what Accept writes to).

- [ ] **Step 1: Write the failing tests**

In `web/src/api/rest.test.ts` replace the tests `"sends accept with class_id and candidate only when given"` and
`"sends setMask with class_id only when given"` with:

```ts
  const CLICKS = { type: "clicks" as const, points: [[5, 6, 1]] as [number, number, 0 | 1][] };

  it("sends prompt with the object id, or null for a new object", async () => {
    const fetchMock = mockFetch(jsonResponse({ candidates: [] }), jsonResponse({ candidates: [] }));
    await api.prompt("c1", 3, CLICKS, 2);
    await api.prompt("c1", 3, CLICKS, null);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/clips/c1/prompt");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ frame: 3, prompt: CLICKS, object_id: 2 });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ frame: 3, prompt: CLICKS, object_id: null });
  });

  it("sends accept to an object, or to a new object with its class, and candidates only when given", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({}));
    const text = { type: "text" as const, phrase: "cup" };
    await api.accept("c1", 3, CLICKS, { objectId: 2 }, { candidate: 1 });
    await api.accept("c1", 3, text, { objectId: null, classId: 4 }, { candidates: [0, 2] });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ frame: 3, prompt: CLICKS, object_id: 2, candidate: 1 });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
      frame: 3, prompt: text, object_id: null, class_id: 4, candidate: 0, candidates: [0, 2],
    });
  });

  it("sends setMask to an object, or null and the class of a new object", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({}));
    await api.setMask("c1", 3, "abc==", { objectId: 2 });
    await api.setMask("c1", 3, "abc==", { objectId: null, classId: 1 });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/clips/c1/mask");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ frame: 3, mask_png_b64: "abc==", object_id: 2 });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ frame: 3, mask_png_b64: "abc==", object_id: null, class_id: 1 });
  });
```

Replace `web/src/hooks/usePrompting.test.tsx` with:

```tsx
// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/rest";
import type { AcceptResult, Candidate, ClipDetail } from "../api/types";
import type { Selection } from "../lib/activeObject";
import type { ClipHandle } from "./useClip";
import { usePrompting } from "./usePrompting";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { prompt: vi.fn(), accept: vi.fn() },
}));

const cand = (area: number): Candidate => ({ score: 0.9, bbox: [10, 10, 50, 50], area, phrase: null, mask_png_b64: "AAAA" });
const baseDetail = (): ClipDetail => ({
  clip: {
    id: "c1", project_id: "p", source: { type: "video", name: "a.mp4" }, fps: 30, stride: 1, width: 640, height: 480,
    frame_count: 10, created_at: "", deleted_frames: [], objects: [{ id: 1, class_id: 0, keyframes: [], propagated_ranges: [] }],
  },
  states: Array(10).fill("empty"),
  classes: [{ id: 0, name: "a", color: "#ff0000" }, { id: 1, name: "b", color: "#00ff00" }],
  object_states: { "1": Array(10).fill("empty") },
});
const accepted = (ids: number[]): AcceptResult => ({ ...baseDetail(), object_ids: ids });
const handle = (detail: ClipDetail): ClipHandle => ({
  detail, error: null, refreshError: null, reload: vi.fn(), apply: vi.fn(), patchFrames: vi.fn(), maskVersion: () => "0.0",
});
const OBJECT_1: Selection = { kind: "object", id: 1 };
type Args = { frame: number; selection: Selection | null; clip: ClipHandle; enabled: boolean; onCreated: (ids: number[]) => void };
const initial = (over: Partial<Args> = {}): Args => ({
  frame: 2, selection: OBJECT_1, clip: handle(baseDetail()), enabled: true, onCreated: vi.fn(), ...over,
});
const setup = (args: Args = initial()) =>
  renderHook((a: Args) => usePrompting({ clipId: "c1", ...a }), { initialProps: args });
const CLICK = { type: "clicks", points: [[5, 6, 1]] };

beforeEach(() => {
  vi.resetAllMocks();
});
afterEach(cleanup);

describe("usePrompting", () => {
  it("previews after every click, for the active object", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    expect(api.prompt).toHaveBeenLastCalledWith("c1", 2, CLICK, 1);
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
    const args = initial();
    const { result, rerender } = setup(args);
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    rerender({ ...args, frame: 3 });
    expect(result.current.draft.points).toEqual([]);
    expect(result.current.candidates).toEqual([]);
    expect(api.prompt).toHaveBeenCalledTimes(1); // the old frame's draft is never sent for the new frame
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    rerender({ ...args, frame: 3, enabled: false });
    expect(result.current.draft.points).toEqual([]);
  });

  it("clears the draft when the active object changes, and previews a new object with no object id", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const args = initial();
    const { result, rerender } = setup(args);
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    rerender({ ...args, selection: { kind: "new", classId: 0 } });
    expect(result.current.draft.points).toEqual([]);
    expect(result.current.candidates).toEqual([]);
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(api.prompt).toHaveBeenLastCalledWith("c1", 2, CLICK, null));
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

  it("accepts into the active object, or creates the new object with its class and reports its id", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    vi.mocked(api.accept).mockResolvedValue(accepted([1]));
    const args = initial();
    const { result, rerender } = setup(args);
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.canAccept).toBe(true));
    await act(() => result.current.accept());
    expect(api.accept).toHaveBeenLastCalledWith("c1", 2, CLICK, { objectId: 1 }, { candidate: 0 });
    expect(args.clip.apply).toHaveBeenCalledWith(accepted([1]), true);
    expect(args.onCreated).not.toHaveBeenCalled();
    expect(result.current.draft.points).toEqual([]);

    vi.mocked(api.accept).mockResolvedValue(accepted([4]));
    rerender({ ...args, selection: { kind: "new", classId: 1 } });
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.canAccept).toBe(true));
    await act(() => result.current.accept());
    expect(api.accept).toHaveBeenLastCalledWith("c1", 2, CLICK, { objectId: null, classId: 1 }, { candidate: 0 });
    expect(args.onCreated).toHaveBeenCalledWith([4]);
  });

  it("needs a class before a new object's first accept", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    const { result } = setup(initial({ selection: { kind: "new", classId: null } }));
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.candidates).toHaveLength(1));
    expect(result.current.canAccept).toBe(false);
    await act(() => result.current.accept());
    expect(api.accept).not.toHaveBeenCalled();
    expect(result.current.error).toMatch(/class/i);
  });

  it("offers Accept all as new objects for several candidates on a New object row only", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(1), cand(2), cand(3)] });
    vi.mocked(api.accept).mockResolvedValue(accepted([5, 6, 7]));
    const submit = (r: { current: ReturnType<typeof usePrompting> }) => {
      act(() => r.current.dispatch({ type: "tool", tool: "text" }));
      act(() => r.current.dispatch({ type: "phrase", phrase: "bolt" }));
      act(() => r.current.dispatch({ type: "submit" }));
    };
    const onObject = setup();
    submit(onObject.result);
    await waitFor(() => expect(onObject.result.current.candidates).toHaveLength(3));
    expect(onObject.result.current.offersAcceptAll).toBe(false);
    onObject.unmount();

    const args = initial({ selection: { kind: "new", classId: 0 } });
    const { result } = setup(args);
    submit(result);
    await waitFor(() => expect(result.current.candidates).toHaveLength(3));
    expect([result.current.offersAcceptAll, result.current.canAcceptAll]).toEqual([true, true]);
    await act(() => result.current.acceptAll());
    expect(api.accept).toHaveBeenLastCalledWith(
      "c1", 2, { type: "text", phrase: "bolt" }, { objectId: null, classId: 0 }, { candidate: 0, candidates: [0, 1, 2] },
    );
    expect(args.onCreated).toHaveBeenCalledWith([5, 6, 7]);
  });

  it("a second accept while one is in flight is ignored", async () => {
    vi.mocked(api.prompt).mockResolvedValue({ candidates: [cand(900)] });
    let resolveAccept: (v: AcceptResult) => void = () => {};
    vi.mocked(api.accept).mockReturnValue(new Promise((r) => (resolveAccept = r)));
    const args = initial();
    const { result } = setup(args);
    act(() => result.current.dispatch({ type: "point", x: 5, y: 6, label: 1 }));
    await waitFor(() => expect(result.current.canAccept).toBe(true));
    act(() => {
      void result.current.accept();
      void result.current.accept();
    });
    expect(api.accept).toHaveBeenCalledTimes(1);
    expect(result.current.canAccept).toBe(false);
    await act(async () => resolveAccept(accepted([1])));
    expect(args.clip.apply).toHaveBeenCalledTimes(1);
    expect(result.current.draft.points).toEqual([]);
  });

  it("re-submitting the same text after a failure retries", async () => {
    vi.mocked(api.prompt)
      .mockRejectedValueOnce(new ApiError(409, "job j9 is still running", "j9"))
      .mockResolvedValueOnce({ candidates: [cand(900)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "tool", tool: "text" }));
    act(() => result.current.dispatch({ type: "phrase", phrase: "bolt" }));
    act(() => result.current.dispatch({ type: "submit" }));
    await waitFor(() => expect(result.current.error).toBe("job j9 is still running"));
    act(() => result.current.dispatch({ type: "submit" }));
    await waitFor(() => expect(api.prompt).toHaveBeenCalledTimes(2));
    expect(result.current.candidates).toHaveLength(1);
    expect(result.current.error).toBeNull();
    expect(result.current.busyJobId).toBeNull();
  });

  it("re-drawing the identical box retries", async () => {
    vi.mocked(api.prompt)
      .mockRejectedValueOnce(new ApiError(409, "job j9 is still running", "j9"))
      .mockResolvedValueOnce({ candidates: [cand(900)] });
    const { result } = setup();
    act(() => result.current.dispatch({ type: "tool", tool: "box" }));
    act(() => result.current.dispatch({ type: "box", box: [10, 10, 50, 50] }));
    await waitFor(() => expect(result.current.error).toBe("job j9 is still running"));
    act(() => result.current.dispatch({ type: "box", box: [10, 10, 50, 50] }));
    await waitFor(() => expect(api.prompt).toHaveBeenCalledTimes(2));
    expect(result.current.candidates).toHaveLength(1);
    expect(result.current.error).toBeNull();
    expect(result.current.busyJobId).toBeNull();
  });
});
```

Replace `web/src/hooks/useBrush.test.tsx` with:

```tsx
// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { MaskResult } from "../api/types";
import type { Selection } from "../lib/activeObject";
import { loadBase } from "../lib/brushCanvas";
import type { FrameStore } from "../lib/frameCache";
import type { Rgb } from "../lib/mask";
import { useBrush, type Brush } from "./useBrush";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { setMask: vi.fn() },
}));

vi.mock("../lib/brushCanvas", () => ({
  createLayer: () => ({ canvas: document.createElement("canvas"), ctx: null }),
  loadBase: vi.fn(async () => null),
  encodeLayer: () => ({ pngB64: "UE5H", any: true }),
}));

const store = { image: async () => null } as unknown as FrameStore;
const rgb: Rgb = [255, 0, 0];
const saved = { object_id: 7 } as MaskResult;
const OBJECT_2: Selection = { kind: "object", id: 2 };

type Args = { active: boolean; selection: Selection | null };
const setup = (over: Partial<Args> = {}, onCreated = vi.fn()) =>
  renderHook(
    (a: Args) =>
      useBrush({
        clipId: "c1", frame: 3, width: 4, height: 4, rgb, store, maskUrl: null,
        active: a.active, selection: a.selection, onCreated, apply: vi.fn(),
      }),
    { initialProps: { active: true, selection: OBJECT_2, ...over } },
  );

const stroke = (result: { current: Brush }) => {
  act(() => result.current.start("paint", 0, 0));
  act(() => result.current.extend(1, 1));
  act(() => result.current.end());
};

beforeEach(() => {
  vi.resetAllMocks();
});
afterEach(cleanup);

describe("useBrush", () => {
  it("blocks save while a stroke is in progress, then saves to the active object and clears the draft", async () => {
    vi.mocked(api.setMask).mockResolvedValue(saved);
    const onCreated = vi.fn();
    const { result } = setup({}, onCreated);
    act(() => result.current.start("paint", 0, 0));
    expect(result.current.canSave).toBe(false);
    await act(async () => result.current.save());
    expect(api.setMask).not.toHaveBeenCalled();

    act(() => result.current.end());
    expect(result.current.canSave).toBe(true);
    await act(async () => result.current.save());
    expect(api.setMask).toHaveBeenCalledWith("c1", 3, "UE5H", { objectId: 2 });
    expect(onCreated).not.toHaveBeenCalled();
    expect(result.current.draft.strokes).toHaveLength(0);
    expect(result.current.dirty).toBe(false);
  });

  it("needs a class for a new object, then creates it and reports its id", async () => {
    vi.mocked(api.setMask).mockResolvedValue(saved);
    const onCreated = vi.fn();
    const { result, rerender } = setup({ selection: { kind: "new", classId: null } }, onCreated);
    stroke(result);
    expect(result.current.canSave).toBe(false);
    expect(result.current.saveBlockedReason).toBe("Choose a class for the new object before saving.");

    rerender({ active: true, selection: { kind: "new", classId: 3 } });
    expect(result.current.saveBlockedReason).toBeNull();
    expect(result.current.canSave).toBe(true);
    await act(async () => result.current.save());
    expect(api.setMask).toHaveBeenCalledWith("c1", 3, "UE5H", { objectId: null, classId: 3 });
    expect(onCreated).toHaveBeenCalledWith(7);
  });

  it("keeps committed strokes when the tool becomes inactive, but drops an already-clean draft", () => {
    const { result, rerender } = setup();
    stroke(result);
    expect(result.current.draft.strokes).toHaveLength(1);
    rerender({ active: false, selection: OBJECT_2 });
    expect(result.current.draft.strokes).toHaveLength(1);
    act(() => result.current.discard());
    expect(result.current.draft.strokes).toHaveLength(0);
    rerender({ active: false, selection: OBJECT_2 });
    expect(result.current.draft.strokes).toHaveLength(0);
  });

  it("notifies the canvas once the base mask finishes loading, and start() refuses to paint until then", async () => {
    vi.mocked(loadBase).mockImplementation(async () => {
      await Promise.resolve();
      return { fake: true } as unknown as ImageData;
    });
    const { result } = renderHook(() =>
      useBrush({
        clipId: "c1", frame: 3, width: 4, height: 4, rgb, store, maskUrl: "http://x/mask.png",
        active: true, selection: OBJECT_2, onCreated: vi.fn(), apply: vi.fn(),
      }),
    );
    expect(result.current.loading).toBe(true);
    act(() => result.current.start("paint", 0, 0));
    expect(result.current.dirty).toBe(false); // start() was refused: no stroke began

    const before = result.current.baseVersion;
    await act(async () => {
      await new Promise((r) => setTimeout(r)); // flush the microtask queue past loadBase's await tick
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.baseVersion).toBe(before + 1);
  });

  it("ignores undo while a stroke is in progress, so the in-progress stroke is not lost", () => {
    const { result } = setup();
    act(() => result.current.start("paint", 0, 0));
    act(() => result.current.undo()); // must be a no-op: a stroke is in progress
    act(() => result.current.end());
    expect(result.current.draft.strokes).toHaveLength(1);
    act(() => result.current.undo()); // now that the stroke is committed, undo removes it
    expect(result.current.draft.strokes).toHaveLength(0);
  });
});
```

In `web/src/components/PromptPanel.test.tsx`: in the `prompting()` helper add
`offersAcceptAll: false, canAcceptAll: false, acceptAll: vi.fn(async () => {}),` before `...over`; pass
`target="Accept writes to #1 bolt."` to every `<PromptPanel ... />` render; add:

```tsx
  it("names what Accept writes to, and offers Accept all as new objects on a New object row", () => {
    const c = { score: 0.91, bbox: [0, 0, 9, 9] as [number, number, number, number], area: 1234, phrase: "bolt", mask_png_b64: "" };
    const p = prompting({
      draft: { ...emptyDraft("text"), query: "bolt" }, candidates: [c, c], canAccept: true, offersAcceptAll: true, canAcceptAll: true,
    });
    render(<PromptPanel prompting={p} disabledReason={null} target="Accept creates a new bolt object." />);
    expect(screen.getByText("Accept creates a new bolt object.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Accept selected/ }));
    expect(p.accept).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Accept all as new objects" }));
    expect(p.acceptAll).toHaveBeenCalled();
    cleanup();
    render(<PromptPanel prompting={prompting({ candidates: [c, c], canAccept: true })} disabledReason={null} target="Accept writes to #1 bolt." />);
    expect(screen.queryByRole("button", { name: "Accept all as new objects" })).toBeNull();
  });
```

In `web/src/pages/Label.test.tsx`:
- in "right-click adds a negative point; prompting is off while the engine is not loaded" (~line 162): expect
  `api.prompt` `toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 0]] }, 1)` (the active object's
  id is the fourth argument).
- in the "previews a click on the canvas and accepts it" test: expect
  `api.prompt` `toHaveBeenCalledWith("c1", 25, { type: "clicks", points: [[100, 50, 1]] }, 1)`; replace the `accepted`
  line with `const accepted = { ...oneObject(detail.states.map((s, i) => (i === 25 ? "keyframe" : s))), object_ids: [1] };`;
  after the accept click add
  `expect(api.accept).toHaveBeenCalledWith("c1", 25, { type: "clicks", points: [[100, 50, 1]] }, { objectId: 1 }, { candidate: 0 });`
  inside the existing `waitFor` (replace `toHaveBeenCalledTimes(1)` with this assertion).
- in both brush tests use `vi.mocked(api.setMask).mockResolvedValue({ ...detail, object_id: 1 });` and expect
  `toHaveBeenCalledWith("c1", 0, "UE5H", { objectId: 1 })`.
- add to `describe("Label page (prompting)", ...)`:

```ts
  it("creates a new object from its first accepted mask and makes it active", async () => {
    vi.mocked(api.getClip).mockResolvedValue(detail);
    vi.mocked(api.prompt).mockResolvedValue({
      candidates: [{ score: 0.9, bbox: [80, 30, 120, 70], area: 1200, phrase: null, mask_png_b64: "AAAA" }],
    });
    vi.mocked(api.accept).mockResolvedValue({ ...two, object_ids: [2] });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.keyDown(window, { key: "n" });
    expect(screen.getByText("Accept creates a new connector object.")).toBeTruthy();
    const canvas = screen.getByLabelText("Frame");
    fireEvent.mouseDown(canvas, { clientX: 108, clientY: 58, button: 0 });
    fireEvent.mouseUp(canvas, { clientX: 108, clientY: 58, button: 0 });
    await waitFor(() => expect(api.prompt).toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 1]] }, null));
    const accept = screen.getByRole("button", { name: /^Accept/ }) as HTMLButtonElement;
    await waitFor(() => expect(accept.disabled).toBe(false));
    fireEvent.click(accept);
    await waitFor(() =>
      expect(api.accept).toHaveBeenCalledWith("c1", 0, { type: "clicks", points: [[100, 50, 1]] }, { objectId: null, classId: 0 }, { candidate: 0 }),
    );
    const created = await within(screen.getByRole("list", { name: "Objects" })).findByRole("button", { name: /^#2 / });
    expect(created.getAttribute("aria-pressed")).toBe("true");
  });
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && node node_modules/.bin/vitest run src/api/rest.test.ts src/hooks/usePrompting.test.tsx src/hooks/useBrush.test.tsx src/components/PromptPanel.test.tsx src/pages/Label.test.tsx`
Expected: FAIL — the API still sends the legacy bodies, the hooks take `classId`, there is no accept-all.

- [ ] **Step 3: Implement**

`web/src/api/rest.ts`:
- type import: add `AcceptResult`, `MaskResult`, `WriteTarget`.
- after the `patch` helper add:

```ts
/** `object_id` is always sent (the server requires the key; null = a new object); `class_id` only for a new object. */
const targetBody = (t: WriteTarget) =>
  t.objectId === null ? { object_id: null, class_id: t.classId } : { object_id: t.objectId };
```

- replace `prompt`, `accept`, `setMask` in `api` with:

```ts
  /** `objectId`: a clicks prompt without a box derives its box from that object's mask here; null = a fresh one */
  prompt: (clipId: string, frame: number, prompt: PromptSpec, objectId: number | null) =>
    request<{ candidates: Candidate[] }>(`/api/clips/${clipId}/prompt`, post({ frame, prompt, object_id: objectId })),
  /** `candidates` (a new object only): each listed candidate becomes its own new object */
  accept: (
    clipId: string, frame: number, prompt: PromptSpec, target: WriteTarget,
    opts: { candidate?: number; candidates?: number[] } = {},
  ) =>
    request<AcceptResult>(
      `/api/clips/${clipId}/accept`,
      post({
        frame,
        prompt,
        ...targetBody(target),
        candidate: opts.candidate ?? 0,
        ...(opts.candidates === undefined ? {} : { candidates: opts.candidates }),
      }),
    ),
  setMask: (clipId: string, frame: number, maskPngB64: string, target: WriteTarget) =>
    request<MaskResult>(`/api/clips/${clipId}/mask`, post({ frame, mask_png_b64: maskPngB64, ...targetBody(target) })),
```

Replace `web/src/hooks/usePrompting.ts` with:

```ts
import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type Dispatch } from "react";
import { ApiError, api } from "../api/rest";
import type { Candidate } from "../api/types";
import { writeTarget, type Selection } from "../lib/activeObject";
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
  /** several candidates (a text prompt) on a "New object" row: "Accept all as new objects" is shown */
  offersAcceptAll: boolean;
  canAcceptAll: boolean;
  accept: () => Promise<void>;
  /** every candidate becomes its own new object of the new object's class */
  acceptAll: () => Promise<void>;
  clear: () => void;
};

type Options = {
  clipId: string;
  frame: number;
  enabled: boolean;
  clip: ClipHandle;
  /** the Objects panel's active row; null until the clip loads */
  selection: Selection | null;
  /** the ids a "New object" accept created (first one first), so the page can make one active */
  onCreated: (ids: number[]) => void;
};

const NO_MASK = "No usable mask: nothing was found, or the mask is too small or too large. Add a click or try another prompt.";

export function usePrompting({ clipId, frame, enabled, clip, selection, onCreated }: Options): Prompting {
  const [draft, dispatch] = useReducer(draftReducer, undefined, () => emptyDraft());
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyJobId, setBusyJobId] = useState<string | null>(null);
  const [accepting, setAccepting] = useState(false);
  const acceptingRef = useRef(false);
  const [attempt, setAttempt] = useState(0);
  const requestId = useRef(0);
  const draftIsStale = useRef(false);
  const objectId = selection?.kind === "object" ? selection.id : null;
  const target = useMemo(() => writeTarget(selection), [selection]);

  // "submit" (text) and "box" can repeat identical content after a failed preview; bump a counter so the
  // preview effect re-requests even though the prompt itself did not change. Everything else that mutates
  // content already changes the prompt, and candidate/phrase/undo/tool must not re-request on their own.
  const wrappedDispatch = useCallback<Dispatch<DraftAction>>((action) => {
    if (action.type === "submit" || action.type === "box") setAttempt((a) => a + 1);
    dispatch(action);
  }, []);

  const promptKey = JSON.stringify([draft.tool, draft.points, draft.box, draft.query]);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- promptKey covers every field toPrompt reads
  const prompt = useMemo(() => toPrompt(draft), [promptKey]);

  // a draft belongs to one frame and one object, and only exists while prompting is possible
  useEffect(() => {
    draftIsStale.current = true; // the effect below runs in this same commit and still sees the old draft
    dispatch({ type: "clear" });
    setError(null);
  }, [clipId, frame, enabled, objectId]);

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
      .prompt(clipId, frame, prompt, objectId)
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
  }, [clipId, frame, enabled, prompt, attempt, objectId]);

  const canAccept =
    enabled && prompt !== null && !busy && !accepting && draft.candidate < candidates.length && target !== null;
  const offersAcceptAll = objectId === null && candidates.length > 1;
  const canAcceptAll = canAccept && offersAcceptAll;

  const apply = clip.apply;
  const send = useCallback(
    async (choice: { candidate: number; candidates?: number[] }) => {
      if (acceptingRef.current) return; // a second call in the same tick is ignored
      acceptingRef.current = true;
      setAccepting(true);
      try {
        if (prompt === null || candidates.length === 0) return;
        if (target === null) {
          setError("Choose a class for the new object before accepting its first mask.");
          return;
        }
        const next = await api.accept(clipId, frame, prompt, target, choice);
        apply(next, true);
        if (target.objectId === null) onCreated(next.object_ids);
        dispatch({ type: "clear" });
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setBusyJobId(e instanceof ApiError && e.status === 409 ? e.jobId : null);
      } finally {
        acceptingRef.current = false;
        setAccepting(false);
      }
    },
    [prompt, candidates.length, target, clipId, frame, apply, onCreated],
  );
  const accept = useCallback(() => send({ candidate: draft.candidate }), [send, draft.candidate]);
  const acceptAll = useCallback(() => send({ candidate: 0, candidates: candidates.map((_, i) => i) }), [send, candidates]);

  const clear = useCallback(() => {
    dispatch({ type: "clear" });
    setError(null);
  }, []);

  return {
    draft, dispatch: wrappedDispatch, candidates, busy, error, busyJobId, canAccept, offersAcceptAll, canAcceptAll,
    accept, acceptAll, clear,
  };
}
```

`web/src/hooks/useBrush.ts`:
- imports: add `import { writeTarget, type Selection } from "../lib/activeObject";`
- `Brush.canSave` doc: `/** true when save() will actually do something: dirty, not busy, no stroke in progress, and a class is chosen for a new object. */`
- in `Options` replace `classId: number | null;` and `needsClass: boolean;` with:

```ts
  /** the Objects panel's active row: Save writes to that object, or creates the new object with its class */
  selection: Selection | null;
  /** the id of the object a "New object" save created, so the page can make it active */
  onCreated: (id: number) => void;
```

- the `maskUrl` option doc: `/** the active object's own stored mask on this frame (versioned url); null when it has none or on a "New object" row */`
- destructure: `const { clipId, frame, width, height, rgb, store, maskUrl, active, selection, onCreated, apply } = o;`
- replace everything from the comment `// Mirrors usePrompting's canAccept: ...` to the end of the `save` callback with:

```ts
  // Mirrors usePrompting's canAccept: a "New object" row needs a class before its first mask can be saved.
  const target = useMemo(() => writeTarget(selection), [selection]);
  const saveBlockedReason = dirty && target === null ? "Choose a class for the new object before saving." : null;
  const canSave = dirty && !busy && !live && target !== null;

  const save = useCallback(async () => {
    // A stroke in progress (stroke.current !== null) must not be saved: encoding the layer now and clearing the
    // draft, only to have the in-flight end() commit the whole stroke into the emptied draft, would leave a stuck
    // "not saved" state. canSave already reflects this and the class check, but save() guards itself too.
    if (!dirty || busy || stroke.current !== null || target === null) return;
    const encoded = encodeLayer(layer.canvas);
    if (encoded === null) {
      setError("This browser cannot export the painted mask.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.setMask(clipId, frame, encoded.pngB64, target);
      apply(result, true);
      if (target.objectId === null) onCreated(result.object_id);
      setDraft((d) => emptyBrush(d.size));
    } catch (e: unknown) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }, [dirty, busy, layer, clipId, frame, target, apply, onCreated]);
```

`web/src/components/PromptPanel.tsx`:
- `type Props = { prompting: Prompting; disabledReason: string | null; /** what Accept does, as a sentence naming the active object */ target: string };`
- signature `export function PromptPanel({ prompting, disabledReason, target }: Props)`, and destructure
  `offersAcceptAll, canAcceptAll, acceptAll` from `prompting` as well.
- after the hint/disabled paragraph add `{!disabled && <p className="note text-ink">{target}</p>}`
- the Accept button's text becomes `{candidates.length > 1 ? "Accept selected" : "Accept"} <kbd aria-hidden>Enter</kbd>`
- after the Accept/Clear/Undo row add:

```tsx
      {offersAcceptAll && (
        <button type="button" disabled={!canAcceptAll} className="btn" onClick={() => void acceptAll()}>
          Accept all as new objects
        </button>
      )}
```

`web/src/pages/Label.tsx`:
- objects import: add `className`.
- replace the `usePrompting(...)` line with:

```ts
  const selectCreated = useCallback(
    (ids: number[]) => {
      if (ids.length > 0) select(ids[0]);
    },
    [select],
  );
  const prompting = usePrompting({
    clipId, frame, enabled: disabledReason === null && detail !== null && mode === "prompt", clip,
    selection: objects.selection, onCreated: selectCreated,
  });
```

- in `useBrush({...})` replace the interim `classId: ..., needsClass: ..., apply, // interim until Task 4` with
  `selection: objects.selection, onCreated: select, apply,`
- after `const { draft, dispatch } = prompting;` (after the early returns) add:

```ts
  const newClass = className(detail, objects.classId);
  const promptTarget =
    objects.active !== null
      ? `Accept writes to ${objectLabel(detail, objects.active)}.`
      : newClass === null
        ? "Choose a class for the new object in the Objects panel."
        : `Accept creates a new ${newClass} object.`;
```

- render `<PromptPanel prompting={prompting} disabledReason={disabledReason} target={promptTarget} />`.

- [ ] **Step 4: Run the tests**

Run: `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit`
Expected: all pass. `grep -n "objects\[0\]" web/src/pages/Label.tsx web/src/hooks/*.ts` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add web/src/api/rest.ts web/src/api/rest.test.ts web/src/hooks/usePrompting.ts web/src/hooks/usePrompting.test.tsx web/src/hooks/useBrush.ts web/src/hooks/useBrush.test.tsx web/src/components/PromptPanel.tsx web/src/components/PromptPanel.test.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git commit -m "Label page: prompts, accepts and brush saves go to the active object; accept all as new objects" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 5: Timeline with one lane per object

**Files:**
- Create: `web/src/lib/lanes.ts`, `web/src/lib/lanes.test.ts`
- Modify: `web/src/components/Timeline.tsx` (rewrite), `web/src/lib/objects.ts`, `web/src/pages/Label.tsx`
- Test: `web/src/components/Timeline.test.tsx` (rewrite), `web/src/lib/objects.test.ts`, `web/src/pages/Label.test.tsx`

**Interfaces:**
- Consumes: `laneOf`, `sortedObjects`, `objectLabel`, `classColor` (Tasks 1–2); `objects.select`, `hidden`,
  `activeId` (Task 3).
- Produces:
  - `lib/lanes.ts`: constants `LANE_H = 16`, `ACTIVE_H = 30`, `LANE_GAP = 2`, `GUTTER = 72`, `MAX_LANES = 6`,
    `MAX_VIEW = 120`; `type Lane = { id: number; label: string; color: string; states: FrameState[]; hidden: boolean }`;
    `type LaneBox = { id: number | null; top: number; height: number }`;
    `type LaneLayout = { boxes: LaneBox[]; content: number; view: number }`;
    `layoutLanes(ids: readonly (number | null)[], activeId: number | null): LaneLayout`;
    `laneAt(boxes: readonly LaneBox[], y: number): LaneBox | null`;
    `clampScroll(scroll: number, layout: LaneLayout): number`;
    `scrollToShow(scroll: number, layout: LaneLayout, id: number | null): number`.
  - `lib/objects.ts`: `lanesFor(detail: ClipDetail, hidden: ReadonlySet<number>): Lane[]`.
  - `Timeline` props: `{ states: FrameState[]; lanes: Lane[]; activeId: number | null; frame: number; thumb?; range?; onSeek; onRange?; onSelect?: (id: number) => void }`
    (the `color` prop is gone: each lane carries its class colour).

- [ ] **Step 1: Write the failing tests**

Create `web/src/lib/lanes.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { ACTIVE_H, GUTTER, LANE_GAP, LANE_H, MAX_LANES, MAX_VIEW, clampScroll, laneAt, layoutLanes, scrollToShow } from "./lanes";

describe("lane layout", () => {
  it("uses the spec's sizes", () => {
    expect([LANE_H, ACTIVE_H, LANE_GAP, GUTTER, MAX_LANES, MAX_VIEW]).toEqual([16, 30, 2, 72, 6, 120]);
  });
  it("stacks lanes in order: the active one 30 px, the others 16 px, 2 px apart", () => {
    expect(layoutLanes([1, 2, 3], 2)).toEqual({
      boxes: [{ id: 1, top: 0, height: 16 }, { id: 2, top: 18, height: 30 }, { id: 3, top: 50, height: 16 }],
      content: 66,
      view: 66,
    });
  });
  it("draws every lane short when no object is active, and the placeholder lane of an empty clip tall", () => {
    expect(layoutLanes([1, 2], null).boxes.map((b) => b.height)).toEqual([16, 16]);
    expect(layoutLanes([null], null)).toEqual({ boxes: [{ id: null, top: 0, height: 30 }], content: 30, view: 30 });
  });
  it("grows up to six lanes, then keeps the view at that height", () => {
    const six = layoutLanes([1, 2, 3, 4, 5, 6], 1);
    expect([six.content, six.view]).toEqual([120, 120]);
    const eight = layoutLanes([1, 2, 3, 4, 5, 6, 7, 8], 8);
    expect([eight.content, eight.view]).toEqual([156, 120]); // 7 × (16 + 2) + 30
  });
});

describe("hit testing and scrolling", () => {
  const eight = layoutLanes([1, 2, 3, 4, 5, 6, 7, 8], 8);
  it("finds the lane under a y position, and none in a gap", () => {
    const three = layoutLanes([1, 2, 3], 2);
    expect(laneAt(three.boxes, 0)?.id).toBe(1);
    expect(laneAt(three.boxes, 15.9)?.id).toBe(1);
    expect(laneAt(three.boxes, 17)).toBeNull();
    expect(laneAt(three.boxes, 18)?.id).toBe(2);
    expect(laneAt(three.boxes, 47.9)?.id).toBe(2);
    expect(laneAt(three.boxes, 70)).toBeNull();
  });
  it("clamps the scroll offset to the lane stack", () => {
    expect(clampScroll(-5, eight)).toBe(0);
    expect(clampScroll(999, eight)).toBe(36); // 156 − 120
    expect(clampScroll(10, layoutLanes([1, 2], 1))).toBe(0);
  });
  it("scrolls just enough to show a lane", () => {
    expect(scrollToShow(0, eight, 8)).toBe(36);
    expect(scrollToShow(36, eight, 1)).toBe(0);
    expect(scrollToShow(10, eight, 3)).toBe(10);
    expect(scrollToShow(10, eight, null)).toBe(10);
  });
});
```

Append to `web/src/lib/objects.test.ts` (add `lanesFor` to its import):

```ts
describe("lanesFor", () => {
  it("gives one lane per object in id order, named, coloured and marked hidden", () => {
    const d = detail();
    d.clip.objects.reverse();
    expect(lanesFor(d, new Set([2]))).toEqual([
      { id: 1, label: "#1 cup", color: "#ff0000", states: S("keed"), hidden: false },
      { id: 2, label: "#2 bowl", color: "#00ff00", states: S("emed"), hidden: true },
    ]);
  });
});
```

Replace `web/src/components/Timeline.test.tsx` with:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FrameState } from "../api/types";
import type { Lane } from "../lib/lanes";
import { stubCanvasAndObservers } from "../test/setup-dom";
import { Timeline } from "./Timeline";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);
const states = S("kmmeeedeee");
const lane = (id: number, label: string, st: FrameState[], hidden = false): Lane => ({ id, label, color: "#e6194b", states: st, hidden });
const two = [lane(1, "#1 connector", S("kmmeeedeee")), lane(2, "#2 cup", S("eekmmedeee"))];
// The canvas spans clientX 10..182: a 72 px name gutter, then 10 px per frame, so frame f is centred at 87 + 10f.
const X = (f: number) => 87 + 10 * f;
// Lanes start 6 px down: with #1 active, lane 1 spans y 6..36 and lane 2 y 38..54.
const noLanes = { states, lanes: [] as Lane[], activeId: null, frame: 0 };

/** a 2D context that records every call and every globalAlpha it is given */
function recordCanvas() {
  const calls: { name: string; args: unknown[] }[] = [];
  const alphas: number[] = [];
  const ctx = new Proxy(
    {},
    {
      get: (_t, name) => (...args: unknown[]) => {
        calls.push({ name: String(name), args });
      },
      set: (_t, name, value) => {
        if (name === "globalAlpha") alphas.push(value as number);
        return true;
      },
    },
  );
  HTMLCanvasElement.prototype.getContext = (() => ctx) as unknown as HTMLCanvasElement["getContext"];
  return { calls, alphas };
}

beforeEach(() => {
  stubCanvasAndObservers();
  HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ left: 10, top: 0, width: 172, height: 60 }) as DOMRect;
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Timeline", () => {
  it("seeks on click and while dragging with the left button held", () => {
    const onSeek = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), button: 0, buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(5);
    fireEvent.mouseMove(strip, { clientX: X(8), buttons: 1 });
    expect(onSeek).toHaveBeenLastCalledWith(8);
    fireEvent.mouseMove(strip, { clientX: X(1), buttons: 0 });
    expect(onSeek).toHaveBeenCalledTimes(2);
  });
  it("ignores a press in the name gutter", () => {
    const onSeek = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} />);
    fireEvent.mouseDown(screen.getByLabelText("Timeline"), { clientX: 40, button: 0, buttons: 1 });
    expect(onSeek).not.toHaveBeenCalled();
  });
  it("marks a range with Shift-drag instead of seeking, in either direction", () => {
    const onSeek = vi.fn();
    const onRange = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} onRange={onRange} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), button: 0, buttons: 1, shiftKey: true });
    expect(onRange).toHaveBeenLastCalledWith([5, 5]);
    fireEvent.mouseMove(strip, { clientX: X(2), buttons: 1, shiftKey: true });
    expect(onRange).toHaveBeenLastCalledWith([2, 5]);
    fireEvent.mouseUp(strip);
    fireEvent.mouseMove(strip, { clientX: X(8), buttons: 1 }); // a plain drag afterwards seeks again
    expect(onSeek).toHaveBeenCalledTimes(1);
    expect(onSeek).toHaveBeenLastCalledWith(8);
    expect(onRange).toHaveBeenCalledTimes(2);
  });
  it("scrubs normally after a Shift-drag that was released outside the strip", () => {
    const onSeek = vi.fn();
    const onRange = vi.fn();
    render(<Timeline {...noLanes} onSeek={onSeek} onRange={onRange} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), button: 0, buttons: 1, shiftKey: true });
    fireEvent.mouseLeave(strip); // the button is let go somewhere else: the strip never sees a mouseup
    fireEvent.mouseUp(window);
    onRange.mockClear();
    fireEvent.mouseDown(strip, { clientX: X(1), button: 0, buttons: 1 });
    fireEvent.mouseMove(strip, { clientX: X(8), buttons: 1 });
    expect(onRange).not.toHaveBeenCalled();
    expect(onSeek).toHaveBeenLastCalledWith(8);
  });
  it("shows the hovered frame, its state and its thumbnail, and hides them again on leave", () => {
    const { container } = render(<Timeline {...noLanes} thumb={{ url: (i) => `/thumb/${i}`, aspect: 4 / 3 }} onSeek={() => {}} />);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseMove(strip, { clientX: X(6), buttons: 0 });
    expect(screen.getByText("frame 6")).toBeTruthy();
    expect(screen.getByText("deleted")).toBeTruthy();
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/thumb/6");
    fireEvent.mouseLeave(strip);
    expect(screen.queryByText("frame 6")).toBeNull();
  });
  it("shows a single lane named No objects yet for a clip without objects", () => {
    render(<Timeline {...noLanes} onSeek={() => {}} />);
    expect(screen.getByText("No objects yet")).toBeTruthy();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
  it("is found by its label (it cannot be operated by keyboard — the global shortcuts cover seeking and selection)", () => {
    render(<Timeline {...noLanes} frame={4} onSeek={() => {}} />);
    expect(screen.getByLabelText("Timeline")).toBeTruthy();
  });
});

describe("Timeline lanes", () => {
  it("names each lane in the gutter; clicking a name, or inside another lane, selects that object", () => {
    const onSelect = vi.fn();
    const onSeek = vi.fn();
    render(<Timeline states={states} lanes={two} activeId={1} frame={0} onSeek={onSeek} onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: "#1 connector" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "#2 cup" }));
    expect(onSelect).toHaveBeenLastCalledWith(2);
    const strip = screen.getByLabelText("Timeline");
    fireEvent.mouseDown(strip, { clientX: X(5), clientY: 42, button: 0, buttons: 1 });
    expect(onSelect).toHaveBeenCalledTimes(2);
    expect(onSelect).toHaveBeenLastCalledWith(2);
    expect(onSeek).toHaveBeenLastCalledWith(5); // and it seeks, as a click always did
    fireEvent.mouseDown(strip, { clientX: X(3), clientY: 20, button: 0, buttons: 1 });
    expect(onSelect).toHaveBeenCalledTimes(2); // the active lane: seek only
    expect(onSeek).toHaveBeenLastCalledWith(3);
    fireEvent.mouseDown(strip, { clientX: X(4), clientY: 37, button: 0, buttons: 1 });
    expect(onSelect).toHaveBeenCalledTimes(2); // the gap between two lanes selects nothing
  });
  it("lists every object's state on the hovered frame", () => {
    render(<Timeline states={states} lanes={two} activeId={1} frame={0} onSeek={() => {}} />);
    fireEvent.mouseMove(screen.getByLabelText("Timeline"), { clientX: X(2), clientY: 20, buttons: 0 });
    expect(screen.getByText("frame 2")).toBeTruthy();
    expect(screen.getAllByRole("listitem").map((li) => li.textContent)).toEqual(["#1 connectormasked", "#2 cupkeyframe"]);
  });
  it("hatches a deleted frame across every lane and draws hidden lanes at 40 % opacity", () => {
    Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 172 });
    try {
      const { calls, alphas } = recordCanvas();
      render(<Timeline states={states} lanes={[two[0], { ...two[1], hidden: true }]} activeId={1} frame={0} onSeek={() => {}} />);
      // frame 6 is deleted: its hatching is clipped to one rect per lane, at x = 72 + 6 × 10
      const hatchTops = calls.filter((c) => c.name === "rect" && c.args[0] === 132).map((c) => c.args[1] as number);
      expect([...new Set(hatchTops)].sort((a, b) => a - b)).toEqual([6, 38]);
      expect(alphas).toContain(0.4);
      expect(screen.getByRole("button", { name: "#2 cup" }).style.opacity).toBe("0.4");
      // no onSelect (the selection is locked): the names are disabled
      expect((screen.getByRole("button", { name: "#2 cup" }) as HTMLButtonElement).disabled).toBe(true);
    } finally {
      delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientWidth;
    }
  });
  it("grows to six lanes, then scrolls the lanes and keeps the active one in view", () => {
    const many = (n: number) => Array.from({ length: n }, (_, i) => lane(i + 1, `#${i + 1} part`, S("eeeeeeeeee")));
    const { rerender } = render(<Timeline states={states} lanes={many(3)} activeId={1} frame={0} onSeek={() => {}} />);
    const strip = screen.getByLabelText("Timeline");
    expect(strip.style.height).toBe("90px"); // 6 + (30 + 2 + 16 + 2 + 16) + 18
    rerender(<Timeline states={states} lanes={many(6)} activeId={1} frame={0} onSeek={() => {}} />);
    expect(strip.style.height).toBe("144px"); // 6 + 120 + 18
    rerender(<Timeline states={states} lanes={many(8)} activeId={8} frame={0} onSeek={() => {}} />);
    expect(strip.style.height).toBe("144px"); // capped at six lanes' worth
    // lane 8 sits at 7 × 18 = 126..156 in the stack: scrolled by 156 − 120 = 36 to show it
    expect(screen.getByRole("button", { name: "#8 part" }).style.top).toBe("90px");
    expect(screen.getByRole("button", { name: "#1 part" }).style.top).toBe("-36px");
  });
});
```

Add to `describe("Label page (objects)", ...)` in `web/src/pages/Label.test.tsx`:

```ts
  it("shows one timeline lane per object; a lane's name selects its object", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "#2 cup" })); // the name in the timeline's gutter
    expect(row(/^#2 cup/).getAttribute("aria-pressed")).toBe("true");
  });
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && node node_modules/.bin/vitest run src/lib/lanes.test.ts src/lib/objects.test.ts src/components/Timeline.test.tsx src/pages/Label.test.tsx`
Expected: FAIL — `./lanes` and `lanesFor` do not exist; the Timeline has no lanes or gutter.

- [ ] **Step 3: Implement**

Create `web/src/lib/lanes.ts`:

```ts
import type { FrameState } from "../api/types";

/** Timeline geometry (spec §9), in CSS px. */
export const LANE_H = 16; // not smaller: the lane names beside it stay text-xs
export const ACTIVE_H = 30;
export const LANE_GAP = 2;
export const GUTTER = 72;
export const MAX_LANES = 6;
/** the lane viewport's height cap: six lanes with the active one among them; beyond that the lanes scroll */
export const MAX_VIEW = ACTIVE_H + (MAX_LANES - 1) * (LANE_H + LANE_GAP);

/** One object's lane: its name ("#2 cup"), class colour, per-frame states, and whether it is hidden on the canvas. */
export type Lane = { id: number; label: string; color: string; states: FrameState[]; hidden: boolean };
/** A lane's place in the stack, from the top of the lane area, before scrolling. `id` null = the empty clip's lane. */
export type LaneBox = { id: number | null; top: number; height: number };
export type LaneLayout = { boxes: LaneBox[]; content: number; view: number };

/** Stack the lanes in the given order: the active lane (and the single placeholder lane) tall, the others short. */
export function layoutLanes(ids: readonly (number | null)[], activeId: number | null): LaneLayout {
  const boxes: LaneBox[] = [];
  let top = 0;
  for (const id of ids) {
    const height = id === null || id === activeId ? ACTIVE_H : LANE_H;
    boxes.push({ id, top, height });
    top += height + LANE_GAP;
  }
  const content = Math.max(0, top - LANE_GAP);
  return { boxes, content, view: Math.min(content, MAX_VIEW) };
}

/** The lane under `y` (lane-area coordinates, scroll applied), or null in a gap or past the last lane. */
export function laneAt(boxes: readonly LaneBox[], y: number): LaneBox | null {
  for (const b of boxes) if (y >= b.top && y < b.top + b.height) return b;
  return null;
}

export function clampScroll(scroll: number, layout: LaneLayout): number {
  return Math.max(0, Math.min(layout.content - layout.view, Math.round(scroll)));
}

/** The smallest change of `scroll` that shows lane `id` entirely; unchanged (but clamped) when it already shows. */
export function scrollToShow(scroll: number, layout: LaneLayout, id: number | null): number {
  const box = layout.boxes.find((b) => b.id === id);
  if (box === undefined) return clampScroll(scroll, layout);
  if (box.top < scroll) return clampScroll(box.top, layout);
  if (box.top + box.height > scroll + layout.view) return clampScroll(box.top + box.height - layout.view, layout);
  return clampScroll(scroll, layout);
}
```

Append to `web/src/lib/objects.ts` (and add `import type { Lane } from "./lanes";` to its imports):

```ts
/** The Timeline's lanes: one per object in id order (spec §9). */
export function lanesFor(detail: ClipDetail, hidden: ReadonlySet<number>): Lane[] {
  return sortedObjects(detail).map((o) => ({
    id: o.id,
    label: objectLabel(detail, o),
    color: classColor(detail, o.class_id),
    states: laneOf(detail, o.id),
    hidden: hidden.has(o.id),
  }));
}
```

Replace `web/src/components/Timeline.tsx` with:

```tsx
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { FrameState } from "../api/types";
import { GUTTER, clampScroll, laneAt, layoutLanes, scrollToShow, type Lane } from "../lib/lanes";
import { frameAtX, runs } from "../lib/timeline";

type Props = {
  /** the whole clip per frame: sets the frame count, and is drawn as the single lane of a clip without objects */
  states: FrameState[];
  /** one lane per object, in object-id order (`lanesFor`) */
  lanes: Lane[];
  /** the active object: its lane is drawn 30 px tall; null on a "New object" row (every lane short) */
  activeId: number | null;
  frame: number;
  /** thumbnail shown while hovering; omit to show the frame number only */
  thumb?: { url: (frame: number) => string; aspect: number };
  /** frames marked for clearing (inclusive), drawn as a bracket over the lanes */
  range?: [number, number] | null;
  onSeek: (frame: number) => void;
  /** Shift-drag marks a range instead of seeking; omit to disable marking */
  onRange?: (range: [number, number]) => void;
  /** a click on a lane, or on its name in the gutter, selects that object; omit while the selection is locked */
  onSelect?: (id: number) => void;
};

const TOP = 6; // room above the lanes for the playhead's flag
const RULER = 18; // the frame-number ruler under the lanes
const THUMB_W = 176;
const HIDDEN_ALPHA = 0.4;
// canvas cannot read the Tailwind theme; these mirror tailwind.config.js
const INK = "#E9E7E2";
const DIM = "#A3A5AA";
const LINE = "#4A4D52";
const WELL = "#1E1F21";
const PANEL = "#36383C";
const KEY = "#F59E0B";
const STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
const WORDS: Record<FrameState, string> = { keyframe: "keyframe", masked: "masked", empty: "no mask", deleted: "deleted" };
const NO_OBJECTS = "No objects yet";

/**
 * One object's frames in a band `h` px tall at `y`. State is told apart by shape and texture as well as colour: a
 * filled run is masked, a hollow run has no mask, a hatched run is deleted, and a diamond marks a keyframe.
 */
function drawLane(ctx: CanvasRenderingContext2D, states: readonly FrameState[], y: number, h: number, color: string, px: number) {
  for (const r of runs(states)) {
    const x = GUTTER + r.start * px;
    const w = (r.end - r.start + 1) * px;
    if (r.state === "masked" || r.state === "keyframe") {
      ctx.fillStyle = color;
      ctx.fillRect(x, y, w, h);
    } else if (r.state === "deleted") {
      ctx.fillStyle = WELL;
      ctx.fillRect(x, y, w, h);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.strokeStyle = DIM;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let hx = x - h; hx < x + w; hx += 5) {
        ctx.moveTo(hx, y + h);
        ctx.lineTo(hx + h, y);
      }
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.strokeStyle = LINE;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, Math.max(w - 1, 1), h - 1);
    }
  }
  if (px >= 5) {
    // frames are discrete: notch every boundary while there is room for it
    ctx.fillStyle = PANEL;
    for (let i = 1; i < states.length; i++) ctx.fillRect(Math.round(GUTTER + i * px), y, 1, h);
  }
  // keyframe diamonds, on top and scaled to the lane, outlined so they hold against any class colour
  const dy = Math.min(7, h / 2);
  const dx = (dy * 6) / 7;
  const mid = y + h / 2;
  states.forEach((s, i) => {
    if (s !== "keyframe") return;
    const cx = GUTTER + i * px + px / 2;
    ctx.beginPath();
    ctx.moveTo(cx, mid - dy);
    ctx.lineTo(cx + dx, mid);
    ctx.lineTo(cx, mid + dy);
    ctx.lineTo(cx - dx, mid);
    ctx.closePath();
    ctx.fillStyle = KEY;
    ctx.fill();
    ctx.strokeStyle = WELL;
    ctx.lineWidth = h >= 20 ? 1.5 : 1;
    ctx.stroke();
  });
}

/** One lane per object (spec §9): the active lane tall, the names in a gutter on the left, the playhead across all. */
export function Timeline({ states, lanes, activeId, frame, thumb, range = null, onSeek, onRange, onSelect }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [fontsReady, setFontsReady] = useState(false);
  const [scroll, setScroll] = useState(0);
  const anchor = useRef<number | null>(null); // first frame of a Shift-drag in progress

  const idsKey = lanes.map((l) => l.id).join(",");
  const layout = useMemo(
    () => layoutLanes(lanes.length === 0 ? [null] : lanes.map((l) => l.id), activeId),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- idsKey stands for the lane ids, all layoutLanes reads
    [idsKey, activeId],
  );
  const height = TOP + layout.view + RULER;

  // keep the active lane in view whenever it (or the set of lanes) changes; the wheel scrolls freely in between
  useEffect(() => setScroll((s) => scrollToShow(s, layout, activeId)), [layout, activeId]);

  useEffect(() => {
    const el = wrap.current;
    if (el === null) return;
    const observer = new ResizeObserver(() => setWidth(el.clientWidth));
    observer.observe(el);
    setWidth(el.clientWidth);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    // on the window, because a Shift-drag usually ends with the pointer off this narrow strip
    const release = () => {
      anchor.current = null;
    };
    window.addEventListener("mouseup", release);
    return () => window.removeEventListener("mouseup", release);
  }, []);

  useEffect(() => {
    let closed = false;
    void document.fonts?.ready.then(() => {
      if (!closed) setFontsReady(true); // the ruler labels were drawn in the fallback face until now
    });
    return () => {
      closed = true;
    };
  }, []);

  useEffect(() => {
    const c = canvas.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (states.length === 0) return;
    const strip = Math.max(0, width - GUTTER);
    const px = strip / states.length;
    const view = layout.view;

    // the lanes, clipped to the lane viewport (it scrolls once there are more than six)
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, TOP, strip, view);
    ctx.clip();
    for (const box of layout.boxes) {
      const y = TOP + box.top - scroll;
      if (y + box.height <= TOP || y >= TOP + view) continue;
      const lane = box.id === null ? undefined : lanes.find((l) => l.id === box.id);
      ctx.globalAlpha = lane?.hidden ? HIDDEN_ALPHA : 1;
      drawLane(ctx, lane?.states ?? states, y, box.height, lane?.color ?? DIM, px);
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    if (layout.content > view) {
      // a thin scrollbar: where the view sits in the lane stack
      ctx.fillStyle = DIM;
      ctx.fillRect(width - 3, TOP + (scroll * view) / layout.content, 3, (view * view) / layout.content);
    }

    // ruler
    const rulerTop = TOP + view;
    const step = STEPS.find((s) => s * px >= 56) ?? STEPS[STEPS.length - 1];
    ctx.fillStyle = DIM;
    ctx.font = '500 12px "Barlow Semi Condensed", sans-serif';
    ctx.textBaseline = "alphabetic";
    for (let i = 0; i < states.length; i += step) {
      const x = GUTTER + i * px + px / 2;
      ctx.fillRect(Math.round(x), rulerTop + 1, 1, 3);
      ctx.textAlign = i === 0 ? "left" : "center";
      ctx.fillText(String(i), i === 0 ? GUTTER : x, height - 3);
    }

    if (range !== null && range[0] <= range[1]) {
      // the marked range, across every lane: a light wash with a bracket at each end
      const x0 = GUTTER + range[0] * px;
      const x1 = GUTTER + (range[1] + 1) * px;
      ctx.fillStyle = "rgba(233, 231, 226, 0.3)";
      ctx.fillRect(x0, TOP, x1 - x0, view);
      ctx.fillStyle = INK;
      ctx.fillRect(x0, TOP - 4, x1 - x0, 2);
      ctx.fillRect(x0, TOP - 4, 2, view + 8);
      ctx.fillRect(x1 - 2, TOP - 4, 2, view + 8);
      ctx.fillRect(x0, TOP + view + 2, x1 - x0, 2);
    }

    if (hover !== null && hover !== frame) {
      ctx.fillStyle = "rgba(233, 231, 226, 0.25)";
      ctx.fillRect(GUTTER + hover * px, TOP, Math.max(px, 1), view);
    }

    // playhead across every lane: a flag and a light line. Where frames are wide enough the line stops at the lanes
    // and an outline takes over, so it never hides a keyframe diamond on the current frame.
    const cx = GUTTER + frame * px + px / 2;
    ctx.fillStyle = INK;
    if (px >= 14) {
      ctx.fillRect(cx - 1, 0, 2, TOP);
      ctx.fillRect(cx - 1, TOP + view, 2, 4);
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.strokeRect(GUTTER + frame * px + 1, TOP + 1, px - 2, view - 2);
    } else {
      ctx.fillRect(cx - 1, 0, 2, TOP + view + 4);
    }
    ctx.beginPath();
    ctx.moveTo(cx - 5, 0);
    ctx.lineTo(cx + 5, 0);
    ctx.lineTo(cx, 6);
    ctx.closePath();
    ctx.fill();
  }, [states, lanes, layout, scroll, frame, width, height, hover, range, fontsReady]);

  const point = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    return { x, y: e.clientY - rect.top, frame: frameAtX(x - GUTTER, rect.width - GUTTER, states.length) };
  };

  const px = states.length > 0 ? Math.max(0, width - GUTTER) / states.length : 0;
  const tipW = Math.min(THUMB_W, width);
  const tipLeft = hover === null ? 0 : Math.max(0, Math.min(width - tipW, GUTTER + hover * px + px / 2 - tipW / 2));

  return (
    <div
      ref={wrap}
      className="relative w-full select-none"
      onWheel={(e) => {
        if (layout.content > layout.view) setScroll((s) => clampScroll(s + e.deltaY, layout));
      }}
    >
      <canvas
        ref={canvas}
        aria-label="Timeline"
        className="block w-full cursor-pointer"
        style={{ height }}
        onMouseDown={(e) => {
          if (e.button !== 0) return;
          const p = point(e);
          anchor.current = null; // never inherit a drag whose release this strip did not see
          if (p.x < GUTTER) return; // the names in the gutter are buttons of their own
          if (e.shiftKey && onRange !== undefined) {
            anchor.current = p.frame;
            onRange([p.frame, p.frame]);
            return;
          }
          const inLanes = p.y >= TOP && p.y < TOP + layout.view;
          const box = inLanes ? laneAt(layout.boxes, p.y - TOP + scroll) : null;
          if (box !== null && box.id !== null && box.id !== activeId) onSelect?.(box.id);
          onSeek(p.frame);
        }}
        onMouseMove={(e) => {
          const p = point(e);
          setHover(p.x < GUTTER ? null : p.frame);
          if (!(e.buttons & 1)) anchor.current = null; // the button was released outside the strip
          else if (anchor.current !== null) onRange?.([Math.min(anchor.current, p.frame), Math.max(anchor.current, p.frame)]);
          else onSeek(p.frame);
        }}
        onMouseLeave={() => setHover(null)}
      />
      <div className="absolute left-0 overflow-hidden" style={{ top: TOP, width: GUTTER, height: layout.view }}>
        {layout.boxes.map((b) => {
          const lane = b.id === null ? undefined : lanes.find((l) => l.id === b.id);
          const style = { top: b.top - scroll, height: b.height, opacity: lane?.hidden ? HIDDEN_ALPHA : 1 };
          if (lane === undefined) {
            return (
              <span key="none" className="absolute left-0 flex w-full items-center truncate pr-1 font-cond text-xs text-dim" style={style}>
                {NO_OBJECTS}
              </span>
            );
          }
          const active = lane.id === activeId;
          return (
            // not in the tab order: O / Shift+O and the Objects panel select objects from the keyboard
            // every name at text-xs (never shrunk); the active one told apart by colour and aria-pressed
            <button key={lane.id} type="button" tabIndex={-1} aria-pressed={active} title={lane.label}
              disabled={onSelect === undefined}
              className={`absolute left-0 flex w-full items-center truncate pr-1 text-left font-cond text-xs leading-none ${active ? "text-ink" : onSelect === undefined ? "text-dim" : "text-dim hover:text-ink"}`}
              style={style} onClick={() => onSelect?.(lane.id)}>
              {lane.label}
            </button>
          );
        })}
      </div>
      {hover !== null && states.length > 0 && (
        <div className="pointer-events-none absolute z-10 overflow-hidden rounded border border-line bg-well shadow-lg"
          style={{ left: tipLeft, bottom: height + 6, width: tipW }}>
          {thumb !== undefined && (
            <img src={thumb.url(hover)} alt="" className="block w-full bg-well object-contain" style={{ aspectRatio: thumb.aspect }} />
          )}
          <p className="flex justify-between px-1.5 py-0.5 font-cond text-xs">
            <span>{`frame ${hover}`}</span>
            {lanes.length === 0 && <span className="text-dim">{WORDS[states[hover] ?? "empty"]}</span>}
          </p>
          {lanes.length > 0 && (
            <ul className="px-1.5 pb-0.5 font-cond text-xs">
              {lanes.map((l) => (
                <li key={l.id} className="flex justify-between gap-2">
                  <span className="truncate">{l.label}</span>
                  <span className="text-dim">{WORDS[l.states[hover] ?? "empty"]}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
```

`web/src/pages/Label.tsx`:
- objects import: add `lanesFor`.
- after the prefetch effect (still before the early returns — it is a hook) add:
  `const lanes = useMemo(() => (detail === null ? [] : lanesFor(detail, hidden)), [detail, hidden]);`
- replace the `<Timeline ... />` element with:

```tsx
        <Timeline
          states={detail.states}
          lanes={lanes}
          activeId={activeId}
          frame={frame}
          thumb={{ url: (i) => thumbUrl(clipId, i), aspect: width / height }}
          range={range}
          onSeek={lockReason === null ? seek : () => {}}
          onRange={setRange}
          onSelect={lockReason === null ? select : undefined}
        />
```

- [ ] **Step 4: Run the tests**

Run: `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/lanes.ts web/src/lib/lanes.test.ts web/src/lib/objects.ts web/src/lib/objects.test.ts web/src/components/Timeline.tsx web/src/components/Timeline.test.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git commit -m "Timeline: one lane per object, a name gutter, lane selection, scrolling past six lanes" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 6: Propagate "All objects / Selected" and Clear "Active object / All objects"

**Files:**
- Modify: `web/src/components/PropagatePanel.tsx`, `web/src/components/CleanupPanel.tsx`, `web/src/pages/Label.tsx`
- Create: `web/src/components/PropagatePanel.test.tsx`, `web/src/components/CleanupPanel.test.tsx`
- Test: `web/src/pages/Label.test.tsx`

**Interfaces:**
- Consumes: `api.propagate(..., objects)`, `api.clearRange(..., objects)` (Task 1); `activeLane`, `objects`
  (Task 3); job result fields `skipped: number[]`, `lost: {"<id>": frame}`, `lost_at` (server).
- Produces:
  - `PropagatePanel`: `export type PropagateScope = "all" | "selected"`; new props `scope: PropagateScope`,
    `selectedLabel: string | null`, `objectName: (id: number) => string`, `onScope: (scope: PropagateScope) => void`,
    `ownResult: boolean` (the job was started from this page: only then are its object ids this clip's, so only then
    are the skipped and lost objects named; otherwise the panel shows `Lost an object at frame ${lost_at}.`).
  - `CleanupPanel`: `export type ClearScope = "active" | "all"`; new props `lanes: FrameState[][]` (the lanes the
    clear acts on), `scope: ClearScope`, `activeLabel: string | null`, `onScope: (scope: ClearScope) => void`.
  - `Label`: propagate sends `"all"` or `[activeId]`; clear-range sends `[activeId]` or `"all"`; the propagate-blocked
    reason checks the active lane for "Selected" and any object for "All objects".

- [ ] **Step 1: Write the failing tests**

Create `web/src/components/PropagatePanel.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { JobHandle } from "../hooks/useJob";
import { idleJob } from "../lib/jobReducer";
import { PropagatePanel } from "./PropagatePanel";

type Props = ComponentProps<typeof PropagatePanel>;
const job = (over: Partial<JobHandle> = {}): JobHandle => ({ ...idleJob, transport: "ws", ...over });
const props = (over: Partial<Props> = {}): Props => ({
  job: job(), running: false, foreign: false, starting: false, blockedReason: null, follow: true, error: null,
  scope: "all", selectedLabel: "#2 cup", objectName: (id) => (id === 2 ? "#2 cup" : `#${id}`), ownResult: true,
  onScope: vi.fn(), onStart: vi.fn(), onStop: vi.fn(), onFollow: vi.fn(), ...over,
});
const group = () => within(screen.getByRole("group", { name: "Objects to propagate" }));

afterEach(cleanup);

describe("PropagatePanel", () => {
  it("offers All objects and the selected object", () => {
    const p = props();
    render(<PropagatePanel {...p} />);
    expect((group().getByRole("radio", { name: "All objects" }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(group().getByRole("radio", { name: "Selected: #2 cup" }));
    expect(p.onScope).toHaveBeenCalledWith("selected");
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    expect(p.onStart).toHaveBeenCalledWith("forward");
  });
  it("disables Selected on a New object row, with the reason, and shows All objects chosen", () => {
    render(<PropagatePanel {...props({ scope: "selected", selectedLabel: null })} />);
    const selected = group().getByRole("radio", { name: "Selected object" }) as HTMLInputElement;
    expect(selected.disabled).toBe(true);
    expect((group().getByRole("radio", { name: "All objects" }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getByText(/A new object has no mask yet/)).toBeTruthy();
  });
  it("names the skipped objects and each lost object when the run ends", () => {
    const result = { processed: 20, written: 30, skipped: [3], lost: { "2": 33, "5": 12 }, lost_at: 12 };
    render(<PropagatePanel {...props({ job: job({ status: "done", result }) })} />);
    expect(screen.getByText("Skipped #3: no mask on the start frame.")).toBeTruthy();
    const lost = screen.getAllByText(/^Lost /).map((el) => el.textContent);
    expect(lost).toEqual([
      "Lost #5 at frame 12. Correct it there and propagate again.",
      "Lost #2 cup at frame 33. Correct it there and propagate again.",
    ]);
  });
  it("names no object of a run this page did not start: its ids are another clip's", () => {
    const result = { processed: 20, written: 30, skipped: [3], lost: { "2": 33 }, lost_at: 33 };
    render(<PropagatePanel {...props({ ownResult: false, job: job({ status: "done", result }) })} />);
    expect(screen.queryByText(/^Skipped/)).toBeNull();
    expect(screen.getAllByText(/^Lost /).map((el) => el.textContent)).toEqual(["Lost an object at frame 33."]);
  });
});
```

Create `web/src/components/CleanupPanel.test.tsx`:

```tsx
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FrameState } from "../api/types";
import { CleanupPanel } from "./CleanupPanel";

const LETTER: Record<string, FrameState> = { e: "empty", m: "masked", k: "keyframe", d: "deleted" };
const S = (s: string): FrameState[] => s.split("").map((c) => LETTER[c]);
type Props = ComponentProps<typeof CleanupPanel>;
const props = (over: Partial<Props> = {}): Props => ({
  frame: 0, states: S("kmmde"), lanes: [S("kmmde")], scope: "active", activeLabel: "#1 cup", range: [0, 4],
  clearBlockedReason: null, lockReason: null, error: null, onScope: vi.fn(), onRange: vi.fn(), onToggleDelete: vi.fn(),
  onPrune: vi.fn(), onClearRange: vi.fn(), ...over,
});
const group = () => within(screen.getByRole("group", { name: "Objects to clear" }));

afterEach(cleanup);

describe("CleanupPanel", () => {
  it("clears the active object by default and offers every object", () => {
    const p = props();
    render(<CleanupPanel {...p} />);
    expect((group().getByRole("radio", { name: "Active object (#1 cup)" }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(group().getByRole("radio", { name: "All objects" }));
    expect(p.onScope).toHaveBeenCalledWith("all");
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frames 0 to 4" }));
    expect(p.onClearRange).toHaveBeenCalledWith(0, 4);
  });
  it("counts the masks and keyframes of every lane it would clear", () => {
    render(<CleanupPanel {...props({ scope: "all", lanes: [S("kmmde"), S("emkde")] })} />);
    expect(screen.getByText("Removes 3 propagated masks, 2 keyframes and any mask under the 1 deleted frame.")).toBeTruthy();
  });
  it("has nothing to clear for a New object row", () => {
    render(<CleanupPanel {...props({ lanes: [], activeLabel: null })} />);
    expect(group().getByRole("radio", { name: "Active object" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Clear masks in frames 0 to 4" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("There is nothing to clear in frames 0 to 4.")).toBeTruthy();
  });
});
```

In `web/src/pages/Label.test.tsx`:
- "propagates from a keyframe, follows the streamed frames...": expect
  `api.propagate` `toHaveBeenCalledWith("c1", 0, "forward", "all")`; the `done` event becomes
  `{ seq: 2, type: "done", processed: 39, written: 38, lost_at: 33, lost: { "1": 33 }, skipped: [], cancelled: false }`;
  replace `/lost the object at frame 33/i` with `/lost #1 connector at frame 33/i`.
- "marks a range with [ and ]...": expect `api.clearRange` `toHaveBeenCalledWith("c1", 1, 11, [1])`.
- "prunes after confirmation and clears a range": expect `toHaveBeenCalledWith("c1", 5, 9, [1])`.
- add to `describe("Label page (objects)", ...)`:

```ts
  it("propagates the selected object alone when chosen", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    vi.mocked(api.propagate).mockResolvedValue({ job_id: "j3" });
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(within(screen.getByRole("group", { name: "Objects to propagate" })).getByRole("radio", { name: "Selected: #1 connector" }));
    fireEvent.click(screen.getByRole("button", { name: "Forward ▶" }));
    await waitFor(() => expect(api.propagate).toHaveBeenCalledWith("c1", 0, "forward", [1]));
  });

  it("checks the selected object's own mask before propagating it alone, and any object's for All objects", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(row(/^#2 cup/));
    for (let i = 0; i < 10; i++) fireEvent.keyDown(window, { key: "ArrowRight" }); // frame 10: #1 masked, #2 empty
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(within(screen.getByRole("group", { name: "Objects to propagate" })).getByRole("radio", { name: "Selected: #2 cup" }));
    expect((screen.getByRole("button", { name: "Forward ▶" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/#2 cup has no mask on this frame/)).toBeTruthy();
  });

  it("clears the active object's masks by default, or every object's when chosen", async () => {
    vi.mocked(api.getClip).mockResolvedValue(two);
    vi.mocked(api.clearRange).mockResolvedValue(two);
    renderLabel();
    await screen.findByText("bench.mp4");
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frame 0" }));
    await waitFor(() => expect(api.clearRange).toHaveBeenLastCalledWith("c1", 0, 0, [1]));
    fireEvent.click(within(screen.getByRole("group", { name: "Objects to clear" })).getByRole("radio", { name: "All objects" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear masks in frame 0" }));
    await waitFor(() => expect(api.clearRange).toHaveBeenLastCalledWith("c1", 0, 0, "all"));
  });
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd web && node node_modules/.bin/vitest run src/components/PropagatePanel.test.tsx src/components/CleanupPanel.test.tsx src/pages/Label.test.tsx`
Expected: FAIL — no scope groups, no lost/skipped notes, calls without `objects`.

- [ ] **Step 3: Implement**

Replace `web/src/components/PropagatePanel.tsx` with:

```tsx
import type { Direction } from "../api/types";
import type { JobHandle } from "../hooks/useJob";

/** "All objects" (every object with a mask on the start frame) or the selected object alone. */
export type PropagateScope = "all" | "selected";

type Props = {
  job: JobHandle;
  running: boolean;
  /** true while `job` is a propagation this page did not start (attached from a 409 naming another clip's job) */
  foreign: boolean;
  /** true while a start request (including its interactive retries) is in flight and has no job yet to show */
  starting: boolean;
  /** why propagation cannot start from here, or null */
  blockedReason: string | null;
  follow: boolean;
  error: string | null;
  scope: PropagateScope;
  /** "#2 cup" for the active object; null on a "New object" row, where only "All objects" is possible */
  selectedLabel: string | null;
  /** how an object id in the job result is named ("#2 cup", or "#2" once it is gone) */
  objectName: (id: number) => string;
  onScope: (scope: PropagateScope) => void;
  /** the job was started from this page, so the ids in its result are this clip's objects and can be named */
  ownResult: boolean;
  onStart: (direction: Direction) => void;
  onStop: () => void;
  onFollow: (follow: boolean) => void;
};

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
const ids = (v: unknown): number[] => (Array.isArray(v) ? v.filter((x): x is number => typeof x === "number") : []);
/** the result's `lost` (object id, a string key in JSON → the frame it was lost on), nearest frame first */
const lostList = (v: unknown): [number, number][] => {
  if (v === null || typeof v !== "object") return [];
  return Object.entries(v as Record<string, unknown>)
    .filter((e): e is [string, number] => typeof e[1] === "number")
    .map(([id, f]): [number, number] => [Number(id), f])
    .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
};

export function PropagatePanel(p: Props) {
  const { job, running, foreign, starting, blockedReason, follow, error, selectedLabel } = p;
  const blocked = running || starting || blockedReason !== null;
  const scope: PropagateScope = selectedLabel === null ? "all" : p.scope;
  // another page's job names another clip's objects: say only where it lost one
  const skipped = p.ownResult ? ids(job.result?.skipped) : [];
  const lost = p.ownResult ? lostList(job.result?.lost) : [];
  const lostAt = p.ownResult ? null : num(job.result?.lost_at);
  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Propagate</h2>
      <fieldset className="flex flex-col gap-1" disabled={running || starting}>
        <legend className="sr-only">Objects to propagate</legend>
        <label className="flex items-center gap-1.5 text-xs">
          <input type="radio" name="propagate-scope" checked={scope === "all"} onChange={() => p.onScope("all")} />
          All objects
        </label>
        <label className="flex items-center gap-1.5 text-xs">
          <input type="radio" name="propagate-scope" checked={scope === "selected"} disabled={selectedLabel === null}
            onChange={() => p.onScope("selected")} />
          {selectedLabel === null ? "Selected object" : `Selected: ${selectedLabel}`}
        </label>
        {selectedLabel === null && <p className="note">A new object has no mask yet: select an object to propagate it alone.</p>}
      </fieldset>
      <div className="flex gap-1">
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => p.onStart("backward")}>◀ Backward</button>
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => p.onStart("both")}>Both</button>
        <button type="button" className="btn flex-1" disabled={blocked} onClick={() => p.onStart("forward")}>Forward ▶</button>
        <button type="button" className="btn btn-danger" disabled={!running} onClick={p.onStop}>Stop</button>
      </div>
      {!running && blockedReason !== null && <p className="note">{blockedReason}</p>}
      <label className="flex items-center gap-1.5 text-xs text-dim">
        <input type="checkbox" checked={follow} onChange={(e) => p.onFollow(e.target.checked)} />
        Follow the propagation
      </label>

      {running && (
        <>
          <p className="note text-ink">
            {foreign
              ? "The GPU is busy with a propagation that was not started from this page. Stop cancels it."
              : "Propagating. Stop keeps the masks written so far."}
          </p>
          <progress value={job.total === null ? undefined : job.done} max={job.total ?? undefined} />
          <p className="note">
            {job.done} of {job.total ?? "?"} frames{job.transport === "poll" ? ". Live updates dropped; checking every second." : ""}
          </p>
        </>
      )}
      {job.logs.length > 0 && <p className="note">{job.logs[job.logs.length - 1]}</p>}
      {job.status === "done" && (
        <p className="note text-ink">
          Done: {num(job.result?.processed) ?? 0} frames processed, {num(job.result?.written) ?? 0} masks written.
        </p>
      )}
      {job.status === "cancelled" && <p className="note text-ink">Stopped. The masks written so far were kept.</p>}
      {skipped.length > 0 && <p className="note">Skipped {skipped.map(p.objectName).join(", ")}: no mask on the start frame.</p>}
      {lost.map(([id, f]) => (
        <p key={id} className="note text-ink">
          Lost {p.objectName(id)} at frame {f}. Correct it there and propagate again.
        </p>
      ))}
      {lostAt !== null && <p className="note text-ink">Lost an object at frame {lostAt}.</p>}
      {job.status === "error" && <p role="alert" className="text-xs text-danger">{job.error}</p>}
      {error !== null && <p role="alert" className="text-xs text-danger">{error}</p>}
    </section>
  );
}
```

`web/src/components/CleanupPanel.tsx`:
- after `export type Range = [number, number];` add
  `/** what "Clear masks" removes: the active object's masks (default) or every object's */`
  `export type ClearScope = "active" | "all";`
- in `Props` add after `states`:

```ts
  /** the lanes "Clear masks" acts on: [the active object's lane], every object's lane, or none (a New object row) */
  lanes: FrameState[][];
  scope: ClearScope;
  /** "#1 cup"; null on a "New object" row */
  activeLabel: string | null;
  onScope: (scope: ClearScope) => void;
```

  and document `states` as `/** the whole clip's states: Delete / Restore, and the deleted frames a clear covers */`.
- destructure `lanes, scope, activeLabel, onScope` in the signature.
- replace the four lines from `const marked = valid ? states.slice(start, end + 1) : [];` to
  `const deleted = marked.filter((s) => s === "deleted").length;` with:

```ts
  const slice = (lane: FrameState[]) => (valid ? lane.slice(start, end + 1) : []);
  const count = (want: FrameState) => lanes.reduce((n, lane) => n + slice(lane).filter((s) => s === want).length, 0);
  const masks = count("masked");
  const keyframes = count("keyframe");
  // deleted frames are the clip's (deleted in every lane); they only matter when some object is cleared
  const deleted = lanes.length === 0 ? 0 : slice(states).filter((s) => s === "deleted").length;
```

- after the range inputs row (the `<div className="flex items-center gap-1">...</div>`) and before the Clear button
  add:

```tsx
          <fieldset className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            <legend className="sr-only">Objects to clear</legend>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="clear-scope" checked={scope === "active"} onChange={() => onScope("active")} />
              {activeLabel === null ? "Active object" : `Active object (${activeLabel})`}
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="clear-scope" checked={scope === "all"} onChange={() => onScope("all")} />
              All objects
            </label>
          </fieldset>
```

`web/src/pages/Label.tsx`:
- imports: `import { CleanupPanel, type ClearScope, type Range } from "../components/CleanupPanel";`,
  `import { PropagatePanel, type PropagateScope } from "../components/PropagatePanel";`, add `ObjectSelection` to the
  `../api/types` import, and `sortedObjects` to the objects import.
- after `const [mode, setMode] = ...` add:

```ts
  const [propagateScope, setPropagateScope] = useState<PropagateScope>("all");
  const [clearScope, setClearScope] = useState<ClearScope>("active");
```

- after the `activeLane` line add
  `const scope: PropagateScope = activeId === null ? "all" : propagateScope; // "Selected" needs an existing object`
- in `startPropagation` replace `retryInteractive(() => api.propagate(clipId, frame, direction)).then(` with:

```ts
      const which: ObjectSelection = scope === "selected" && activeId !== null ? [activeId] : "all";
      retryInteractive(() => api.propagate(clipId, frame, direction, which)).then(
```

  and its dependency list becomes `[clipId, frame, attach, starting, scope, activeId]`.
- replace `clearRange` with:

```ts
  const clearRange = useCallback(
    (start: number, end: number) => {
      const which: ObjectSelection | null = clearScope === "all" ? "all" : activeId === null ? null : [activeId];
      if (which === null) return; // a new object has no masks yet
      // the mark goes only once the masks are gone, so a failed request can be retried without marking again
      const cleared = api.clearRange(clipId, start, end, which).then((next) => {
        setRange(null);
        return next;
      });
      cleanup(cleared, true);
    },
    [cleanup, clipId, clearScope, activeId],
  );
```

- replace the `propagateBlocked` declaration (after the early returns) with:

```ts
  // "Selected" needs the active object's own mask here; "All objects" needs any object's (the server skips the rest)
  const fromState = scope === "selected" ? (activeLane[frame] ?? "empty") : state;
  const propagateBlocked = !engineReady(health)
    ? "The SAM3 engine is not loaded."
    : state === "deleted"
      ? "This frame is deleted. Restore it, or go to a masked frame, to propagate from it."
      : !isLabeled(fromState)
        ? scope === "selected" && objects.active !== null
          ? `${objectLabel(detail, objects.active)} has no mask on this frame. Go to a frame where it has one, or prompt here, to propagate it.`
          : "This frame has no mask. Go to a masked frame, or prompt here, to propagate from it."
        : null;
  const objectName = (id: number) => {
    const o = detail.clip.objects.find((x) => x.id === id);
    return o === undefined ? `#${id}` : objectLabel(detail, o);
  };
  const clearLanes =
    clearScope === "all" ? sortedObjects(detail).map((o) => laneOf(detail, o.id)) : activeId === null ? [] : [activeLane];
  const activeLabel = objects.active === null ? null : objectLabel(detail, objects.active);
```

- `<PropagatePanel ...>` gains `scope={scope} selectedLabel={activeLabel} objectName={objectName} onScope={setPropagateScope}`
  and `ownResult={ownJob.current}`.
- `<CleanupPanel ...>` gains `lanes={clearLanes} scope={clearScope} activeLabel={activeLabel} onScope={setClearScope}`.

- [ ] **Step 4: Run the tests**

Run: `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/PropagatePanel.tsx web/src/components/PropagatePanel.test.tsx web/src/components/CleanupPanel.tsx web/src/components/CleanupPanel.test.tsx web/src/pages/Label.tsx web/src/pages/Label.test.tsx
git commit -m "Label page: propagate all objects or the selected one; clear the active object or all" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 7: Server — remove the legacy shims

**Files:**
- Modify: `server/vsp/api/labeling.py`, `server/vsp/api/clips.py`, `server/vsp/labeling/propagate.py`,
  `scripts/seed_demo.py`, `README.md` (the "Several objects per clip (server API)" section only)
- Test: `server/tests/test_api_labeling.py`, `server/tests/test_api_export.py`, `server/tests/test_propagate.py`,
  `server/tests/test_api_core.py` (a comment only)

**Interfaces:**
- Produces (HTTP):
  - `POST /prompt`, `/accept`, `/mask`: `object_id` is **required** and nullable (`int | None` with no default): a
    body without the key is a 422. `null` = a new object (`/accept`, `/mask`: `class_id` required; `/prompt`: no box
    derived from an existing mask); an id targets that object. Responses unchanged.
  - `GET /api/clips/{id}/masks/{idx}` no longer exists (404).
  - Propagation `frame` events are `{type: "frame", frame, objects: [{id, has_mask}], done, total}` — no top-level
    `has_mask`.
  - `grep -rn "legacy shim" server` prints nothing.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_api_labeling.py`: delete `test_legacy_bodies_without_object_id_keep_one_object` and
`test_legacy_accept_with_unknown_class_on_an_existing_object_writes_nothing`, and add:

```python
def test_bodies_must_name_the_object(tmp_path):
    """object_id is required (null = a new object): a body without it is a 422 that writes nothing, and the old
    first-object mask route is gone."""
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        ok = client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        assert ok.status_code == 200
        assert client.post(f"/api/clips/{cid}/prompt", json={"frame": 0, "prompt": CLICK}).status_code == 422
        assert client.post(f"/api/clips/{cid}/accept", json={"frame": 3, "prompt": CLICK}).status_code == 422
        info = client.get(f"/api/clips/{cid}").json()["clip"]
        m = np.zeros((info["height"], info["width"]), dtype=bool)
        m[5:15, 5:25] = True
        b64 = base64.b64encode(encode_mask_png(m)).decode("ascii")
        assert client.post(f"/api/clips/{cid}/mask", json={"frame": 3, "mask_png_b64": b64}).status_code == 422
        assert client.get(f"/api/clips/{cid}").json()["states"][3] == "empty"  # nothing was written
        assert client.get(f"/api/clips/{cid}/masks/0").status_code == 404
        assert client.get(f"/api/clips/{cid}/objects/1/masks/0").status_code == 200


def test_new_object_accept_with_an_unknown_class_writes_nothing(tmp_path):
    with _client(tmp_path, FakeEngine()) as client:
        cid = _clip(client, tmp_path)
        client.post(f"/api/clips/{cid}/accept", json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})
        before = client.get(f"/api/clips/{cid}").json()
        bad = client.post(f"/api/clips/{cid}/accept", json={"frame": 3, "prompt": CLICK, "object_id": None, "class_id": 9})
        assert bad.status_code == 400
        after = client.get(f"/api/clips/{cid}").json()
        assert after["states"] == before["states"] and after["clip"]["objects"] == before["clip"]["objects"]
```

In `test_propagate_objects_body_and_events` replace the last line with
`assert all("objects" in e and "has_mask" not in e for e in frame_events)`.

In `server/tests/test_propagate.py` (`test_all_objects_share_one_engine_call_per_chunk`) replace
`assert frame_events[0]["has_mask"] is True and frame_events[-1]["done"] == 29` with
`assert "has_mask" not in frame_events[0] and frame_events[-1]["done"] == 29`.

Every request that still relies on the absent-key behaviour must now name its object. In
`server/tests/test_api_labeling.py` make these exact replacements (the `replace_all` ones in every test of the file):

| Before | After |
|---|---|
| `json={"frame": 5, "prompt": CLICK}).json()["candidates"]` | `json={"frame": 5, "prompt": CLICK, "object_id": None}).json()["candidates"]` |
| `json={"frame": 5, "prompt": CLICK, "class_id": 0})` | `json={"frame": 5, "prompt": CLICK, "object_id": None, "class_id": 0})` |
| `json={"frame": 12, "prompt": {"type": "clicks", "points": [[44, 30, 1], [40, 22, 0]]}},` | `json={"frame": 12, "prompt": {"type": "clicks", "points": [[44, 30, 1], [40, 22, 0]]}, "object_id": 1},` |
| (replace all) `json={"frame": 0, "prompt": CLICK, "class_id": 0})` | `json={"frame": 0, "prompt": CLICK, "object_id": None, "class_id": 0})` |
| `json={"frame": 1, "prompt": CLICK, "class_id": 9})` | `json={"frame": 1, "prompt": CLICK, "object_id": None, "class_id": 9})` |
| `json={"frame": 0, "prompt": tiny})` | `json={"frame": 0, "prompt": tiny, "object_id": None})` |
| `json={"frame": 1, "prompt": tiny})` | `json={"frame": 1, "prompt": tiny, "object_id": 1})` |
| `json={"frame": 0, "prompt": {"type": "clicks"}})` | `json={"frame": 0, "prompt": {"type": "clicks"}, "object_id": None})` |
| (replace all) `json={"frame": 0, "prompt": CLICK})` | `json={"frame": 0, "prompt": CLICK, "object_id": None})` |
| `json={"frame": 1, "prompt": CLICK})` | `json={"frame": 1, "prompt": CLICK, "object_id": 1})` |
| `json={"frame": 0, "prompt": inverted})` | `json={"frame": 0, "prompt": inverted, "object_id": None})` |
| `json={"frame": 0, "prompt": bad_point})` | `json={"frame": 0, "prompt": bad_point, "object_id": None})` |
| `json={"frame": 2, "mask_png_b64": b64}).status_code == 400` | `json={"frame": 2, "mask_png_b64": b64, "object_id": None}).status_code == 400` |
| `json={"frame": 2, "mask_png_b64": b64, "class_id": 0})` | `json={"frame": 2, "mask_png_b64": b64, "object_id": None, "class_id": 0})` |
| `json={"frame": 2, "mask_png_b64": "nope"})` | `json={"frame": 2, "mask_png_b64": "nope", "object_id": 1})` |
| `json={"frame": 99, "mask_png_b64": b64})` | `json={"frame": 99, "mask_png_b64": b64, "object_id": 1})` |
| `json={"frame": -1, "mask_png_b64": b64})` | `json={"frame": -1, "mask_png_b64": b64, "object_id": 1})` |
| `json={"frame": 3, "mask_png_b64": b64})` (the busy one in the paint test) | `json={"frame": 3, "mask_png_b64": b64, "object_id": 1})` |

(The last row must not touch the new `test_bodies_must_name_the_object`, which deliberately omits the key: apply it
before adding that test, or edit the paint test's line by hand.)

In `server/tests/test_api_export.py` add `"object_id": None, ` before `"class_id": 0` in both `/accept` calls (lines
~61 and ~125).

In `server/tests/test_api_core.py` (~line 115) replace the comment above the `/masks/0` 404 assertion,
`# legacy shim (removed in plan B): a clip with no objects also 404s on the un-scoped mask route`, with
`# the un-scoped mask route no longer exists` (the assertion stays), so `grep -rn "legacy shim" server` prints nothing.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest server/tests/test_api_labeling.py server/tests/test_propagate.py -q`
Expected: FAIL — bodies without `object_id` are still accepted (200), `/masks/0` still answers 200, and frame events
still carry `has_mask`.

- [ ] **Step 3: Implement**

`server/vsp/api/labeling.py`:
- import line: `from ..labeling.objects import delete_object, find_object, relabel_object` (drop only `check_class`,
  which only the shims used; `find_object` keeps `/prompt`'s 404 for an unknown object).
- replace `PromptBody` and `MaskBody` with:

```python
class PromptBody(BaseModel):
    frame: int = Field(ge=0)
    prompt: PromptSpec
    # required, null allowed: null = a new object (no click box derived from an existing mask); an id = that object
    object_id: int | None


class MaskBody(BaseModel):
    frame: int = Field(ge=0)
    mask_png_b64: str
    object_id: int | None  # required, null allowed: null = a new object, which needs class_id
    class_id: int | None = None
```

- replace the three route functions `prompt_preview`, `prompt_accept` and `paint_mask` with:

```python
@router.post("/prompt")
def prompt_preview(clip_id: str, body: PromptBody, state: State) -> dict:
    clip = state.store.read_clip(clip_id)
    engine = require_engine(state)
    with gpu_lease(state):
        if body.object_id is not None:
            find_object(clip, body.object_id)  # 404 for an unknown object, before running any inference
        detections = preview(state.store, engine, clip, body.frame, body.prompt, object_id=body.object_id)
    return {"candidates": [candidate_view(d) for d in detections]}


@router.post("/accept")
def prompt_accept(clip_id: str, body: AcceptBody, state: State) -> dict:
    state.store.read_clip(clip_id)  # 404 before requiring the engine, same order as every other route here
    engine = require_engine(state)
    with gpu_lease(state):
        clip, ids = accept(
            state.store, engine, clip_id, body.frame, body.prompt,
            object_id=body.object_id, class_id=body.class_id, candidate=body.candidate, candidates=body.candidates,
        )
        return {**clip_detail(state, clip), "object_ids": ids}


@router.post("/mask")
def paint_mask(clip_id: str, body: MaskBody, state: State) -> dict:
    """A hand-painted mask for one frame of one object (brush spec). The frame becomes a `paint` keyframe."""
    state.store.read_clip(clip_id)
    mask = decode_mask_png(body.mask_png_b64)
    with gpu_lease(state):  # a propagation writes the very masks this replaces
        clip, object_id = set_mask(
            state.store, clip_id, body.frame, mask, object_id=body.object_id, class_id=body.class_id,
        )
        return {**clip_detail(state, clip), "object_id": object_id}
```

`server/vsp/api/clips.py`: delete the `get_mask` route (the whole `@router.get("/api/clips/{clip_id}/masks/{idx}")`
function).

`server/vsp/labeling/propagate.py`: in `_run_direction`'s `emit({...})` for `"frame"` delete the line
`"has_mask": any(r["has_mask"] for r in report),  # legacy shim: removed in plan B`.

`scripts/seed_demo.py`: in the `/accept` call add `"object_id": None, ` before `"class_id": cls["id"]`.

`README.md`, section "### Several objects per clip (server API)":
- replace its first two paragraphs (from "A clip can hold more than one labelled object ..." through "... rather than
  falling back to the first one.") with:

```markdown
A clip can hold more than one labelled object (`clip.objects`, each with its own `id`, `class_id`, keyframes and
`propagated_ranges`; ids are never reused). The web Label page edits them one active object at a time (see "Label
page: several objects" below).

`/prompt`, `/accept` and `/mask` require an `object_id` key (a body without it is a 422): `null` means a new object
(`/accept` and `/mask` then need `class_id`; `/prompt` derives no click box from an existing mask), an id targets
that object.
```

- in the bullet on `/prompt` and `/accept` replace "take an optional `object_id`" with "take the required
  `object_id`"; in the `/mask` bullet replace "takes the same optional `object_id`/`class_id`" with "takes the same
  required `object_id` and optional `class_id`"; at the end of the propagate bullet add "There is no clip-level
  `has_mask`: a frame's overall state follows from the objects' lanes."
- delete the line "The legacy `GET /api/clips/{id}/masks/{idx}` still serves the first object's mask (404 if the clip
  has none)."

- [ ] **Step 4: Run the tests**

Run: `uv run pytest server/tests -q -m "not gpu"` and `uv run ruff check server` — all pass, clean.
`grep -rn "legacy shim" server` and `grep -rn '"/api/clips/{clip_id}/masks' server` print nothing.
Run the web suite too (`cd web && node node_modules/.bin/vitest run`) — unchanged, green.

- [ ] **Step 5: Commit**

```bash
git add server/vsp/api/labeling.py server/vsp/api/clips.py server/vsp/labeling/propagate.py server/tests/test_api_labeling.py server/tests/test_api_export.py server/tests/test_propagate.py server/tests/test_api_core.py scripts/seed_demo.py README.md
git commit -m "API: object_id required on prompt, accept and mask; legacy mask route and frame has_mask removed" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

### Task 8: README (Label page, keys) and manual QA

**Files:** Modify `README.md` (Web GUI part only).

- [ ] **Step 1: Document the Label page.** Insert a section after "### Demo data" and before "### Exporting":

```markdown
### Label page: several objects

A clip can hold any number of objects, each with its own class (two cups and a bowl are three objects). The
**Objects** panel in the right sidebar lists them as `#id class` with the number of frames each has a mask on; the
highlighted row is the **active object**, which every prompt, accept, brush save, "Selected" propagation and
"Active object" clear acts on. Click a row (or its lane name in the timeline, or press `O` / `Shift+O`) to change it.
The eye hides an object from the frame (display only, for this page session). The class select relabels the active
object at once — also during a propagation. **+ New object** (`N`) starts a row "New object (class)": the object is
created by its first accepted or painted mask. A text prompt that finds several candidates on a New object row offers
**Accept all as new objects**, one object per candidate. **Delete object** asks first, naming how many frames lose
their mask; it is refused while a propagation or an export runs.

The frame shows every visible object in its class colour; only the active object gets the Mask outline. The timeline
has one lane per object (the active lane tall), a name gutter on the left, and the playhead across all lanes; past six
lanes the lanes scroll with the mouse wheel. Hovering a frame lists every object's state on it. The Propagate panel
runs **All objects** (every object with a mask on the current frame; the others are named as skipped) or **Selected**
only; lost objects are named one by one. Cleanup's "Clear masks" acts on the **Active object** (default) or **All
objects**. `K`, `G` and "has a mask here" follow the active object's lane.
```

- [ ] **Step 2: Keyboard line.** In "### Keyboard", the second line starts with
  ``next / previous keyframe · `G` next empty frame · ``; replace that text with
  ``next / previous keyframe of the active object · `G` its next empty frame · `O` / `Shift+O` next / previous object · `N` new object · ``
  (the rest of the paragraph is unchanged).

- [ ] **Step 3: Manual QA.** In "### Manual QA checklist", in item 6 replace "and a new prompt on the clip still lands
  on the same object and class." with "and a new prompt lands on the active object with its class.", and append after
  the last item (numbering continues from it; 20 onwards at the time of writing):

```markdown
20. Several objects: on one clip label a cup (#1), press `N`, choose another class and label it (#2). The Objects
    panel lists both with frame counts; the timeline shows two lanes, the active one tall; clicking a lane or its name
    selects that object, and `O` / `Shift+O` cycle. Hide #1 with its eye: it leaves the frame and its lane dims.
    Only the active object carries the Mask outline; brushing #2 keeps #1 visible underneath.
21. Propagate "All objects" from a frame where both have masks: both lanes fill in one run. Then "Selected: #2" from
    another of its keyframes: only lane 2 changes. Let one object leave the frame: the panel names it as lost while
    the other continues; start from a frame where only #1 has a mask with "All objects": #2 is named as skipped.
22. Shift-drag a stretch and clear it with "Active object", then again with "All objects"; check the note counts the
    right masks each time. Relabel #2 during a propagation (allowed); try to delete #1 during a propagation and during
    an export (refused), then delete it (the dialog names its frame count) and check #2 becomes active.
23. Text-prompt "cup" on a New object row with several cups in view: "Accept all as new objects" creates one object
    per candidate. With seven or more objects the timeline stops growing at six lanes, scrolls with the wheel, and
    `O` onto an off-screen object scrolls its lane into view.
```

- [ ] **Step 4: Final verification.** Run `cd web && node node_modules/.bin/vitest run`,
  `node node_modules/.bin/tsc --noEmit`, `node node_modules/.bin/vite build` (the page still builds),
  `uv run pytest server/tests -q -m "not gpu"`, `uv run ruff check server`. Then run the manual QA items 20–23 (and
  the existing 2, 3, 6, 12, 16, 17 that touch the Label page) against real footage with `uv run vsp serve`, and report
  any item that could not be run (for example no GPU).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Docs: Label page with several objects per clip (objects panel, lanes, keys, manual QA)" -m "Co-Authored-By: ..." -m "Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6"
```

---

## Spec coverage (self-review)

| Spec | Where |
|---|---|
| §7 `object_states` consumed, aggregate `states` kept for ClipStats/Clips | Task 1 |
| §7 per-object mask route; old route removed | Tasks 2, 7 |
| §7 `/prompt`, `/accept`, `/mask` with `object_id`; accept-all `candidates`; relabel PATCH; DELETE (409 surfaced) | Tasks 3, 4, 7 |
| §7 `clear-range` / `propagate` `objects` | Tasks 1, 6 |
| §8 Objects panel (swatch, `#id class`, count, eye, active row, class select, New object, Delete with count, Add class) | Task 3 |
| §8 keys `O` / `Shift+O` / `N`; `K`, `G`, `hasMask`, prefetch, propagate-blocked on the active lane | Tasks 3, 2, 6 |
| §8 canvas: all visible objects, active outlined, one composite per frame keyed as specified, 48-frame LRU | Task 2 |
| §8 brush: base from the active object's URL, active left out of the composite, `/mask` with id or null + class | Tasks 2, 4 |
| §8 prompting: `object_id` on `/prompt` and `/accept`, class-sending logic removed, Accept selected / all | Task 4 |
| §8 Propagate All / Selected (disabled with reason on New object), skipped and lost named | Task 6 |
| §8 Cleanup Active / All (default active) | Task 6 |
| §8 `patchFrames` new event shape, per-object lanes, aggregate recompute | Task 1 |
| §8 deleting the active object selects the next (or New object); a maskless object stays listed | Task 3 |
| §9 lanes 16/30/2 px (names text-xs), 72 px gutter, hatching across lanes, 40 % hidden, select + seek, full-height playhead, 6 lanes then scroll keeping the active in view, per-object hover, Shift-drag range, "No objects yet" | Task 5 |
| §13 relabel during propagation; delete 409 during propagation/export; Selected disabled on New object | Tasks 3, 6 |
| §14 plan B tests; README Label page, keys, manual QA | Tasks 1–8 |
