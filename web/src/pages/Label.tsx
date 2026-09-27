import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, INTERACTIVE, api, frameUrl, objectMaskUrl, retryInteractive, thumbUrl } from "../api/rest";
import type { ClipDetail, Direction, JobEvent, ObjectSelection } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { BrushPanel } from "../components/BrushPanel";
import { CleanupPanel, type ClearScope, type Range } from "../components/CleanupPanel";
import { ClipStats } from "../components/ClipStats";
import { FrameCanvas } from "../components/FrameCanvas";
import { ObjectsPanel } from "../components/ObjectsPanel";
import { OverlayPanel } from "../components/OverlayPanel";
import { PromptPanel } from "../components/PromptPanel";
import { PropagatePanel, type PropagateScope } from "../components/PropagatePanel";
import { Timeline } from "../components/Timeline";
import { ToolRail, type RailTool } from "../components/ToolRail";
import { Transport } from "../components/Transport";
import { useBrush } from "../hooks/useBrush";
import { useClip } from "../hooks/useClip";
import { HealthContext, engineReady } from "../hooks/useHealth";
import { useJob } from "../hooks/useJob";
import { useKeyboard, type KeyAction } from "../hooks/useKeyboard";
import { useObjects } from "../hooks/useObjects";
import { usePlayback } from "../hooks/usePlayback";
import { usePrompting } from "../hooks/usePrompting";
import { initialSelection } from "../lib/activeObject";
import { FrameStore } from "../lib/frameCache";
import { isTerminal } from "../lib/jobReducer";
import { hexToRgb } from "../lib/mask";
import { FALLBACK_COLOR, className, classColor, isLabeled, labeledCount, laneOf, lanesFor, maskParts, objectLabel, parseFrameEvent, sortedObjects, type FrameEvent } from "../lib/objects";
import { loadStyle, outlineFor, saveStyle, type OverlayStyle } from "../lib/overlayStyle";
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
  const [overlayStyle, setOverlayStyle] = useState<OverlayStyle>(loadStyle);
  const [fitSignal, setFitSignal] = useState(0);
  const [range, setRange] = useState<Range | null>(null); // frames marked for clearing
  const [jobId, setJobId] = useState<string | null>(() => sessionStorage.getItem(jobKey(clipId)));
  const [follow, setFollow] = useState(true);
  const [propagateError, setPropagateError] = useState<string | null>(null);
  const [cleanupError, setCleanupError] = useState<string | null>(null);
  const [classColorError, setClassColorError] = useState<string | null>(null);
  const [mode, setMode] = useState<"prompt" | "brush">("prompt");
  const [propagateScope, setPropagateScope] = useState<PropagateScope>("all");
  const [clearScope, setClearScope] = useState<ClearScope>("active");

  const detail = clip.detail;
  const frameCount = detail?.clip.frame_count ?? 0;
  const fps = detail?.clip.fps ?? 30;
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });

  const objects = useObjects(clipId, detail);
  const { hidden, select, step: stepObject, startNew, setNewClass, toggleHidden } = objects;
  const activeId = objects.activeId;
  // K, G and every "has a mask here" check follow the active object's lane, not the whole clip's states
  const activeLane = useMemo(() => (detail === null ? [] : laneOf(detail, activeId)), [detail, activeId]);
  const scope: PropagateScope = activeId === null ? "all" : propagateScope; // "Selected" needs an existing object
  const [objectsError, setObjectsError] = useState<string | null>(null);

  // ---- propagation job -------------------------------------------------------------------------------------------
  const { patchFrames, reload, apply, update } = clip;
  // true when this page itself started or resumed (from its own sessionStorage key) the current job; false when it
  // was attached from a 409 naming some other job holding the GPU lock (possibly another clip's) — that job must
  // never patch this clip's frames or move this view, and its lost_at must not be seeked to when it ends.
  const ownJob = useRef(true);
  // F3: a re-attach replays every event from seq 0, so `frame` events can arrive many per macrotask; buffer own-job
  // ones and apply the whole batch (one setFrame, one patchFrames) on the next animation frame instead of per event.
  const frameEventBuffer = useRef<FrameEvent[]>([]);
  /** the highest `done` the playhead has followed in this job: a lagging object's re-emitted earlier frames (same or
   * lower `done`) must not pull the view back */
  const followedDone = useRef(-1);
  const flushHandle = useRef<number | null>(null);
  const followRef = useRef(follow);
  followRef.current = follow;
  const frameCountRef = useRef(frameCount);
  frameCountRef.current = frameCount;

  const flushFrameEvents = useCallback(() => {
    flushHandle.current = null;
    const events = frameEventBuffer.current;
    frameEventBuffer.current = [];
    if (events.length === 0) return;
    patchFrames(events);
    if (followRef.current) {
      // follow the event with the highest `done` (the first one to reach it), not the last event of the batch
      let lead: FrameEvent | null = null;
      for (const e of events) if ((e.done ?? 0) > (lead?.done ?? followedDone.current)) lead = e;
      if (lead !== null) {
        followedDone.current = lead.done ?? 0;
        setFrame(clampFrame(lead.frame, frameCountRef.current));
      }
    }
  }, [patchFrames]);

  const onJobEvent = useCallback(
    (event: JobEvent) => {
      const parsed = parseFrameEvent(event);
      if (parsed === null) return;
      if (!ownJob.current) return; // a job this page did not start must not drive this clip
      frameEventBuffer.current.push(parsed);
      if (flushHandle.current === null) flushHandle.current = requestAnimationFrame(flushFrameEvents);
    },
    [flushFrameEvents],
  );
  const job = useJob(jobId, onJobEvent);
  const running = jobId !== null && !isTerminal(job.status);
  const foreign = running && !ownJob.current;

  // A stale job's still-buffered events must never land on the next job; also cancels on unmount.
  useEffect(() => {
    return () => {
      if (flushHandle.current !== null) cancelAnimationFrame(flushHandle.current);
      flushHandle.current = null;
      frameEventBuffer.current = [];
      followedDone.current = -1;
    };
  }, [jobId]);

  useEffect(() => {
    if (jobId === null || !isTerminal(job.status)) return;
    const own = ownJob.current;
    if (flushHandle.current !== null) {
      // apply the last streamed frames now, so a late flush cannot override the seek to `lost_at` below
      cancelAnimationFrame(flushHandle.current);
      flushFrameEvents();
    }
    if (own) sessionStorage.removeItem(jobKey(clipId));
    const lostAt = own ? job.result?.lost_at : undefined; // a foreign job's lost_at is none of this clip's business
    void reload().then(() => {
      if (typeof lostAt === "number") setFrame(lostAt);
    });
  }, [jobId, job.status, job.result, clipId, reload, flushFrameEvents]);

  useEffect(() => {
    if (!running || job.transport !== "poll") return;
    const id = setInterval(() => void reload(), 1000); // no frame events without the socket
    return () => clearInterval(id);
  }, [running, job.transport, reload]);

  const attach = useCallback(
    (id: string, own: boolean) => {
      ownJob.current = own;
      if (own) sessionStorage.setItem(jobKey(clipId), id);
      setJobId(id);
      setPlaying(false);
    },
    [clipId],
  );

  const [starting, setStarting] = useState(false);
  const startPropagation = useCallback(
    (direction: Direction) => {
      if (starting) return; // a second click while the request (including its interactive retries) is in flight
      setPropagateError(null);
      setStarting(true);
      const which: ObjectSelection = scope === "selected" && activeId !== null ? [activeId] : "all";
      retryInteractive(() => api.propagate(clipId, frame, direction, which)).then(
        (r) => {
          setStarting(false);
          attach(r.job_id, true);
        },
        (e: unknown) => {
          setStarting(false);
          if (e instanceof ApiError && e.status === 409 && e.jobId !== null && e.jobId !== INTERACTIVE) attach(e.jobId, false);
          else setPropagateError(message(e));
        },
      );
    },
    [clipId, frame, attach, starting, scope, activeId],
  );

  const stopPropagation = useCallback(() => {
    if (jobId !== null) api.cancelJob(jobId).catch((e: unknown) => setPropagateError(message(e)));
  }, [jobId]);

  // ---- prompting -------------------------------------------------------------------------------------------------
  const state = detail?.states[frame] ?? "empty";
  const disabledReason = !engineReady(health)
    ? "The SAM3 engine is not loaded."
    : running
      ? "A propagation is running. Stop it to prompt."
      : state === "deleted"
        ? "This frame is deleted. Restore it to label it."
        : playing
          ? "Pause playback to prompt."
          : null;
  const color = detail === null ? FALLBACK_COLOR : classColor(detail, objects.classId);
  // hexToRgb allocates a new array every call: memoised so its identity is stable across renders that don't
  // change `color` (e.g. a health poll), since it feeds effect dependency arrays (useBrush's re-base effect).
  const rgb = useMemo(() => hexToRgb(color), [color]);
  const outline = useMemo(() => outlineFor(overlayStyle, rgb), [overlayStyle, rgb]);
  const changeStyle = useCallback((s: OverlayStyle) => {
    setOverlayStyle(s);
    saveStyle(s);
  }, []);
  const hasMask = (i: number) => isLabeled(activeLane[i]);
  const maskVersion = clip.maskVersion;
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

  const brushReason = running
    ? "A propagation is running. Stop it to paint."
    : state === "deleted"
      ? "This frame is deleted. Restore it to paint on it."
      : playing
        ? "Pause playback to paint."
        : null;
  const brush = useBrush({
    clipId, frame, width: detail?.clip.width ?? 1, height: detail?.clip.height ?? 1, rgb, store,
    maskUrl: activeId !== null && hasMask(frame) ? objectMaskUrl(clipId, activeId, frame, maskVersion(frame)) : null,
    active: mode === "brush" && brushReason === null && detail !== null,
    selection: objects.selection, onCreated: select, apply,
  });
  const lockReason = brush.dirty ? `Save or discard the brush strokes on frame ${frame} first.` : null;
  // while the active object is being edited, its preview or brush layer is drawn in its place
  const editing = (mode === "brush" && brushReason === null) || prompting.candidates.length > 0;

  // `busyJobId` outlives the 409 that produced it (only a later preview clears it), so each id is handled once;
  // otherwise the page would re-attach to that long-finished job every time a job of its own ends.
  const handledBusyJob = useRef<string | null>(null);
  useEffect(() => {
    const busy = prompting.busyJobId;
    if (busy === null || busy === INTERACTIVE || running || handledBusyJob.current === busy) return;
    handledBusyJob.current = busy;
    attach(busy, false);
  }, [prompting.busyJobId, running, attach]);

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

  // ---- navigation and keyboard -----------------------------------------------------------------------------------
  const step = useCallback((delta: number) => setFrame((f) => clampFrame(f + delta, frameCount)), [frameCount]);
  const seek = useCallback((f: number) => setFrame(clampFrame(f, frameCount)), [frameCount]);
  const jump = (target: number | null) => {
    if (target !== null) setFrame(target);
  };

  useKeyboard((action: KeyAction) => {
    if (detail === null) return;
    if (lockReason !== null && !["accept", "clear", "undo", "brushSize", "overlay"].includes(action.type)) return;
    switch (action.type) {
      case "accept":
        if (mode === "brush") void brush.save();
        else if (prompting.canAccept) void prompting.accept();
        break;
      case "clear":
        if (mode === "brush") brush.discard();
        else prompting.clear();
        break;
      case "undo":
        if (mode === "brush") brush.undo();
        else prompting.dispatch({ type: "undo" });
        break;
      case "brushSize":
        if (mode === "brush") brush.grow(action.dir);
        break;
      case "step":
        step(action.delta);
        break;
      case "play":
        if (!running) setPlaying((p) => !p);
        break;
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
      case "delete":
        toggleDelete();
        break;
      case "overlay":
        setOverlay((o) => !o);
        break;
      case "mark":
        // the other edge follows when it would otherwise end up on the wrong side
        setRange((r) =>
          action.edge === "start" ? [frame, Math.max(frame, r?.[1] ?? frame)] : [Math.min(frame, r?.[0] ?? frame), frame],
        );
        break;
    }
  });

  // ---- prefetch --------------------------------------------------------------------------------------------------
  useEffect(() => {
    if (detail === null || running) return; // during a job the view follows the stream; nothing to look ahead for
    const ahead = playing ? 30 : 4;
    for (let i = frame + 1; i <= Math.min(frame + ahead, frameCount - 1); i++) {
      void store.image(frameUrl(clipId, i));
      if (!overlay) continue;
      // the parts FrameCanvas will ask for on that frame, so playback hits the composite cache
      const parts = maskParts(detail, {
        clipId, frame: i, version: maskVersion(i), hidden, activeId, excludeActive: editing, style: overlayStyle,
      });
      if (parts.length > 0) void store.composite(parts);
    }
  }, [store, clipId, frame, playing, overlay, detail, overlayStyle, activeId, editing, frameCount, maskVersion, running, hidden]);

  const lanes = useMemo(() => (detail === null ? [] : lanesFor(detail, hidden)), [detail, hidden]);

  const addClass = useCallback(
    async (name: string) => {
      if (detail === null) return;
      const created = await api.addClass(detail.clip.project_id, name);
      // against the latest detail, not the `detail` captured above: a lane patch from a running propagation may
      // land while this request is in flight, and a stale snapshot here would roll it back
      update((d) => ({ ...d, classes: [...d.classes, created] }));
      if (objects.selection?.kind === "new") setNewClass(created.id); // an existing object is relabelled only on request
    },
    [detail, update, objects.selection, setNewClass],
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
  const saveClassColor = useCallback(
    (color: string) => {
      if (detail === null || objects.classId === null) return;
      setClassColorError(null);
      api
        .setClassColor(detail.clip.project_id, objects.classId, color)
        .then(() => reload())
        .catch((e: unknown) => setClassColorError(message(e)));
    },
    [detail, objects.classId, reload],
  );
  const deleteObject = useCallback(() => {
    const target = objects.active;
    if (detail === null || target === null) return;
    const n = labeledCount(laneOf(detail, target.id));
    if (!window.confirm(`Delete ${objectLabel(detail, target)} and its masks on ${n} ${n === 1 ? "frame" : "frames"}?`)) return;
    setObjectsError(null);
    api.deleteObject(clipId, target.id).then((next) => apply(next, true), (e: unknown) => setObjectsError(message(e)));
  }, [detail, objects.active, clipId, apply]);

  const back = (
    <Link to="/" aria-label="Back to clips" className="flex items-center gap-1 font-cond text-dim hover:text-ink">
      <span aria-hidden>‹</span> Clips
    </Link>
  );

  if (clip.error !== null) {
    return (
      <div className="flex h-full flex-col">
        <AppHeader>{back}</AppHeader>
        <div className="p-4">
          <p role="alert" className="alert">{clip.error}</p>
        </div>
      </div>
    );
  }
  if (detail === null) {
    return (
      <div className="flex h-full flex-col">
        <AppHeader>{back}</AppHeader>
        <p className="p-4 text-sm text-dim">Loading the clip…</p>
      </div>
    );
  }

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
  const { draft, dispatch } = prompting;
  const newClass = className(detail, objects.classId);
  const promptTarget =
    objects.active !== null
      ? `Accept writes to ${objectLabel(detail, objects.active)}.`
      : newClass === null
        ? "Choose a class for the new object in the Objects panel."
        : `Accept creates a new ${newClass} object.`;
  const { source, width, height, stride } = detail.clip;
  const onTool = (t: RailTool) => {
    if (t === "brush") {
      prompting.clear();
      setMode("brush");
    } else {
      setMode("prompt");
      dispatch({ type: "tool", tool: t });
    }
  };

  return (
    <div className="flex h-full flex-col">
      <AppHeader>
        {back}
        <h1 className="truncate font-medium">{source.name}</h1>
        <span className="shrink-0 font-cond text-dim">
          {width}×{height}, {fps} fps{stride > 1 ? `, every ${stride} frames` : ""}
        </span>
      </AppHeader>
      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1">
          <FrameCanvas
            store={store}
            width={width}
            height={height}
            frameUrl={frameUrl(clipId, frame)}
            masks={
              overlay
                ? maskParts(detail, {
                    clipId, frame, version: maskVersion(frame), hidden, activeId, excludeActive: editing,
                    style: overlayStyle,
                  })
                : []
            }
            color={color}
            outline={outline}
            tool={mode === "prompt" && disabledReason === null ? draft.tool : null}
            draft={draft}
            candidates={prompting.candidates}
            onPoint={(x, y, label) => dispatch({ type: "point", x, y, label })}
            onBox={(box) => dispatch({ type: "box", box })}
            onPickCandidate={(index) => dispatch({ type: "candidate", index })}
            fitSignal={fitSignal}
            brush={
              mode === "brush" && brushReason === null
                ? {
                    size: brush.draft.size, layer: brush.layer, baseVersion: brush.baseVersion,
                    onStart: brush.start, onExtend: brush.extend, onEnd: brush.end,
                  }
                : null
            }
          />
        </div>
        {/* every control sits to the right of the frame: the tools first, then the inspector */}
        <ToolRail tool={mode === "brush" ? "brush" : draft.tool} disabled={disabledReason !== null}
          brushDisabled={brushReason !== null || detail === null} locked={lockReason !== null} onTool={onTool}
          onFit={() => setFitSignal((n) => n + 1)} />
        <aside className="flex w-80 shrink-0 flex-col gap-5 overflow-y-auto border-l border-line bg-panel p-3">
          {clip.refreshError !== null && (
            <p role="alert" className="alert text-xs">Could not refresh the clip: {clip.refreshError}</p>
          )}
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
          {mode === "brush" ? (
            <BrushPanel brush={brush} frame={frame} disabledReason={brushReason} />
          ) : (
            <PromptPanel prompting={prompting} disabledReason={disabledReason} target={promptTarget} />
          )}
          <PropagatePanel
            job={job}
            running={running}
            foreign={foreign}
            starting={starting}
            blockedReason={lockReason ?? propagateBlocked}
            follow={follow}
            error={propagateError}
            scope={scope}
            selectedLabel={activeLabel}
            objectName={objectName}
            onScope={setPropagateScope}
            ownResult={ownJob.current}
            onStart={startPropagation}
            onStop={stopPropagation}
            onFollow={setFollow}
          />
          <OverlayPanel style={overlayStyle} classColor={color} activeClassName={newClass} classColorError={classColorError}
            onStyle={changeStyle} onClassColor={saveClassColor} />
          <CleanupPanel
            frame={frame}
            states={detail.states}
            lanes={clearLanes}
            scope={clearScope}
            activeLabel={activeLabel}
            onScope={setClearScope}
            range={range}
            clearBlockedReason={running ? "Masks cannot be cleared while a propagation runs." : null}
            lockReason={lockReason}
            error={cleanupError}
            onRange={setRange}
            onToggleDelete={toggleDelete}
            onPrune={prune}
            onClearRange={clearRange}
          />
        </aside>
      </div>
      <footer className="flex flex-col gap-2 border-t border-line bg-panel px-3 pb-2 pt-2">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1">
          <Transport
            frame={frame}
            frameCount={frameCount}
            playing={playing}
            overlay={overlay}
            name={detail.clip.original_names?.[String(frame)] ?? null}
            disabled={lockReason !== null}
            onStep={step}
            onTogglePlay={() => {
              if (!running) setPlaying((p) => !p);
            }}
            onToggleOverlay={() => setOverlay((o) => !o)}
          />
          {/* whole-clip counts: coloured neutrally, not with the active object's class colour */}
          <ClipStats states={detail.states} color={FALLBACK_COLOR} />
        </div>
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
      </footer>
    </div>
  );
}
