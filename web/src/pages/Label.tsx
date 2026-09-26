import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, INTERACTIVE, api, frameUrl, maskUrl, retryInteractive, thumbUrl } from "../api/rest";
import type { ClipDetail, Direction, JobEvent } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { BrushPanel } from "../components/BrushPanel";
import { CleanupPanel, type Range } from "../components/CleanupPanel";
import { ClipStats } from "../components/ClipStats";
import { FrameCanvas } from "../components/FrameCanvas";
import { ObjectPanel } from "../components/ObjectPanel";
import { OverlayPanel } from "../components/OverlayPanel";
import { PromptPanel } from "../components/PromptPanel";
import { PropagatePanel } from "../components/PropagatePanel";
import { Timeline } from "../components/Timeline";
import { ToolRail, type RailTool } from "../components/ToolRail";
import { Transport } from "../components/Transport";
import { useBrush } from "../hooks/useBrush";
import { useClip } from "../hooks/useClip";
import { HealthContext, engineReady } from "../hooks/useHealth";
import { useJob } from "../hooks/useJob";
import { useKeyboard, type KeyAction } from "../hooks/useKeyboard";
import { usePlayback } from "../hooks/usePlayback";
import { usePrompting } from "../hooks/usePrompting";
import { FrameStore, MASK_ALPHA } from "../lib/frameCache";
import { isTerminal } from "../lib/jobReducer";
import { hexToRgb } from "../lib/mask";
import { parseFrameEvent, type FrameEvent } from "../lib/objects";
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
  const [classId, setClassId] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => sessionStorage.getItem(jobKey(clipId)));
  const [follow, setFollow] = useState(true);
  const [propagateError, setPropagateError] = useState<string | null>(null);
  const [cleanupError, setCleanupError] = useState<string | null>(null);
  const [mode, setMode] = useState<"prompt" | "brush">("prompt");

  const detail = clip.detail;
  const frameCount = detail?.clip.frame_count ?? 0;
  const fps = detail?.clip.fps ?? 30;
  usePlayback({ playing, fps, frame, frameCount, setFrame, setPlaying });

  const object = detail?.clip.objects[0] ?? null;
  useEffect(() => {
    if (classId === null && detail !== null) setClassId(object?.class_id ?? detail.classes[0]?.id ?? null);
  }, [classId, detail, object]);

  // ---- propagation job -------------------------------------------------------------------------------------------
  const { patchFrames, reload, apply } = clip;
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
      retryInteractive(() => api.propagate(clipId, frame, direction)).then(
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
    [clipId, frame, attach, starting],
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
  const shownClass = classId ?? object?.class_id ?? null;
  const color = detail?.classes.find((c) => c.id === shownClass)?.color ?? "#22c55e";
  // hexToRgb allocates a new array every call: memoised so its identity is stable across renders that don't
  // change `color` (e.g. a health poll), since it feeds effect dependency arrays (useBrush's re-base effect).
  const rgb = useMemo(() => hexToRgb(color), [color]);
  const outline = useMemo(() => outlineFor(overlayStyle, rgb), [overlayStyle, rgb]);
  const changeStyle = useCallback((s: OverlayStyle) => {
    setOverlayStyle(s);
    saveStyle(s);
  }, []);
  const hasMask = (i: number) => detail !== null && (detail.states[i] === "masked" || detail.states[i] === "keyframe");
  const maskVersion = clip.maskVersion;
  const prompting = usePrompting({ clipId, frame, enabled: disabledReason === null && detail !== null && mode === "prompt", clip, classId });

  const brushReason = running
    ? "A propagation is running. Stop it to paint."
    : state === "deleted"
      ? "This frame is deleted. Restore it to paint on it."
      : playing
        ? "Pause playback to paint."
        : null;
  const brush = useBrush({
    clipId, frame, width: detail?.clip.width ?? 1, height: detail?.clip.height ?? 1, rgb, store,
    maskUrl: hasMask(frame) ? maskUrl(clipId, frame, maskVersion(frame)) : null,
    active: mode === "brush" && brushReason === null && detail !== null,
    classId, needsClass: object === null, apply,
  });
  const lockReason = brush.dirty ? `Save or discard the brush strokes on frame ${frame} first.` : null;

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
      // the mark goes only once the masks are gone, so a failed request can be retried without marking again
      const cleared = api.clearRange(clipId, start, end).then((next) => {
        setRange(null);
        return next;
      });
      cleanup(cleared, true);
    },
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
      if (overlay && hasMask(i)) void store.layer(maskUrl(clipId, i, maskVersion(i)), rgb, MASK_ALPHA, outline);
    }
    // hasMask is derived from detail
  }, [store, clipId, frame, playing, overlay, detail, rgb, outline, frameCount, maskVersion, running]);

  const addClass = useCallback(
    async (name: string) => {
      if (detail === null) return;
      const created = await api.addClass(detail.clip.project_id, name);
      apply({ ...detail, classes: [...detail.classes, created] }, false);
      setClassId(created.id);
    },
    [detail, apply],
  );

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

  const propagateBlocked = !engineReady(health)
    ? "The SAM3 engine is not loaded."
    : state === "deleted"
      ? "This frame is deleted. Restore it, or go to a masked frame, to propagate from it."
      : state !== "keyframe" && state !== "masked"
        ? "This frame has no mask. Go to a masked frame, or prompt here, to propagate from it."
        : null;
  const { draft, dispatch } = prompting;
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
            maskUrl={overlay && hasMask(frame) ? maskUrl(clipId, frame, maskVersion(frame)) : null}
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
          <ObjectPanel detail={detail} classId={classId} onClassId={setClassId} onAddClass={addClass} />
          {mode === "brush" ? (
            <BrushPanel brush={brush} frame={frame} disabledReason={brushReason} />
          ) : (
            <PromptPanel prompting={prompting} disabledReason={disabledReason} />
          )}
          <PropagatePanel
            job={job}
            running={running}
            foreign={foreign}
            starting={starting}
            blockedReason={lockReason ?? propagateBlocked}
            follow={follow}
            error={propagateError}
            onStart={startPropagation}
            onStop={stopPropagation}
            onFollow={setFollow}
          />
          <OverlayPanel style={overlayStyle} classColor={color} onStyle={changeStyle} />
          <CleanupPanel
            frame={frame}
            states={detail.states}
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
          <ClipStats states={detail.states} color={color} />
        </div>
        <Timeline
          states={detail.states}
          frame={frame}
          color={color}
          thumb={{ url: (i) => thumbUrl(clipId, i), aspect: width / height }}
          range={range}
          onSeek={lockReason === null ? seek : () => {}}
          onRange={setRange}
        />
      </footer>
    </div>
  );
}
