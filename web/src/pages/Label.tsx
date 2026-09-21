import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
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
  const { patchFrames, reload, apply } = clip;
  // true when this page itself started or resumed (from its own sessionStorage key) the current job; false when it
  // was attached from a 409 naming some other job holding the GPU lock (possibly another clip's) — that job must
  // never patch this clip's frames or move this view, and its lost_at must not be seeked to when it ends.
  const ownJob = useRef(true);
  // F3: a re-attach replays every event from seq 0, so `frame` events can arrive many per macrotask; buffer own-job
  // ones and apply the whole batch (one setFrame, one patchFrames) on the next animation frame instead of per event.
  const frameEventBuffer = useRef<{ frame: number; hasMask: boolean }[]>([]);
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
    if (followRef.current) setFrame(clampFrame(events[events.length - 1].frame, frameCountRef.current));
  }, [patchFrames]);

  const onJobEvent = useCallback(
    (event: JobEvent) => {
      if (event.type !== "frame" || typeof event.frame !== "number") return;
      if (!ownJob.current) return; // a job this page did not start must not drive this clip
      frameEventBuffer.current.push({ frame: event.frame, hasMask: event.has_mask === true });
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
    };
  }, [jobId]);

  useEffect(() => {
    if (jobId === null || !isTerminal(job.status)) return;
    const own = ownJob.current;
    if (own) sessionStorage.removeItem(jobKey(clipId));
    const lostAt = own ? job.result?.lost_at : undefined; // a foreign job's lost_at is none of this clip's business
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
      ? "A propagation is running — Stop it to prompt."
      : state === "deleted"
        ? "This frame is deleted — restore it to label it."
        : playing
          ? "Pause playback to prompt."
          : null;
  const prompting = usePrompting({ clipId, frame, enabled: disabledReason === null && detail !== null, clip, classId });

  useEffect(() => {
    if (prompting.busyJobId !== null && prompting.busyJobId !== INTERACTIVE && !running) attach(prompting.busyJobId, false);
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
            foreign={foreign}
            starting={starting}
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
