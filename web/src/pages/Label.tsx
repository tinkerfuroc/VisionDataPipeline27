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
