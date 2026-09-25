import type { ReactNode } from "react";
import type { Instance } from "../api/types";
import { PredictionOverlay } from "./PredictionOverlay";

type Props = {
  src: string; name: string; width: number; height: number; groundTruth: Instance[]; predictions: Instance[];
  showGt: boolean; showPred: boolean; caption: ReactNode; onOpen?: () => void;
};

export function GalleryCard({ src, name, width, height, groundTruth, predictions, showGt, showPred, caption, onOpen }: Props) {
  const body = (
    <>
      <div className="relative w-full overflow-hidden rounded bg-well" style={{ aspectRatio: `${width} / ${height}` }}>
        <img src={src} alt={name} loading="lazy" className="absolute inset-0 h-full w-full object-contain" />
        <PredictionOverlay width={width} height={height} groundTruth={groundTruth} predictions={predictions}
          showGt={showGt} showPred={showPred} scores={false} />
      </div>
      <div className="mt-1 flex items-center gap-1.5 font-cond text-xs text-dim">{caption}</div>
    </>
  );
  return onOpen ? (
    <button type="button" aria-label={`Open ${name}`} onClick={onOpen}
      className="snap-start rounded border border-line bg-panel p-1.5 text-left hover:border-dim">
      {body}
    </button>
  ) : (
    <div className="snap-start rounded border border-line bg-panel p-1.5">{body}</div>
  );
}
