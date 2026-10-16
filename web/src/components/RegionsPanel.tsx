import type { RegionRange } from "../api/types";
import { addRangeFrom, problem, rangeIndexAt, removeRange, setBounds, splitAt, updateRange } from "../lib/regions";

type Props = {
  regions: RegionRange[];
  frame: number;
  frameCount: number;
  width: number;
  height: number;
  /** what a drag on the frame draws */
  draw: "block" | "crop";
  dirty: boolean;
  saving: boolean;
  error: string | null;
  onDraw: (d: "block" | "crop") => void;
  onChange: (regions: RegionRange[]) => void;
  onSave: () => void;
  onDiscard: () => void;
};

/** The Regions tool: the frame range the current frame belongs to, its crop and blocked boxes (drawn on the
 * frame), saved together with one request. Excluded areas are invisible to SAM3 and cut out of exports. */
export function RegionsPanel(p: Props) {
  const index = rangeIndexAt(p.regions, p.frame);
  const range = index === null ? null : p.regions[index];
  const issue = problem(p.regions, p.width, p.height, p.frameCount);
  return (
    <section aria-label="Regions" className="flex flex-col gap-2">
      <h2 className="heading">Regions</h2>
      <p className="note">Grey areas are hidden from SAM3 and cut out of exports. Drag on the frame to draw.</p>
      {range === null || index === null ? (
        <>
          <p className="note text-ink">No region on this frame</p>
          <button type="button" className="btn" onClick={() => p.onChange(addRangeFrom(p.regions, p.frame, p.frameCount).regions)}>
            Add region from here
          </button>
        </>
      ) : (
        <>
          <div className="flex items-center gap-1">
            <input aria-label="Start frame" type="number" min={0} max={p.frameCount - 1} className="field w-16" value={range.start}
              onChange={(e) => p.onChange(setBounds(p.regions, index, Number(e.target.value), range.end, p.frameCount))} />
            <span className="text-dim">to</span>
            <input aria-label="End frame" type="number" min={0} max={p.frameCount - 1} className="field w-16" value={range.end}
              onChange={(e) => p.onChange(setBounds(p.regions, index, range.start, Number(e.target.value), p.frameCount))} />
          </div>
          <div className="flex gap-1">
            <button type="button" className="btn flex-1"
              onClick={() => p.onChange(setBounds(p.regions, index, p.frame, range.end, p.frameCount))}>
              Set start to this frame
            </button>
            <button type="button" className="btn flex-1"
              onClick={() => p.onChange(setBounds(p.regions, index, range.start, p.frame, p.frameCount))}>
              Set end to this frame
            </button>
          </div>
          <fieldset className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            <legend className="sr-only">A drag draws</legend>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="region-draw" checked={p.draw === "block"} onChange={() => p.onDraw("block")} />
              Blocked box
            </label>
            <label className="flex items-center gap-1.5">
              <input type="radio" name="region-draw" checked={p.draw === "crop"} onChange={() => p.onDraw("crop")} />
              Crop
            </label>
          </fieldset>
          <p className="note">
            {range.crop ? `Crop ${range.crop[2] - range.crop[0]}×${range.crop[3] - range.crop[1]}` : "No crop"}, {range.blocks.length} blocked
          </p>
          <div className="flex gap-1">
            <button type="button" className="btn flex-1" disabled={p.frame === range.start}
              onClick={() => p.onChange(splitAt(p.regions, index, p.frame).regions)}>
              New range from here
            </button>
            <button type="button" className="btn" onClick={() => p.onChange(updateRange(p.regions, index, { crop: null, blocks: [] }))}>
              Clear
            </button>
            <button type="button" className="btn" onClick={() => p.onChange(removeRange(p.regions, index))}>Remove range</button>
          </div>
        </>
      )}
      {issue !== null && <p className="note text-ink">{issue}</p>}
      {p.error !== null && <p role="alert" className="text-xs text-danger">{p.error}</p>}
      <div className="flex gap-1">
        <button type="button" disabled={!p.dirty || p.saving || issue !== null}
          className={`btn flex-1 ${p.dirty && issue === null ? "btn-on" : ""}`} onClick={p.onSave}>
          Save regions
        </button>
        <button type="button" className="btn" disabled={!p.dirty || p.saving} onClick={p.onDiscard}>Discard</button>
      </div>
      {p.dirty && <p className="note">Unsaved changes</p>}
    </section>
  );
}
