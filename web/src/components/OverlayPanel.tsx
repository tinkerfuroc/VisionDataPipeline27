import { DEFAULT_STYLE, MAX_THICKNESS, clampThickness, type OverlayStyle } from "../lib/overlayStyle";

type Props = {
  style: OverlayStyle;
  /** the shown class's colour, hex: what "Class colour" means and where a custom pick starts */
  classColor: string;
  onStyle: (style: OverlayStyle) => void;
};

/** Mask outline colour and width. Display only: the class colour saved in the project, and exports, are untouched. */
export function OverlayPanel({ style, classColor, onStyle }: Props) {
  const custom = style.color !== null;
  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Mask outline</h2>
      <label className="flex items-center gap-2 text-xs">
        <input type="checkbox" checked={!custom}
          onChange={(e) => onStyle({ ...style, color: e.target.checked ? null : classColor })} />
        Class colour
      </label>
      <label className="flex items-center gap-2 text-xs">
        <span className="w-16">Colour</span>
        <input aria-label="Outline colour" type="color" value={style.color ?? classColor} disabled={!custom}
          className="h-7 w-12 cursor-pointer disabled:cursor-default disabled:opacity-50"
          onChange={(e) => onStyle({ ...style, color: e.target.value })} />
      </label>
      <label className="flex items-center gap-2 text-xs">
        <span className="w-16">Thickness</span>
        <input aria-label="Outline thickness" type="range" min={0} max={MAX_THICKNESS} value={style.thickness}
          className="flex-1" onChange={(e) => onStyle({ ...style, thickness: clampThickness(Number(e.target.value)) })} />
        <span className="w-12 text-right text-dim">{style.thickness === 0 ? "off" : `${style.thickness} px`}</span>
      </label>
      <div className="flex items-center justify-between gap-2">
        <p className="note">Only changes how masks look here; saved class colours and exports stay the same.</p>
        <button type="button" className="btn shrink-0"
          disabled={style.color === DEFAULT_STYLE.color && style.thickness === DEFAULT_STYLE.thickness}
          onClick={() => onStyle(DEFAULT_STYLE)}>
          Reset
        </button>
      </div>
    </section>
  );
}
