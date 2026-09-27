import { useEffect, useRef, useState } from "react";
import { DEFAULT_STYLE, MAX_THICKNESS, clampThickness, type OverlayStyle } from "../lib/overlayStyle";

const SAVE_DELAY_MS = 300;

type Props = {
  style: OverlayStyle;
  /** the active class's saved colour, hex: the outline's default when it follows the class */
  classColor: string;
  /** the active class's name; null disables the Class colour row (no object, and no class chosen for a new one) */
  activeClassName: string | null;
  /** a failed class-colour save, shown under the row */
  classColorError: string | null;
  onStyle: (style: OverlayStyle) => void;
  /** called ~300ms after the last change to the Class colour picker, with the final colour */
  onClassColor: (color: string) => void;
};

/**
 * The mask fill (the active class's colour, saved in the project) and the outline (a per-browser display
 * preference; only the active object gets one). The class colour is shown locally at once -- the native picker
 * fires onChange continuously while the user drags -- and saved ~300 ms after the last change; a class switch
 * (by name, which is unique per project) drops any pending save and previews the new class's own colour.
 */
export function OverlayPanel({ style, classColor, activeClassName, classColorError, onStyle, onClassColor }: Props) {
  const custom = style.color !== null;
  const [preview, setPreview] = useState(classColor);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    setPreview(classColor);
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    // classColor is not a dependency: it is only read as this effect's starting value on a genuine class switch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeClassName]);
  // a colour saved elsewhere (the Clips page, another tab) arrives with a reload; follow it unless a save is pending
  useEffect(() => {
    if (timer.current === null) setPreview(classColor);
  }, [classColor]);
  useEffect(() => () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
  }, []);

  const changeClassColor = (color: string) => {
    setPreview(color);
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      timer.current = null;
      onClassColor(color);
    }, SAVE_DELAY_MS);
  };

  return (
    <section className="flex flex-col gap-2">
      <h2 className="heading">Mask colours</h2>

      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-2 text-xs">
          <span>Class colour</span>
          <span className="text-dim">{activeClassName ?? "choose a class"}</span>
        </div>
        <input
          aria-label="Class colour"
          type="color"
          value={preview}
          disabled={activeClassName === null}
          title={activeClassName === null ? "Choose a class first." : undefined}
          className="h-7 w-12 cursor-pointer disabled:cursor-default disabled:opacity-50"
          onChange={(e) => changeClassColor(e.target.value)}
        />
        {classColorError !== null && <p role="alert" className="text-xs text-danger">{classColorError}</p>}
        <p className="note">Saved in the project: it recolours this class on every clip, and new exports' mask images.</p>
      </div>

      <div className="flex flex-col gap-2 border-t border-line pt-2">
        <span className="text-xs text-dim">Outline</span>
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={!custom}
            onChange={(e) => onStyle({ ...style, color: e.target.checked ? null : classColor })} />
          Outline follows class colour
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
          <p className="note">Only for this browser: it does not touch the saved class colour or exports.</p>
          <button type="button" className="btn shrink-0"
            disabled={style.color === DEFAULT_STYLE.color && style.thickness === DEFAULT_STYLE.thickness}
            onClick={() => onStyle(DEFAULT_STYLE)}>
            Reset
          </button>
        </div>
      </div>
    </section>
  );
}
