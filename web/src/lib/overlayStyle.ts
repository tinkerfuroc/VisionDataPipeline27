import { hexToRgb, type Outline, type Rgb } from "./mask";

/** How the Label page draws a mask's boundary. A viewing preference only: nothing here reaches the server. */
export type OverlayStyle = {
  /** outline colour as `#rrggbb`, or null to use the class colour */
  color: string | null;
  /** outline width in image pixels; 0 draws no outline */
  thickness: number;
};

export const MAX_THICKNESS = 12;
export const DEFAULT_STYLE: OverlayStyle = { color: null, thickness: 5 };

const KEY = "vsp.overlayStyle";
const HEX = /^#[0-9a-f]{6}$/i;

export function clampThickness(v: number): number {
  return Number.isFinite(v) ? Math.max(0, Math.min(MAX_THICKNESS, Math.round(v))) : DEFAULT_STYLE.thickness;
}

/** Parse a stored style, keeping each valid field and defaulting the rest. */
export function parseStyle(raw: string | null): OverlayStyle {
  if (raw === null) return DEFAULT_STYLE;
  try {
    const v = JSON.parse(raw) as Partial<Record<keyof OverlayStyle, unknown>>;
    return {
      color: typeof v.color === "string" && HEX.test(v.color) ? v.color.toLowerCase() : null,
      thickness: typeof v.thickness === "number" ? clampThickness(v.thickness) : DEFAULT_STYLE.thickness,
    };
  } catch {
    return DEFAULT_STYLE;
  }
}

/** Storage can be missing or throw (private windows, blocked site data); the style then lasts for the session only. */
export function loadStyle(): OverlayStyle {
  try {
    return parseStyle(localStorage.getItem(KEY));
  } catch {
    return DEFAULT_STYLE;
  }
}

export function saveStyle(style: OverlayStyle): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(style));
  } catch {
    // not remembered; still applied
  }
}

export function outlineFor(style: OverlayStyle, classRgb: Rgb): Outline {
  return { rgb: style.color === null ? classRgb : hexToRgb(style.color), edge: style.thickness };
}
