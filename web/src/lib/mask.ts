export type Rgb = [number, number, number];

export function hexToRgb(hex: string): Rgb {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (m === null) return [0, 255, 0];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Turn a decoded mask (white = object) into a coloured overlay, in place: object → rgb at `alpha`, rest transparent. */
export function tintMask(rgba: Uint8ClampedArray, rgb: Rgb, alpha: number): Uint8ClampedArray {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255);
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i] > 127) {
      rgba[i] = rgb[0];
      rgba[i + 1] = rgb[1];
      rgba[i + 2] = rgb[2];
      rgba[i + 3] = a;
    } else {
      rgba[i + 3] = 0;
    }
  }
  return rgba;
}
