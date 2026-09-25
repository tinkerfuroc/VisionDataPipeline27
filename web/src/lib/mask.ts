export type Rgb = [number, number, number];

export function hexToRgb(hex: string): Rgb {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (m === null) return [0, 255, 0];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** How the mask boundary is drawn: `edge` solid pixels in `rgb`, just inside the object. */
export type Outline = { rgb: Rgb; edge: number };

/**
 * Turn a decoded mask (white = object) into a coloured overlay, in place: object → rgb at `alpha`, rest transparent.
 *
 * With `size` and an `outline` of positive width, the object's outermost `outline.edge` pixels are drawn solid in
 * `outline.rgb`: a flat tint alone disappears on an object of a similar hue, and where the mask ends is exactly
 * what the operator is judging.
 */
export function tintMask(
  rgba: Uint8ClampedArray,
  rgb: Rgb,
  alpha: number,
  size?: { width: number; height: number },
  outline?: Outline,
): Uint8ClampedArray {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255);
  const n = rgba.length / 4;
  const inside = new Uint8Array(n);
  for (let i = 0; i < n; i++) inside[i] = rgba[i * 4] > 127 ? 1 : 0;

  // depth[i]: 0 outside; 1 on the object's outermost ring, 2 on the next ring in, ...; `edge + 1` for the interior
  const depth = new Uint8Array(n);
  let edge = 0;
  if (size !== undefined && outline !== undefined && size.width * size.height === n) {
    const { width, height } = size;
    edge = Math.max(0, Math.min(254, Math.round(outline.edge)));
    const interior = edge + 1;
    for (let i = 0; i < n; i++) depth[i] = inside[i] === 1 ? interior : 0;
    // Erode one ring at a time: ring k is every interior pixel with a 4-neighbour in ring k - 1 (ring 0 = outside).
    // Off-image neighbours do not count, so an object cut off by the frame gets no edge along the image border.
    for (let ring = 1; ring <= edge; ring++) {
      const found: number[] = [];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = y * width + x;
          if (depth[i] !== interior) continue;
          const prev = ring - 1;
          if (
            (x > 0 && depth[i - 1] === prev) ||
            (x < width - 1 && depth[i + 1] === prev) ||
            (y > 0 && depth[i - width] === prev) ||
            (y < height - 1 && depth[i + width] === prev)
          ) {
            found.push(i);
          }
        }
      }
      for (const i of found) depth[i] = ring; // after the scan, so a ring never grows from pixels found in the same pass
    }
  }

  const line = outline?.rgb ?? rgb;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (inside[i] === 0) {
      rgba[o + 3] = 0;
    } else if (edge > 0 && depth[i] <= edge) {
      rgba[o] = line[0];
      rgba[o + 1] = line[1];
      rgba[o + 2] = line[2];
      rgba[o + 3] = 255;
    } else {
      rgba[o] = rgb[0];
      rgba[o + 1] = rgb[1];
      rgba[o + 2] = rgb[2];
      rgba[o + 3] = a;
    }
  }
  return rgba;
}
