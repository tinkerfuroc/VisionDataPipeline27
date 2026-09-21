export type Rgb = [number, number, number];

export function hexToRgb(hex: string): Rgb {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (m === null) return [0, 255, 0];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const HALO: Rgb = [255, 255, 255];
const HALO_ALPHA = 230;

/**
 * Turn a decoded mask (white = object) into a coloured overlay, in place: object → rgb at `alpha`, rest transparent.
 *
 * With `size`, the object also gets a boundary the eye can follow on any background: its outermost `edge` pixels are
 * drawn in the solid class colour, and the one pixel outside it in white. A flat tint alone disappears on an object
 * of a similar hue, and where the mask ends is exactly what the operator is judging.
 */
export function tintMask(
  rgba: Uint8ClampedArray,
  rgb: Rgb,
  alpha: number,
  size?: { width: number; height: number },
): Uint8ClampedArray {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255);
  const n = rgba.length / 4;
  const inside = new Uint8Array(n);
  for (let i = 0; i < n; i++) inside[i] = rgba[i * 4] > 127 ? 1 : 0;

  // depth[i]: 0 outside; 1 on the object's outermost ring, 2 on the next ring in, ...; `edge + 1` for the interior
  const depth = new Uint8Array(n);
  let edge = 0;
  if (size !== undefined && size.width * size.height === n) {
    const { width, height } = size;
    edge = Math.max(2, Math.round(Math.max(width, height) / 400));
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
    // halo: outside pixels that touch the object (8-neighbourhood)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (inside[i] === 1) continue;
        let near = false;
        for (let dy = -1; dy <= 1 && !near; dy++) {
          for (let dx = -1; dx <= 1 && !near; dx++) {
            const xx = x + dx;
            const yy = y + dy;
            near = xx >= 0 && yy >= 0 && xx < width && yy < height && inside[yy * width + xx] === 1;
          }
        }
        if (near) depth[i] = 255;
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const d = depth[i];
    if (d === 255) {
      rgba[o] = HALO[0];
      rgba[o + 1] = HALO[1];
      rgba[o + 2] = HALO[2];
      rgba[o + 3] = HALO_ALPHA;
    } else if (inside[i] === 1) {
      rgba[o] = rgb[0];
      rgba[o + 1] = rgb[1];
      rgba[o + 2] = rgb[2];
      rgba[o + 3] = edge > 0 && d <= edge ? 255 : a;
    } else {
      rgba[o + 3] = 0;
    }
  }
  return rgba;
}
