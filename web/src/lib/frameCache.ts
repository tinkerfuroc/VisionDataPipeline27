import { tintMask, type Outline, type Rgb } from "./mask";

export class LruCache<V> {
  private readonly map = new Map<string, V>();
  constructor(private readonly max: number) {}

  get(key: string): V | undefined {
    const value = this.map.get(key);
    if (value !== undefined) {
      this.map.delete(key);
      this.map.set(key, value);
    }
    return value;
  }

  set(key: string, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.max) this.map.delete(this.map.keys().next().value as string);
  }

  delete(key: string): void {
    this.map.delete(key);
  }

  get size(): number {
    return this.map.size;
  }
}

export function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

/** Decode a mask image (white = object) into a canvas holding the coloured, semi-transparent, outlined overlay. */
export function tintToCanvas(img: HTMLImageElement, rgb: Rgb, alpha: number, outline: Outline): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  tintMask(data.data, rgb, alpha, { width: canvas.width, height: canvas.height }, outline);
  ctx.putImageData(data, 0, 0);
  return canvas;
}

/** Shared alpha for the stored mask overlay; must match every caller so prefetched layers are cache hits. */
export const MASK_ALPHA = 0.45;
/** Alpha for a text-prompt candidate preview overlay. */
export const PREVIEW_ALPHA = 0.55;

/** Small on purpose: a decoded 1080p frame or layer is about 8 MB. */
const MAX_IMAGES = 64;
/** One composite per frame, however many objects it has. */
const MAX_LAYERS = 48;
/** Preview keys are whole base64 data URLs (one per candidate, per prompt); kept tiny and separate so a run
 * of text-prompt previews can never evict prefetched playback mask layers. */
const MAX_PREVIEWS = 12;

/** One object's stored mask on one frame, as drawn: filled in `rgb`, plus the configurable outline for the active
 * object only (`null` = fill only). */
export type MaskPart = { url: string; rgb: Rgb; outline: Outline | null };

/** Everything a composite layer depends on: each part's URL (object id, frame, mask version) in draw order, its class
 * colour and its outline, and the fill alpha. Equal keys mean an identical layer. */
export function compositeKey(parts: readonly MaskPart[], alpha: number): string {
  const one = (p: MaskPart) =>
    `${p.url}|${p.rgb.join(",")}|${p.outline === null ? "-" : `${p.outline.rgb.join(",")}/${p.outline.edge}`}`;
  return `${alpha};${parts.map(one).join(";")}`;
}

/** Draw tinted layers into one canvas, in order (later on top); a single layer is returned as is. The inputs are
 * released: only the composite is cached. */
export function composeLayers(layers: HTMLCanvasElement[]): HTMLCanvasElement {
  if (layers.length === 1) return layers[0];
  const out = document.createElement("canvas");
  out.width = layers[0].width;
  out.height = layers[0].height;
  const ctx = out.getContext("2d")!;
  for (const layer of layers) {
    ctx.drawImage(layer, 0, 0, out.width, out.height);
    layer.width = 0;
  }
  return out;
}

export class FrameStore {
  private readonly images = new LruCache<Promise<HTMLImageElement | null>>(MAX_IMAGES);
  private readonly layers = new LruCache<Promise<HTMLCanvasElement | null>>(MAX_LAYERS);
  private readonly previews = new LruCache<Promise<HTMLCanvasElement | null>>(MAX_PREVIEWS);

  constructor(
    private readonly load: (url: string) => Promise<HTMLImageElement | null> = loadImage,
    private readonly tint: (img: HTMLImageElement, rgb: Rgb, alpha: number, outline: Outline) => HTMLCanvasElement = tintToCanvas,
    private readonly compose: (layers: HTMLCanvasElement[]) => HTMLCanvasElement = composeLayers,
  ) {}

  image(url: string): Promise<HTMLImageElement | null> {
    let pending = this.images.get(url);
    if (pending === undefined) {
      pending = this.load(url).then((img) => {
        if (img === null) this.images.delete(url); // a failed load must not stick
        return img;
      });
      this.images.set(url, pending);
    }
    return pending;
  }

  /**
   * A frame's visible objects, tinted and drawn in order into ONE cached layer, so the 48-entry cache holds 48 frames
   * whatever the object count. A mask that fails to load is left out and the partial layer is not cached (the next
   * request fetches again); with no parts, or nothing loaded, the result is null.
   */
  composite(parts: readonly MaskPart[], alpha: number = MASK_ALPHA): Promise<HTMLCanvasElement | null> {
    if (parts.length === 0) return Promise.resolve(null);
    const key = compositeKey(parts, alpha);
    let pending = this.layers.get(key);
    if (pending === undefined) {
      pending = Promise.all(parts.map((p) => this.load(p.url))).then((imgs) => {
        const tinted: HTMLCanvasElement[] = [];
        imgs.forEach((img, i) => {
          const p = parts[i];
          if (img !== null) tinted.push(this.tint(img, p.rgb, alpha, p.outline ?? { rgb: p.rgb, edge: 0 }));
        });
        if (tinted.length < parts.length) this.layers.delete(key);
        return tinted.length === 0 ? null : this.compose(tinted);
      });
      this.layers.set(key, pending);
    }
    return pending;
  }

  /** A text-prompt candidate, cached separately so previews cannot evict stored mask layers. */
  preview(dataUrl: string, rgb: Rgb, alpha: number, outline: Outline): Promise<HTMLCanvasElement | null> {
    const key = `${dataUrl}|${rgb.join(",")}|${alpha}|${outline.rgb.join(",")}|${outline.edge}`;
    let pending = this.previews.get(key);
    if (pending === undefined) {
      pending = this.load(dataUrl).then((img) => {
        if (img === null) {
          this.previews.delete(key);
          return null;
        }
        return this.tint(img, rgb, alpha, outline);
      });
      this.previews.set(key, pending);
    }
    return pending;
  }
}
