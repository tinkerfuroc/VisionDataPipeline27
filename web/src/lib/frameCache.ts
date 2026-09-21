import { tintMask, type Rgb } from "./mask";

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

/** Decode a mask image (white = object) into a canvas holding the coloured, semi-transparent overlay. */
export function tintToCanvas(img: HTMLImageElement, rgb: Rgb, alpha: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  tintMask(data.data, rgb, alpha);
  ctx.putImageData(data, 0, 0);
  return canvas;
}

/** Shared alpha for the stored mask overlay; must match every caller so prefetched layers are cache hits. */
export const MASK_ALPHA = 0.45;
/** Alpha for a text-prompt candidate preview overlay. */
export const PREVIEW_ALPHA = 0.55;

/** Small on purpose: a decoded 1080p frame or layer is about 8 MB. */
const MAX_IMAGES = 64;
const MAX_LAYERS = 48;
/** Preview keys are whole base64 data URLs (one per candidate, per prompt); kept tiny and separate so a run
 * of text-prompt previews can never evict prefetched playback mask layers. */
const MAX_PREVIEWS = 12;

export class FrameStore {
  private readonly images = new LruCache<Promise<HTMLImageElement | null>>(MAX_IMAGES);
  private readonly layers = new LruCache<Promise<HTMLCanvasElement | null>>(MAX_LAYERS);
  private readonly previews = new LruCache<Promise<HTMLCanvasElement | null>>(MAX_PREVIEWS);

  constructor(
    private readonly load: (url: string) => Promise<HTMLImageElement | null> = loadImage,
    private readonly tint: (img: HTMLImageElement, rgb: Rgb, alpha: number) => HTMLCanvasElement = tintToCanvas,
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

  layer(url: string, rgb: Rgb, alpha: number): Promise<HTMLCanvasElement | null> {
    return this.layerFrom(this.layers, url, rgb, alpha);
  }

  /** Like `layer()`, but cached separately so text-prompt candidate previews cannot evict stored mask layers. */
  preview(dataUrl: string, rgb: Rgb, alpha: number): Promise<HTMLCanvasElement | null> {
    return this.layerFrom(this.previews, dataUrl, rgb, alpha);
  }

  private layerFrom(
    cache: LruCache<Promise<HTMLCanvasElement | null>>,
    url: string,
    rgb: Rgb,
    alpha: number,
  ): Promise<HTMLCanvasElement | null> {
    const key = `${url}|${rgb.join(",")}|${alpha}`;
    let pending = cache.get(key);
    if (pending === undefined) {
      pending = this.load(url).then((img) => {
        if (img === null) {
          cache.delete(key);
          return null;
        }
        return this.tint(img, rgb, alpha);
      });
      cache.set(key, pending);
    }
    return pending;
  }
}
