import { baseLayerData, layerToMaskData } from "./brush";
import type { FrameStore } from "./frameCache";
import type { Rgb } from "./mask";

/** The DOM half of the brush: an offscreen layer canvas, the stored mask as its base, and PNG encoding.
 * Everything here tolerates a missing 2D context (jsdom) by returning null. */

export function createLayer(width: number, height: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D | null } {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return { canvas, ctx: canvas.getContext("2d") };
}

export async function loadBase(store: FrameStore, url: string, rgb: Rgb, width: number, height: number): Promise<ImageData | null> {
  const img = await store.image(url);
  if (img === null) return null;
  const { canvas, ctx } = createLayer(width, height);
  if (ctx === null) return null;
  ctx.drawImage(img, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height);
  const out = ctx.createImageData(width, height);
  out.data.set(baseLayerData(pixels, rgb));
  canvas.width = 0; // release the scratch bitmap
  return out;
}

export function encodeLayer(layer: HTMLCanvasElement): { pngB64: string; any: boolean } | null {
  const ctx = layer.getContext("2d");
  if (ctx === null) return null;
  const { width, height } = layer;
  const { data, any } = layerToMaskData(ctx.getImageData(0, 0, width, height).data);
  const { canvas, ctx: scratch } = createLayer(width, height);
  if (scratch === null) return null;
  const img = scratch.createImageData(width, height);
  img.data.set(data);
  scratch.putImageData(img, 0, 0);
  const url = canvas.toDataURL("image/png");
  canvas.width = 0;
  const prefix = "data:image/png;base64,";
  if (!url.startsWith(prefix)) return null;
  return { pngB64: url.slice(prefix.length), any };
}
