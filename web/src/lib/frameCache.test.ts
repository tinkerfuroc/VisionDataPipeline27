import { describe, expect, it, vi } from "vitest";
import { FrameStore, LruCache, compositeKey, type MaskPart } from "./frameCache";
import type { Outline } from "./mask";

const OUT: Outline = { rgb: [255, 255, 255], edge: 5 };
type Load = (url: string) => Promise<HTMLImageElement | null>;

const img = (name: string) => ({ name }) as unknown as HTMLImageElement;
const canvas = (name: string) => ({ name }) as unknown as HTMLCanvasElement;
const part = (id: number, over: Partial<MaskPart> = {}): MaskPart => ({
  url: `/api/clips/c1/objects/${id}/masks/4?v=0.0`, rgb: [id, 0, 0], outline: null, ...over,
});
/** a store whose tint names each layer after its image, and whose compose records what it was given */
const setup = (load: Load = vi.fn<Load>(async (url) => img(url))) => {
  const tint = vi.fn((i: HTMLImageElement) => canvas(`tinted ${(i as unknown as { name: string }).name}`));
  const compose = vi.fn((layers: HTMLCanvasElement[]) => canvas(`composite of ${layers.length}`));
  return { store: new FrameStore(load, tint, compose), load, tint, compose };
};

describe("LruCache", () => {
  it("evicts the least recently used entry; get() refreshes", () => {
    const c = new LruCache<number>(2);
    c.set("a", 1);
    c.set("b", 2);
    expect(c.get("a")).toBe(1);
    c.set("c", 3);
    expect(c.get("b")).toBeUndefined();
    expect(c.get("a")).toBe(1);
    expect(c.get("c")).toBe(3);
    expect(c.size).toBe(2);
  });
});

describe("FrameStore images", () => {
  it("loads each URL once, even for concurrent requests", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const store = new FrameStore(load, vi.fn());
    const [a, b] = await Promise.all([store.image("/f/1"), store.image("/f/1")]);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(1);
  });
  it("does not cache a failed load, so the next request retries", async () => {
    const load = vi.fn<Load>().mockResolvedValueOnce(null).mockResolvedValueOnce(img("ok"));
    const store = new FrameStore(load, vi.fn());
    expect(await store.image("/f/1")).toBeNull();
    expect(await store.image("/f/1")).not.toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
  });
});

describe("FrameStore composite layers", () => {
  it("composites a frame's visible objects into one cached layer", async () => {
    const { store, load, tint, compose } = setup();
    const parts = [part(1), part(2, { outline: OUT })];
    const a = await store.composite(parts, 0.45);
    const b = await store.composite(parts.map((p) => ({ ...p })), 0.45);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(2);
    expect(tint).toHaveBeenCalledTimes(2);
    expect(compose).toHaveBeenCalledTimes(1);
    expect(compose.mock.calls[0][0]).toEqual([canvas(`tinted ${parts[0].url}`), canvas(`tinted ${parts[1].url}`)]);
  });
  it("outlines the active object only; the others are fill only", async () => {
    const { store, tint } = setup();
    await store.composite([part(1), part(2, { outline: OUT })], 0.45);
    expect(tint).toHaveBeenNthCalledWith(1, img(part(1).url), [1, 0, 0], 0.45, { rgb: [1, 0, 0], edge: 0 });
    expect(tint).toHaveBeenNthCalledWith(2, img(part(2).url), [2, 0, 0], 0.45, OUT);
  });
  it("keys the layer by mask version, object order, active object, class colours and outline style", async () => {
    const { store, compose } = setup();
    const base = [part(1), part(2, { outline: OUT })];
    const variants: MaskPart[][] = [
      base,
      [part(1), part(2, { outline: OUT, url: "/api/clips/c1/objects/2/masks/4?v=0.1" })], // a new mask version
      [part(2, { outline: OUT }), part(1)], // another order
      [part(1, { outline: OUT }), part(2)], // another active object
      [part(1, { rgb: [9, 9, 9] }), part(2, { outline: OUT })], // a relabelled object
      [part(1), part(2, { outline: { ...OUT, edge: 2 } })], // another outline thickness
      [part(1), part(2, { outline: { ...OUT, rgb: [0, 0, 0] } })], // another outline colour
    ];
    for (const v of variants) await store.composite(v, 0.45);
    await store.composite(base.map((p) => ({ ...p })), 0.45); // equal content: a cache hit
    expect(compose).toHaveBeenCalledTimes(variants.length);
    expect(new Set(variants.map((v) => compositeKey(v, 0.45))).size).toBe(variants.length);
    expect(compositeKey(base, 0.45)).not.toBe(compositeKey(base, 0.55));
  });
  it("holds 48 frames however many objects each has", async () => {
    const { store, compose } = setup();
    const frame = (i: number) => [1, 2, 3].map((id) => part(id, { url: `/api/clips/c1/objects/${id}/masks/${i}?v=0.0` }));
    for (let i = 0; i < 49; i++) await store.composite(frame(i), 0.45);
    expect(compose).toHaveBeenCalledTimes(49);
    await store.composite(frame(48), 0.45);
    await store.composite(frame(1), 0.45);
    expect(compose).toHaveBeenCalledTimes(49); // frames 1..48 are all still cached
    await store.composite(frame(0), 0.45);
    expect(compose).toHaveBeenCalledTimes(50); // frame 0 was the one evicted
  });
  it("skips a mask that fails to load and does not keep the partial layer", async () => {
    const load = vi.fn<Load>(async (url) => img(url));
    load.mockImplementationOnce(async (url) => img(url)).mockImplementationOnce(async () => null);
    const { store, compose } = setup(load);
    const parts = [part(1), part(2)];
    await store.composite(parts, 0.45);
    expect(compose).toHaveBeenLastCalledWith([canvas(`tinted ${parts[0].url}`)]);
    await store.composite(parts, 0.45);
    expect(load).toHaveBeenCalledTimes(4); // not cached: both masks are fetched again
    expect(compose).toHaveBeenLastCalledWith([canvas(`tinted ${parts[0].url}`), canvas(`tinted ${parts[1].url}`)]);
  });
  it("returns null without loading anything for no parts, and null when no mask loads", async () => {
    const { store, load, compose } = setup(vi.fn<Load>(async () => null));
    expect(await store.composite([], 0.45)).toBeNull();
    expect(load).not.toHaveBeenCalled();
    expect(await store.composite([part(1)], 0.45)).toBeNull();
    expect(compose).not.toHaveBeenCalled();
  });
  it("previews never evict composite layers", async () => {
    const { store, compose, tint } = setup();
    await store.composite([part(1)], 0.45);
    for (let i = 0; i < 20; i++) await store.preview(`data:image/png;base64,p${i}`, [0, 255, 0], 0.55, OUT);
    await store.composite([part(1)], 0.45);
    expect(compose).toHaveBeenCalledTimes(1);
    expect(tint).toHaveBeenCalledTimes(21);
  });
  it("tints a preview once per colour, alpha and outline", async () => {
    const { store, tint } = setup();
    await store.preview("data:p", [255, 0, 0], 0.55, OUT);
    await store.preview("data:p", [255, 0, 0], 0.55, OUT);
    await store.preview("data:p", [255, 0, 0], 0.55, { ...OUT, edge: 0 });
    await store.preview("data:p", [0, 255, 0], 0.55, OUT);
    expect(tint).toHaveBeenCalledTimes(3);
  });
});
