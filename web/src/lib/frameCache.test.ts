import { describe, expect, it, vi } from "vitest";
import { FrameStore, LruCache } from "./frameCache";
import type { Outline } from "./mask";

const OUT: Outline = { rgb: [255, 255, 255], edge: 5 };

const img = (name: string) => ({ name }) as unknown as HTMLImageElement;
const canvas = (name: string) => ({ name }) as unknown as HTMLCanvasElement;

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

describe("FrameStore", () => {
  it("loads each URL once, even for concurrent requests", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const store = new FrameStore(load, vi.fn());
    const [a, b] = await Promise.all([store.image("/f/1"), store.image("/f/1")]);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(1);
  });
  it("does not cache a failed load, so the next request retries", async () => {
    const load = vi.fn<(url: string) => Promise<HTMLImageElement | null>>().mockResolvedValueOnce(null).mockResolvedValueOnce(img("ok"));
    const store = new FrameStore(load, vi.fn());
    expect(await store.image("/f/1")).toBeNull();
    expect(await store.image("/f/1")).not.toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
  });
  it("tints a mask once per colour and alpha", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const tint = vi.fn(() => canvas("layer"));
    const store = new FrameStore(load, tint);
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, OUT);
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, OUT);
    await store.layer("/m/1?v=0.0", [0, 255, 0], 0.45, OUT);
    expect(tint).toHaveBeenCalledTimes(2);
  });
  it("re-tints when the outline colour or thickness changes", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const tint = vi.fn(() => canvas("layer"));
    const store = new FrameStore(load, tint);
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, OUT);
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, { ...OUT, rgb: [0, 0, 0] });
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, { ...OUT, edge: 0 });
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, { rgb: [255, 255, 255], edge: 5 });
    expect(tint).toHaveBeenCalledTimes(3);
    expect(tint).toHaveBeenLastCalledWith(img("/m/1?v=0.0"), [255, 0, 0], 0.45, { ...OUT, edge: 0 });
  });
  it("returns null for a mask that cannot be loaded", async () => {
    const store = new FrameStore(vi.fn(async () => null), vi.fn());
    expect(await store.layer("/m/9?v=0.0", [1, 2, 3], 0.5, OUT)).toBeNull();
  });
  it("previews never evict mask layers", async () => {
    const load = vi.fn(async (url: string) => img(url));
    const tint = vi.fn(() => canvas("layer"));
    const store = new FrameStore(load, tint);
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, OUT);
    for (let i = 0; i < 20; i++) {
      await store.preview(`data:image/png;base64,p${i}`, [0, 255, 0], 0.55, OUT);
    }
    tint.mockClear();
    await store.layer("/m/1?v=0.0", [255, 0, 0], 0.45, OUT);
    expect(tint).not.toHaveBeenCalled();
  });
});
