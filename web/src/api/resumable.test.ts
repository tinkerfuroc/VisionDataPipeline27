// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, UploadCancelled } from "./rest";
import { forgetUpload, rememberedUploads, sameFile, startResumableUpload, uploaderEnv } from "./resumable";

type Call = { method: string; url: string; size?: number };

/** A fake server for one upload ("u1"), with hooks to fail particular requests. */
function fakeServer(options: { received?: number; reportReceived?: number; failPut?: (offset: number, n: number) => Response | Error | null } = {}) {
  let received = options.received ?? 0;
  const calls: Call[] = [];
  let putCount = 0;
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const size = init?.body instanceof Blob ? init.body.size : undefined;
    calls.push({ method, url, size });
    if (method === "POST") return json({ upload_id: "u1", chunk_size: 4, received: 0 }, 201);
    if (method === "GET") return json({ upload_id: "u1", project_id: "p1", name: "a.mp4", size: 10, received: options.reportReceived ?? received, job_id: null });
    if (method === "DELETE") return new Response(null, { status: 204 });
    const offset = Number(new URL(url, "http://x").searchParams.get("offset"));
    const failure = options.failPut?.(offset, putCount++) ?? null;
    if (failure instanceof Error) throw failure;
    if (failure !== null) return failure;
    if (offset !== received) return json({ detail: "resync", received }, 409);
    received += size ?? 0;
    return json(received === 10 ? { received, job_id: "j1" } : { received });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, puts: () => calls.filter((c) => c.method === "PUT").map((c) => new URL(c.url, "http://x").searchParams.get("offset")) };
}

const file = (name = "a.mp4", size = 10, lastModified = 111) => new File([new Uint8Array(size)], name, { lastModified });

const realEnv = { ...uploaderEnv };
let waits: number[];
beforeEach(() => {
  localStorage.clear();
  waits = [];
  uploaderEnv.chunkSize = 4;
  uploaderEnv.online = () => Promise.resolve();
  uploaderEnv.waitToRetry = (ms: number) => {
    waits.push(ms);
    return Promise.resolve();
  };
});
afterEach(() => {
  Object.assign(uploaderEnv, realEnv);
  vi.unstubAllGlobals();
});

describe("startResumableUpload", () => {
  it("sends the file in chunks at the right offsets, reports progress and resolves with the job", async () => {
    const server = fakeServer();
    const progress: number[] = [];
    const handle = startResumableUpload("p1", file(), { stride: 2, fps: 30 }, (sent) => progress.push(sent));
    expect(await handle.promise).toEqual({ job_id: "j1" });
    expect(server.calls[0]).toMatchObject({ method: "POST", url: "/api/projects/p1/uploads" });
    expect(server.puts()).toEqual(["0", "4", "8"]);
    expect(progress).toEqual([0, 4, 8, 10]);
    expect(rememberedUploads()).toEqual([]); // forgotten once complete
  });

  it("remembers an upload while it runs", async () => {
    fakeServer({ failPut: () => new Response(JSON.stringify({ detail: "nope" }), { status: 413 }) });
    const handle = startResumableUpload("p1", file(), { stride: 1, fps: 30 });
    await expect(handle.promise).rejects.toBeInstanceOf(ApiError);
    expect(rememberedUploads()).toEqual([{ upload_id: "u1", project_id: "p1", name: "a.mp4", size: 10, lastModified: 111 }]);
  });

  it("retries a chunk after network errors and 5xx with growing waits", async () => {
    const server = fakeServer({
      failPut: (offset, n) => (offset === 4 && n < 3 ? (n === 2 ? new Response("{}", { status: 502 }) : new TypeError("network")) : null),
    });
    await startResumableUpload("p1", file(), { stride: 1, fps: 30 }).promise;
    expect(waits).toEqual([1000, 2000]); // put #1 network error, put #2 a 502, put #3 succeeds
    expect(server.puts()).toEqual(["0", "4", "4", "4", "8"]);
  });

  it("keeps waiting 30 s between tries once the backoff tops out", async () => {
    fakeServer({ failPut: (_offset, n) => (n < 8 ? new TypeError("network") : null) });
    await startResumableUpload("p1", file(), { stride: 1, fps: 30 }).promise;
    expect(waits).toEqual([1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000]);
  });

  it("resyncs to the server's count on a 409", async () => {
    // GET says 4, but the server meanwhile has 8 (another tab sent a chunk): its 409 moves the client to 8
    const server = fakeServer({ received: 8, reportReceived: 4 });
    await startResumableUpload("p1", file(), { stride: 1, fps: 30 }, undefined, {
      upload_id: "u1", project_id: "p1", name: "a.mp4", size: 10, lastModified: 111,
    }).promise;
    expect(server.puts()).toEqual(["4", "8"]);
  });

  it("stops on another 4xx with the server's message", async () => {
    fakeServer({ failPut: () => new Response(JSON.stringify({ detail: "too big" }), { status: 413 }) });
    await expect(startResumableUpload("p1", file(), { stride: 1, fps: 30 }).promise).rejects.toMatchObject({
      status: 413, message: "too big",
    });
  });

  it("waits for the browser to come back online before sending", async () => {
    let goOnline!: () => void;
    uploaderEnv.online = () => new Promise<void>((resolve) => (goOnline = resolve));
    const server = fakeServer();
    const handle = startResumableUpload("p1", file(), { stride: 1, fps: 30 });
    await new Promise((r) => setTimeout(r, 10));
    expect(server.puts()).toEqual([]);
    uploaderEnv.online = () => Promise.resolve();
    goOnline();
    await handle.promise;
    expect(server.puts()).toEqual(["0", "4", "8"]);
  });

  it("abort rejects with UploadCancelled, deletes the upload and forgets it", async () => {
    let release!: () => void;
    uploaderEnv.online = () => new Promise<void>((resolve) => (release = resolve));
    const server = fakeServer();
    const handle = startResumableUpload("p1", file(), { stride: 1, fps: 30 });
    await new Promise((r) => setTimeout(r, 10)); // created, waiting before the first chunk
    handle.abort();
    release();
    await expect(handle.promise).rejects.toBeInstanceOf(UploadCancelled);
    await new Promise((r) => setTimeout(r, 0));
    expect(server.calls.some((c) => c.method === "DELETE" && c.url === "/api/uploads/u1")).toBe(true);
    expect(rememberedUploads()).toEqual([]);
  });

  it("resumes a remembered upload from the server's count", async () => {
    const server = fakeServer({ received: 8 });
    const progress: number[] = [];
    await startResumableUpload("p1", file(), { stride: 1, fps: 30 }, (sent) => progress.push(sent), {
      upload_id: "u1", project_id: "p1", name: "a.mp4", size: 10, lastModified: 111,
    }).promise;
    expect(server.calls.map((c) => c.method)).toEqual(["GET", "PUT"]);
    expect(server.puts()).toEqual(["8"]);
    expect(progress).toEqual([8, 10]);
  });

  it("starts over quietly when the server has expired the upload", async () => {
    const server = fakeServer();
    const realFetch = globalThis.fetch;
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) =>
      (init?.method ?? "GET") === "GET"
        ? new Response(JSON.stringify({ detail: "upload u0 expired" }), { status: 404 })
        : (realFetch as typeof fetch)(url, init)));
    localStorage.setItem("vdp.uploads", JSON.stringify([{ upload_id: "u0", project_id: "p1", name: "a.mp4", size: 10, lastModified: 111 }]));
    const result = await startResumableUpload("p1", file(), { stride: 1, fps: 30 }, undefined, rememberedUploads()[0]).promise;
    expect(result).toEqual({ job_id: "j1" });
    expect(server.calls[0].method).toBe("POST");
    expect(rememberedUploads()).toEqual([]);
  });
});

describe("cancel and already-complete uploads", () => {
  const rem = { upload_id: "u1", project_id: "p1", name: "a.mp4", size: 10, lastModified: 111 };
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

  it("abort while the create call is pending still deletes the upload once it exists", async () => {
    let answer!: () => void;
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      calls.push(`${method} ${url}`);
      if (method === "POST") {
        await new Promise<void>((resolve) => (answer = resolve));
        return json({ upload_id: "u1", chunk_size: 4, received: 0 }, 201);
      }
      return new Response(null, { status: 204 });
    }));
    const handle = startResumableUpload("p1", file(), { stride: 1, fps: 30 });
    await new Promise((r) => setTimeout(r, 10));
    handle.abort();
    answer();
    await expect(handle.promise).rejects.toBeInstanceOf(UploadCancelled);
    await new Promise((r) => setTimeout(r, 10));
    expect(calls).toContain("DELETE /api/uploads/u1");
    expect(rememberedUploads()).toEqual([]);
  });

  it("abort cancels the chunk in flight and does not retry it", async () => {
    let signal: AbortSignal | undefined;
    let puts = 0;
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "POST") return json({ upload_id: "u1", chunk_size: 4, received: 0 }, 201);
      if (method === "DELETE") return new Response(null, { status: 204 });
      puts += 1;
      signal = init?.signal ?? undefined;
      return new Promise<Response>((_, reject) =>
        signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))));
    }));
    const handle = startResumableUpload("p1", file(), { stride: 1, fps: 30 });
    await new Promise((r) => setTimeout(r, 10));
    expect(signal?.aborted).toBe(false);
    handle.abort();
    await expect(handle.promise).rejects.toBeInstanceOf(UploadCancelled);
    await new Promise((r) => setTimeout(r, 10));
    expect(signal?.aborted).toBe(true);
    expect(puts).toBe(1);
    expect(waits).toEqual([]);
  });

  it("abort while waiting offline rejects promptly", async () => {
    uploaderEnv.online = () => new Promise<void>(() => {}); // never comes back
    fakeServer();
    const handle = startResumableUpload("p1", file(), { stride: 1, fps: 30 });
    await new Promise((r) => setTimeout(r, 10));
    handle.abort();
    await expect(handle.promise).rejects.toBeInstanceOf(UploadCancelled);
  });

  it("a 409 that moves nothing waits before trying again", async () => {
    const server = fakeServer({ received: 4, reportReceived: 4 });
    let first = true;
    const inner = globalThis.fetch as typeof fetch;
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === "PUT" && first) {
        first = false; // the server still says 4 at the offset the client has
        return json({ detail: "resync", received: 4 }, 409);
      }
      return inner(url, init);
    }));
    await startResumableUpload("p1", file(), { stride: 1, fps: 30 }, undefined, rem).promise;
    expect(waits).toEqual([1000]);
    expect(server.puts()).toEqual(["4", "8"]);
  });

  it("a 409 that carries the job resolves with it", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      calls.push(method);
      if (method === "POST") return json({ upload_id: "u1", chunk_size: 4, received: 0 }, 201);
      return json({ detail: "done", received: 10, job_id: "j9" }, 409);
    }));
    expect(await startResumableUpload("p1", file(), { stride: 1, fps: 30 }).promise).toEqual({ job_id: "j9" });
    expect(calls).toEqual(["POST", "PUT"]);
    expect(rememberedUploads()).toEqual([]);
  });

  it("resuming an upload that already has its job resolves without sending", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      calls.push(init?.method ?? "GET");
      return json({ upload_id: "u1", project_id: "p1", name: "a.mp4", size: 10, received: 10, job_id: "j5" });
    }));
    localStorage.setItem("vdp.uploads", JSON.stringify([rem]));
    expect(await startResumableUpload("p1", file(), { stride: 1, fps: 30 }, undefined, rem).promise).toEqual({ job_id: "j5" });
    expect(calls).toEqual(["GET"]);
    expect(rememberedUploads()).toEqual([]);
  });
});

describe("remembered uploads", () => {
  it("matches a file by name, size and modified time", () => {
    const r = { upload_id: "u1", project_id: "p1", name: "a.mp4", size: 10, lastModified: 111 };
    expect(sameFile(r, file())).toBe(true);
    expect(sameFile(r, file("a.mp4", 11))).toBe(false);
    expect(sameFile(r, file("a.mp4", 10, 222))).toBe(false);
  });

  it("forgets one upload and survives junk in storage", () => {
    localStorage.setItem("vdp.uploads", "not json");
    expect(rememberedUploads()).toEqual([]);
    localStorage.setItem("vdp.uploads", JSON.stringify([{ upload_id: "a" }, { upload_id: "b" }]));
    forgetUpload("a");
    expect(rememberedUploads().map((r) => r.upload_id)).toEqual(["b"]);
  });
});
