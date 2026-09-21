import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, detailText, maskUrl, request, retryInteractive } from "./rest";
import type { ExportRequest } from "./types";

function mockFetch(...responses: Response[]) {
  const fn = vi.fn();
  for (const r of responses) fn.mockResolvedValueOnce(r);
  vi.stubGlobal("fetch", fn);
  return fn;
}
const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

describe("request", () => {
  it("returns the parsed JSON body", async () => {
    mockFetch(jsonResponse({ ok: 1 }));
    expect(await request<{ ok: number }>("/api/x")).toEqual({ ok: 1 });
  });

  it("returns undefined for 204", async () => {
    mockFetch(new Response(null, { status: 204 }));
    expect(await request<void>("/api/x", { method: "DELETE" })).toBeUndefined();
  });

  it("raises ApiError with the string detail and the job id of a 409", async () => {
    mockFetch(jsonResponse({ detail: "job abc is still running", job_id: "abc" }, 409));
    const err = await request<never>("/api/x").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.message).toBe("job abc is still running");
    expect(err.jobId).toBe("abc");
  });

  it("flattens a FastAPI 422 detail list", async () => {
    mockFetch(
      jsonResponse(
        { detail: [{ loc: ["body", "prompt", "box"], msg: "Field required" }, { loc: ["body", "frame"], msg: "too small" }] },
        422,
      ),
    );
    const err = await request<never>("/api/x").catch((e) => e);
    expect(err.status).toBe(422);
    expect(err.message).toBe("prompt.box: Field required; frame: too small");
    expect(err.jobId).toBeNull();
  });

  it("falls back to the status line when the error body is not JSON", async () => {
    mockFetch(new Response("<html>bad gateway</html>", { status: 502, statusText: "Bad Gateway" }));
    const err = await request<never>("/api/x").catch((e) => e);
    expect(err.message).toBe("502 Bad Gateway");
  });
});

describe("detailText", () => {
  it("uses the fallback for bodies without a detail", () => {
    expect(detailText({ nope: 1 }, "fallback")).toBe("fallback");
    expect(detailText(null, "fallback")).toBe("fallback");
  });
});

describe("retryInteractive", () => {
  const noSleep = () => Promise.resolve();

  it("retries while the 409 names the interactive lease, then succeeds", async () => {
    const fn = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new ApiError(409, "busy", "interactive"))
      .mockRejectedValueOnce(new ApiError(409, "busy", "interactive"))
      .mockResolvedValueOnce("ok");
    expect(await retryInteractive(fn, 10, 1, noSleep)).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("does not retry a 409 that names a real job", async () => {
    const fn = vi.fn<() => Promise<string>>().mockRejectedValue(new ApiError(409, "busy", "abc123"));
    await expect(retryInteractive(fn, 10, 1, noSleep)).rejects.toMatchObject({ jobId: "abc123" });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("gives up after the configured number of attempts", async () => {
    const fn = vi.fn<() => Promise<string>>().mockRejectedValue(new ApiError(409, "busy", "interactive"));
    await expect(retryInteractive(fn, 3, 1, noSleep)).rejects.toBeInstanceOf(ApiError);
    expect(fn).toHaveBeenCalledTimes(3);
  });
});

describe("api", () => {
  it("posts a propagate request with the server's field names", async () => {
    const fetchMock = mockFetch(jsonResponse({ job_id: "j1" }, 202));
    expect(await api.propagate("c1", 7, "both")).toEqual({ job_id: "j1" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clips/c1/propagate");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ from_frame: 7, direction: "both" });
  });

  it("sends accept with class_id and candidate only when given", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({}));
    await api.accept("c1", 3, { type: "clicks", points: [[5, 6, 1]] }, { classId: 2, candidate: 1 });
    await api.accept("c1", 3, { type: "clicks", points: [[5, 6, 1]] });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      frame: 3, prompt: { type: "clicks", points: [[5, 6, 1]] }, class_id: 2, candidate: 1,
    });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
      frame: 3, prompt: { type: "clicks", points: [[5, 6, 1]] }, candidate: 0,
    });
  });

  it("uses `from` as the prune field name and uploads videos as multipart", async () => {
    const fetchMock = mockFetch(jsonResponse({}), jsonResponse({ job_id: "j2" }, 202));
    await api.prune("c1", 12);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ from: 12 });
    await api.importVideo("p1", new File(["x"], "a.mp4"), 2);
    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toBe("/api/projects/p1/clips/import-video");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("stride")).toBe("2");
    expect(init.headers).toBeUndefined();
  });

  it("versions mask URLs", () => {
    expect(maskUrl("c1", 4, "2.1")).toBe("/api/clips/c1/masks/4?v=2.1");
  });

  it("posts an export request with the server's field names", async () => {
    const fetchMock = mockFetch(jsonResponse({ job_id: "j3" }, 202));
    const body: ExportRequest = {
      clips: ["a", "b"],
      formats: ["coco", "masks"],
      options: { split_mode: "per_clip", val_clips: ["b"], val_fraction: 0.2, stride: 1, include_empty: true },
    };
    expect(await api.startExport("p1", body)).toEqual({ job_id: "j3" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/projects/p1/export");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(body);
  });
});
