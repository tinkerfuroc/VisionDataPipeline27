// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { browser, toLogin } from "./auth";
import { api, request } from "./rest";

const settles = (p: Promise<unknown>) =>
  Promise.race([
    p.then(() => "settled", () => "settled"),
    new Promise((resolve) => setTimeout(() => resolve("pending"), 20)),
  ]);

let go: MockInstance<(url: string) => void>;
beforeEach(() => {
  go = vi.spyOn(browser, "go").mockImplementation(() => {});
  window.history.replaceState(null, "", "/label/c1?f=3");
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("login redirects", () => {
  it("toLogin keeps the current page as next", () => {
    toLogin();
    expect(go).toHaveBeenCalledWith("/login?next=%2Flabel%2Fc1%3Ff%3D3");
  });

  it("toLogin keeps the hash, so a phone upload link returns to its page", () => {
    window.history.replaceState(null, "", "/#/upload?project=p1");
    toLogin();
    expect(go).toHaveBeenCalledWith("/login?next=%2F%23%2Fupload%3Fproject%3Dp1");
  });

  it("a 401 login required sends the browser to log in and never settles", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "login required" }), { status: 401 })));
    expect(await settles(request("/api/projects"))).toBe("pending");
    expect(go).toHaveBeenCalledWith("/login?next=%2Flabel%2Fc1%3Ff%3D3");
  });

  it("any other 401 is an ordinary error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "nope" }), { status: 401 })));
    await expect(request("/api/projects")).rejects.toMatchObject({ status: 401, message: "nope" });
    expect(go).not.toHaveBeenCalled();
  });

  it("an upload refused for login sends the browser to log in", async () => {
    class Xhr {
      static last: Xhr;
      upload = { onprogress: null as unknown };
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      onabort: (() => void) | null = null;
      status = 0;
      statusText = "";
      responseText = "";
      constructor() { Xhr.last = this; }
      open() {}
      setRequestHeader() {}
      send() {}
      abort() {}
    }
    vi.stubGlobal("XMLHttpRequest", Xhr);
    const handle = api.uploadClip("p1", new File(["x"], "a.mp4"), { stride: 1, fps: 30 });
    Xhr.last.status = 401;
    Xhr.last.responseText = JSON.stringify({ detail: "login required" });
    Xhr.last.onload!();
    expect(await settles(handle.promise)).toBe("pending");
    expect(go).toHaveBeenCalledWith("/login?next=%2Flabel%2Fc1%3Ff%3D3");
  });

  it("logout posts to the server", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    await api.logout();
    expect(fetchMock).toHaveBeenCalledWith("/api/auth/logout", { method: "POST" });
  });
});
