// @vitest-environment jsdom
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePoll } from "./usePoll";

function setHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
}

beforeEach(() => setHidden(false));
afterEach(() => {
  cleanup();
  setHidden(false);
});

describe("usePoll", () => {
  it("fetches immediately while the page is visible", async () => {
    const fetcher = vi.fn(async () => 1);
    const { result } = renderHook(() => usePoll(fetcher, null, []));
    await waitFor(() => expect(result.current.data).toBe(1));
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("skips the fetch while the tab is hidden but keeps rescheduling", async () => {
    const fetcher = vi.fn(async () => 1);
    setHidden(true);
    renderHook(() => usePoll(fetcher, 15, []));
    await new Promise((r) => setTimeout(r, 80)); // several reschedules' worth
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("refreshes at once when the tab becomes visible again", async () => {
    const fetcher = vi.fn(async () => 1);
    setHidden(true);
    renderHook(() => usePoll(fetcher, 60000, []));
    await new Promise((r) => setTimeout(r, 20));
    expect(fetcher).not.toHaveBeenCalled();

    setHidden(false);
    document.dispatchEvent(new Event("visibilitychange"));
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  });
});
