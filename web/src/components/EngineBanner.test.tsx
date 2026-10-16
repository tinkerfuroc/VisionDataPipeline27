// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/rest";
import type { Health } from "../api/types";
import { EngineBanner } from "./EngineBanner";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { loadEngine: vi.fn() },
}));

const health = (engine: Health["engine"]): Health => ({ status: "ok", version: "0", engine });
afterEach(cleanup);

describe("EngineBanner", () => {
  it("shows nothing when the engine is loaded or health is still unknown", () => {
    const { container, rerender } = render(
      <EngineBanner health={health({ loaded: true, device: "cuda:0", error: null })} unreachable={false} />,
    );
    expect(container.textContent).toBe("");
    rerender(<EngineBanner health={null} unreachable={false} />);
    expect(container.textContent).toBe("");
  });
  it("says the model is loading", () => {
    render(<EngineBanner health={health({ loaded: false, device: null, error: null })} unreachable={false} />);
    expect(screen.getByRole("status").textContent).toContain("Loading the SAM3 model");
  });
  it("shows the load error with the two repair commands", () => {
    render(<EngineBanner health={health({ loaded: false, device: null, error: "no CUDA device" })} unreachable={false} />);
    const text = screen.getByRole("alert").textContent ?? "";
    expect(text).toContain("no CUDA device");
    expect(text).toContain("uv run vdp verify-weights");
    expect(text).toContain("uv run vdp convert-weights");
  });
  it("says when the server cannot be reached", () => {
    render(<EngineBanner health={null} unreachable={true} />);
    expect(screen.getByRole("alert").textContent).toContain("Cannot reach the server");
  });
  it("says SAM3 is paused for training, and Reload now loads it and refreshes health", async () => {
    vi.mocked(api.loadEngine).mockResolvedValue({ started: true });
    const refresh = vi.fn();
    render(
      <EngineBanner health={health({ loaded: false, device: "cuda:0", error: null, paused: true })} unreachable={false}
        refresh={refresh} />,
    );
    const text = screen.getByRole("status").textContent ?? "";
    expect(text).toContain("SAM3 paused for training");
    expect(text).toContain("memory pressure");
    expect(text).not.toContain("Loading the SAM3 model");
    fireEvent.click(screen.getByRole("button", { name: "Reload now" }));
    await waitFor(() => expect(refresh).toHaveBeenCalled());
    expect(api.loadEngine).toHaveBeenCalledTimes(1);
  });
});
