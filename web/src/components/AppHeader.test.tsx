// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { browser } from "../api/auth";
import { api } from "../api/rest";
import type { Health } from "../api/types";
import { HealthContext } from "../hooks/useHealth";
import { AppHeader } from "./AppHeader";

vi.mock("../api/rest", async (original) => ({
  ...(await original<typeof import("../api/rest")>()),
  api: { logout: vi.fn() },
}));

const health = (auth?: Health["auth"]): Health => ({
  status: "ok", version: "0", engine: { loaded: true, device: "cuda:0", error: null }, auth,
});

function renderHeader(h: Health) {
  return render(
    <MemoryRouter>
      <HealthContext.Provider value={{ health: h, unreachable: false }}>
        <AppHeader />
      </HealthContext.Provider>
    </MemoryRouter>,
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("AppHeader", () => {
  it("shows no Log out button when login is off or the server predates it", () => {
    renderHeader(health({ enabled: false }));
    expect(screen.queryByRole("button", { name: "Log out" })).toBeNull();
    cleanup();
    renderHeader(health());
    expect(screen.queryByRole("button", { name: "Log out" })).toBeNull();
  });

  it("logs out, then goes to the login page", async () => {
    vi.mocked(api.logout).mockResolvedValue(undefined);
    const go = vi.spyOn(browser, "go").mockImplementation(() => {});
    renderHeader(health({ enabled: true }));
    fireEvent.click(screen.getByRole("button", { name: "Log out" }));
    await waitFor(() => expect(go).toHaveBeenCalledWith("/login"));
    expect(api.logout).toHaveBeenCalled();
  });

  it("goes to the login page even when the logout request fails", async () => {
    vi.mocked(api.logout).mockRejectedValue(new Error("offline"));
    const go = vi.spyOn(browser, "go").mockImplementation(() => {});
    renderHeader(health({ enabled: true }));
    fireEvent.click(screen.getByRole("button", { name: "Log out" }));
    await waitFor(() => expect(go).toHaveBeenCalledWith("/login"));
  });
});
