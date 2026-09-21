import { vi } from "vitest";

/** jsdom has no canvas implementation and no ResizeObserver; components only need them not to throw. */
export function stubCanvasAndObservers(): void {
  const ctx = new Proxy({}, { get: () => () => undefined, set: () => true });
  HTMLCanvasElement.prototype.getContext = (() => ctx) as unknown as HTMLCanvasElement["getContext"];
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
}
