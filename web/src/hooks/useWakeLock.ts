import { useEffect } from "react";

/** Keep the screen on while `active` (where the browser supports it), so a phone does not lock mid-upload. The
 * browser drops the lock when the page is hidden; it is taken again when the page becomes visible. */
export function useWakeLock(active: boolean): void {
  useEffect(() => {
    if (!active || !("wakeLock" in navigator)) return;
    let lock: WakeLockSentinel | null = null;
    let stopped = false;
    const acquire = () => {
      if (document.visibilityState !== "visible") return;
      navigator.wakeLock.request("screen").then(
        (l) => {
          if (stopped) void l.release();
          else lock = l;
        },
        () => {
          /* refused (e.g. battery saver): uploads still resume after a lock */
        },
      );
    };
    acquire();
    document.addEventListener("visibilitychange", acquire);
    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", acquire);
      void lock?.release();
    };
  }, [active]);
}
