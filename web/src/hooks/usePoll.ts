import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Fetch now, then every `intervalMs` (null = once) until unmount or a deps change. The latest fetcher is kept in
 * a ref, so an inline closure does not restart the loop on every render; `refresh()` fetches immediately.
 */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number | null, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const [nonce, setNonce] = useState(0);
  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const value = await fetcherRef.current();
        if (closed) return;
        setData(value);
        setError(null);
      } catch (e) {
        if (closed) return;
        setError(e instanceof Error ? e.message : String(e));
      }
      if (!closed && intervalMs !== null) timer = setTimeout(() => void tick(), intervalMs);
    };
    void tick();
    return () => {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, nonce, ...deps]);

  return { data, error, refresh };
}
