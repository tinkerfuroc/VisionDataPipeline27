import { useEffect, useReducer, useRef, useState } from "react";
import { WS_LOGIN_REQUIRED, toLogin } from "../api/auth";
import { ApiError, api, wsUrl } from "../api/rest";
import type { JobEvent } from "../api/types";
import { idleJob, isTerminal, jobReducer, type JobView } from "../lib/jobReducer";

export type JobHandle = JobView & { transport: "ws" | "poll" };

const POLL_MS = 1000;

/** Follow one job: WebSocket events while the socket lives, status polling if it drops before the job ends. */
export function useJob(jobId: string | null, onEvent?: (event: JobEvent) => void): JobHandle {
  const [view, dispatch] = useReducer(jobReducer, idleJob);
  const [transport, setTransport] = useState<"ws" | "poll">("ws");
  // Reset DURING RENDER (not only in the effect below) so the commit where `jobId` changes never exposes
  // the previous job's terminal status/result to a consumer such as Label.tsx's terminal-job effect.
  // `dispatch`/`setTransport` called here only take effect on React's next (replay) pass of this
  // component, so `view`/`transport` captured above are still stale for the remainder of THIS pass;
  // `effectiveView`/`effectiveTransport` make the value this same call returns already reflect the reset.
  const [seenJobId, setSeenJobId] = useState(jobId);
  let effectiveView = view;
  let effectiveTransport = transport;
  if (seenJobId !== jobId) {
    setSeenJobId(jobId);
    setTransport("ws");
    dispatch({ type: "reset" });
    effectiveView = idleJob;
    effectiveTransport = "ws";
  }
  const callback = useRef(onEvent);
  callback.current = onEvent;

  useEffect(() => {
    dispatch({ type: "reset" });
    setTransport("ws");
    if (jobId === null) return;

    let closed = false;
    let finished = false;
    let nextSeq = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async (): Promise<void> => {
      if (closed || finished) return;
      try {
        const snapshot = await api.getJob(jobId);
        if (closed) return;
        dispatch({ type: "snapshot", snapshot });
        if (isTerminal(snapshot.status)) {
          finished = true;
          return;
        }
      } catch (e) {
        if (closed) return;
        if (e instanceof ApiError && e.status === 404) {
          finished = true;
          const message = "the job no longer exists (was the server restarted?)";
          dispatch({ type: "event", event: { seq: nextSeq, type: "error", message } });
          return;
        }
      }
      timer = setTimeout(() => void poll(), POLL_MS);
    };

    const ws = new WebSocket(wsUrl(`/ws/jobs/${jobId}`));
    ws.onmessage = (m: MessageEvent) => {
      if (closed) return;
      const event = JSON.parse(String(m.data)) as JobEvent;
      if (event.seq < nextSeq) return;
      nextSeq = event.seq + 1;
      if (event.type === "done" || event.type === "cancelled" || event.type === "error") finished = true;
      dispatch({ type: "event", event });
      callback.current?.(event);
    };
    ws.onclose = (e: CloseEvent) => {
      if (closed || finished) return;
      if (e.code === WS_LOGIN_REQUIRED) {
        toLogin(); // the session expired: polling would only get 401s
        return;
      }
      setTransport("poll");
      void poll();
    };

    return () => {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
      ws.close();
    };
  }, [jobId]);

  return { ...effectiveView, transport: effectiveTransport };
}
