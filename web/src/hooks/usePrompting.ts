import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type Dispatch } from "react";
import { ApiError, api } from "../api/rest";
import type { Candidate } from "../api/types";
import { writeTarget, type Selection } from "../lib/activeObject";
import { draftReducer, emptyDraft, toPrompt, type Draft, type DraftAction } from "../lib/promptState";
import type { ClipHandle } from "./useClip";

export type Prompting = {
  draft: Draft;
  dispatch: Dispatch<DraftAction>;
  candidates: Candidate[];
  busy: boolean;
  error: string | null;
  /** id of the running GPU job a 409 named, so the page can attach to it */
  busyJobId: string | null;
  canAccept: boolean;
  /** several candidates (a text prompt) on a "New object" row: "Accept all as new objects" is shown */
  offersAcceptAll: boolean;
  canAcceptAll: boolean;
  accept: () => Promise<void>;
  /** every candidate becomes its own new object of the new object's class */
  acceptAll: () => Promise<void>;
  clear: () => void;
};

type Options = {
  clipId: string;
  frame: number;
  enabled: boolean;
  clip: ClipHandle;
  /** the Objects panel's active row; null until the clip loads */
  selection: Selection | null;
  /** the ids a "New object" accept created (first one first), so the page can make one active */
  onCreated: (ids: number[]) => void;
};

const NO_MASK = "No usable mask: nothing was found, or the mask is too small or too large. Add a click or try another prompt.";

export function usePrompting({ clipId, frame, enabled, clip, selection, onCreated }: Options): Prompting {
  const [draft, dispatch] = useReducer(draftReducer, undefined, () => emptyDraft());
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyJobId, setBusyJobId] = useState<string | null>(null);
  const [accepting, setAccepting] = useState(false);
  const acceptingRef = useRef(false);
  const [attempt, setAttempt] = useState(0);
  const requestId = useRef(0);
  const draftIsStale = useRef(false);
  const objectId = selection?.kind === "object" ? selection.id : null;
  const target = useMemo(() => writeTarget(selection), [selection]);

  // "submit" (text) and "box" can repeat identical content after a failed preview; bump a counter so the
  // preview effect re-requests even though the prompt itself did not change. Everything else that mutates
  // content already changes the prompt, and candidate/phrase/undo/tool must not re-request on their own.
  const wrappedDispatch = useCallback<Dispatch<DraftAction>>((action) => {
    if (action.type === "submit" || action.type === "box") setAttempt((a) => a + 1);
    dispatch(action);
  }, []);

  const promptKey = JSON.stringify([draft.tool, draft.points, draft.box, draft.query]);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- promptKey covers every field toPrompt reads
  const prompt = useMemo(() => toPrompt(draft), [promptKey]);

  // a draft belongs to one frame and one object, and only exists while prompting is possible
  useEffect(() => {
    draftIsStale.current = true; // the effect below runs in this same commit and still sees the old draft
    dispatch({ type: "clear" });
    setError(null);
  }, [clipId, frame, enabled, objectId]);

  useEffect(() => {
    const id = ++requestId.current; // also invalidates any request still in flight
    const stale = draftIsStale.current;
    draftIsStale.current = false;
    if (prompt === null || !enabled || stale) {
      setCandidates([]);
      setBusy(false);
      return;
    }
    setBusy(true);
    api
      .prompt(clipId, frame, prompt, objectId)
      .then((r) => {
        if (id !== requestId.current) return;
        setCandidates(r.candidates);
        setError(r.candidates.length === 0 ? NO_MASK : null);
        setBusyJobId(null);
        setBusy(false);
      })
      .catch((e: unknown) => {
        if (id !== requestId.current) return;
        setCandidates([]);
        setError(e instanceof Error ? e.message : String(e));
        setBusyJobId(e instanceof ApiError && e.status === 409 ? e.jobId : null);
        setBusy(false);
      });
  }, [clipId, frame, enabled, prompt, attempt, objectId]);

  const canAccept =
    enabled && prompt !== null && !busy && !accepting && draft.candidate < candidates.length && target !== null;
  const offersAcceptAll = objectId === null && candidates.length > 1;
  const canAcceptAll = canAccept && offersAcceptAll;

  const apply = clip.apply;
  const send = useCallback(
    async (choice: { candidate: number; candidates?: number[] }) => {
      if (acceptingRef.current) return; // a second call in the same tick is ignored
      acceptingRef.current = true;
      setAccepting(true);
      try {
        if (prompt === null || candidates.length === 0) return;
        if (target === null) {
          setError("Choose a class for the new object before accepting its first mask.");
          return;
        }
        const next = await api.accept(clipId, frame, prompt, target, choice);
        apply(next, true);
        if (target.objectId === null) onCreated(next.object_ids);
        dispatch({ type: "clear" });
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setBusyJobId(e instanceof ApiError && e.status === 409 ? e.jobId : null);
      } finally {
        acceptingRef.current = false;
        setAccepting(false);
      }
    },
    [prompt, candidates.length, target, clipId, frame, apply, onCreated],
  );
  const accept = useCallback(() => send({ candidate: draft.candidate }), [send, draft.candidate]);
  const acceptAll = useCallback(() => send({ candidate: 0, candidates: candidates.map((_, i) => i) }), [send, candidates]);

  const clear = useCallback(() => {
    dispatch({ type: "clear" });
    setError(null);
  }, []);

  return {
    draft, dispatch: wrappedDispatch, candidates, busy, error, busyJobId, canAccept, offersAcceptAll, canAcceptAll,
    accept, acceptAll, clear,
  };
}
