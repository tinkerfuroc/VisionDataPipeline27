import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type Dispatch } from "react";
import { ApiError, api } from "../api/rest";
import type { Candidate } from "../api/types";
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
  accept: () => Promise<void>;
  clear: () => void;
};

type Options = { clipId: string; frame: number; enabled: boolean; clip: ClipHandle; classId: number | null };

const NO_MASK = "No usable mask: nothing was found, or the mask is too small or too large. Add a click or try another prompt.";

export function usePrompting({ clipId, frame, enabled, clip, classId }: Options): Prompting {
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

  // a draft belongs to one frame, and only exists while prompting is possible
  useEffect(() => {
    draftIsStale.current = true; // the effect below runs in this same commit and still sees the old frame's draft
    dispatch({ type: "clear" });
    setError(null);
  }, [clipId, frame, enabled]);

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
      .prompt(clipId, frame, prompt)
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
  }, [clipId, frame, enabled, prompt, attempt]);

  const object = clip.detail?.clip.objects[0] ?? null;
  const needsClass = object === null;
  const canAccept =
    enabled && prompt !== null && !busy && !accepting && draft.candidate < candidates.length && !(needsClass && classId === null);

  const apply = clip.apply;
  const accept = useCallback(async () => {
    if (acceptingRef.current) return; // a second call in the same tick is ignored
    acceptingRef.current = true;
    setAccepting(true);
    try {
      if (prompt === null || candidates.length === 0) return;
      if (needsClass && classId === null) {
        setError("Choose a class before accepting the first mask.");
        return;
      }
      const sendClass = classId !== null && (object === null || object.class_id !== classId);
      const next = await api.accept(clipId, frame, prompt, { classId: sendClass ? classId : undefined, candidate: draft.candidate });
      apply(next, true);
      dispatch({ type: "clear" });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusyJobId(e instanceof ApiError && e.status === 409 ? e.jobId : null);
    } finally {
      acceptingRef.current = false;
      setAccepting(false);
    }
  }, [prompt, candidates.length, needsClass, classId, object, clipId, frame, draft.candidate, apply]);

  const clear = useCallback(() => {
    dispatch({ type: "clear" });
    setError(null);
  }, []);

  return { draft, dispatch: wrappedDispatch, candidates, busy, error, busyJobId, canAccept, accept, clear };
}
