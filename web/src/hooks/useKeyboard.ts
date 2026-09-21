import { useEffect, useRef } from "react";

export type KeyAction =
  | { type: "accept" }
  | { type: "clear" }
  | { type: "undo" }
  | { type: "step"; delta: number }
  | { type: "play" }
  | { type: "keyframe"; dir: 1 | -1 }
  | { type: "nextEmpty" }
  | { type: "delete" }
  | { type: "overlay" };

type KeyLike = { key: string; shiftKey: boolean; ctrlKey: boolean; metaKey: boolean; altKey: boolean };

/** Spec §7 keyboard map. Ctrl / Meta / Alt combinations belong to the browser. */
export function keyAction(e: KeyLike): KeyAction | null {
  if (e.ctrlKey || e.metaKey || e.altKey) return null;
  switch (e.key) {
    case "Enter":
      return { type: "accept" };
    case "Escape":
      return { type: "clear" };
    case "ArrowRight":
      return { type: "step", delta: e.shiftKey ? 10 : 1 };
    case "ArrowLeft":
      return { type: "step", delta: e.shiftKey ? -10 : -1 };
    case " ":
      return { type: "play" };
    case "Delete":
      return { type: "delete" };
  }
  switch (e.key.toLowerCase()) {
    case "z":
      return { type: "undo" };
    case "k":
      return { type: "keyframe", dir: e.shiftKey ? -1 : 1 };
    case "g":
      return { type: "nextEmpty" };
    case "m":
      return { type: "overlay" };
  }
  return null;
}

const isField = (t: EventTarget | null): boolean =>
  t instanceof HTMLElement && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);

export function useKeyboard(onAction: (action: KeyAction) => void): void {
  const handler = useRef(onAction);
  handler.current = onAction;

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isField(e.target)) return;
      const action = keyAction(e);
      if (action === null) return;
      e.preventDefault();
      // Enter / Space would otherwise also activate whichever button was clicked last
      if (document.activeElement instanceof HTMLElement && document.activeElement.tagName === "BUTTON") document.activeElement.blur();
      handler.current(action);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
