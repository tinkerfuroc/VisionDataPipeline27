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
  | { type: "overlay" }
  /** mark the current frame as the first / last frame of the range to clear */
  | { type: "mark"; edge: "start" | "end" }
  /** `-` shrinks, `=` grows the brush */
  | { type: "brushSize"; dir: 1 | -1 }
  /** `O` / `Shift+O`: next / previous object */
  | { type: "object"; dir: 1 | -1 }
  /** `N`: start a new object */
  | { type: "newObject" };

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
    case "[":
      return { type: "mark", edge: "start" };
    case "]":
      return { type: "mark", edge: "end" };
    case "-":
      return { type: "brushSize", dir: -1 };
    case "=":
    case "+":
      return { type: "brushSize", dir: 1 };
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
    case "o":
      return { type: "object", dir: e.shiftKey ? -1 : 1 };
    case "n":
      return { type: "newObject" };
  }
  return null;
}

// A non-text <input> (a checkbox, a button, ...) has no text cursor to protect, so it must not swallow shortcuts.
const NON_TEXT_INPUT_TYPES = new Set(["checkbox", "radio", "button", "submit", "range", "file", "color"]);

export function isField(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  if (t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable) return true;
  if (t.tagName === "INPUT") return !NON_TEXT_INPUT_TYPES.has((t as HTMLInputElement).type.toLowerCase());
  return false;
}

export function useKeyboard(onAction: (action: KeyAction) => void): void {
  const handler = useRef(onAction);
  handler.current = onAction;

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isField(e.target)) return;
      const action = keyAction(e);
      if (action === null) return;
      // Space on a focused checkbox or radio must keep toggling it (they are not text fields, so they get here)
      const t = e.target;
      if (action.type === "play" && t instanceof HTMLInputElement && (t.type === "checkbox" || t.type === "radio")) return;
      e.preventDefault();
      // Enter / Space would otherwise also activate whichever button was clicked last
      if (
        (action.type === "accept" || action.type === "play") &&
        document.activeElement instanceof HTMLElement &&
        document.activeElement.tagName === "BUTTON"
      ) {
        document.activeElement.blur();
      }
      handler.current(action);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
