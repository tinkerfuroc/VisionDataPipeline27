import type { Box, Point, PromptSpec } from "../api/types";

export type Tool = "clicks" | "box" | "text";

export type Draft = {
  tool: Tool;
  points: Point[];
  box: Box | null;
  /** what is typed in the text box */
  phrase: string;
  /** the phrase that was submitted; only this is sent to the server */
  query: string;
  /** index of the chosen candidate (text prompts can return several) */
  candidate: number;
};

export const emptyDraft = (tool: Tool = "clicks"): Draft => ({ tool, points: [], box: null, phrase: "", query: "", candidate: 0 });

export type DraftAction =
  | { type: "tool"; tool: Tool }
  | { type: "point"; x: number; y: number; label: 0 | 1 }
  | { type: "box"; box: Box }
  | { type: "undo" }
  | { type: "phrase"; phrase: string }
  | { type: "submit" }
  | { type: "candidate"; index: number }
  | { type: "clear" };

export function draftReducer(d: Draft, a: DraftAction): Draft {
  switch (a.type) {
    case "tool":
      return a.tool === d.tool ? d : { ...emptyDraft(a.tool), phrase: d.phrase };
    case "point":
      if (d.tool === "text" || (d.tool === "box" && d.box === null)) return d;
      return { ...d, points: [...d.points, [a.x, a.y, a.label]], candidate: 0 };
    case "box":
      return d.tool === "box" ? { ...d, box: a.box, points: [], candidate: 0 } : d;
    case "undo":
      if (d.points.length > 0) return { ...d, points: d.points.slice(0, -1), candidate: 0 };
      if (d.box !== null) return { ...d, box: null, candidate: 0 };
      return d;
    case "phrase":
      return { ...d, phrase: a.phrase };
    case "submit":
      return d.tool === "text" ? { ...d, query: d.phrase.trim(), candidate: 0 } : d;
    case "candidate":
      return { ...d, candidate: Math.max(0, a.index) };
    case "clear":
      return emptyDraft(d.tool);
  }
}

/** The prompt to preview/accept for this draft, or null when there is nothing to send yet. */
export function toPrompt(d: Draft): PromptSpec | null {
  if (d.tool === "clicks") return d.points.length > 0 ? { type: "clicks", points: d.points } : null;
  if (d.tool === "box") return d.box !== null ? { type: "box", box: d.box, points: d.points } : null;
  return d.query !== "" ? { type: "text", phrase: d.query } : null;
}
