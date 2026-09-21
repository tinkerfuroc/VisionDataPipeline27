import { describe, expect, it } from "vitest";
import { draftReducer, emptyDraft, toPrompt, type Draft, type DraftAction } from "./promptState";

const run = (d: Draft, ...actions: DraftAction[]) => actions.reduce(draftReducer, d);
const pt = (x: number, y: number, label: 0 | 1 = 1): DraftAction => ({ type: "point", x, y, label });

describe("draftReducer", () => {
  it("collects positive and negative clicks", () => {
    const d = run(emptyDraft(), pt(5, 6), pt(7, 8, 0));
    expect(d.points).toEqual([[5, 6, 1], [7, 8, 0]]);
    expect(toPrompt(d)).toEqual({ type: "clicks", points: [[5, 6, 1], [7, 8, 0]] });
  });
  it("sends nothing for an empty draft", () => {
    expect(toPrompt(emptyDraft("clicks"))).toBeNull();
    expect(toPrompt(emptyDraft("box"))).toBeNull();
    expect(toPrompt(emptyDraft("text"))).toBeNull();
  });
  it("ignores clicks in box mode until a box exists, then treats them as refinement", () => {
    const noBox = run(emptyDraft("box"), pt(1, 1));
    expect(noBox.points).toEqual([]);
    const d = run(emptyDraft("box"), { type: "box", box: [10, 10, 50, 50] }, pt(20, 20), pt(40, 40, 0));
    expect(toPrompt(d)).toEqual({ type: "box", box: [10, 10, 50, 50], points: [[20, 20, 1], [40, 40, 0]] });
  });
  it("drawing a new box discards the refinement clicks", () => {
    const d = run(emptyDraft("box"), { type: "box", box: [10, 10, 50, 50] }, pt(20, 20), { type: "box", box: [0, 0, 9, 9] });
    expect(d.box).toEqual([0, 0, 9, 9]);
    expect(d.points).toEqual([]);
  });
  it("ignores a box outside box mode", () => {
    expect(run(emptyDraft("clicks"), { type: "box", box: [1, 2, 3, 4] }).box).toBeNull();
  });
  it("undo removes the last click, then the box, then does nothing", () => {
    let d = run(emptyDraft("box"), { type: "box", box: [10, 10, 50, 50] }, pt(20, 20));
    d = draftReducer(d, { type: "undo" });
    expect(d.points).toEqual([]);
    expect(d.box).not.toBeNull();
    d = draftReducer(d, { type: "undo" });
    expect(d.box).toBeNull();
    expect(draftReducer(d, { type: "undo" })).toBe(d);
  });
  it("text prompts are sent only after submit, trimmed", () => {
    let d = run(emptyDraft("text"), { type: "phrase", phrase: "  red connector " });
    expect(toPrompt(d)).toBeNull();
    d = draftReducer(d, { type: "submit" });
    expect(toPrompt(d)).toEqual({ type: "text", phrase: "red connector" });
    expect(run(d, pt(1, 1)).points).toEqual([]);
  });
  it("remembers the chosen candidate and resets it when the prompt changes", () => {
    let d = run(emptyDraft("text"), { type: "phrase", phrase: "bolt" }, { type: "submit" }, { type: "candidate", index: 2 });
    expect(d.candidate).toBe(2);
    d = draftReducer(d, { type: "submit" });
    expect(d.candidate).toBe(0);
  });
  it("switching tools starts over but keeps the typed phrase; clear keeps the tool", () => {
    const d = run(emptyDraft("clicks"), pt(1, 1), { type: "phrase", phrase: "bolt" }, { type: "tool", tool: "text" });
    expect(d).toEqual({ ...emptyDraft("text"), phrase: "bolt" });
    expect(draftReducer(d, { type: "tool", tool: "text" })).toBe(d);
    expect(run(d, { type: "submit" }, { type: "clear" })).toEqual(emptyDraft("text"));
  });
});
