import { describe, expect, it } from "vitest";
import type { JobEvent, JobSnapshot } from "../api/types";
import { idleJob, isTerminal, jobReducer, type JobView } from "./jobReducer";

const feed = (...events: JobEvent[]): JobView =>
  events.reduce((v, event) => jobReducer(v, { type: "event", event }), idleJob);
const snapshot = (over: Partial<JobSnapshot>): JobSnapshot => ({
  id: "j", kind: "propagate", status: "running", result: null, error: null, created_at: "", n_events: 0, ...over,
});

describe("jobReducer", () => {
  it("tracks propagation progress and the last streamed frame", () => {
    const v = feed(
      { seq: 0, type: "progress", done: 0, total: 40 },
      { seq: 1, type: "frame", frame: 12, has_mask: true, done: 1, total: 40 },
    );
    expect(v).toMatchObject({ status: "running", done: 1, total: 40, lastFrame: 12, nextSeq: 2 });
  });
  it("keeps total unknown for import progress events", () => {
    expect(feed({ seq: 0, type: "progress", done: 17 })).toMatchObject({ done: 17, total: null });
  });
  it("ignores events it has already seen (the socket replays from seq 0 on reconnect)", () => {
    const v = feed({ seq: 0, type: "progress", done: 5, total: 9 }, { seq: 1, type: "progress", done: 6, total: 9 });
    expect(jobReducer(v, { type: "event", event: { seq: 0, type: "progress", done: 5, total: 9 } })).toBe(v);
  });
  it("keeps the last 50 log lines", () => {
    const events = Array.from({ length: 60 }, (_, i): JobEvent => ({ seq: i, type: "log", message: `m${i}` }));
    const v = feed(...events);
    expect(v.logs).toHaveLength(50);
    expect(v.logs[0]).toBe("m10");
  });
  it("finishes with the result spread into the final event", () => {
    const v = feed({ seq: 0, type: "done", processed: 11, written: 11, lost_at: null });
    expect(v.status).toBe("done");
    expect(v.result).toEqual({ processed: 11, written: 11, lost_at: null });
  });
  it("reports cancellation and errors", () => {
    expect(feed({ seq: 0, type: "cancelled", cancelled: true }).status).toBe("cancelled");
    const failed = feed({ seq: 0, type: "error", message: "PropagationFailed: boom" });
    expect(failed).toMatchObject({ status: "error", error: "PropagationFailed: boom" });
  });
  it("takes a terminal state from a polled snapshot when no final event arrived", () => {
    const running = jobReducer(idleJob, { type: "snapshot", snapshot: snapshot({ status: "running" }) });
    expect(running.status).toBe("running");
    const done = jobReducer(running, { type: "snapshot", snapshot: snapshot({ status: "done", result: { clip_id: "c" } }) });
    expect(done).toMatchObject({ status: "done", result: { clip_id: "c" } });
  });
  it("never leaves a terminal state except through reset", () => {
    const done = feed({ seq: 0, type: "done" });
    expect(jobReducer(done, { type: "snapshot", snapshot: snapshot({ status: "running" }) })).toBe(done);
    expect(jobReducer(done, { type: "event", event: { seq: 1, type: "progress", done: 1 } }).status).toBe("done");
    expect(jobReducer(done, { type: "reset" })).toBe(idleJob);
  });
  it("knows which states are terminal", () => {
    expect(["done", "error", "cancelled"].every((s) => isTerminal(s as JobView["status"]))).toBe(true);
    expect(["idle", "pending", "running"].some((s) => isTerminal(s as JobView["status"]))).toBe(false);
  });
});
