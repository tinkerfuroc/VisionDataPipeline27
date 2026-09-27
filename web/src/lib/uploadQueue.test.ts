import { describe, expect, it } from "vitest";
import {
  cancelAll,
  cancelCurrent,
  dismissFailed,
  emptyUploadQueue,
  failCurrent,
  finishCurrent,
  queueFiles,
  removeWaiting,
  startNext,
  updateProgress,
  type UploadQueueState,
} from "./uploadQueue";

const file = (name: string, size = 10) => new File([new Uint8Array(size)], name);
const snapshot = (over: Partial<{ projectId: string; stride: number; fps: number; maxBytes: number | null }> = {}) => ({
  projectId: "p1",
  stride: 1,
  fps: 30,
  maxBytes: null,
  ...over,
});
const idGen = (start = 0) => {
  let n = start;
  return () => n++;
};

describe("queueFiles", () => {
  it("queues every valid file, snapshotting the project id, stride and fps", () => {
    const { state, skipped } = queueFiles(emptyUploadQueue, [file("a.mp4"), file("b.zip")], snapshot({ stride: 2, fps: 15 }), idGen());
    expect(skipped).toEqual([]);
    expect(state.waiting.map((i) => i.file.name)).toEqual(["a.mp4", "b.zip"]);
    expect(state.waiting.every((i) => i.projectId === "p1" && i.stride === 2 && i.fps === 15)).toBe(true);
  });

  it("skips an invalid file and reports why, but still queues the valid ones in the same selection", () => {
    const { state, skipped } = queueFiles(emptyUploadQueue, [file("notes.txt"), file("a.mp4")], snapshot(), idGen());
    expect(skipped).toEqual([{ name: "notes.txt", reason: "That file is not a video or a .zip of images." }]);
    expect(state.waiting.map((i) => i.file.name)).toEqual(["a.mp4"]);
  });

  it("a later change to the snapshot does not affect files already queued", () => {
    const first = queueFiles(emptyUploadQueue, [file("a.mp4")], snapshot({ stride: 1 }), idGen()).state;
    const second = queueFiles(first, [file("b.mp4")], snapshot({ stride: 5 }), idGen(1)).state;
    expect(second.waiting.find((i) => i.file.name === "a.mp4")?.stride).toBe(1);
    expect(second.waiting.find((i) => i.file.name === "b.mp4")?.stride).toBe(5);
  });

  it("assigns ids in the order given by nextId", () => {
    const gen = idGen(5);
    const { state } = queueFiles(emptyUploadQueue, [file("a.mp4"), file("b.mp4")], snapshot(), gen);
    expect(state.waiting.map((i) => i.id)).toEqual([5, 6]);
  });
});

describe("startNext", () => {
  it("moves the first waiting file into current, in order", () => {
    const queued = queueFiles(emptyUploadQueue, [file("a.mp4", 100), file("b.mp4")], snapshot(), idGen()).state;
    const started = startNext(queued);
    expect(started?.item.file.name).toBe("a.mp4");
    expect(started?.state.current).toEqual({ ...started!.item, sent: 0, total: 100 });
    expect(started?.state.waiting.map((i) => i.file.name)).toEqual(["b.mp4"]);
  });

  it("does nothing when something is already running", () => {
    const running: UploadQueueState = {
      current: { id: 0, file: file("a.mp4"), projectId: "p1", stride: 1, fps: 30, sent: 0, total: 10 },
      waiting: [{ id: 1, file: file("b.mp4"), projectId: "p1", stride: 1, fps: 30 }],
      failed: [],
    };
    expect(startNext(running)).toBeNull();
  });

  it("does nothing when nothing is waiting", () => {
    expect(startNext(emptyUploadQueue)).toBeNull();
  });
});

describe("updateProgress / finishCurrent / failCurrent / cancelCurrent", () => {
  const started = () => startNext(queueFiles(emptyUploadQueue, [file("a.mp4", 100)], snapshot(), idGen()).state)!;

  it("updates progress on the current file", () => {
    const { state, item } = started();
    const next = updateProgress(state, item.id, 40, 100);
    expect(next.current?.sent).toBe(40);
  });

  it("ignores progress from a stale (no longer current) id", () => {
    const { state, item } = started();
    const next = updateProgress(state, item.id + 1, 40, 100);
    expect(next).toBe(state);
  });

  it("finishing the current file clears it and lets the queue move on to the next one", () => {
    const { state, item } = started();
    const finished = finishCurrent(state, item.id);
    expect(finished.current).toBeNull();
    const advanced = startNext(finished);
    expect(advanced).toBeNull(); // nothing else was waiting in this fixture
  });

  it("a failure moves the current file into failed with its message and clears current, so the queue moves on", () => {
    const withWaiting = queueFiles(emptyUploadQueue, [file("a.mp4"), file("b.mp4")], snapshot(), idGen()).state;
    const { state, item } = startNext(withWaiting)!;
    const failed = failCurrent(state, item.id, "the server refused it");
    expect(failed.current).toBeNull();
    expect(failed.failed).toEqual([{ id: item.id, file: item.file, projectId: item.projectId, stride: item.stride, fps: item.fps, message: "the server refused it" }]);
    const advanced = startNext(failed);
    expect(advanced?.item.file.name).toBe("b.mp4");
  });

  it("cancelling the current file clears it without adding a failed entry, so the queue moves on", () => {
    const withWaiting = queueFiles(emptyUploadQueue, [file("a.mp4"), file("b.mp4")], snapshot(), idGen()).state;
    const { state, item } = startNext(withWaiting)!;
    const cancelled = cancelCurrent(state, item.id);
    expect(cancelled.current).toBeNull();
    expect(cancelled.failed).toEqual([]);
    const advanced = startNext(cancelled);
    expect(advanced?.item.file.name).toBe("b.mp4");
  });

  it("ignores finish/fail/cancel for a stale id", () => {
    const { state, item } = started();
    expect(finishCurrent(state, item.id + 1)).toBe(state);
    expect(failCurrent(state, item.id + 1, "x")).toBe(state);
    expect(cancelCurrent(state, item.id + 1)).toBe(state);
  });
});

describe("cancelAll", () => {
  it("clears the current file and every waiting file", () => {
    const withWaiting = queueFiles(emptyUploadQueue, [file("a.mp4"), file("b.mp4"), file("c.mp4")], snapshot(), idGen()).state;
    const { state } = startNext(withWaiting)!;
    const cleared = cancelAll(state);
    expect(cleared.current).toBeNull();
    expect(cleared.waiting).toEqual([]);
  });
});

describe("removeWaiting / dismissFailed", () => {
  it("removes one waiting file by id, leaving the others in order", () => {
    const { state } = queueFiles(emptyUploadQueue, [file("a.mp4"), file("b.mp4"), file("c.mp4")], snapshot(), idGen());
    const removed = removeWaiting(state, state.waiting[1].id);
    expect(removed.waiting.map((i) => i.file.name)).toEqual(["a.mp4", "c.mp4"]);
  });

  it("dismisses one failed file by id, leaving the others", () => {
    const withWaiting = queueFiles(emptyUploadQueue, [file("a.mp4"), file("b.mp4")], snapshot(), idGen()).state;
    const first = startNext(withWaiting)!;
    const afterFirstFail = failCurrent(first.state, first.item.id, "boom");
    const second = startNext(afterFirstFail)!;
    const afterSecondFail = failCurrent(second.state, second.item.id, "bang");
    expect(afterSecondFail.failed).toHaveLength(2);
    const dismissed = dismissFailed(afterSecondFail, afterSecondFail.failed[0].id);
    expect(dismissed.failed.map((f) => f.message)).toEqual(["bang"]);
  });
});
