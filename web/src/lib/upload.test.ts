import { describe, expect, it } from "vitest";
import { formatBytes, importKind, uploadPercent, uploadProblem } from "./upload";

describe("importKind", () => {
  it("recognises video files and zips by extension, ignoring case", () => {
    expect(importKind("bench.mp4")).toBe("video");
    expect(importKind("RUN.MOV")).toBe("video");
    expect(importKind("frames.zip")).toBe("zip");
    expect(importKind("notes.txt")).toBeNull();
    expect(importKind("noextension")).toBeNull();
  });
});

describe("uploadProblem", () => {
  const ok = { name: "bench.mp4", size: 10, stride: 1, fps: 30, maxBytes: 1000 };
  it("says what stops the upload, or null", () => {
    expect(uploadProblem(ok)).toBeNull();
    expect(uploadProblem({ ...ok, name: null })).toBe("Choose a video or a .zip of images.");
    expect(uploadProblem({ ...ok, name: "notes.txt" })).toBe("That file is not a video or a .zip of images.");
    expect(uploadProblem({ ...ok, stride: 0 })).toBe("Keep every Nth frame needs a whole number of 1 or more.");
    expect(uploadProblem({ ...ok, name: "run.zip", fps: 0 })).toBe("Frames per second must be above 0.");
    expect(uploadProblem({ ...ok, size: 2000 })).toBe("That file is 2.0 kB; the limit is 1000 B.");
    expect(uploadProblem({ ...ok, size: 2000, maxBytes: null })).toBeNull(); // no limit known: let the server decide
  });
});

describe("formatBytes and uploadPercent", () => {
  it("scales units and reports whole percents", () => {
    expect(formatBytes(3)).toBe("3 B");
    expect(formatBytes(1536)).toBe("1.5 kB");
    expect(formatBytes(8 * 2 ** 30)).toBe("8.0 GB");
    expect(uploadPercent(50, 200)).toBe(25);
    expect(uploadPercent(0, 0)).toBeNull();
  });
});
