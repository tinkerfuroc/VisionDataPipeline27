import { describe, expect, it } from "vitest";
import { exportLabel, slug } from "./exportName";

describe("export names", () => {
  it("slugs like the server", () => {
    expect(slug("Cups v2")).toBe("cups-v2");
    expect(slug("  Tassen / Schalen (final)  ")).toBe("tassen-schalen-final");
    expect(slug("--weird__name..")).toBe("weird__name");
    expect(slug("a".repeat(60))).toBe("a".repeat(48));
    expect(slug("x".repeat(47) + "-yz")).toBe("x".repeat(47));
    expect(slug("杯子")).toBe("");
  });
  it("labels an export by name, else by id", () => {
    expect(exportLabel({ id: "cups-v2", name: "Cups v2" })).toBe("Cups v2");
    expect(exportLabel({ id: "20260926-104433-a1b2c3", name: null })).toBe("20260926-104433-a1b2c3");
  });
});
