// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PredictionOverlay } from "./PredictionOverlay";

afterEach(cleanup);

describe("PredictionOverlay", () => {
  it("draws a detection run's boxes as unfilled rectangles and masks as polygons", () => {
    const box = { class_id: 0, score: 0.9, box: [0.1, 0.2, 0.5, 0.6], polygons: [[0.1, 0.2, 0.5, 0.2, 0.5, 0.6, 0.1, 0.6]] };
    const mask = { class_id: 0, polygons: [[0.2, 0.2, 0.8, 0.2, 0.8, 0.8]] };
    const { container } = render(
      <PredictionOverlay width={100} height={50} groundTruth={[mask]} predictions={[box]} showGt showPred scores={false} />,
    );
    const rect = container.querySelector("rect")!;
    expect([rect.getAttribute("x"), rect.getAttribute("y"), rect.getAttribute("width"), rect.getAttribute("height")])
      .toEqual(["10", "10", "40", "20"]);
    expect(rect.getAttribute("fill")).toBe("none");
    expect(container.querySelectorAll("polygon")).toHaveLength(1);  // the ground-truth mask
  });
});
