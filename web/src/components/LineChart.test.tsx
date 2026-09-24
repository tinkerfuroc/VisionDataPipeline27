// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { LineChart } from "./LineChart";

afterEach(cleanup);

describe("LineChart", () => {
  it("draws one path per series, a legend for two, and a tooltip on hover", () => {
    const { container } = render(
      <LineChart
        title="Mask mAP"
        series={[
          { name: "mAP50-95", color: "#3987e5", points: [[1, 0.1], [2, 0.4], [3, 0.5]] },
          { name: "mAP50", color: "#d95926", points: [[1, 0.2], [2, 0.6], [3, 0.7]] },
        ]}
      />,
    );
    expect(container.querySelectorAll("path[data-series]")).toHaveLength(2);
    expect(screen.getByText("mAP50")).toBeTruthy();
    fireEvent.mouseMove(container.querySelector("rect[data-hit]")!, { clientX: 10000, clientY: 10 });
    const tip = screen.getByRole("tooltip");
    expect(tip.textContent).toContain("Epoch 3");
    expect(tip.textContent).toContain("0.70");
  });
  it("has no legend for one series and says so when empty", () => {
    const { container, rerender } = render(<LineChart title="Box" series={[{ name: "Box", color: "#3987e5", points: [[1, 0.3]] }]} />);
    expect(container.querySelectorAll("[data-legend]")).toHaveLength(0);
    rerender(<LineChart title="Box" series={[{ name: "Box", color: "#3987e5", points: [] }]} />);
    expect(screen.getByText("No epochs yet.")).toBeTruthy();
  });
});
