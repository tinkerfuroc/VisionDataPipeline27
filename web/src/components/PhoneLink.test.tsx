// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PhoneLink } from "./PhoneLink";

afterEach(cleanup);

describe("PhoneLink", () => {
  it("reveals the phone page's address with the project, and a QR code of it", () => {
    render(<PhoneLink projectId="p1" />);
    expect(screen.queryByText(/\/upload\?project=p1/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Upload from a phone" }));
    const url = `${window.location.origin}/upload?project=p1`;
    expect(screen.getByText(url)).toBeTruthy();
    const img = screen.getByRole("img", { name: "QR code for the phone upload page" }) as HTMLImageElement;
    expect(img.src.startsWith("data:image/")).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Hide" }));
    expect(screen.queryByText(url)).toBeNull();
  });
});
