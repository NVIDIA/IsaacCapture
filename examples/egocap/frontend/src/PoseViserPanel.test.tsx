/** The 3D view panel: it renders nothing when the feature is off, embeds one stable iframe when ready, and
 * never reloads that iframe for anything but a changed URL/backend or an explicit retry. */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { PoseViserPanel } from "./PoseViserPanel";
import type { VisualizationInfo } from "./captureProtocol";

afterEach(cleanup);

const READY: VisualizationInfo = { status: "ready", url: "http://lab:8085/", diagnostic: null };

function panel(visualization: VisualizationInfo | null, backendInstanceId: string | null = "b1") {
  return <PoseViserPanel visualization={visualization} backendInstanceId={backendInstanceId} />;
}

describe("PoseViserPanel", () => {
  it("renders nothing when the backend reports no 3D view or a disabled one", () => {
    const { container, rerender } = render(panel(null));
    expect(container.innerHTML).toBe("");
    rerender(panel({ status: "disabled", url: null, diagnostic: null }));
    expect(container.innerHTML).toBe("");
  });

  it("embeds the advertised URL in a titled iframe with a single Reset view button", () => {
    render(panel(READY));
    const frame = screen.getByTitle("3D tracking view");
    expect(frame.getAttribute("src")).toBe("http://lab:8085/");
    expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual(["Reset view"]);
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("keeps the same iframe element through tracking loss, diagnostics and unrelated rerenders", () => {
    const { rerender } = render(panel(READY));
    const before = screen.getByTitle("3D tracking view");
    rerender(panel({ ...READY, diagnostic: "tracking is stale" }));
    rerender(panel({ ...READY, diagnostic: null }));
    expect(screen.getByTitle("3D tracking view")).toBe(before);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows the tracking diagnostic as a note while the view stays embedded", () => {
    render(panel({ ...READY, diagnostic: "tracking is stale" }));
    expect(screen.getByRole("status").textContent).toBe("tracking is stale");
    expect(screen.getByTitle("3D tracking view")).toBeTruthy();
  });

  it("rebuilds the iframe only for a changed URL, a new backend instance or Reset view", () => {
    const { rerender } = render(panel(READY));
    const first = screen.getByTitle("3D tracking view");
    rerender(panel({ ...READY, url: "http://lab:9000/" }));
    const second = screen.getByTitle("3D tracking view");
    expect(second).not.toBe(first);
    rerender(panel({ ...READY, url: "http://lab:9000/" }, "b2"));
    const third = screen.getByTitle("3D tracking view");
    expect(third).not.toBe(second);
    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
    expect(screen.getByTitle("3D tracking view")).not.toBe(third);
  });

  it("offers to resume shortcuts only while the iframe holds focus", () => {
    render(panel(READY));
    const frame = screen.getByTitle("3D tracking view") as HTMLIFrameElement;
    expect(screen.queryByRole("button", { name: "Resume shortcuts" })).toBeNull();
    window.dispatchEvent(new Event("blur")); // focus moved elsewhere, not into the iframe
    expect(screen.queryByRole("button", { name: "Resume shortcuts" })).toBeNull();

    frame.focus();
    fireEvent.blur(window);
    expect(screen.getByRole("status").textContent).toContain("Shortcuts are paused");
    fireEvent.click(screen.getByRole("button", { name: "Resume shortcuts" }));
    expect(screen.queryByRole("button", { name: "Resume shortcuts" })).toBeNull();

    frame.focus();
    fireEvent.blur(window);
    fireEvent.focus(window); // the user clicked back into the page
    expect(screen.queryByRole("button", { name: "Resume shortcuts" })).toBeNull();
  });

  it("says the view is unavailable, with the reason, and offers no iframe", () => {
    render(panel({ status: "unavailable", url: null, diagnostic: "the 3D view is bound to loopback" }));
    expect(screen.getByText(/3D view unavailable: the 3D view is bound to loopback/)).toBeTruthy();
    expect(screen.queryByTitle("3D tracking view")).toBeNull();
  });
});
