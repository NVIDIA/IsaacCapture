/**
 * The cameras-first operator page: the status bar, the camera stage, the control dock and the side drawer that
 * holds the 3D view and raw data. useCaptureSession is mocked so each test controls the snapshot directly.
 */
import { afterEach, describe, expect, it, vi, type Mock } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import type { PreviewSnapshot } from "./captureProtocol";
import type { RecordingSnapshot } from "./recordingTypes";
import { useCaptureSession, type CaptureSessionState } from "./useCaptureSession";

vi.mock("./useCaptureSession", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./useCaptureSession")>();
  return { ...actual, useCaptureSession: vi.fn() };
});

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

function snapshot(overrides: Partial<PreviewSnapshot> = {}): PreviewSnapshot {
  return {
    version: 1,
    type: 3,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station",
    profile_label: "Profile",
    connecting: false,
    started: { session_id: "s1", running: true, mode: 0, streams: [] },
    status: { session_id: "s1", type: 3, running: true, startup_stage: null, message: null, status: null },
    stopped: null,
    error_message: null,
    binary_generation: 1,
    tracking: null,
    cameras: [],
    visualization: null,
    ...overrides,
  };
}

function recording(overrides: Partial<RecordingSnapshot> = {}): RecordingSnapshot {
  return {
    batchId: "b1",
    episodePhase: "IDLE",
    currentEpisodeId: null,
    currentEpisodeElapsedMs: null,
    taskText: "Place the object",
    taskDetails: "",
    defaultOperator: null,
    autoFillOperatorName: false,
    savedCount: 3,
    lastResult: null,
    readyToRecord: true,
    revision: 1,
    ...overrides,
  };
}

function mount(overrides: Partial<CaptureSessionState> = {}) {
  (useCaptureSession as unknown as Mock).mockReturnValue({
    phase: "connected",
    snapshot: snapshot(),
    receivedAtMs: Date.now(),
    receivedAtMonotonic: performance.now(),
    diagnostic: null,
    trackingSample: null,
    recordingSnapshot: recording(),
    recordingSnapshotReceivedAtMonotonic: performance.now(),
    retry: vi.fn(),
    sendRecordingCommand: vi.fn().mockRejectedValue(new Error("not connected")),
    operatorToken: "op",
    setOperatorToken: vi.fn(),
    clearOperatorToken: vi.fn(),
    controlToken: "token",
    setControlToken: vi.fn(),
    clearControlToken: vi.fn(),
    ...overrides,
  });
  return render(<App />);
}

describe("operator page layout", () => {
  it("puts the status bar, the camera stage, the controls and the side panel on one page", () => {
    const { container } = mount();
    expect(screen.getByRole("region", { name: "Operator status" })).not.toBeNull();
    expect(container.querySelector(".op-stage .camera-grid")).not.toBeNull();
    expect(container.querySelector(".op-dock .recording-panel")).not.toBeNull();
    expect(container.querySelector(".op-body > .op-main")).not.toBeNull();
    expect(container.querySelector(".op-body > .op-sidebar")).not.toBeNull();
    expect(screen.getByRole("separator", { name: "Resize side panel" })).not.toBeNull();
  });

  it("always shows the 3D view above the details, with nothing to toggle", () => {
    const { container } = mount({ snapshot: snapshot({ visualization: { status: "ready", url: "http://lab:8085/", diagnostic: null } }) });
    const panels = Array.from(container.querySelectorAll(".op-sidebar .op-panel")).map((p) => p.getAttribute("aria-label"));
    expect(panels).toEqual(["3D view", "Details"]);
    expect(container.querySelector(".op-sidebar iframe")).not.toBeNull();
    expect(screen.getByText("Tracking data")).not.toBeNull();
    expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
    expect(screen.queryByRole("group", { name: "More views" })).toBeNull();
  });

  it("shows no sidebar or divider while the feed is unavailable", () => {
    const { container } = mount({ phase: "unavailable", snapshot: null, recordingSnapshot: null });
    expect(container.querySelector(".op-sidebar")).toBeNull();
    expect(screen.queryByRole("separator")).toBeNull();
  });

  it("frames the cameras in red only for a live take", () => {
    const live = mount({ recordingSnapshot: recording({ episodePhase: "RECORDING", currentEpisodeElapsedMs: 5000 }) });
    expect(live.container.querySelector(".op-stage")?.className).toContain("op-stage--recording");
    cleanup();
    const idle = mount();
    expect(idle.container.querySelector(".op-stage")?.className).not.toContain("op-stage--recording");
    cleanup();
    const stale = mount({
      phase: "reconnecting",
      recordingSnapshot: recording({ episodePhase: "RECORDING", currentEpisodeElapsedMs: 5000 }),
    });
    expect(stale.container.querySelector(".op-stage")?.className).not.toContain("op-stage--recording");
    expect(screen.getByText("NO FEED")).not.toBeNull();
  });

  it("says SET UP, not READY, until an operator and control token are entered", () => {
    mount({ operatorToken: null, controlToken: null });
    expect(screen.getByText("SET UP")).not.toBeNull();
    expect(screen.queryByText("READY")).toBeNull();
  });

  it("shows STOPPING at once when Stop is pressed, before the backend finishes", async () => {
    mount({
      recordingSnapshot: recording({ episodePhase: "RECORDING", currentEpisodeId: "e1", currentEpisodeElapsedMs: 8000 }),
      sendRecordingCommand: vi.fn().mockReturnValue(new Promise(() => {})), // the backend is still writing the take
    });
    expect(screen.getByText("REC")).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^Stop\/Save/ }));
    await waitFor(() => expect(screen.getByText("STOPPING")).not.toBeNull());
    expect(screen.queryByText("REC")).toBeNull();
  });
});


describe("resizing the side panel", () => {
  const widthOf = (container: HTMLElement) =>
    (container.querySelector(".op-body") as HTMLElement).style.getPropertyValue("--op-sidebar-width");
  const handle = () => screen.getByRole("separator", { name: "Resize side panel" });

  function windowWidth(width: number) {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
  }

  it("starts at the default width and describes itself to assistive tech", () => {
    windowWidth(1600);
    const { container } = mount();
    expect(widthOf(container)).toBe("460px");
    expect(handle().getAttribute("aria-valuenow")).toBe("460");
    expect(handle().getAttribute("aria-valuemin")).toBe("400");
    expect(handle().getAttribute("aria-valuemax")).toBe("1000");
    expect(handle().getAttribute("aria-orientation")).toBe("vertical");
  });

  it("starts at the remembered width", () => {
    windowWidth(1600);
    window.localStorage.setItem("egocap.sidebarWidth", "620");
    const { container } = mount();
    expect(widthOf(container)).toBe("620px");
  });

  it("widens the panel with ArrowLeft, narrows it with ArrowRight, and takes bigger steps with Shift", () => {
    windowWidth(1600);
    const { container } = mount();
    fireEvent.keyDown(handle(), { key: "ArrowLeft" });
    expect(widthOf(container)).toBe("484px");
    fireEvent.keyDown(handle(), { key: "ArrowLeft", shiftKey: true });
    expect(widthOf(container)).toBe("580px");
    fireEvent.keyDown(handle(), { key: "ArrowRight" });
    expect(widthOf(container)).toBe("556px");
  });

  it("jumps to the limits with Home and End, and never past them", () => {
    windowWidth(1600);
    const { container } = mount();
    fireEvent.keyDown(handle(), { key: "Home" });
    expect(widthOf(container)).toBe("400px");
    fireEvent.keyDown(handle(), { key: "ArrowRight", shiftKey: true });
    expect(widthOf(container)).toBe("400px");
    fireEvent.keyDown(handle(), { key: "End" });
    expect(widthOf(container)).toBe("1000px");
    fireEvent.keyDown(handle(), { key: "ArrowLeft", shiftKey: true });
    expect(widthOf(container)).toBe("1000px");
  });

  it("ignores other keys", () => {
    windowWidth(1600);
    const { container } = mount();
    fireEvent.keyDown(handle(), { key: "a" });
    expect(widthOf(container)).toBe("460px");
  });

  it("resets to the default on double-click", () => {
    windowWidth(1600);
    const { container } = mount();
    fireEvent.keyDown(handle(), { key: "End" });
    fireEvent.doubleClick(handle());
    expect(widthOf(container)).toBe("460px");
  });

  it("follows the pointer while dragging: pulling left widens the panel, and the page marks the drag", () => {
    windowWidth(1600);
    const { container } = mount();
    const body = container.querySelector(".op-body") as HTMLElement;
    fireEvent.pointerDown(handle(), { pointerId: 1, clientX: 1140 });
    expect(body.className).toContain("op-body--resizing");
    fireEvent.pointerMove(handle(), { pointerId: 1, clientX: 1000 });
    expect(widthOf(container)).toBe("600px"); // 1600 - 1000: the divider is now 600 px from the right edge
    fireEvent.pointerMove(handle(), { pointerId: 1, clientX: 1300 });
    expect(widthOf(container)).toBe("400px"); // pushed right past the minimum: held there
    fireEvent.pointerUp(handle(), { pointerId: 1 });
    expect(body.className).not.toContain("op-body--resizing");
  });

  it("ignores pointer movement when no drag started", () => {
    windowWidth(1600);
    const { container } = mount();
    fireEvent.pointerMove(handle(), { pointerId: 1, clientX: 900 });
    expect(widthOf(container)).toBe("460px");
  });

  it("remembers the width once a drag ends, not on every move", () => {
    windowWidth(1600);
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    mount();
    setItem.mockClear();
    fireEvent.pointerDown(handle(), { pointerId: 1, clientX: 1140 });
    fireEvent.pointerMove(handle(), { pointerId: 1, clientX: 1000 });
    fireEvent.pointerMove(handle(), { pointerId: 1, clientX: 900 });
    expect(setItem).not.toHaveBeenCalled();
    fireEvent.pointerUp(handle(), { pointerId: 1 });
    expect(window.localStorage.getItem("egocap.sidebarWidth")).toBe("700");
  });

  it("does not overwrite the remembered width when the window refits", () => {
    windowWidth(1600);
    window.localStorage.setItem("egocap.sidebarWidth", "900");
    mount();
    windowWidth(1000);
    fireEvent(window, new Event("resize"));
    expect(window.localStorage.getItem("egocap.sidebarWidth")).toBe("900");
  });

  it("refits to a smaller window so the main column keeps its room", () => {
    windowWidth(1600);
    const { container } = mount();
    fireEvent.keyDown(handle(), { key: "End" });
    expect(widthOf(container)).toBe("1000px");
    windowWidth(1200);
    fireEvent(window, new Event("resize"));
    expect(widthOf(container)).toBe("640px"); // 1200 - 560
    windowWidth(1024);
  });
});
