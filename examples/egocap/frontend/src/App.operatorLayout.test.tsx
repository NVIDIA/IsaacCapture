// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The cameras-first operator page: the status bar, the camera stage, the control dock and the side drawer that
 * holds the 3D view and raw data. useCaptureSession is mocked so each test controls the snapshot directly.
 */
import { afterEach, describe, expect, it, vi, type Mock } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import App from "./App";
import type { PreviewSnapshot } from "./captureProtocol";
import { DAY_REPORT_FIXTURE, UPTIME_FIXTURE } from "./dayReportTestDoubles";
import type { RecordingSnapshot } from "./recordingTypes";
import { OPERATOR_SETTINGS_FIXTURE, STATIONS_FIXTURE } from "./settingsTestDoubles";
import { SETUP_HELD, SETUP_READY } from "./setupTestDoubles";
import { useCaptureSession, type CaptureSessionState } from "./useCaptureSession";

vi.mock("./useCaptureSession", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./useCaptureSession")>();
  return { ...actual, useCaptureSession: vi.fn() };
});

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  vi.unstubAllGlobals();
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
    notReadyReason: null,
    revision: 1,
    ...overrides,
  };
}

/** Set what the capture session reports from the next render on. */
function stubSession(overrides: Partial<CaptureSessionState> = {}) {
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
}

function mount(overrides: Partial<CaptureSessionState> = {}) {
  stubSession(overrides);
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

/** Answer the routes a test names and leave every other request unanswered, as a backend that is still starting would. */
function stubRoutes(routes: Record<string, unknown>) {
  const fetched = vi.fn(async (url: string, _init?: RequestInit) => (url in routes ? new Response(JSON.stringify(routes[url]), { status: 200 }) : new Response("{}", { status: 503 })));
  vi.stubGlobal("fetch", fetched);
  return fetched;
}

/** How many times a route has been read. */
const readsOf = (fetched: ReturnType<typeof stubRoutes>, route: string) => fetched.mock.calls.filter(([url]) => url === route).length;

describe("station uptime in the status strip", () => {
  it("shows the station's read-only chip beside its name once today's uptime is read", async () => {
    stubRoutes({ "/api/station-uptime": UPTIME_FIXTURE });
    const { container } = mount();
    const chip = await screen.findByRole("img", { name: /^Station down for 5m 00s: Left wrist/ });
    expect(chip.textContent).toBe("Down · 5m 00s");
    expect(container.querySelector(".status-strip .status-station + .station-uptime")).not.toBeNull();
    expect(container.querySelector(".station-uptime")?.textContent).toBe("Down · 5m 00sactive 2h 05m · down 17m · 2 blockers");
    // Active and down time are the backend's to decide: the strip offers nothing to press.
    expect(container.querySelector(".status-strip button")).toBeNull();
  });

  it("reads the uptime again as soon as a take starts, is held or is saved, not on the chip's own timer", async () => {
    const fetched = stubRoutes({ "/api/station-uptime": UPTIME_FIXTURE });
    const { rerender } = mount();
    await waitFor(() => expect(readsOf(fetched, "/api/station-uptime")).toBe(1));
    const changes: Array<Partial<RecordingSnapshot>> = [
      { episodePhase: "RECORDING", currentEpisodeElapsedMs: 0 },
      { readyToRecord: false, notReadyReason: "Left wrist: cam_left: no recent frame" },
      { savedCount: 4 },
    ];
    for (const [index, change] of changes.entries()) {
      stubSession({ recordingSnapshot: recording(change) });
      rerender(<App />);
      await waitFor(() => expect(readsOf(fetched, "/api/station-uptime")).toBe(index + 2));
    }
    // A snapshot that changes none of them is no reason to read.
    stubSession({ recordingSnapshot: recording({ savedCount: 4, revision: 9 }) });
    rerender(<App />);
    await screen.findByRole("img", { name: /^Station down/ });
    expect(readsOf(fetched, "/api/station-uptime")).toBe(4);
  });

  it("shows no chip while the uptime cannot be read", async () => {
    const fetched = stubRoutes({});
    const { container } = mount();
    await waitFor(() => expect(readsOf(fetched, "/api/station-uptime")).toBe(1));
    expect(container.querySelector(".station-uptime")).toBeNull();
  });
});

describe("start-of-day rail", () => {
  it("shows four steps under the header and reads the flow with the control token", async () => {
    const fetched = stubRoutes({ "/api/setup": SETUP_HELD });
    const { container } = mount();
    await screen.findByRole("button", { name: /Step 2, Devices: needs attention/ });
    expect(container.querySelector(".app-header + .step-rail")).not.toBeNull();
    expect(Array.from(container.querySelectorAll(".step-rail .step-label")).map((label) => label.firstChild?.textContent)).toEqual(["Station", "Devices", "Record", "Wrap up"]);
    const [, init] = fetched.mock.calls.find(([url]) => url === "/api/setup") ?? [];
    expect(new Headers(init?.headers).get("X-Control-Token")).toBe("token");
  });

  it("opens Settings on the Devices tab over a page that takes no input, and reads the flow again on close", async () => {
    const fetched = stubRoutes({ "/api/setup": SETUP_HELD });
    const { container } = mount();
    const devices = await screen.findByRole("button", { name: /Step 2, Devices: needs attention/ });
    expect(container.querySelector(".op-body")?.hasAttribute("inert")).toBe(false);
    fireEvent.click(devices);
    const dialog = screen.getByRole("dialog", { name: "Settings" });
    expect(dialog.querySelector(".settings-tab--on")?.textContent).toBe("Devices");
    // The page's own read is shown at once; the tab then reads the flow itself.
    expect(within(dialog).getByRole("status").textContent).toBe("Recording is held: Left wrist: cam_left: no recent frame.");
    // The page behind a dialog is inert, which is also what suspends the recording shortcuts.
    expect(container.querySelector(".op-body")?.hasAttribute("inert")).toBe(true);
    await waitFor(() => expect(readsOf(fetched, "/api/setup")).toBe(2));
    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(container.querySelector(".op-body")?.hasAttribute("inert")).toBe(false);
    // Closing reads the flow again, so a fix made while the dialog was open shows on the rail at once.
    await waitFor(() => expect(readsOf(fetched, "/api/setup")).toBe(3));
  });

  it("opens Settings on the Station tab for the station step, as the only dialog", async () => {
    stubRoutes({ "/api/setup": SETUP_HELD });
    mount();
    fireEvent.click(await screen.findByRole("button", { name: /Step 1, Station/ }));
    const settings = screen.getByRole("dialog", { name: "Settings" });
    expect(settings.querySelector(".settings-tab--on")?.textContent).toBe("Station");
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
  });

  it("gives Settings the flow, so the station in use shows where its headset connects", async () => {
    const station = { id: "st_1", name: "Bench A", camera_type: "oak_d", camera_roles: { ego: null, wrist_left: null, wrist_right: null }, problems: [] };
    stubRoutes({
      "/api/setup": { ...SETUP_READY, headset: { address: "10.0.0.42", saved: "10.0.0.7", changed: true } },
      "/api/stations": { ...STATIONS_FIXTURE, stations: [station], selected: "st_1", label: "Bench A" },
      "/api/operator-settings": OPERATOR_SETTINGS_FIXTURE,
      "/api/tools": { tools: [] },
      "/api/task-files": { files: [], imported: [] },
    });
    mount();
    fireEvent.click(await screen.findByRole("button", { name: /Step 1, Station/ }));
    const stations = await screen.findByRole("region", { name: "Stations" });
    expect(within(stations).getByText("This host's address changed from 10.0.0.7 to 10.0.0.42. Enter the new one in the headset.")).toBeTruthy();
    fireEvent.click(within(stations).getByRole("button", { name: "Edit" }));
    expect(within(stations).getByText("Headset connects to 10.0.0.42")).toBeTruthy();
  });

  it("shows Record as a step that opens nothing and takes no focus", async () => {
    stubRoutes({ "/api/setup": SETUP_READY });
    const { container } = mount();
    await screen.findByRole("button", { name: /Step 1, Station/ });
    expect(screen.queryByRole("button", { name: /Step 3/ })).toBeNull();
    const record = container.querySelectorAll(".step-rail-item")[2].firstElementChild as HTMLElement;
    expect(record.textContent).toBe("3Record: to do");
    expect(record.getAttribute("aria-current")).toBe("step");
    const focused = document.activeElement;
    fireEvent.click(record);
    expect(document.activeElement).toBe(focused);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(container.querySelector(".op-body")?.hasAttribute("inert")).toBe(false);
  });

  it("opens the end-of-day report from the last step", async () => {
    stubRoutes({ "/api/setup": SETUP_READY });
    const { container } = mount();
    fireEvent.click(await screen.findByRole("button", { name: /Step 4, Wrap up/ }));
    expect(screen.getByRole("dialog", { name: "End of day" })).not.toBeNull();
    expect(container.querySelector(".op-body")?.hasAttribute("inert")).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(container.querySelector(".op-body")?.hasAttribute("inert")).toBe(false);
  });

  it("moves focus into the end-of-day report when it opens and back to the Wrap up step when it closes", async () => {
    stubRoutes({ "/api/setup": SETUP_READY });
    mount();
    const wrapUp = await screen.findByRole("button", { name: /Step 4, Wrap up/ });
    wrapUp.focus();
    fireEvent.click(wrapUp);
    const dialog = screen.getByRole("dialog", { name: "End of day" });
    expect(document.activeElement).toBe(dialog);
    // Focus sent to the page behind is brought back into the dialog.
    wrapUp.focus();
    expect(document.activeElement).toBe(dialog);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(wrapUp);
  });

  it("starts the end-of-day report afresh at every opening", async () => {
    const fetched = stubRoutes({ "/api/setup": SETUP_READY, "/api/day-report": DAY_REPORT_FIXTURE });
    mount();
    const wrapUp = await screen.findByRole("button", { name: /Step 4, Wrap up/ });
    fireEvent.click(wrapUp);
    fireEvent.change(await screen.findByLabelText("Notes"), { target: { value: "half a thought" } });
    // Unsaved notes hold the dialog open until the operator says to discard them.
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.click(wrapUp);
    // A new dialog: it shows nothing of the last one until its own report arrives.
    expect(screen.queryByLabelText("Notes")).toBeNull();
    expect(((await screen.findByLabelText("Notes")) as HTMLTextAreaElement).value).toBe("left glove battery");
    expect(readsOf(fetched, "/api/day-report")).toBe(2);
  });

  it("suspends the recording shortcuts while the end-of-day report is open", async () => {
    stubRoutes({ "/api/setup": SETUP_READY });
    const send = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    mount({ sendRecordingCommand: send, recordingSnapshot: recording({ episodePhase: "RECORDING", currentEpisodeId: "e1", currentEpisodeElapsedMs: 0 }) });
    fireEvent.click(await screen.findByRole("button", { name: /Step 4, Wrap up/ }));
    for (const key of ["a", "w", "s", "r", "t", " "]) fireEvent.keyDown(window, { key });
    expect(send).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    fireEvent.keyDown(window, { key: "w" });
    expect(send).toHaveBeenCalledTimes(1);
  });

  it("shows no rail while the flow cannot be read", async () => {
    const fetched = stubRoutes({});
    const { container } = mount();
    await waitFor(() => expect(readsOf(fetched, "/api/setup")).toBe(1));
    expect(container.querySelector(".step-rail")).toBeNull();
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
