/**
 * Component-level tests for render/focus/image-event paths that reducer
 * and parser tests can't reach: CameraTile's retry-outcome ordering, the
 * capture tree's native-disclosure expansion/focus preservation across a
 * stale transition, and its diagnostic-vs-never-had-data distinction.
 * useCaptureSession is mocked so each test controls phase/snapshot/tracking
 * directly rather than driving a real socket.
 */
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi, type Mock } from "vitest";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import App from "./App";
import { useCaptureSession, type AcceptedTracking, type CaptureSessionState } from "./useCaptureSession";
import { Axes, HandJointOrder, PositionUnit, QuaternionOrder, type PreviewSnapshot } from "./captureProtocol";
import type { ControllerDisplay, DecodedTracking, HandDisplay } from "./trackingDecoder";

vi.mock("./useCaptureSession", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./useCaptureSession")>();
  return { ...actual, useCaptureSession: vi.fn() };
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

function makeSnapshot(overrides: Partial<PreviewSnapshot> = {}): PreviewSnapshot {
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

const ABSENT_CONTROLLER: ControllerDisplay = {
  present: false,
  gripPosition: null,
  gripOrientation: null,
  gripValid: false,
  aimPosition: null,
  aimOrientation: null,
  aimValid: false,
  primaryClick: null,
  secondaryClick: null,
  thumbstickX: null,
  thumbstickY: null,
  thumbstickClick: null,
  menuClick: null,
  squeezeValue: null,
  triggerValue: null,
};
const ABSENT_HAND: HandDisplay = { present: false, joints: [] };

function makeTracking(overrides: Partial<DecodedTracking> = {}): DecodedTracking {
  return {
    head: { present: false, position: null, orientation: null, isValid: false, isTracked: false },
    controllerLeft: ABSENT_CONTROLLER,
    controllerRight: ABSENT_CONTROLLER,
    handLeft: ABSENT_HAND,
    handRight: ABSENT_HAND,
    ...overrides,
  };
}

/** A minimal but validly-shaped descriptor -- these tests never assert on
 * descriptor-derived label text, only on tracking values/availability. */
function makeTrackingSample(decoded: DecodedTracking, overrides: Partial<AcceptedTracking> = {}): AcceptedTracking {
  return {
    decoded,
    descriptor: {
      kind: "tracking",
      stream_id: 0,
      payload_size: 1995,
      fields: [],
      positionUnit: PositionUnit.METERS,
      quaternionOrder: QuaternionOrder.XYZW,
      axes: Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD,
      handJointOrder: HandJointOrder.OPENXR_26,
    },
    sequence: 1n,
    snapshotAgeMs: 10,
    snapshotReceivedAtMonotonic: performance.now(),
    ...overrides,
  };
}

function makeState(overrides: Partial<CaptureSessionState> = {}): CaptureSessionState {
  // Partial<CaptureSessionState> widens operatorToken/controlToken to include
  // `undefined` for TS's purposes even though no real caller passes it
  // explicitly -- resolved before the return so the literal below has no
  // duplicate keys (a hard TS error, unlike plain JS) and still satisfies
  // the required, non-optional CaptureSessionState contract.
  const {
    operatorToken: operatorTokenOverride,
    controlToken: controlTokenOverride,
    ...restOverrides
  } = overrides;
  return {
    phase: "connected",
    snapshot: makeSnapshot(),
    receivedAtMs: Date.now(),
    receivedAtMonotonic: performance.now(),
    diagnostic: null,
    trackingSample: null,
    recordingSnapshot: null,
    recordingSnapshotReceivedAtMonotonic: null,
    retry: vi.fn(),
    sendRecordingCommand: vi.fn().mockRejectedValue(new Error("not connected")),
    operatorToken: operatorTokenOverride === undefined ? "test-operator" : operatorTokenOverride,
    setOperatorToken: vi.fn(),
    clearOperatorToken: vi.fn(),
    controlToken: controlTokenOverride === undefined ? "test-control-token" : controlTokenOverride,
    setControlToken: vi.fn(),
    clearControlToken: vi.fn(),
    ...restOverrides,
  };
}

function mockState(state: CaptureSessionState): void {
  (useCaptureSession as unknown as Mock).mockReturnValue(state);
}

describe("CameraTile retry outcome ordering", () => {
  const camera = { stream_id: 1, name: "head_camera", url: "/stream1?a=1", sequence: "5", age_ms: 100, diagnostic: null };

  /** Advances fake time and re-mocks the hook with a fresh
   * receivedAtMonotonic, simulating the real ~1/s snapshot heartbeat that
   * keeps the camera's effective sample age from growing unbounded --
   * without this, CameraGrid's own staleness math would mark the camera
   * stale partway through a multi-second retry sequence purely from the
   * test's fake-timer advances, independent of anything under test. */
  function advanceAndRefreshHeartbeat(rerender: (ui: ReactElement) => void, ms: number): void {
    act(() => {
      vi.advanceTimersByTime(ms);
    });
    mockState(makeState({ snapshot: makeSnapshot({ cameras: [camera] }), receivedAtMonotonic: performance.now() }));
    rerender(<App />);
  }

  it("a final successful automatic retry does not leave Retry image visible", () => {
    vi.useFakeTimers();
    mockState(makeState({ snapshot: makeSnapshot({ cameras: [camera] }) }));
    const { rerender } = render(<App />);
    let img = screen.getByAltText("head_camera") as HTMLImageElement;

    // Three failures, each scheduling the next automatic retry -- the
    // third (final, budget-exhausting) retry's own request then succeeds.
    for (let i = 0; i < 3; i++) {
      fireEvent.error(img);
      advanceAndRefreshHeartbeat(rerender, 1000);
      img = screen.getByAltText("head_camera") as HTMLImageElement;
    }
    // The third retry's request is now in flight -- its outcome isn't
    // known yet, so the button must not appear just because the attempt
    // count reached the limit. This is the actual window the premature-
    // outcome bug showed through: a stale currentRequestFailed=true left
    // over from the PREVIOUS (already-superseded) request's failure.
    expect(screen.queryByText("Retry image")).toBeNull();

    fireEvent.load(img);
    expect(screen.queryByText("Retry image")).toBeNull();
  });

  it("shows Retry image once the budget is exhausted and a request is still failed", () => {
    vi.useFakeTimers();
    mockState(makeState({ snapshot: makeSnapshot({ cameras: [camera] }) }));
    const { rerender } = render(<App />);
    let img = screen.getByAltText("head_camera") as HTMLImageElement;

    for (let i = 0; i < 3; i++) {
      fireEvent.error(img);
      advanceAndRefreshHeartbeat(rerender, 1000);
      img = screen.getByAltText("head_camera") as HTMLImageElement;
    }
    // The fourth (post-budget) request also fails -- no further automatic
    // retry is scheduled, and the manual control must appear.
    act(() => {
      fireEvent.error(img);
    });

    expect(screen.getByText("Retry image")).toBeTruthy();
  });

  it("manual retry resets state and no stray automatic retry fires afterward", () => {
    vi.useFakeTimers();
    mockState(makeState({ snapshot: makeSnapshot({ cameras: [camera] }) }));
    const { rerender } = render(<App />);
    let img = screen.getByAltText("head_camera") as HTMLImageElement;

    for (let i = 0; i < 3; i++) {
      fireEvent.error(img);
      advanceAndRefreshHeartbeat(rerender, 1000);
      img = screen.getByAltText("head_camera") as HTMLImageElement;
    }
    act(() => {
      fireEvent.error(img);
    });
    const srcAfterExhaustion = img.src;

    act(() => {
      fireEvent.click(screen.getByText("Retry image"));
    });
    expect(screen.queryByText("Retry image")).toBeNull();

    // No pending timer should still be armed from before the manual retry --
    // advancing well past the retry interval must not change src again on
    // its own (that would mean a stray automatic retry fired).
    advanceAndRefreshHeartbeat(rerender, 5000);
    img = screen.getByAltText("head_camera") as HTMLImageElement;
    expect(img.src).not.toBe(srcAfterExhaustion);
    const srcAfterWait = img.src;
    advanceAndRefreshHeartbeat(rerender, 5000);
    img = screen.getByAltText("head_camera") as HTMLImageElement;
    expect(img.src).toBe(srcAfterWait);
  });
});

/** Finds the `<summary>` for a named tree branch (e.g. "Left controller",
 * "Left hand") by its name span, which is unique per branch. */
function branchSummary(name: string): HTMLElement {
  return screen.getByText(name, { selector: ".tree-summary-name" }).closest("summary") as HTMLElement;
}

function branchValue(summary: HTMLElement): string {
  return summary.querySelector(".tree-summary-value")?.textContent ?? "";
}

/** Reads a labeled row's cell values, scoped to one branch's `<details>` --
 * every branch's table is always present in the DOM (collapsed ones are
 * merely hidden, not removed), so an unscoped query would match the same
 * row label in a sibling branch too. */
function rowCells(branchSummary: HTMLElement, rowLabel: string): string[] {
  const details = branchSummary.closest("details") as HTMLElement;
  const th = within(details).getByText(rowLabel);
  const row = th.closest("tr") as HTMLTableRowElement;
  return Array.from(row.querySelectorAll("td")).map((td) => td.textContent ?? "");
}

/** Opens the top-level "Controllers" branch and returns its own `<details>`
 * element -- a nested side's disclosure is only actually reachable/visible
 * to a real browser user once this ancestor is open, so every test that
 * interacts with "Left controller"/"Right controller" must do this first
 * rather than clicking/focusing a descendant hidden under a closed parent. */
function openControllersGroup(): HTMLDetailsElement {
  const groupDetails = branchSummary("Controllers").closest("details") as HTMLDetailsElement;
  if (!groupDetails.open) fireEvent.click(branchSummary("Controllers"));
  expect(groupDetails.open).toBe(true);
  return groupDetails;
}

describe("Controller branch expansion/focus across a stale transition", () => {
  it("preserves expansion state and the same summary element across live -> stale -> live", () => {
    const liveTracking = makeTracking({
      controllerLeft: { ...ABSENT_CONTROLLER, present: true, gripValid: true, gripPosition: [1, 2, 3], gripOrientation: [0, 0, 0, 1] },
    });
    const baseSnapshot = makeSnapshot({ tracking: { sequence: "1", age_ms: 10, diagnostic: null } });

    mockState(
      makeState({
        phase: "connected",
        snapshot: baseSnapshot,
        trackingSample: makeTrackingSample(liveTracking),
        receivedAtMonotonic: performance.now(),
      }),
    );
    const { rerender } = render(<App />);

    const groupDetails = openControllersGroup();
    const toggle = branchSummary("Left controller");
    const toggleDetails = toggle.closest("details") as HTMLDetailsElement;
    expect(branchValue(toggle)).toBe("present");
    fireEvent.click(toggle);
    expect(toggleDetails.open).toBe(true);
    expect(rowCells(toggle, "Grip position")).toEqual(["1.000", "2.000", "3.000", "—"]);
    toggle.focus();
    expect(document.activeElement).toBe(toggle);

    // Connection drops: App passes tracking=null to CaptureTree while
    // !presentation.live, so ControllerBranch receives controller=null.
    mockState(
      makeState({
        phase: "reconnecting",
        snapshot: baseSnapshot,
        trackingSample: makeTrackingSample(liveTracking),
        receivedAtMonotonic: performance.now(),
      }),
    );
    rerender(<App />);

    const toggleWhileStale = branchSummary("Left controller");
    // Same DOM node -- never unmounted, so expansion/focus survive. Native
    // `open` must stay true too: finding the row text or matching node
    // identity alone doesn't prove the disclosure is actually still
    // expanded, since a collapsed `<details>` keeps its descendants in the
    // DOM (just hidden).
    expect(toggleWhileStale).toBe(toggle);
    expect(groupDetails.open).toBe(true);
    expect(toggleDetails.open).toBe(true);
    expect(document.activeElement).toBe(toggle);
    expect(branchValue(toggle)).toBe("unavailable");
    // Detail stays expanded but every value is masked -- not the stale
    // "1.000, 2.000, 3.000" from before the connection dropped.
    expect(rowCells(toggleWhileStale, "Grip position")).toEqual(["—"]);

    // Connection recovers with fresh data.
    mockState(
      makeState({
        phase: "connected",
        snapshot: baseSnapshot,
        trackingSample: makeTrackingSample(liveTracking),
        receivedAtMonotonic: performance.now(),
      }),
    );
    rerender(<App />);

    const toggleAfterRecovery = branchSummary("Left controller");
    expect(toggleAfterRecovery).toBe(toggle);
    expect(groupDetails.open).toBe(true);
    expect(toggleDetails.open).toBe(true);
    expect(document.activeElement).toBe(toggle);
    expect(branchValue(toggle)).toBe("present");
    // Still expanded (never collapsed by the stale transition), showing
    // real data again.
    expect(rowCells(toggleAfterRecovery, "Grip position")).toEqual(["1.000", "2.000", "3.000", "—"]);
  });

  it("never shows a decoded-but-absent controller's retained field values as measurements", () => {
    // present=false but with nonzero retained fields and a true `gripValid`
    // flag -- exactly the leaked-measurement shape the display must mask.
    const absentButPopulated: ControllerDisplay = {
      ...ABSENT_CONTROLLER,
      present: false,
      gripValid: true,
      gripPosition: [9, 9, 9],
      gripOrientation: [0, 0, 0, 1],
      thumbstickX: 0.5,
      triggerValue: 0.9,
    };
    const tracking = makeTracking({ controllerLeft: absentButPopulated });
    mockState(
      makeState({
        phase: "connected",
        snapshot: makeSnapshot({ tracking: { sequence: "1", age_ms: 10, diagnostic: null } }),
        trackingSample: makeTrackingSample(tracking),
        receivedAtMonotonic: performance.now(),
      }),
    );
    render(<App />);

    openControllersGroup();
    const toggle = branchSummary("Left controller");
    fireEvent.click(toggle);
    expect((toggle.closest("details") as HTMLDetailsElement).open).toBe(true);
    expect(rowCells(toggle, "Grip position")).toEqual(["—"]);
    expect(rowCells(toggle, "Thumbstick X")).toEqual(["—"]);
  });

  it("keeps the explicit invalid label distinct from the unavailable dash for a present controller", () => {
    const presentInvalid: ControllerDisplay = {
      ...ABSENT_CONTROLLER,
      present: true,
      gripValid: false,
      gripPosition: [1, 2, 3],
      gripOrientation: [0, 0, 0, 1],
    };
    const tracking = makeTracking({ controllerLeft: presentInvalid });
    mockState(
      makeState({
        phase: "connected",
        snapshot: makeSnapshot({ tracking: { sequence: "1", age_ms: 10, diagnostic: null } }),
        trackingSample: makeTrackingSample(tracking),
        receivedAtMonotonic: performance.now(),
      }),
    );
    render(<App />);

    openControllersGroup();
    const toggle = branchSummary("Left controller");
    fireEvent.click(toggle);
    expect((toggle.closest("details") as HTMLDetailsElement).open).toBe(true);
    expect(rowCells(toggle, "Grip position")).toEqual(["invalid"]);
  });
});

describe("CaptureTree staleness advances with wall-clock time alone", () => {
  it("masks an accepted sample once it crosses the stale threshold without any new props arriving", () => {
    vi.useFakeTimers();
    const liveTracking = makeTracking({
      controllerLeft: { ...ABSENT_CONTROLLER, present: true, gripValid: true, gripPosition: [1, 2, 3], gripOrientation: [0, 0, 0, 1] },
    });
    const fixedState = makeState({
      phase: "connected",
      snapshot: makeSnapshot({ tracking: { sequence: "1", age_ms: 10, diagnostic: null } }),
      trackingSample: makeTrackingSample(liveTracking, { snapshotAgeMs: 10 }),
      receivedAtMonotonic: performance.now(),
    });
    mockState(fixedState);
    render(<App />);

    const groupDetails = openControllersGroup();
    const toggle = branchSummary("Left controller");
    const toggleDetails = toggle.closest("details") as HTMLDetailsElement;
    fireEvent.click(toggle);
    expect(toggleDetails.open).toBe(true);
    expect(rowCells(toggle, "Grip position")).toEqual(["1.000", "2.000", "3.000", "—"]);
    toggle.focus();
    expect(document.activeElement).toBe(toggle);

    // No new snapshot/sample arrives -- the mocked hook's return value
    // never changes -- but wall-clock time advances past the 2s stale
    // threshold. Without CaptureTree's own once-per-second tick, this
    // would stay showing the live grip position indefinitely.
    act(() => {
      vi.advanceTimersByTime(2_500);
    });

    expect(branchValue(toggle)).toBe("unavailable");
    expect(rowCells(toggle, "Grip position")).toEqual(["—"]);
    // Expansion and focus must not be lost by this transition either --
    // actual `.open` state, not just node identity or row text.
    expect(branchSummary("Left controller")).toBe(toggle);
    expect(groupDetails.open).toBe(true);
    expect(toggleDetails.open).toBe(true);
    expect(document.activeElement).toBe(toggle);
  });
});

describe("CaptureTree diagnostic vs never-had-data", () => {
  it("shows the generic empty label when no tracking sample has ever arrived", () => {
    mockState(makeState({ snapshot: makeSnapshot({ tracking: null }) }));
    render(<App />);
    expect(screen.getByText("No tracking sample yet.")).toBeTruthy();
  });

  it("shows the backend diagnostic instead of the generic label when one is present", () => {
    mockState(
      makeState({
        snapshot: makeSnapshot({
          tracking: { sequence: null, age_ms: null, diagnostic: "tracking sample exceeded the preview size limit and was dropped" },
        }),
      }),
    );
    render(<App />);
    expect(screen.getByText("tracking sample exceeded the preview size limit and was dropped")).toBeTruthy();
    expect(screen.queryByText("No tracking sample yet.")).toBeNull();
  });
});
