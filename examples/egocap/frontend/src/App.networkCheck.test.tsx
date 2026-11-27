// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The network check as the operator page shows it: driven from the status snapshot whether or not a take is running,
 * never a pass once the feed is stale, and its retry button sends the real recording command. useCaptureSession is
 * mocked so each test controls the snapshot and phase directly.
 */
import { afterEach, describe, expect, it, vi, type Mock } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import type { PreviewSnapshot } from "./captureProtocol";
import { NetworkCheckState, type NetworkCheckStatus, type StatusSnapshot } from "./captureStatus";
import type { RecordingSnapshot } from "./recordingTypes";
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

function networkCheck(overrides: Partial<NetworkCheckStatus> = {}): NetworkCheckStatus {
  return {
    state: NetworkCheckState.READY,
    generation: "g1",
    endpoint: "capture host to station switch",
    legs_tested: ["camera media delivery"],
    legs_not_tested: ["recorder disk throughput"],
    window_s: 10,
    valid_for_s: 60,
    phase: null,
    progress_s: null,
    completed_at_unix_ns: 1,
    reason: null,
    cameras: [],
    ...overrides,
  };
}

function snapshot(check: NetworkCheckStatus | null | "absent"): PreviewSnapshot {
  const status: StatusSnapshot = { updated_at_ns: "1", providers: [], devices: [], cameras: [], network_check: null };
  if (check === "absent") delete (status as Partial<StatusSnapshot>).network_check;
  else status.network_check = check;
  return {
    version: 1,
    type: 3,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station",
    profile_label: "Profile",
    connecting: false,
    started: { session_id: "s1", running: true, mode: 0, streams: [] },
    status: { session_id: "s1", type: 3, running: true, startup_stage: null, message: null, status },
    stopped: null,
    error_message: null,
    binary_generation: 1,
    tracking: null,
    cameras: [],
    visualization: null,
  };
}

function recording(overrides: Partial<RecordingSnapshot> = {}): RecordingSnapshot {
  return {
    batchId: "batch-7",
    episodePhase: "IDLE",
    currentEpisodeId: null,
    currentEpisodeElapsedMs: null,
    taskText: "Place the object",
    taskDetails: "",
    defaultOperator: null,
    autoFillOperatorName: false,
    savedCount: 0,
    lastResult: null,
    readyToRecord: true,
    notReadyReason: null,
    revision: 1,
    ...overrides,
  };
}

function session(overrides: Partial<CaptureSessionState> = {}): CaptureSessionState {
  return {
    phase: "connected",
    snapshot: snapshot(networkCheck()),
    receivedAtMs: Date.now(),
    receivedAtMonotonic: performance.now(),
    diagnostic: null,
    trackingSample: null,
    recordingSnapshot: recording(),
    recordingSnapshotReceivedAtMonotonic: performance.now(),
    retry: vi.fn(),
    sendRecordingCommand: vi.fn().mockResolvedValue({ outcome: "ok", reason: null }),
    operatorToken: "op",
    setOperatorToken: vi.fn(),
    clearOperatorToken: vi.fn(),
    controlToken: "token",
    setControlToken: vi.fn(),
    clearControlToken: vi.fn(),
    ...overrides,
  };
}

function setSession(state: CaptureSessionState): void {
  (useCaptureSession as unknown as Mock).mockReturnValue(state);
}

const panel = () => screen.queryByRole("region", { name: "Network check" });
const runButton = () => screen.getByRole("button", { name: /Run (test again|network test)/ }) as HTMLButtonElement;

describe("network check on the operator page", () => {
  it("shows no panel when the snapshot carries no network_check", () => {
    setSession(session({ snapshot: snapshot(null) }));
    render(<App />);
    expect(panel()).toBeNull();
  });

  it("shows no panel when the key is absent from the status", () => {
    setSession(session({ snapshot: snapshot("absent") }));
    render(<App />);
    expect(panel()).toBeNull();
  });

  it("shows the panel while idle and while recording", () => {
    setSession(session());
    const idle = render(<App />);
    expect(screen.getByText("Network test passed")).toBeTruthy();
    idle.unmount();

    setSession(
      session({ recordingSnapshot: recording({ episodePhase: "RECORDING", currentEpisodeId: "e1", currentEpisodeElapsedMs: 10 }) }),
    );
    render(<App />);
    expect(screen.getByText("Network test passed")).toBeTruthy();
  });

  it("follows the status from checking to ready and then to not current when the feed goes stale", () => {
    setSession(
      session({ snapshot: snapshot(networkCheck({ state: NetworkCheckState.CHECKING, phase: "measuring", progress_s: 3 })) }),
    );
    const { rerender, container } = render(<App />);
    expect(screen.getByText("Network test in progress (measuring, 3 of 10 seconds)")).toBeTruthy();
    expect(runButton().disabled).toBe(true);

    setSession(session({ snapshot: snapshot(networkCheck()) }));
    rerender(<App />);
    expect(screen.getByText("Network test passed")).toBeTruthy();
    expect(container.querySelector(".netcheck--ok")).not.toBeNull();
    expect(runButton().disabled).toBe(false);

    // The same retained READY snapshot, but the feed has gone stale.
    setSession(session({ phase: "reconnecting", snapshot: snapshot(networkCheck()) }));
    rerender(<App />);
    expect(screen.getByText("Network test: status unavailable or out of date")).toBeTruthy();
    expect(screen.queryByText("Network test passed")).toBeNull();
    expect(container.querySelector(".netcheck--ok")).toBeNull();
    expect(runButton().disabled).toBe(true);
  });

  it("does not draw a pass from the retained snapshot of an unavailable feed", () => {
    setSession(session({ phase: "unavailable", snapshot: snapshot(networkCheck()) }));
    const { container } = render(<App />);
    expect(screen.queryByText("Network test passed")).toBeNull();
    expect(container.querySelector(".netcheck--ok")).toBeNull();
  });

  it("returns to a pass only once the feed is live again", () => {
    setSession(session({ phase: "reconnecting", snapshot: snapshot(networkCheck()) }));
    const { rerender } = render(<App />);
    expect(screen.queryByText("Network test passed")).toBeNull();
    setSession(session({ snapshot: snapshot(networkCheck()) }));
    rerender(<App />);
    expect(screen.getByText("Network test passed")).toBeTruthy();
  });

  it("retry sends run_network_check with the current batch id", async () => {
    const sendRecordingCommand = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    setSession(session({ sendRecordingCommand }));
    render(<App />);
    fireEvent.click(runButton());
    await waitFor(() => expect(sendRecordingCommand).toHaveBeenCalledTimes(1));
    expect(sendRecordingCommand).toHaveBeenCalledWith("run_network_check", { batch_id: "batch-7" });
  });

  it("retry uses the batch id of the latest recording snapshot", async () => {
    const sendRecordingCommand = vi.fn().mockResolvedValue({ outcome: "ok", reason: null });
    setSession(session({ sendRecordingCommand }));
    const { rerender } = render(<App />);
    setSession(session({ sendRecordingCommand, recordingSnapshot: recording({ batchId: "batch-8", revision: 2 }) }));
    rerender(<App />);
    fireEvent.click(runButton());
    await waitFor(() => expect(sendRecordingCommand).toHaveBeenCalledWith("run_network_check", { batch_id: "batch-8" }));
  });

  it("disables retry with an explanation while a take is recording, and enables it when idle", () => {
    const reason = "Finish the current take before checking the connection";
    setSession(
      session({ recordingSnapshot: recording({ episodePhase: "RECORDING", currentEpisodeId: "e1", currentEpisodeElapsedMs: 10 }) }),
    );
    const { rerender } = render(<App />);
    expect(runButton().disabled).toBe(true);
    expect(runButton().title).toBe(reason);

    setSession(session());
    rerender(<App />);
    expect(runButton().disabled).toBe(false);
    expect(runButton().title).toBe("");
  });

  it("disables retry when there is no recording snapshot to name a batch", () => {
    setSession(session({ recordingSnapshot: null, recordingSnapshotReceivedAtMonotonic: null }));
    render(<App />);
    expect(runButton().disabled).toBe(true);
  });

  it("shows the server's reason when the retry is rejected", async () => {
    const sendRecordingCommand = vi.fn().mockResolvedValue({ outcome: "rejected", reason: "a take is recording" });
    setSession(session({ sendRecordingCommand }));
    render(<App />);
    fireEvent.click(runButton());
    expect(await screen.findByText("a take is recording")).toBeTruthy();
  });
});
