// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The operator status bar: the recording indicator's rules, the timer, counts, last result and device health. */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import { DeviceState } from "./captureStatus";
import type { CaptureStatusResponse } from "./captureStatus";
import { OperatorBar, deriveRecIndicator } from "./OperatorBar";
import type { RecordingSnapshot } from "./recordingTypes";

afterEach(cleanup);

function snap(overrides: Partial<RecordingSnapshot> = {}): RecordingSnapshot {
  return {
    batchId: "b1",
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
    revision: 1,
    ...overrides,
  };
}

describe("deriveRecIndicator", () => {
  it("waits for the first snapshot rather than inventing a state", () => {
    expect(deriveRecIndicator(null, true)).toMatchObject({ label: "WAITING", variant: "idle" });
  });

  it("labels a stale feed as such, whatever the retained snapshot says", () => {
    expect(deriveRecIndicator(snap({ episodePhase: "RECORDING" }), false)).toMatchObject({ label: "NO FEED", variant: "stale" });
  });

  it.each([
    ["RECORDING", "REC", "recording"],
    ["STARTING", "STARTING", "busy"],
    ["STOPPING", "STOPPING", "busy"],
    ["FINALIZING", "SAVING", "busy"],
    ["RECOVERY_REQUIRED", "RECOVERY", "warning"],
  ] as const)("shows backend phase %s as %s", (episodePhase, label, variant) => {
    expect(deriveRecIndicator(snap({ episodePhase }), true)).toMatchObject({ label, variant });
  });

  it("is READY only when the backend would admit a take and this browser may start one", () => {
    expect(deriveRecIndicator(snap(), true)).toMatchObject({ label: "READY", variant: "ready" });
    expect(deriveRecIndicator(snap({ readyToRecord: false }), true)).toMatchObject({ label: "NOT READY" });
    expect(deriveRecIndicator(snap({ readyToRecord: false, taskText: "" }), true).hint).toContain("Set a task");
  });

  it("asks for whatever setup is missing instead of claiming READY", () => {
    const none = deriveRecIndicator(snap(), true, { operatorSet: false, controlTokenSet: false });
    expect(none).toMatchObject({ label: "SET UP", variant: "warning" });
    expect(none.hint).toBe("Enter an operator name and the control token below");
    expect(deriveRecIndicator(snap(), true, { operatorSet: false, controlTokenSet: true }).hint).toBe("Enter an operator name below");
    expect(deriveRecIndicator(snap(), true, { operatorSet: true, controlTokenSet: false }).hint).toBe("Enter the control token below");
  });

  it("shows a command in flight at once, since the backend's phase only changes when it finishes", () => {
    const recording = snap({ episodePhase: "RECORDING" });
    expect(deriveRecIndicator(recording, true, undefined, "stopping")).toMatchObject({ label: "STOPPING", variant: "busy" });
    expect(deriveRecIndicator(snap(), true, undefined, "starting").label).toBe("STARTING");
    expect(deriveRecIndicator(recording, true, undefined, "discarding").label).toBe("DISCARDING");
    expect(deriveRecIndicator(snap(), true, undefined, "restarting").label).toBe("NEW BATCH");
    expect(deriveRecIndicator(recording, true, undefined, "uncertain")).toMatchObject({ label: "UNKNOWN", variant: "warning" });
    // editing a task is not a take-level command, so it leaves the indicator alone
    expect(deriveRecIndicator(recording, true, undefined, "editingTask").label).toBe("REC");
  });

  it("lets a stale feed win over an in-flight command", () => {
    expect(deriveRecIndicator(snap(), false, undefined, "stopping").label).toBe("NO FEED");
  });
});

const device = (id: string, status: number, error = "") => ({
  id,
  path: `/${id}`,
  type: "headset",
  description: id,
  provider_id: "p",
  status,
  reason: 0,
  error,
});
const status = (devices: unknown[], cameras: unknown[] = []) =>
  ({ status: { providers: [], devices, cameras } }) as unknown as CaptureStatusResponse;

function bar(props: Partial<Parameters<typeof OperatorBar>[0]> = {}) {
  return render(
    <OperatorBar
      response={status([device("openxr/headset", DeviceState.CONNECTED)])}
      live={true}
      recordingSnapshot={snap()}
      recordingSnapshotReceivedAtMonotonic={performance.now()}
      operatorName="op"
      hasControlToken={true}
      pendingPhase="idle"
      {...props}
    />,
  );
}

describe("OperatorBar", () => {
  it("shows the state, the task and the saved count", () => {
    bar({ recordingSnapshot: snap({ savedCount: 12 }) });
    const region = screen.getByRole("region", { name: "Operator status" });
    expect(within(region).getByText("READY")).not.toBeNull();
    expect(within(region).getByText("Place the object")).not.toBeNull();
    expect(within(region).getByText("12")).not.toBeNull();
  });

  it("shows the task details under the task, and nothing when there are none", () => {
    const { unmount } = bar();
    expect(document.querySelector(".op-task-details")).toBeNull();
    unmount();
    bar({ recordingSnapshot: snap({ taskDetails: "Left hand, blue cup" }) });
    expect(screen.getByText("Left hand, blue cup")).not.toBeNull();
  });

  it("flags an unset task rather than leaving a blank", () => {
    bar({ recordingSnapshot: snap({ taskText: "", readyToRecord: false }) });
    expect(screen.getByText("No task set")).not.toBeNull();
  });

  it("counts up the take in progress and shows nothing while idle", () => {
    const { rerender } = bar({
      recordingSnapshot: snap({ episodePhase: "RECORDING", currentEpisodeElapsedMs: 42_000 }),
      recordingSnapshotReceivedAtMonotonic: performance.now(),
    });
    expect(screen.getByText("REC")).not.toBeNull();
    expect(screen.getByText("0:42")).not.toBeNull();
    rerender(
      <OperatorBar
        response={null}
        live={true}
        recordingSnapshot={snap()}
        recordingSnapshotReceivedAtMonotonic={performance.now()}
        operatorName="op"
        hasControlToken={true}
        pendingPhase="idle"
      />,
    );
    expect(screen.queryByText("0:42")).toBeNull();
  });

  it("shows no running timer next to NO FEED", () => {
    bar({
      live: false,
      recordingSnapshot: snap({ episodePhase: "RECORDING", currentEpisodeElapsedMs: 42_000 }),
      recordingSnapshotReceivedAtMonotonic: performance.now(),
    });
    expect(screen.getByText("NO FEED")).not.toBeNull();
    expect(screen.queryByText("0:42")).toBeNull();
  });

  it("summarises the last take, with its issues on hover", () => {
    bar({
      recordingSnapshot: snap({
        lastResult: { episodeId: "e1", disposition: "REJECTED", issues: ["dropped frames", "short clip"], durationMs: 41_000 },
      }),
    });
    const last = screen.getByText(/Rejected/);
    expect(last.textContent).toBe("Rejected · 0:41 · 2 issues");
    expect(last.closest(".op-last")?.getAttribute("title")).toBe("dropped frames; short clip");
  });

  it("never says all devices are fine from a stale feed", () => {
    bar({ live: false });
    expect(screen.queryByText("All devices OK")).toBeNull();
    expect(screen.getByText("Last known: no faults")).not.toBeNull();
  });

  it("says every device is fine, or counts the ones that are not and lists them first", () => {
    const { unmount } = bar();
    expect(screen.getByText("All devices OK")).not.toBeNull();
    unmount();
    bar({
      response: status(
        [device("openxr/headset", DeviceState.CONNECTED), device("manus/hand/left", DeviceState.DISCONNECTED, "glove asleep")],
        [],
      ),
    });
    expect(screen.getByText("1 need attention")).not.toBeNull();
    const chips = within(screen.getByRole("list", { name: "Device health" })).getAllByRole("listitem");
    expect(chips[0].textContent).toContain("Glove L");
    expect(chips[0].textContent).toContain("glove asleep");
    expect(chips[1].textContent).toBe("Headset");
  });

  it("says so when no device has been reported yet", () => {
    bar({ response: null });
    expect(screen.getByText("none reported yet")).not.toBeNull();
  });
});
