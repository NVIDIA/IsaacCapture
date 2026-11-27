// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The network check panel: what each state says, and that a pass is never drawn from a result that is not current. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NetworkCheckState, type NetworkCheckCamera, type NetworkCheckStatus } from "./captureStatus";
import { NetworkCheckPanel, headlineFor } from "./NetworkCheckPanel";
import type { RecordingCommandResult } from "./recordingTypes";

afterEach(cleanup);

function camera(overrides: Partial<NetworkCheckCamera> = {}): NetworkCheckCamera {
  return {
    stream_id: 1,
    name: "head_camera",
    passed: true,
    expected_fps: 30,
    usable_frames: 300,
    delivered_fps: 29.94,
    missing_frames: 0,
    max_capture_gap_ms: 40,
    max_arrival_gap_ms: 55,
    start_silence_ms: 10,
    end_silence_ms: 12,
    age_median_ms: 80,
    age_max_ms: 120,
    age_growth_ms: 1,
    reasons: [],
    ...overrides,
  };
}

function check(overrides: Partial<NetworkCheckStatus> = {}): NetworkCheckStatus {
  return {
    state: NetworkCheckState.READY,
    generation: "g1",
    endpoint: "capture host to station switch",
    legs_tested: ["camera media delivery"],
    legs_not_tested: ["recorder disk throughput", "headset link"],
    window_s: 10,
    valid_for_s: 60,
    phase: null,
    progress_s: null,
    completed_at_unix_ns: 1,
    reason: null,
    cameras: [camera()],
    ...overrides,
  };
}

const OK: RecordingCommandResult = { outcome: "ok", reason: null };

describe("NetworkCheckPanel", () => {
  it("renders nothing when the server runs no check", () => {
    const { container } = render(<NetworkCheckPanel check={null} live={true} onRun={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the starting phase while a check begins", () => {
    render(<NetworkCheckPanel check={check({ state: NetworkCheckState.CHECKING, phase: "starting" })} live={true} />);
    expect(screen.getByText("Network test in progress (connecting to camera streams)")).toBeTruthy();
  });

  it("shows measuring progress against the window", () => {
    render(
      <NetworkCheckPanel
        check={check({ state: NetworkCheckState.CHECKING, phase: "measuring", progress_s: 4, window_s: 10 })}
        live={true}
      />,
    );
    expect(screen.getByText("Network test in progress (measuring, 4 of 10 seconds)")).toBeTruthy();
  });

  it("draws a current READY result as a pass", () => {
    const { container } = render(<NetworkCheckPanel check={check()} live={true} onRun={vi.fn()} />);
    expect(screen.getByText("Network test passed")).toBeTruthy();
    expect(container.querySelector(".netcheck--ok")).not.toBeNull();
  });

  it("shows UNSUITABLE with the server's reason", () => {
    const { container } = render(
      <NetworkCheckPanel
        check={check({ state: NetworkCheckState.UNSUITABLE, reason: "head_camera drops frames" })}
        live={true}
      />,
    );
    expect(screen.getByText("Network test failed")).toBeTruthy();
    expect(screen.getByText("head_camera drops frames")).toBeTruthy();
    expect(container.querySelector(".netcheck--bad")).not.toBeNull();
  });

  it("shows STALE as out of date, not as a pass", () => {
    const { container } = render(<NetworkCheckPanel check={check({ state: NetworkCheckState.STALE })} live={true} />);
    expect(screen.getByText("Network test expired; run it again before recording")).toBeTruthy();
    expect(container.querySelector(".netcheck--ok")).toBeNull();
  });

  it("shows NOT_TESTED as not run yet", () => {
    render(<NetworkCheckPanel check={check({ state: NetworkCheckState.NOT_TESTED, cameras: [] })} live={true} />);
    expect(screen.getByText("Network test has not run for this capture session")).toBeTruthy();
  });

  it.each([
    NetworkCheckState.NOT_TESTED,
    NetworkCheckState.CHECKING,
    NetworkCheckState.READY,
    NetworkCheckState.UNSUITABLE,
    NetworkCheckState.STALE,
  ])("when the feed is not live, state %i reads 'status not current' and is never green", (state) => {
    const { container } = render(
      <NetworkCheckPanel check={check({ state, reason: "x" })} live={false} onRun={vi.fn()} />,
    );
    expect(screen.getByText("Network test: status unavailable or out of date")).toBeTruthy();
    expect(container.querySelector(".netcheck--ok")).toBeNull();
    expect(container.querySelector(".netcheck--idle")).not.toBeNull();
    expect(screen.queryByText("Network test passed")).toBeNull();
    // An old failure reason is not presented as current either.
    expect(screen.queryByText("x")).toBeNull();
  });

  it("headlineFor gives idle tone to a READY result that is not live", () => {
    expect(headlineFor(check(), false)).toEqual({ label: "Network test: status unavailable or out of date", tone: "idle" });
    expect(headlineFor(check(), true).tone).toBe("ok");
  });

  describe("camera rows", () => {
    it("shows pass, fail and not judged with reasons", () => {
      render(
        <NetworkCheckPanel
          check={check({
            cameras: [
              camera({ stream_id: 1, name: "head_camera", passed: true }),
              camera({ stream_id: 2, name: "left_wrist", passed: false, reasons: ["too many missing frames", "late"] }),
              camera({ stream_id: 3, name: "right_wrist", passed: null }),
            ],
          })}
          live={true}
        />,
      );
      const rows = screen.getAllByRole("listitem");
      expect(rows).toHaveLength(3);
      expect(rows[0].textContent).toContain("Passed");
      expect(rows[1].textContent).toContain("Failed");
      expect(rows[1].textContent).toContain("too many missing frames; late");
      expect(rows[2].textContent).toContain("Not measured");
    });

    it("marks a live READY pass as Passed with the pass class", () => {
      const { container } = render(<NetworkCheckPanel check={check()} live={true} />);
      expect(screen.getByText("Passed")).toBeTruthy();
      expect(container.querySelector(".netcheck-camera--pass")).not.toBeNull();
      expect(container.querySelector(".netcheck-camera--last-known")).toBeNull();
    });

    it("marks a live UNSUITABLE failing camera as Failed with the fail class", () => {
      const { container } = render(
        <NetworkCheckPanel
          check={check({
            state: NetworkCheckState.UNSUITABLE,
            reason: "head_camera drops frames",
            cameras: [camera({ passed: false, reasons: ["late"] })],
          })}
          live={true}
        />,
      );
      expect(screen.getByText("Failed")).toBeTruthy();
      expect(container.querySelector(".netcheck-camera--fail")).not.toBeNull();
    });

    it.each([
      ["a READY check that is not live", NetworkCheckState.READY, false],
      ["a live STALE check", NetworkCheckState.STALE, true],
    ])("shows %s as last known, never in the pass class", (_name, state, live) => {
      const { container } = render(<NetworkCheckPanel check={check({ state })} live={live} />);
      expect(screen.getByText("Passed (last known)")).toBeTruthy();
      expect(container.querySelector(".netcheck-camera--last-known")).not.toBeNull();
      expect(container.querySelector(".netcheck-camera--pass")).toBeNull();
    });

    it("labels retained failed and unmeasured cameras as last known too", () => {
      render(
        <NetworkCheckPanel
          check={check({
            cameras: [
              camera({ stream_id: 2, name: "left_wrist", passed: false }),
              camera({ stream_id: 3, name: "right_wrist", passed: null }),
            ],
          })}
          live={false}
        />,
      );
      expect(screen.getByText("Failed (last known)")).toBeTruthy();
      expect(screen.getByText("Not measured (last known)")).toBeTruthy();
    });

    it("renders null measurements as unknown, never zero", () => {
      render(
        <NetworkCheckPanel
          check={check({
            cameras: [
              camera({
                passed: null,
                delivered_fps: null,
                missing_frames: null,
                max_arrival_gap_ms: null,
                start_silence_ms: null,
                end_silence_ms: null,
                age_median_ms: null,
                age_max_ms: null,
              }),
            ],
          })}
          live={true}
        />,
      );
      const row = screen.getByRole("listitem").textContent ?? "";
      expect(row).toContain("Delivery rate: unknown (configured: 30 fps)");
      expect(row).toContain("Missing frames: unknown");
      expect(row).toContain("Longest gap without usable frames: unknown");
      expect(row).toContain("Capture-to-receive delay: median unknown; maximum unknown");
      expect(row).not.toMatch(/\b0 ms|Missing frames: 0|^0\b/);
    });

    it("renders real numbers with their units, and a genuine zero as zero", () => {
      render(<NetworkCheckPanel check={check({ cameras: [camera({ missing_frames: 0 })] })} live={true} />);
      const row = screen.getByRole("listitem").textContent ?? "";
      expect(row).toContain("Delivery rate: 29.9 fps (configured: 30 fps)");
      expect(row).toContain("Usable frames received: 300");
      expect(row).toContain("Missing frames: 0");
      expect(row).toContain("Longest gap without usable frames: 55 ms");
      expect(row).toContain("Capture-to-receive delay: median 80 ms; maximum 120 ms");
    });
  });

  it("lists the endpoint and the legs tested and not tested under 'What this tests'", () => {
    render(<NetworkCheckPanel check={check()} live={true} />);
    expect(screen.getByText("Test coverage")).toBeTruthy();
    expect(screen.getByText("Media endpoint: capture host to station switch")).toBeTruthy();
    expect(screen.getByText("Measured path: camera media delivery.")).toBeTruthy();
    expect(screen.getByText("Outside this test: recorder disk throughput; headset link.")).toBeTruthy();
  });

  describe("run button", () => {
    const button = () => screen.getByRole("button") as HTMLButtonElement;

    it("reads 'Check connection' when never tested and 'Check again' otherwise", () => {
      const { unmount } = render(
        <NetworkCheckPanel check={check({ state: NetworkCheckState.NOT_TESTED })} live={true} onRun={vi.fn()} />,
      );
      expect(button().textContent).toBe("Run network test");
      unmount();
      for (const state of [
        NetworkCheckState.CHECKING,
        NetworkCheckState.READY,
        NetworkCheckState.UNSUITABLE,
        NetworkCheckState.STALE,
      ]) {
        const r = render(<NetworkCheckPanel check={check({ state })} live={true} onRun={vi.fn()} />);
        expect(button().textContent).toBe("Run test again");
        r.unmount();
      }
    });

    it("is enabled when live, idle and wired", () => {
      render(<NetworkCheckPanel check={check()} live={true} onRun={vi.fn()} />);
      expect(button().disabled).toBe(false);
    });

    it("is disabled while a check is running", () => {
      render(<NetworkCheckPanel check={check({ state: NetworkCheckState.CHECKING })} live={true} onRun={vi.fn()} />);
      expect(button().disabled).toBe(true);
    });

    it("is disabled when the feed is not live", () => {
      render(<NetworkCheckPanel check={check()} live={false} onRun={vi.fn()} />);
      expect(button().disabled).toBe(true);
    });

    it("is disabled with the blocked reason as its title, and does not send", () => {
      const onRun = vi.fn();
      render(<NetworkCheckPanel check={check()} live={true} onRun={onRun} runBlockedReason="Finish the current take" />);
      expect(button().disabled).toBe(true);
      expect(button().title).toBe("Finish the current take");
      fireEvent.click(button());
      expect(onRun).not.toHaveBeenCalled();
    });

    it("is enabled with no title when the blocked reason is null or omitted", () => {
      const { unmount } = render(<NetworkCheckPanel check={check()} live={true} onRun={vi.fn()} runBlockedReason={null} />);
      expect(button().disabled).toBe(false);
      expect(button().title).toBe("");
      unmount();
      render(<NetworkCheckPanel check={check()} live={true} onRun={vi.fn()} />);
      expect(button().disabled).toBe(false);
    });

    it("is disabled when no handler is supplied", () => {
      render(<NetworkCheckPanel check={check()} live={true} />);
      expect(button().disabled).toBe(true);
    });

    it("is disabled while its own request is in flight and re-enabled after, sending exactly one request", async () => {
      let resolve!: (value: RecordingCommandResult) => void;
      const onRun = vi.fn(() => new Promise<RecordingCommandResult>((r) => (resolve = r)));
      render(<NetworkCheckPanel check={check()} live={true} onRun={onRun} />);
      fireEvent.click(button());
      expect(button().disabled).toBe(true);
      fireEvent.click(button());
      expect(onRun).toHaveBeenCalledTimes(1);
      resolve(OK);
      await waitFor(() => expect(button().disabled).toBe(false));
    });

    it("shows a rejected result's reason, and clears it on the next attempt", async () => {
      const onRun = vi
        .fn<() => Promise<RecordingCommandResult>>()
        .mockResolvedValueOnce({ outcome: "rejected", reason: "a take is recording" })
        .mockResolvedValueOnce(OK);
      render(<NetworkCheckPanel check={check()} live={true} onRun={onRun} />);
      fireEvent.click(button());
      expect(await screen.findByText("a take is recording")).toBeTruthy();
      fireEvent.click(button());
      await waitFor(() => expect(screen.queryByText("a take is recording")).toBeNull());
    });

    it("falls back to a generic reason when a rejection carries none", async () => {
      const onRun = vi.fn().mockResolvedValue({ outcome: "rejected", reason: null });
      render(<NetworkCheckPanel check={check()} live={true} onRun={onRun} />);
      fireEvent.click(button());
      expect(await screen.findByText("Network test did not start")).toBeTruthy();
    });

    it("says the server could not be reached when the request throws", async () => {
      const onRun = vi.fn().mockRejectedValue(new Error("socket closed"));
      render(<NetworkCheckPanel check={check()} live={true} onRun={onRun} />);
      fireEvent.click(button());
      expect(await screen.findByText("Could not confirm whether the network test started; check the reported status")).toBeTruthy();
      await waitFor(() => expect(button().disabled).toBe(false));
    });

    it("shows no refusal after a successful request", async () => {
      const onRun = vi.fn().mockResolvedValue(OK);
      const { container } = render(
        <NetworkCheckPanel check={check({ completed_at_unix_ns: null })} live={true} onRun={onRun} />,
      );
      fireEvent.click(button());
      await waitFor(() => expect(onRun).toHaveBeenCalled());
      await waitFor(() => expect(button().disabled).toBe(false));
      expect(container.querySelector(".netcheck-reason")).toBeNull();
    });
  });

  it("shows when the sample was taken and that it is not watched afterwards", () => {
    const at = Date.UTC(2026, 0, 2, 3, 4, 5);
    render(<NetworkCheckPanel check={check({ completed_at_unix_ns: at * 1e6, valid_for_s: 60 })} live={true} />);
    const expected = `Test completed at ${new Date(at).toLocaleTimeString()}. Results expire after 60 seconds. This test does not monitor subsequent connection changes.`;
    expect(screen.getByText((_, node) => node?.textContent?.replace(/\s+/g, " ").trim() === expected && node.tagName === "P")).toBeTruthy();
  });

  it("omits the sample line when no sample has completed", () => {
    render(<NetworkCheckPanel check={check({ state: NetworkCheckState.NOT_TESTED, completed_at_unix_ns: null })} live={true} />);
    expect(screen.queryByText(/Test completed at/)).toBeNull();
  });
});
