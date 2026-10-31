/**
 * CameraGrid's per-tile rate/latency readout: derived from successive snapshots' camera
 * sequence numbers, shown only for live fresh samples, and dashed until enough history exists.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { CameraGrid } from "./CameraGrid";
import type { PreviewSnapshot } from "./captureProtocol";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

function makeSnapshot(sequence: string | null, ageMs: number | null, fps = 60): PreviewSnapshot {
  return {
    version: 1,
    type: 3,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station",
    profile_label: "Profile",
    connecting: false,
    started: {
      session_id: "s1",
      running: true,
      mode: 0,
      streams: [{ kind: "camera", stream_id: 1, name: "head_camera", width: 1280, height: 992, fps }],
    },
    status: { session_id: "s1", type: 3, running: true, startup_stage: null, message: null, status: null },
    stopped: null,
    error_message: null,
    binary_generation: 1,
    tracking: null,
    cameras: [{ stream_id: 1, name: "head_camera", url: "/stream1?a=1", sequence, age_ms: ageMs, diagnostic: null }],
    visualization: null,
  };
}

describe("CameraGrid rate and latency readout", () => {
  it("shows dashes for the rate until two seconds of history exist, then the measured rate", () => {
    vi.useFakeTimers();
    const t0 = performance.now();
    const { rerender } = render(<CameraGrid snapshot={makeSnapshot("1000", 21)} live={true} receivedAtMonotonic={t0} />);
    expect(screen.getByLabelText("head_camera frame rate and latency").textContent).toBe("— / 60 Hz · 21 ms");

    // Two heartbeats later the sequence advanced by 96 frames over 2 s: 48 Hz.
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    rerender(<CameraGrid snapshot={makeSnapshot("1048", 19)} live={true} receivedAtMonotonic={t0 + 1000} />);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    rerender(<CameraGrid snapshot={makeSnapshot("1096", 23)} live={true} receivedAtMonotonic={t0 + 2000} />);
    expect(screen.getByLabelText("head_camera frame rate and latency").textContent).toBe("48.0 / 60 Hz · 23 ms");
  });

  it("hides the readout when the tile is not live or has never had a sample", () => {
    const t0 = performance.now();
    const { rerender } = render(<CameraGrid snapshot={makeSnapshot(null, null)} live={true} receivedAtMonotonic={t0} />);
    expect(screen.queryByLabelText("head_camera frame rate and latency")).toBeNull();
    rerender(<CameraGrid snapshot={makeSnapshot("5", 10)} live={false} receivedAtMonotonic={t0} />);
    expect(screen.queryByLabelText("head_camera frame rate and latency")).toBeNull();
    expect(screen.getByText("Disconnected")).not.toBeNull();
  });
});
