// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * CameraGrid's per-tile rate/latency readout: derived from successive snapshots' camera
 * sequence numbers, shown only for live fresh samples, and dashed until enough history exists.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { CameraGrid } from "./CameraGrid";
import type { PreviewSnapshot } from "./captureProtocol";

// CameraGrid asks the backend for a LiveKit preview on mount; these tests are about other behavior.
beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async () => ({ status: 404, ok: false, json: async () => ({}) })));
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
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

describe("CameraGrid primary camera", () => {
  function three(names: string[]): PreviewSnapshot {
    const base = makeSnapshot("1", 10);
    return {
      ...base,
      cameras: names.map((name, i) => ({ stream_id: i + 1, name, url: `/stream${i + 1}?a=1`, sequence: "1", age_ms: 10, diagnostic: null })),
    };
  }

  it("gives the head camera the primary slot wherever it is listed", () => {
    const { container } = render(<CameraGrid snapshot={three(["left_wrist", "head_human_camera", "right_wrist"])} live={true} receivedAtMonotonic={performance.now()} />);
    const primary = container.querySelectorAll(".camera-tile--primary");
    expect(primary).toHaveLength(1);
    expect(primary[0].textContent).toContain("head_human_camera");
  });

  it("falls back to the first camera when none is called head", () => {
    const { container } = render(<CameraGrid snapshot={three(["cam_a", "cam_b"])} live={true} receivedAtMonotonic={performance.now()} />);
    expect(container.querySelector(".camera-tile--primary")?.textContent).toContain("cam_a");
  });

  it("tags the grid with how many cameras it holds, capped at four", () => {
    const { container } = render(<CameraGrid snapshot={three(["a", "b", "c", "d", "e"])} live={true} receivedAtMonotonic={performance.now()} />);
    expect(container.querySelector(".camera-grid-items")?.className).toContain("camera-grid-items--n4");
  });
});

describe("Camera role assignments", () => {
  function twoCameras(): PreviewSnapshot {
    const snapshot = makeSnapshot("1", 0);
    return { ...snapshot, cameras: [...snapshot.cameras, { ...snapshot.cameras[0], stream_id: 2, name: "wrist_camera", url: "/stream2" }] };
  }

  it("gives the assigned ego camera the primary tile and shows role badges", () => {
    const { container } = render(<CameraGrid snapshot={twoCameras()} live={true} receivedAtMonotonic={performance.now()} cameraRoles={{ ego: "wrist_camera", wrist_left: "head_camera" }} />);
    const primary = container.querySelectorAll(".camera-tile--primary");
    expect(primary).toHaveLength(1);
    expect(primary[0].textContent).toContain("wrist_camera");
    expect(primary[0].textContent).toContain("Ego (head)");
    expect(screen.getByText("Left wrist")).toBeTruthy();
  });

  it("restores the head-camera fallback after clearing the ego role", () => {
    const snapshot = twoCameras();
    const { container, rerender } = render(<CameraGrid snapshot={snapshot} live={true} receivedAtMonotonic={null} cameraRoles={{ ego: "wrist_camera" }} />);
    expect(container.querySelector(".camera-tile--primary")?.textContent).toContain("wrist_camera");
    rerender(<CameraGrid snapshot={snapshot} live={true} receivedAtMonotonic={null} cameraRoles={{ ego: null }} />);
    expect(container.querySelector(".camera-tile--primary")?.textContent).toContain("head_camera");
    expect(screen.queryByText("Ego (head)")).toBeNull();
  });

  it("keeps an absent ego assignment from hiding the primary camera", () => {
    const { container } = render(<CameraGrid snapshot={twoCameras()} live={true} receivedAtMonotonic={null} cameraRoles={{ ego: "offline_camera" }} />);
    expect(container.querySelectorAll(".camera-tile--primary")).toHaveLength(1);
    expect(container.querySelector(".camera-tile--primary")?.textContent).toContain("head_camera");
  });
});
