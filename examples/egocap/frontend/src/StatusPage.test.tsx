// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The status page reads /api/server-status: configured cameras, session enrichment, sequential polling, staleness. */
import { act, cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CaptureMode, DeviceState, NetworkCheckState } from "./captureStatus";
import { StatusPage } from "./StatusPage";

const POLL_MS = 2000;

const WRIST = {
  name: "wrist",
  transport: "capture_server",
  selected: false,
  device_id: "MX1",
  room: null,
  publisher: null,
  track: null,
  width: 640,
  height: 480,
  fps: 15,
  stream_id: null,
  state: null,
  error: null,
};
const HEAD = {
  name: "head",
  transport: "livekit",
  selected: false,
  device_id: null,
  room: "room-a",
  publisher: "orin",
  track: "head-track",
  width: 1280,
  height: 720,
  fps: 30,
  stream_id: 16,
  state: null,
  error: null,
};

function check(state: number) {
  return {
    state,
    generation: "g",
    endpoint: "e",
    legs_tested: ["camera media delivery"],
    legs_not_tested: ["headset link"],
    window_s: 10,
    valid_for_s: 60,
    phase: null,
    progress_s: null,
    completed_at_unix_ns: null,
    reason: null,
    cameras: [],
  };
}

function payload(overrides: Record<string, unknown> = {}) {
  return {
    mode: CaptureMode.MOCK,
    session_id: null,
    status: null,
    network_check: null,
    cameras: [WRIST, HEAD],
    ...overrides,
  };
}

function ok(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

const fetchMock = vi.fn();

async function settle(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function mountWith(...replies: unknown[]) {
  for (const reply of replies) fetchMock.mockResolvedValueOnce(reply);
  render(<StatusPage />);
  await settle();
}

function cameraRows() {
  return within(screen.getByRole("region", { name: "Cameras configured on CaptureServer" })).queryAllByRole("listitem");
}

beforeEach(() => {
  vi.useFakeTimers();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("StatusPage", () => {
  it("lists the configured physical and LiveKit cameras with no session", async () => {
    await mountWith(ok(payload()));
    expect(fetchMock).toHaveBeenCalledWith("/api/server-status", expect.objectContaining({ cache: "no-store" }));
    const rows = cameraRows();
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("wrist (CaptureServer camera MX1, 640x480 @ 15 fps)");
    expect(rows[0].textContent).toContain("configured, not observed");
    expect(rows[1].textContent).toContain("head (LiveKit room room-a, track head-track, 1280x720 @ 30 fps)");
    expect(rows[1].textContent).toContain("offered (delivery: see the network check)");
    expect(screen.getByText("Providers and devices are reported only while a session runs.")).toBeTruthy();
    expect(screen.getByText(/no session running/)).toBeTruthy();
    expect(screen.getByText("Network check: not configured on this server")).toBeTruthy();
  });

  it("merges a session's observation by name without duplicating a camera", async () => {
    const status = {
      updated_at_ns: "1",
      providers: [],
      devices: [],
      cameras: [],
    };
    await mountWith(
      ok(
        payload({
          session_id: "s1",
          status,
          cameras: [
            { ...WRIST, selected: true, state: DeviceState.FAILED, error: "usb reset" },
            { ...HEAD, selected: true },
          ],
        }),
      ),
    );
    const rows = cameraRows();
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("; in session)");
    expect(rows[0].textContent).toContain("failed");
    expect(rows[0].textContent).toContain("usb reset");
    expect(rows[1].textContent).toContain("offered (delivery: see the network check)");
    expect(screen.getByText(/session s1/)).toBeTruthy();
    expect(screen.queryByText("Providers and devices are reported only while a session runs.")).toBeNull();
  });

  it("reads a connected observed state from the device state", async () => {
    await mountWith(ok(payload({ cameras: [{ ...WRIST, selected: true, state: DeviceState.CONNECTED }] })));
    expect(cameraRows()[0].textContent).toContain("connected");
  });

  it("never starts a second read while one is outstanding", async () => {
    let resolve: (value: unknown) => void = () => {};
    fetchMock.mockReturnValueOnce(new Promise((r) => (resolve = r)));
    render(<StatusPage />);
    await settle(POLL_MS * 10);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    fetchMock.mockResolvedValue(ok(payload()));
    await act(async () => {
      resolve(ok(payload()));
    });
    await settle(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await settle(POLL_MS);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("aborts an in-flight read and clears the timers and the clock when it unmounts", async () => {
    fetchMock.mockImplementation(
      (_url: string, init: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const { unmount } = render(<StatusPage />);
    await settle();
    const signal = fetchMock.mock.calls[0][1].signal as AbortSignal;
    expect(signal.aborted).toBe(false);
    expect(vi.getTimerCount()).toBeGreaterThan(0);
    unmount();
    expect(signal.aborted).toBe(true);
    await settle();
    expect(vi.getTimerCount()).toBe(0);
    await settle(POLL_MS * 5);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("times out a hung read, goes stale without a finished read, and polls again", async () => {
    await mountWith(ok(payload({ network_check: check(NetworkCheckState.READY) })));
    expect(screen.getByText("Network test passed")).toBeTruthy();
    fetchMock.mockImplementation(
      (_url: string, init: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    await settle(POLL_MS);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    // Mid-hang: the clock alone ages the last read past the stale window.
    await settle(POLL_MS * 3);
    expect(screen.getByText(/showing the last known state/)).toBeTruthy();
    expect(screen.queryByText("Network test passed")).toBeNull();
    await settle(5000);
    expect(screen.getByText(/the status read timed out; showing the last known state/)).toBeTruthy();
    await settle(POLL_MS);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("does not schedule another read after an in-flight read is aborted", async () => {
    let rejectRead: (error: unknown) => void = () => {};
    fetchMock.mockImplementationOnce(
      (_url: string, init: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          rejectRead = reject;
          init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const { unmount } = render(<StatusPage />);
    await settle();
    unmount();
    await settle(POLL_MS * 5);
    rejectRead(new Error("late"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("stops reading a passed check as a pass once reads keep failing", async () => {
    await mountWith(ok(payload({ network_check: check(NetworkCheckState.READY) })));
    expect(screen.getByText("Network test passed")).toBeTruthy();
    fetchMock.mockRejectedValue(new Error("network down"));
    await settle(POLL_MS * 2);
    await settle(POLL_MS * 3);
    expect(screen.getByText(/network down; showing the last known state/)).toBeTruthy();
    expect(screen.queryByText("Network test passed")).toBeNull();
    expect(screen.getByText("Network test: status unavailable or out of date")).toBeTruthy();
    expect(cameraRows()).toHaveLength(2);
  });

  it("recovers once a read succeeds again", async () => {
    await mountWith(ok(payload({ network_check: check(NetworkCheckState.READY) })));
    fetchMock.mockRejectedValueOnce(new Error("blip"));
    fetchMock.mockResolvedValue(ok(payload({ network_check: check(NetworkCheckState.READY) })));
    await settle(POLL_MS * 6);
    expect(screen.queryByText(/showing the last known state/)).toBeNull();
    expect(screen.getByText("Network test passed")).toBeTruthy();
  });

  it("shows a 503 as a problem, not as an empty list of cameras", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: "refused" }) });
    render(<StatusPage />);
    await settle();
    expect(screen.getByText(/HTTP 503; showing the last known state/)).toBeTruthy();
    expect(screen.queryByText("no cameras configured")).toBeNull();
    expect(screen.getByText("not read yet")).toBeTruthy();
  });

  it("shows a malformed reply as a problem", async () => {
    await mountWith(ok({ mode: CaptureMode.MOCK }));
    expect(screen.getByText(/showing the last known state/)).toBeTruthy();
    expect(screen.queryByText("no cameras configured")).toBeNull();
  });

  it("says plainly when the server has no cameras configured", async () => {
    await mountWith(ok(payload({ cameras: [] })));
    expect(screen.getByText("no cameras configured")).toBeTruthy();
  });

  it("disables the run button: running the check stays on the recording page", async () => {
    await mountWith(ok(payload({ network_check: check(NetworkCheckState.NOT_TESTED) })));
    const buttons = screen.getAllByRole("button") as HTMLButtonElement[];
    expect(buttons.length).toBeGreaterThan(0);
    expect(buttons.every((b) => b.disabled)).toBe(true);
  });
});
