// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { CameraGrid } from "./CameraGrid";
import type { LiveKitStreamDescriptor, PreviewSnapshot } from "./captureProtocol";
import { STREAMS, FakeRoom } from "./livekitTestDoubles";

vi.mock("livekit-client", async () => {
  const doubles = await import("./livekitTestDoubles");
  return {
    Room: doubles.FakeRoom,
    RoomEvent: {
      TrackSubscribed: "TrackSubscribed",
      TrackUnsubscribed: "TrackUnsubscribed",
      ParticipantDisconnected: "ParticipantDisconnected",
      Reconnecting: "Reconnecting",
      Reconnected: "Reconnected",
      Disconnected: "Disconnected",
    },
    Track: { Kind: { Video: "video", Audio: "audio" } },
  };
});

function snapshot(names: string[], offered: LiveKitStreamDescriptor[] = []): PreviewSnapshot {
  return {
    version: 1,
    type: 3,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station",
    profile_label: "Profile",
    connecting: false,
    started: { session_id: "s1", running: true, mode: 0, streams: offered },
    status: null,
    stopped: null,
    error_message: null,
    binary_generation: 1,
    tracking: null,
    cameras: names.map((name, i) => ({ stream_id: i + 1, name, url: `/stream${i + 1}?a=1`, sequence: null, age_ms: null, diagnostic: null })),
    visualization: null,
  } as PreviewSnapshot;
}

function stubFetch(status: number, body: unknown): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async () => ({ status, ok: status === 200, json: async () => body }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

async function settled(): Promise<void> {
  // Let the hook's fetch and connect promises resolve.
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

beforeEach(() => FakeRoom.reset());
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("CameraGrid with offered LiveKit streams", () => {
  it("renders LiveKit tiles beside MJPEG tiles, with the MJPEG camera keeping the primary slot", async () => {
    stubFetch(200, { token: "tok" });
    const { container } = render(
      <CameraGrid snapshot={snapshot(["head_camera"], STREAMS)} live={true} receivedAtMonotonic={performance.now()} />,
    );
    await settled();
    expect(screen.getByText("head_camera")).not.toBeNull();
    expect(screen.getByText("Head left", { exact: false })).not.toBeNull();
    expect(screen.getByText("Head right", { exact: false })).not.toBeNull();
    expect(screen.queryByText(/Preview only/)).toBeNull();
    expect(container.querySelector(".camera-preview-badge")).toBeNull();
    expect(container.querySelector(".camera-grid-items")?.className).toContain("camera-grid-items--n3");
    const primary = container.querySelectorAll(".camera-tile--primary");
    expect(primary).toHaveLength(1);
    expect(primary[0].textContent).toContain("head_camera");
  });

  it("shows only LiveKit tiles, the first as primary, when CaptureServer has no cameras", async () => {
    stubFetch(200, { token: "tok" });
    const { container } = render(
      <CameraGrid snapshot={snapshot([], STREAMS)} live={false} receivedAtMonotonic={null} />,
    );
    await settled();
    expect(screen.queryByText("No cameras selected.")).toBeNull();
    expect(container.querySelector(".camera-tile--primary")?.textContent).toContain("Head left");
  });

  it("gives two head tiles equal primary weight with the heads-only layout", async () => {
    stubFetch(200, { token: "tok" });
    const { container } = render(
      <CameraGrid snapshot={snapshot([], STREAMS)} live={false} receivedAtMonotonic={null} />,
    );
    await settled();
    const grid = container.querySelector<HTMLElement>(".camera-grid-items");
    expect(grid?.className).toContain("camera-grid-items--heads");
    expect(grid?.className).toContain("camera-grid-items--heads-only");
    expect(grid?.style.getPropertyValue("--head-count")).toBe("2");
    expect(container.querySelectorAll(".camera-tile--primary")).toHaveLength(2);
  });

  it("keeps a wrist tile secondary beside two primary head tiles", async () => {
    stubFetch(200, { token: "tok" });
    const offered = [
      ...STREAMS,
      { ...STREAMS[0], stream_id: 18, name: "Wrist left", track: "wrist" },
    ];
    const { container } = render(
      <CameraGrid snapshot={snapshot([], offered)} live={false} receivedAtMonotonic={null} />,
    );
    await settled();
    const grid = container.querySelector<HTMLElement>(".camera-grid-items");
    expect(grid?.className).toContain("camera-grid-items--heads");
    expect(grid?.className).not.toContain("camera-grid-items--heads-only");
    expect(grid?.style.getPropertyValue("--head-count")).toBe("2");
    expect(grid?.style.getPropertyValue("--side-count")).toBe("1");
    const primary = container.querySelectorAll(".camera-tile--primary");
    expect(primary).toHaveLength(2);
    primary.forEach((tile) => expect(tile.textContent).toContain("Head"));
    expect(screen.getByText("Wrist left", { exact: false }).closest(".camera-tile--primary")).toBeNull();
  });

  it("keeps the single-tile layout when only one LiveKit tile is offered", async () => {
    stubFetch(200, { token: "tok" });
    const { container } = render(
      <CameraGrid snapshot={snapshot([], [STREAMS[0]])} live={false} receivedAtMonotonic={null} />,
    );
    await settled();
    expect(container.querySelector(".camera-grid-items")?.className).toContain("camera-grid-items--n1");
    const primary = container.querySelectorAll(".camera-tile--primary");
    expect(primary).toHaveLength(1);
    expect(primary[0].textContent).toContain("Head left");
  });

  it("does not contact the backend and keeps MJPEG behavior when no streams are offered", async () => {
    const fetchFn = stubFetch(200, { token: "tok" });
    const { container, rerender } = render(<CameraGrid snapshot={null} live={false} receivedAtMonotonic={null} />);
    await settled();
    expect(screen.getByText("No cameras selected.")).not.toBeNull();
    expect(screen.queryByText(/LiveKit preview unavailable/)).toBeNull();
    rerender(<CameraGrid snapshot={snapshot(["a", "b"])} live={true} receivedAtMonotonic={performance.now()} />);
    expect(container.querySelector(".camera-grid-items")?.className).toContain("camera-grid-items--n2");
    expect(screen.queryByText(/Preview only/)).toBeNull();
    expect(FakeRoom.instances).toHaveLength(0);
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("shows the unavailable note with the backend detail when the token cannot be issued", async () => {
    stubFetch(503, { detail: "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set" });
    render(<CameraGrid snapshot={snapshot([], STREAMS)} live={false} receivedAtMonotonic={null} />);
    await settled();
    expect(screen.getByText(/LiveKit preview unavailable: LIVEKIT_API_KEY/)).not.toBeNull();
  });

  it("shows the unavailable note when the backend cannot be reached", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("fetch failed");
      }),
    );
    render(<CameraGrid snapshot={snapshot([], STREAMS)} live={false} receivedAtMonotonic={null} />);
    await settled();
    expect(screen.getByText("LiveKit preview unavailable: Cannot reach the backend")).not.toBeNull();
  });
});
