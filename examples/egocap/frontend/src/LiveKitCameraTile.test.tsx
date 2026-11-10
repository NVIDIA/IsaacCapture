// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { LiveKitCameraTile } from "./LiveKitCameraTile";
import { fakeTrack } from "./livekitTestDoubles";

let frameCallbacks: Array<() => void> = [];

beforeEach(() => {
  vi.useFakeTimers();
  frameCallbacks = [];
  // jsdom has no requestVideoFrameCallback; this one lets a test present frames by hand.
  (HTMLVideoElement.prototype as any).requestVideoFrameCallback = function (cb: () => void) {
    frameCallbacks.push(cb);
    return frameCallbacks.length;
  };
  (HTMLVideoElement.prototype as any).cancelVideoFrameCallback = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  delete (HTMLVideoElement.prototype as any).requestVideoFrameCallback;
  delete (HTMLVideoElement.prototype as any).cancelVideoFrameCallback;
});

function presentFrame(): void {
  const cbs = frameCallbacks;
  frameCallbacks = [];
  act(() => cbs.forEach((cb) => cb()));
}

describe("LiveKitCameraTile placeholders", () => {
  it("shows Connecting while the room connects", () => {
    render(<LiveKitCameraTile name="Head" track={null} phase="connecting" primary={false} />);
    expect(screen.getByText("Connecting")).not.toBeNull();
  });

  it("shows Reconnecting while the SDK reconnects", () => {
    render(<LiveKitCameraTile name="Head" track={null} phase="reconnecting" primary={false} />);
    expect(screen.getByText("Reconnecting")).not.toBeNull();
  });

  it("shows Disconnected in the error phase", () => {
    render(<LiveKitCameraTile name="Head" track={null} phase="error" primary={false} />);
    expect(screen.getByText("Disconnected")).not.toBeNull();
  });

  it("shows Waiting for camera when connected but the track is not published", () => {
    const { container } = render(<LiveKitCameraTile name="Head" track={null} phase="connected" primary={false} />);
    expect(screen.getByText("Waiting for camera")).not.toBeNull();
    expect(container.querySelector("video")).toBeNull();
  });

  it("shows Waiting for video until the first frame is presented, then no placeholder", () => {
    const track = fakeTrack();
    render(<LiveKitCameraTile name="Head" track={track} phase="connected" primary={false} />);
    expect(screen.getByText("Waiting for video")).not.toBeNull();
    presentFrame();
    expect(screen.queryByText("Waiting for video")).toBeNull();
    expect(document.querySelector(".camera-placeholder")).toBeNull();
  });

  it("shows No video frames and a warning tone when frames stop for more than three seconds", () => {
    const track = fakeTrack();
    const { container } = render(<LiveKitCameraTile name="Head" track={track} phase="connected" primary={false} />);
    presentFrame();
    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(screen.queryByText("No video frames")).toBeNull();
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByText("No video frames")).not.toBeNull();
    expect(container.querySelector("figure")?.className).toContain("tone-warning");
    // Frames resuming clears the warning.
    presentFrame();
    expect(screen.queryByText("No video frames")).toBeNull();
  });

  it("falls back to timeupdate when requestVideoFrameCallback is unavailable", () => {
    delete (HTMLVideoElement.prototype as any).requestVideoFrameCallback;
    const track = fakeTrack();
    const { container } = render(<LiveKitCameraTile name="Head" track={track} phase="connected" primary={false} />);
    act(() => {
      container.querySelector("video")!.dispatchEvent(new Event("timeupdate"));
    });
    expect(screen.queryByText("Waiting for video")).toBeNull();
  });
});

describe("LiveKitCameraTile track wiring and labels", () => {
  it("attaches the track to a muted inline video and detaches it on unmount", () => {
    const track = fakeTrack();
    const { container, unmount } = render(<LiveKitCameraTile name="Head" track={track} phase="connected" primary={false} />);
    const video = container.querySelector("video")!;
    expect(track.attach).toHaveBeenCalledWith(video);
    expect(video.muted).toBe(true);
    expect(video.autoplay).toBe(true);
    expect(video.hasAttribute("playsinline")).toBe(true);
    unmount();
    expect(track.detach).toHaveBeenCalledWith(video);
  });

  it("detaches the old track and attaches the new one when the track changes", () => {
    const a = fakeTrack();
    const b = fakeTrack();
    const { container, rerender } = render(<LiveKitCameraTile name="Head" track={a} phase="connected" primary={false} />);
    const video = container.querySelector("video")!;
    rerender(<LiveKitCameraTile name="Head" track={b} phase="connected" primary={false} />);
    expect(a.detach).toHaveBeenCalledWith(video);
    expect(b.attach).toHaveBeenCalledWith(video);
  });

  it("goes back to Waiting for video when a replacement track has not produced a frame", () => {
    const a = fakeTrack();
    const { rerender } = render(<LiveKitCameraTile name="Head" track={a} phase="connected" primary={false} />);
    presentFrame();
    rerender(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} />);
    expect(screen.getByText("Waiting for video")).not.toBeNull();
  });

  it("shows its name and makes no claim about recording", () => {
    render(<LiveKitCameraTile name="Head left" track={null} phase="connected" primary={true} />);
    expect(screen.getByText("Head left", { exact: false })).not.toBeNull();
    expect(screen.queryByText(/Preview only/)).toBeNull();
    expect(document.querySelector(".camera-preview-badge")).toBeNull();
    expect(document.querySelector(".camera-tile--primary")).not.toBeNull();
  });
});
