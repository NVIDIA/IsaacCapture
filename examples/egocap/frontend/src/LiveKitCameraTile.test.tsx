// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { LiveKitCameraTile, displayedFps } from "./LiveKitCameraTile";
import { fakeTrack } from "./livekitTestDoubles";

let frameCallbacks: Array<(now: number, metadata: { presentedFrames: number }) => void> = [];
let presentedTotal = 0;

beforeEach(() => {
  vi.useFakeTimers();
  frameCallbacks = [];
  presentedTotal = 0;
  // jsdom has no requestVideoFrameCallback; this one lets a test present frames by hand.
  (HTMLVideoElement.prototype as any).requestVideoFrameCallback = function (cb: (typeof frameCallbacks)[number]) {
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

/** Presents `frames` frames at once: the browser's cumulative presentedFrames grows by that much. */
function presentFrame(frames = 1): void {
  presentedTotal += frames;
  const cbs = frameCallbacks;
  frameCallbacks = [];
  act(() => cbs.forEach((cb) => cb(0, { presentedFrames: presentedTotal })));
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

describe("displayedFps", () => {
  it("is frames per second over the window", () => {
    expect(displayedFps({ presented: 10, at: 0 }, 40, 1000)).toBe(30);
  });

  it("returns null when the counter went backwards", () => {
    expect(displayedFps({ presented: 40, at: 0 }, 10, 1000)).toBeNull();
  });

  it("returns null for a non-positive interval", () => {
    expect(displayedFps({ presented: 0, at: 1000 }, 5, 1000)).toBeNull();
    expect(displayedFps({ presented: 0, at: 1000 }, 5, 500)).toBeNull();
  });
});

describe("LiveKitCameraTile FPS label", () => {
  function renderTile() {
    return render(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} />);
  }

  it("shows the presented-frame delta per second after two windows", () => {
    renderTile();
    presentFrame(0);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(document.querySelector(".camera-stats")).toBeNull(); // first window only sets the baseline
    presentFrame(30);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(document.querySelector(".camera-stats")?.textContent).toBe("30.0 FPS");
  });

  it("shows 0 FPS when no frame arrives for more than three seconds", () => {
    renderTile();
    presentFrame();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(document.querySelector(".camera-stats")?.textContent).toBe("0 FPS");
  });

  it("shows paused while the document is hidden", () => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    try {
      renderTile();
      presentFrame();
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(document.querySelector(".camera-stats")?.textContent).toBe("paused");
    } finally {
      delete (document as any).hidden;
    }
  });

  it("shows FPS unavailable without requestVideoFrameCallback", () => {
    delete (HTMLVideoElement.prototype as any).requestVideoFrameCallback;
    renderTile();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(document.querySelector(".camera-stats")?.textContent).toBe("FPS unavailable");
  });
});
