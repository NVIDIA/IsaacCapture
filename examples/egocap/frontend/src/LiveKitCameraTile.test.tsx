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

describe("LiveKitCameraTile frame rate and latency", () => {
  /** Present one frame carrying browser metadata, as a real requestVideoFrameCallback would. */
  function presentFrameWith(metadata: Record<string, number>): void {
    const cbs = frameCallbacks as unknown as Array<(now: number, metadata: Record<string, number>) => void>;
    frameCallbacks = [];
    act(() => cbs.forEach((cb) => cb(performance.now(), metadata)));
  }

  /** Thirty frames a second for `seconds`, with the browser's running frame counter and capture times. */
  function stream30Fps(seconds: number, latencyMs: number): void {
    const frames = seconds * 30;
    for (let i = 1; i <= frames; i++) {
      act(() => {
        vi.advanceTimersByTime(1000 / 30);
      });
      const now = performance.now();
      presentFrameWith({ presentationTime: now, presentedFrames: i, captureTime: now - latencyMs });
    }
  }

  it("shows the measured rate against the declared one and the end-to-end latency once a second", () => {
    render(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} advertisedFps={30} />);
    stream30Fps(3, 120);
    const stats = screen.getByLabelText("Head frame rate and latency");
    expect(stats.textContent).toMatch(/^30\.0 \/ 30 Hz · 120 ms$/);
  });

  it("shows a dash for latency when the browser reports no capture time, and no declared rate when none is known", () => {
    render(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} />);
    for (let i = 1; i <= 90; i++) {
      act(() => {
        vi.advanceTimersByTime(1000 / 30);
      });
      presentFrameWith({ presentationTime: performance.now(), presentedFrames: i });
    }
    expect(screen.getByLabelText("Head frame rate and latency").textContent).toMatch(/^30\.0 Hz · —$/);
  });

  it("shows no stats before the first frame, while the rate is still unmeasured, or once the picture is stale", () => {
    render(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} />);
    expect(screen.queryByLabelText("Head frame rate and latency")).toBeNull();
    // One second of frames is not enough span for a rate: the line shows a dash, not a made-up number. The
    // frames land a hair short of the one-second tick that refreshes the line, so nudge the clock past it.
    stream30Fps(1, 100);
    act(() => {
      vi.advanceTimersByTime(20);
    });
    expect(screen.getByLabelText("Head frame rate and latency").textContent).toMatch(/^— Hz · 100 ms$/);
    stream30Fps(2, 100);
    act(() => {
      vi.advanceTimersByTime(20);
    });
    expect(screen.getByLabelText("Head frame rate and latency").textContent).toMatch(/^30\.0 Hz/);
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(screen.getByText("No video frames")).not.toBeNull();
    expect(screen.queryByLabelText("Head frame rate and latency")).toBeNull();
  });

  it("starts the measurement over for a replacement track", () => {
    const { rerender } = render(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} />);
    stream30Fps(3, 100);
    expect(screen.getByLabelText("Head frame rate and latency")).not.toBeNull();
    rerender(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} />);
    expect(screen.queryByLabelText("Head frame rate and latency")).toBeNull();
    stream30Fps(1, 100);
    expect(screen.getByLabelText("Head frame rate and latency").textContent).toMatch(/^— Hz/);
  });
});

describe("LiveKitCameraTile role badge", () => {
  it("shows the assigned role beside the name, and nothing when unassigned", () => {
    const { unmount } = render(<LiveKitCameraTile name="cam4" track={null} phase="connected" primary={false} role="wrist_left" />);
    expect(screen.getByText("Left wrist").className).toBe("camera-role");
    unmount();
    render(<LiveKitCameraTile name="cam4" track={null} phase="connected" primary={false} />);
    expect(document.querySelector(".camera-role")).toBeNull();
  });
});
