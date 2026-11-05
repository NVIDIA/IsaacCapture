import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { LiveKitCameraTile, displayedFps } from "./LiveKitCameraTile";
import { fakeTrack } from "./livekitTestDoubles";

type FrameMetadata = { presentedFrames: number; [field: string]: number };
let frameCallbacks: Array<(now: number, metadata: FrameMetadata) => void> = [];
let presentedTotal = 0;
let presentationClockMs = 0;

beforeEach(() => {
  vi.useFakeTimers();
  frameCallbacks = [];
  presentedTotal = 0;
  presentationClockMs = 0;
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

/**
 * Presents `frames` frames at once: the browser's cumulative presentedFrames grows by that much. The frame's
 * presentationTime is its own, independent of the fake timers' clock: by default a second after the previous call's.
 */
function presentFrame(frames = 1, times: Record<string, number> = {}): void {
  presentedTotal += frames;
  presentationClockMs += 1000;
  const cbs = frameCallbacks;
  frameCallbacks = [];
  act(() =>
    cbs.forEach((cb) => cb(0, { presentedFrames: presentedTotal, presentationTime: presentationClockMs, ...times })),
  );
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

const statsTexts = (): string[] =>
  Array.from(document.querySelectorAll(".camera-stats")).map((e) => e.textContent ?? "");

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
    expect(document.body.textContent).not.toMatch(/fps/); // first window only sets the baseline
    presentFrame(30);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(statsTexts()[0]).toBe("30 fps");
  });

  it("divides by the callback presentationTime endpoints, not by the timer tick", () => {
    renderTile();
    presentFrame(0, { presentationTime: 5000 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    presentFrame(30, { presentationTime: 6000 });
    act(() => {
      vi.advanceTimersByTime(1500); // a late tick: the window's length still comes from the frames
    });
    expect(statsTexts()[0]).toBe("30 fps");
  });

  it("gives no figure for a window with a counter reset and is right again in the next", () => {
    renderTile();
    presentFrame(30, { presentationTime: 1000 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    presentedTotal = 0; // the browser restarted its count
    presentFrame(10, { presentationTime: 2000 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(document.body.textContent).not.toMatch(/fps/);
    presentFrame(30, { presentationTime: 3000 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(statsTexts()[0]).toBe("30 fps");
  });

  it("gives no figure for a window in which no new frame arrived", () => {
    renderTile();
    presentFrame(5, { presentationTime: 1000 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(document.body.textContent).not.toMatch(/fps/);
  });

  it("shows 0 FPS when no frame arrives for more than three seconds", () => {
    renderTile();
    presentFrame();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(statsTexts()[0]).toBe("0 fps");
  });

  it("shows a dash while the document is hidden", () => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    try {
      renderTile();
      presentFrame();
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(statsTexts()[0]).toBe("– fps");
    } finally {
      delete (document as any).hidden;
    }
  });

  describe("visibility changes", () => {
    const setHidden = (hidden: boolean): void => {
      Object.defineProperty(document, "hidden", { configurable: true, value: hidden });
      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });
    };
    afterEach(() => {
      delete (document as any).hidden;
    });

    it("shows a dash at once when the tab is hidden, without waiting for the timer", () => {
      renderTile();
      presentFrame();
      setHidden(true);
      expect(statsTexts()[0]).toBe("– fps");
    });

    it("starts a fresh FPS baseline after the tab returns", () => {
      renderTile();
      presentFrame(0);
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      presentFrame(30);
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(statsTexts()[0]).toBe("30 fps");
      setHidden(true);
      presentFrame(300); // frames presented while hidden must not inflate the first visible window
      setHidden(false);
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(document.body.textContent).not.toMatch(/\d fps/);
      presentFrame(30);
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(statsTexts()[0]).toBe("30 fps");
    });

    it("removes the visibilitychange listener on unmount", () => {
      const remove = vi.spyOn(document, "removeEventListener");
      const { unmount } = renderTile();
      unmount();
      expect(remove).toHaveBeenCalledWith("visibilitychange", expect.any(Function));
      remove.mockRestore();
    });
  });

  it("shows a dash for FPS without requestVideoFrameCallback", () => {
    delete (HTMLVideoElement.prototype as any).requestVideoFrameCallback;
    renderTile();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(statsTexts()[0]).toBe("– fps");
  });
});

describe("LiveKitCameraTile latency", () => {
  function renderTile() {
    return render(<LiveKitCameraTile name="Head" track={fakeTrack()} phase="connected" primary={false} />);
  }

  it("shows one FPS span plus the fixed latency span, with an explanatory title", () => {
    renderTile();
    presentFrame(1);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    presentFrame(30);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    const spans = document.querySelectorAll(".camera-stats");
    expect(spans).toHaveLength(2);
    expect(spans[1].textContent).toBe("– ms latency");
    expect(spans[1].getAttribute("title")).toBeTruthy();
  });

  it("does not use the browser's captureTime and expectedDisplayTime", () => {
    renderTile();
    for (let i = 0; i < 3; i++) presentFrame(1, { captureTime: 1000 + i, expectedDisplayTime: 1070 + i });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    presentFrame(30, { captureTime: 2000, expectedDisplayTime: 2070 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(statsTexts()).toContain("– ms latency");
    expect(statsTexts().some((t) => /\d ms latency/.test(t))).toBe(false);
  });

  it("lists the sorted frame-callback fields in the latency tooltip", () => {
    renderTile();
    presentFrame(1, { rtpTimestamp: 1, expectedDisplayTime: 2 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    presentFrame(30, { rtpTimestamp: 1, expectedDisplayTime: 2 });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    const span = document.querySelectorAll(".camera-stats")[1];
    expect(span.getAttribute("title")).toContain("reports: expectedDisplayTime, presentationTime, presentedFrames, rtpTimestamp.");
  });

  it("has no reports text in the tooltip before any frame", () => {
    renderTile();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    const span = document.querySelectorAll(".camera-stats")[1];
    expect(span?.getAttribute("title") ?? "").not.toContain("reports:");
  });

  it("keeps the fixed latency span beside 0 FPS for a stale stream", () => {
    renderTile();
    presentFrame(1, { captureTime: 1000, expectedDisplayTime: 1070 });
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(statsTexts()).toEqual(["0 fps", "– ms latency"]);
  });

  it("keeps the fixed latency span beside paused while the document is hidden", () => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    try {
      renderTile();
      presentFrame(1);
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(statsTexts()).toEqual(["– fps", "– ms latency"]);
    } finally {
      delete (document as any).hidden;
    }
  });

  it("shows no latency text before any FPS label", () => {
    renderTile();
    expect(document.body.textContent).not.toMatch(/latency/);
  });
});

describe("LiveKitCameraTile measured latency", () => {
  const MS = 1_000_000n;
  /** Remote clock = local + 3 s, so a frame displayed at local 997_072 ms and stamped 1_000_000 us is 72 ms old. */
  const mappingOf = (epoch = 1) => ({ epoch, offsetNs: 3_000n * MS, rttNs: 4n * MS, uncertaintyNs: 2n * MS, acceptedAtNs: 0n });
  const stampedTrack = (stamp: { userTimestamp: bigint } | null = { userTimestamp: 1_000_000_000n }) => {
    const track = fakeTrack();
    track.lookupFrameMetadata = vi.fn(() => stamp ?? undefined);
    return track;
  };
  const frame = { rtpTimestamp: 42, expectedDisplayTime: 997_072 };

  function renderWith(track: any, clock: { mapping: () => any } | null, basis = 4) {
    return render(<LiveKitCameraTile name="Head" track={track} phase="connected" primary={false} clock={clock as any} basis={basis} />);
  }
  const windows = (n = 2): void => {
    for (let i = 0; i < n; i++) {
      act(() => {
        vi.advanceTimersByTime(1000);
      });
    }
  };

  it("shows the mean after the window, with uncertainty and p95 in the title, looking the stamp up by RTP timestamp", () => {
    const track = stampedTrack();
    renderWith(track, { mapping: () => mappingOf() });
    presentFrame(1);
    windows(1);
    presentFrame(1, frame);
    windows(1);
    expect(statsTexts()[1]).toBe("72 ms latency");
    expect(document.querySelectorAll(".camera-stats")[1].getAttribute("title")).toContain("±2.0 ms, 95th percentile 72 ms");
    expect(track.lookupFrameMetadata).toHaveBeenCalledWith({ rtpTimestamp: 42 });
  });

  it("is unavailable without a stamp", () => {
    renderWith(stampedTrack(null), { mapping: () => mappingOf() });
    presentFrame(1, frame);
    windows(1);
    presentFrame(1, frame); // a frame in the second window, so it has an FPS figure beside the latency
    windows(1);
    expect(statsTexts()[1]).toBe("– ms latency");
  });

  it("is unavailable when the basis is not the capture event, even with stamps", () => {
    const track = stampedTrack();
    renderWith(track, { mapping: () => mappingOf() }, 1);
    presentFrame(1, frame);
    windows(1);
    presentFrame(1, frame); // a frame in the second window, so it has an FPS figure beside the latency
    windows(1);
    expect(statsTexts()[1]).toBe("– ms latency");
    expect(track.lookupFrameMetadata).not.toHaveBeenCalled();
  });

  it("is unavailable without a mapping", () => {
    renderWith(stampedTrack(), { mapping: () => null });
    presentFrame(1, frame);
    windows(1);
    presentFrame(1, frame); // a frame in the second window, so it has an FPS figure beside the latency
    windows(1);
    expect(statsTexts()[1]).toBe("– ms latency");
  });

  it("is unavailable for a window whose mapping epoch changed", () => {
    let epoch = 1;
    renderWith(stampedTrack(), { mapping: () => mappingOf(epoch) });
    presentFrame(1);
    windows(1);
    presentFrame(1, frame); // made under epoch 1
    epoch = 2;
    windows(1);
    expect(statsTexts()[1]).toBe("– ms latency");
  });

  it("is unavailable when the stream is stale or the document is hidden", () => {
    renderWith(stampedTrack(), { mapping: () => mappingOf() });
    presentFrame(1, frame);
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(statsTexts()).toEqual(["0 fps", "– ms latency"]);
    cleanup();
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    try {
      renderWith(stampedTrack(), { mapping: () => mappingOf() });
      presentFrame(1, frame);
      windows(1);
      expect(statsTexts()).toEqual(["– fps", "– ms latency"]);
    } finally {
      delete (document as any).hidden;
    }
  });
});
