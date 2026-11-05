/**
 * One live camera tile fed directly by a LiveKit video track.
 *
 * Readiness comes from the track and from decoded frames actually reaching the <video> element, never from a
 * JPEG slot or a CaptureServer camera descriptor. A track that exists but stops producing frames is shown as
 * stale rather than healthy.
 */

import { useEffect, useRef, useState } from "react";
import type { RemoteTrack } from "livekit-client";
import type { LiveKitLatencyEntry } from "./livekitPreview";
import type { LiveKitPhase } from "./useLiveKitPreview";

/** No decoded frame for this long means the picture is frozen. */
const FRAME_STALE_AFTER_MS = 3000;

/** Seconds-long windows keep the rate readable and the state updates to once per second. */
const FPS_WINDOW_MS = 1000;

/** Most recent latency samples kept per window; one per presented frame at typical rates fits comfortably. */
const MAX_LATENCY_SAMPLES = 240;

/** Median and 95th percentile of a window's latency samples in ms, or null when there are none. */
export function latencySummary(samples: number[]): { median: number; p95: number } | null {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const at = (q: number): number => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return { median: at(0.5), p95: at(0.95) };
}

/** The displayed rate for one window from cumulative presented-frame counts; null for an invalid interval. */
export function displayedFps(previous: { presented: number; at: number }, presented: number, at: number): number | null {
  const elapsedMs = at - previous.at;
  const frames = presented - previous.presented;
  if (elapsedMs <= 0 || frames < 0) return null; // a counter reset or a clock oddity: start a new baseline
  return (frames * 1000) / elapsedMs;
}

/** Text shown instead of a picture, or null when a fresh picture is showing. */
function placeholderFor(phase: LiveKitPhase, track: RemoteTrack | null, hasFrame: boolean, stale: boolean): string | null {
  if (phase === "error") return "Disconnected";
  if (phase !== "connected") return phase === "reconnecting" ? "Reconnecting" : "Connecting";
  if (track === null) return "Waiting for camera";
  if (!hasFrame) return "Waiting for video";
  return stale ? "No video frames" : null;
}

/** A named live tile. It makes no claim about recording: whether a stream is saved is the recorder's state. */
export function LiveKitCameraTile({
  name,
  track,
  phase,
  primary,
  pcLatency = null,
}: {
  name: string;
  track: RemoteTrack | null;
  phase: LiveKitPhase;
  primary: boolean;
  /** The backend's measured publisher-stamp-to-receipt delay for this stream, used when the browser has none. */
  pcLatency?: LiveKitLatencyEntry | null;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastFrameAtRef = useRef<number | null>(null);
  // Cumulative presented-frame count of the newest frame callback, and the count at the start of the window.
  const presentedRef = useRef<number | null>(null);
  const baselineRef = useRef<{ presented: number; at: number } | null>(null);
  // Capture-to-expected-display samples (ms) from the browser's own frame metadata, for the current window.
  const latencySamplesRef = useRef<number[]>([]);
  const [fpsLabel, setFpsLabel] = useState<string | null>(null);
  const [latencyLabel, setLatencyLabel] = useState<string | null>(null);
  const [hasFrame, setHasFrame] = useState(false);
  const [stale, setStale] = useState(false);

  // Attach the subscribed track to the element, and detach it when the track changes or the tile goes away.
  useEffect(() => {
    const element = videoRef.current;
    lastFrameAtRef.current = null;
    presentedRef.current = null;
    baselineRef.current = null;
    latencySamplesRef.current = [];
    setFpsLabel(null);
    setLatencyLabel(null);
    setHasFrame(false);
    setStale(false);
    if (element === null || track === null) return;
    track.attach(element);
    return () => {
      track.detach(element);
    };
  }, [track]);

  // Frame liveness: note the time of every presented frame, and judge staleness once a second.
  useEffect(() => {
    const element = videoRef.current;
    if (element === null || track === null) return;
    let stopped = false;
    let callbackId: number | null = null;
    const markFrame = (): void => {
      lastFrameAtRef.current = performance.now();
      setHasFrame(true);
      setStale(false);
    };
    const watchFrames = (): void => {
      if (stopped) return;
      callbackId = element.requestVideoFrameCallback((_now, metadata) => {
        presentedRef.current = metadata.presentedFrames;
        // Only when the browser gives this frame a capture time (WebRTC video, in the same time origin as the
        // expected display time): the sender's capture event as the browser maps it, not a measurement of ours.
        const capture = metadata.captureTime;
        if (typeof capture === "number") {
          const latency = metadata.expectedDisplayTime - capture;
          if (Number.isFinite(latency) && latency >= 0) {
            const samples = latencySamplesRef.current;
            samples.push(latency);
            if (samples.length > MAX_LATENCY_SAMPLES) samples.shift();
          }
        }
        markFrame();
        watchFrames();
      });
    };
    // requestVideoFrameCallback reports presented frames exactly; without it, playback progress is the fallback.
    if (typeof element.requestVideoFrameCallback === "function") watchFrames();
    else element.addEventListener("timeupdate", markFrame);
    const hasFrameCallback = typeof element.requestVideoFrameCallback === "function";
    const timer = setInterval(() => {
      const now = performance.now();
      const last = lastFrameAtRef.current;
      const isStale = last !== null && now - last > FRAME_STALE_AFTER_MS;
      if (isStale) setStale(true);
      // Displayed rate: the change in the browser's own presented-frame count over the window, not a callback tally.
      const presented = presentedRef.current;
      // The window's latency samples are summarized once and cleared, so a stalled stream never keeps an old number.
      const summary = latencySummary(latencySamplesRef.current);
      latencySamplesRef.current = [];
      if (!hasFrameCallback) {
        setFpsLabel("FPS unavailable");
        setLatencyLabel("latency unavailable");
      } else if (document.hidden) {
        baselineRef.current = null; // a hidden tab's scheduling is not comparable: restart after it returns
        setFpsLabel("paused");
        setLatencyLabel(null);
      } else if (isStale) {
        baselineRef.current = null;
        setFpsLabel("0 FPS");
        setLatencyLabel(null);
      } else if (presented !== null) {
        setLatencyLabel(
          summary === null
            ? "latency unavailable"
            : `${summary.median.toFixed(0)} ms (p95 ${summary.p95.toFixed(0)})`,
        );
        const previous = baselineRef.current;
        baselineRef.current = { presented, at: now };
        const fps = previous === null ? null : displayedFps(previous, presented, now);
        if (fps !== null) setFpsLabel(`${fps.toFixed(1)} FPS`);
      }
    }, FPS_WINDOW_MS);
    return () => {
      stopped = true;
      clearInterval(timer);
      if (callbackId !== null) element.cancelVideoFrameCallback(callbackId);
      element.removeEventListener("timeupdate", markFrame);
    };
  }, [track]);

  // The browser's own capture-to-display figure when it has one; otherwise the backend's stamp-to-receipt delay.
  const browserHasLatency = latencyLabel !== null && latencyLabel !== "latency unavailable";
  const pcSummary = pcLatency?.summary ?? null;
  const latencyText = browserHasLatency
    ? latencyLabel
    : pcSummary !== null
      ? `${pcSummary.median_ms.toFixed(0)} ms (p95 ${pcSummary.p95_ms.toFixed(0)}) to PC`
      : (latencyLabel ?? "");
  const latencyTitle = browserHasLatency
    ? "Sender capture time to expected display time, as the browser reports it; not a qualified sensor timestamp"
    : pcSummary !== null
      ? "Publisher stamp to this PC's backend receiving the frame; includes the publisher's clock offset and excludes browser decode and render"
      : (pcLatency?.note ?? "No capture time reaches the browser for this stream");
  const placeholder = placeholderFor(phase, track, hasFrame, stale);
  return (
    <figure className={`camera-tile${primary ? " camera-tile--primary" : ""}${stale ? " tone-warning" : ""}`}>
      {/* The element stays mounted while a track exists so attach/detach never races a re-created node. */}
      {track !== null && <video ref={videoRef} muted autoPlay playsInline aria-label={name} />}
      {placeholder !== null && <div className="camera-placeholder">{placeholder}</div>}
      <figcaption>
        {name}
        {fpsLabel !== null && <span className="camera-stats">{fpsLabel}</span>}
        {latencyLabel !== null && (
          <span className="camera-stats" title={latencyTitle}>
            {latencyText}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
