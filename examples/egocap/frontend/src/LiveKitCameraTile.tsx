// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * One live camera tile fed directly by a LiveKit video track.
 *
 * Readiness comes from the track and from decoded frames actually reaching the <video> element, never from a
 * JPEG slot or a CaptureServer camera descriptor. A track that exists but stops producing frames is shown as
 * stale rather than healthy.
 */

import { useEffect, useRef, useState } from "react";
import type { RemoteTrack } from "livekit-client";
import { latencySample, summarizeLatency } from "./displayLatency";
import type { LatencySample } from "./displayLatency";
import { TIMESTAMP_BASIS_CAPTURE_MONOTONIC_US } from "./captureProtocol";
import type { LiveKitClockSync } from "./liveKitClockSync";
import type { LiveKitPhase } from "./useLiveKitPreview";

/** No decoded frame for this long means the picture is frozen. */
const FRAME_STALE_AFTER_MS = 3000;

/** Seconds-long windows keep the rate readable and the state updates to once per second. */
const FPS_WINDOW_MS = 1000;
/** Recent latency samples kept per window. */
const MAX_LATENCY_SAMPLES = 240;
/** Shown in place of a number that is not available. */
const NO_LATENCY = "– ms";
const NO_FPS = "– FPS";

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
  clock = null,
  basis = 0,
}: {
  name: string;
  track: RemoteTrack | null;
  phase: LiveKitPhase;
  primary: boolean;
  /** Maps the publisher's monotonic clock onto this browser's; latency needs it. */
  clock?: LiveKitClockSync | null;
  /** What the stream's frame timestamp means; only a capture event on the publisher's monotonic clock counts. */
  basis?: number;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastFrameAtRef = useRef<number | null>(null);
  // Cumulative presented-frame count of the newest frame callback, and the count at the start of the window.
  const presentedRef = useRef<number | null>(null);
  const baselineRef = useRef<{ presented: number; at: number } | null>(null);
  // Which fields this browser's frame callback reports (it varies by browser), to read off for the latency work.
  const frameFieldsRef = useRef<string>("");
  const [frameFields, setFrameFields] = useState<string>("");
  // Latency of each presented frame in the current window, and the label shown for the last full window.
  const latencySamplesRef = useRef<LatencySample[]>([]);
  const [latencyLabel, setLatencyLabel] = useState<string>(NO_LATENCY);
  const [latencyDetail, setLatencyDetail] = useState<string>("");
  const [fpsLabel, setFpsLabel] = useState<string | null>(null);
  const [hasFrame, setHasFrame] = useState(false);
  const [stale, setStale] = useState(false);

  // Attach the subscribed track to the element, and detach it when the track changes or the tile goes away.
  useEffect(() => {
    const element = videoRef.current;
    lastFrameAtRef.current = null;
    presentedRef.current = null;
    baselineRef.current = null;
    latencySamplesRef.current = [];
    setLatencyLabel(NO_LATENCY);
    setLatencyDetail("");
    setFpsLabel(null);
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
        frameFieldsRef.current = Object.keys(metadata).sort().join(", ");
        // The publisher's stamp for exactly this frame, by its RTP timestamp, against its expected display time.
        const rtp = (metadata as { rtpTimestamp?: number }).rtpTimestamp;
        if (basis === TIMESTAMP_BASIS_CAPTURE_MONOTONIC_US && typeof rtp === "number" && "lookupFrameMetadata" in track) {
          const stamp = (track as { lookupFrameMetadata: (a: { rtpTimestamp: number }) => { userTimestamp: bigint } | undefined })
            .lookupFrameMetadata({ rtpTimestamp: rtp });
          const sample = latencySample(metadata.expectedDisplayTime, stamp, clock?.mapping() ?? null);
          if (sample !== null) {
            const samples = latencySamplesRef.current;
            samples.push(sample);
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
      if (frameFieldsRef.current !== "") setFrameFields(frameFieldsRef.current);
      // Displayed rate: the change in the browser's own presented-frame count over the window, not a callback tally.
      const presented = presentedRef.current;
      // The window's latency samples are summarized once and cleared; only those made under the clock mapping's
      // current epoch count, so a window spanning a discontinuity has no figure.
      const samples = latencySamplesRef.current;
      latencySamplesRef.current = [];
      const mapping = clock?.mapping() ?? null;
      const summary = mapping === null ? null : summarizeLatency(samples, mapping.epoch);
      const latencyText = summary === null ? NO_LATENCY : `${summary.mean.toFixed(0)} ms`;
      setLatencyDetail(summary === null ? "" : `±${summary.uncertaintyMs.toFixed(1)} ms, 95th percentile ${summary.p95.toFixed(0)} ms`);
      if (!hasFrameCallback) {
        setFpsLabel(NO_FPS);
        setLatencyLabel(NO_LATENCY);
      } else if (document.hidden) {
        baselineRef.current = null; // a hidden tab's scheduling is not comparable: restart after it returns
        setFpsLabel(NO_FPS);
        setLatencyLabel(NO_LATENCY);
      } else if (isStale) {
        baselineRef.current = null;
        setFpsLabel("0 FPS");
        setLatencyLabel(NO_LATENCY);
      } else if (presented !== null) {
        setLatencyLabel(latencyText);
        const previous = baselineRef.current;
        baselineRef.current = { presented, at: now };
        const fps = previous === null ? null : displayedFps(previous, presented, now);
        if (fps !== null) setFpsLabel(`${fps.toFixed(0)} FPS`);
      }
    }, FPS_WINDOW_MS);
    // The timer can be throttled or suspended while the tab is hidden, so the transition itself resets the window:
    // paused on hide, and a fresh baseline (no figure) for the first window after it returns.
    const onVisibility = (): void => {
      baselineRef.current = null;
      latencySamplesRef.current = [];
      if (document.hidden) {
        setFpsLabel(NO_FPS);
        setLatencyLabel(NO_LATENCY);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stopped = true;
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
      if (callbackId !== null) element.cancelVideoFrameCallback(callbackId);
      element.removeEventListener("timeupdate", markFrame);
    };
  }, [track]);

  const placeholder = placeholderFor(phase, track, hasFrame, stale);
  return (
    <figure className={`camera-tile${primary ? " camera-tile--primary" : ""}${stale ? " tone-warning" : ""}`}>
      {/* The element stays mounted while a track exists so attach/detach never races a re-created node. */}
      {track !== null && <video ref={videoRef} muted autoPlay playsInline aria-label={name} />}
      {placeholder !== null && <div className="camera-placeholder">{placeholder}</div>}
      <figcaption>
        {name}
        {fpsLabel !== null && <span className="camera-stats">{fpsLabel}</span>}
        {fpsLabel !== null && (
          <span
            className="camera-stats"
            title={
              latencyLabel === NO_LATENCY
                ? `No latency for the displayed frame: it needs the publisher's per-frame capture stamp (matched by RTP timestamp), a current clock mapping to the publisher, and a stream whose timestamp is a capture event on the publisher's monotonic clock.${
                    frameFields === "" ? "" : ` This browser's frame callback reports: ${frameFields}.`
                  }`
                : `Capture event on the publisher's clock to the frame's expected display on this page, including receive, decode and render (${latencyDetail}); ± is half the clock-exchange round trip plus this browser's timer resolution`
            }
          >
            {latencyLabel}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
