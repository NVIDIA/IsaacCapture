/**
 * One live camera tile fed directly by a LiveKit video track.
 *
 * Readiness comes from the track and from decoded frames actually reaching the <video> element, never from a
 * JPEG slot or a CaptureServer camera descriptor. A track that exists but stops producing frames is shown as
 * stale rather than healthy.
 */

import { useEffect, useRef, useState } from "react";
import type { RemoteTrack } from "livekit-client";
import type { LiveKitPhase } from "./useLiveKitPreview";

/** No decoded frame for this long means the picture is frozen. */
const FRAME_STALE_AFTER_MS = 3000;

/** Seconds-long windows keep the rate readable and the state updates to once per second. */
const FPS_WINDOW_MS = 1000;

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
}: {
  name: string;
  track: RemoteTrack | null;
  phase: LiveKitPhase;
  primary: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastFrameAtRef = useRef<number | null>(null);
  // Cumulative presented-frame count of the newest frame callback, and the count at the start of the window.
  const presentedRef = useRef<number | null>(null);
  const baselineRef = useRef<{ presented: number; at: number } | null>(null);
  const [fpsLabel, setFpsLabel] = useState<string | null>(null);
  const [hasFrame, setHasFrame] = useState(false);
  const [stale, setStale] = useState(false);

  // Attach the subscribed track to the element, and detach it when the track changes or the tile goes away.
  useEffect(() => {
    const element = videoRef.current;
    lastFrameAtRef.current = null;
    presentedRef.current = null;
    baselineRef.current = null;
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
      if (!hasFrameCallback) {
        setFpsLabel("FPS unavailable");
      } else if (document.hidden) {
        baselineRef.current = null; // a hidden tab's scheduling is not comparable: restart after it returns
        setFpsLabel("paused");
      } else if (isStale) {
        baselineRef.current = null;
        setFpsLabel("0 FPS");
      } else if (presented !== null) {
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

  const placeholder = placeholderFor(phase, track, hasFrame, stale);
  return (
    <figure className={`camera-tile${primary ? " camera-tile--primary" : ""}${stale ? " tone-warning" : ""}`}>
      {/* The element stays mounted while a track exists so attach/detach never races a re-created node. */}
      {track !== null && <video ref={videoRef} muted autoPlay playsInline aria-label={name} />}
      {placeholder !== null && <div className="camera-placeholder">{placeholder}</div>}
      <figcaption>
        {name}
        {fpsLabel !== null && <span className="camera-stats">{fpsLabel}</span>}
      </figcaption>
    </figure>
  );
}
