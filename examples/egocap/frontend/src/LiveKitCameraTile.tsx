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

/** Text shown instead of a picture, or null when a fresh picture is showing. */
function placeholderFor(phase: LiveKitPhase, track: RemoteTrack | null, hasFrame: boolean, stale: boolean): string | null {
  if (phase === "error") return "Disconnected";
  if (phase !== "connected") return phase === "reconnecting" ? "Reconnecting" : "Connecting";
  if (track === null) return "Waiting for camera";
  if (!hasFrame) return "Waiting for video";
  return stale ? "No video frames" : null;
}

/** A named preview tile; always labelled preview-only because these feeds are not recorded. */
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
  const [hasFrame, setHasFrame] = useState(false);
  const [stale, setStale] = useState(false);

  // Attach the subscribed track to the element, and detach it when the track changes or the tile goes away.
  useEffect(() => {
    const element = videoRef.current;
    lastFrameAtRef.current = null;
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
      callbackId = element.requestVideoFrameCallback(() => {
        markFrame();
        watchFrames();
      });
    };
    // requestVideoFrameCallback reports presented frames exactly; without it, playback progress is the fallback.
    if (typeof element.requestVideoFrameCallback === "function") watchFrames();
    else element.addEventListener("timeupdate", markFrame);
    const timer = setInterval(() => {
      const last = lastFrameAtRef.current;
      if (last !== null && performance.now() - last > FRAME_STALE_AFTER_MS) setStale(true);
    }, 1000);
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
        <span className="camera-preview-badge">Preview only, not recorded</span>
      </figcaption>
    </figure>
  );
}
