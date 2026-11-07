/**
 * One live camera tile fed directly by a LiveKit video track.
 *
 * Readiness comes from the track and from decoded frames actually reaching the <video> element, never from a
 * JPEG slot or a CaptureServer camera descriptor. A track that exists but stops producing frames is shown as
 * stale rather than healthy. The caption carries the same rate/latency line as a CaptureServer tile, measured
 * from the frames this browser presents (see liveKitFrameStats.ts for what can and cannot be measured).
 */

import { useEffect, useRef, useState } from "react";
import type { RemoteTrack } from "livekit-client";
import { formatCameraStats } from "./cameraRates";
import { ROLE_LABELS } from "./settingsApi";
import { PresentedFrameRate, endToEndLatencyMs, type PresentedFrameMetadata } from "./liveKitFrameStats";
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

/** A named live tile. It makes no claim about recording: whether a stream is saved is the recorder's state. */
export function LiveKitCameraTile({
  name,
  track,
  phase,
  primary,
  advertisedFps = null,
  role = null,
}: {
  name: string;
  track: RemoteTrack | null;
  phase: LiveKitPhase;
  primary: boolean;
  /** The frame rate the publisher declared for this track, shown beside the measured one; null when unknown. */
  advertisedFps?: number | null;
  /** The role this camera fills (ego, a wrist), shown beside its name; null when unassigned. */
  role?: string | null;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastFrameAtRef = useRef<number | null>(null);
  const rateRef = useRef(new PresentedFrameRate());
  const latencyRef = useRef<number | null>(null);
  const [hasFrame, setHasFrame] = useState(false);
  const [stale, setStale] = useState(false);
  const [stats, setStats] = useState<string | null>(null);

  // Attach the subscribed track to the element, and detach it when the track changes or the tile goes away.
  useEffect(() => {
    const element = videoRef.current;
    lastFrameAtRef.current = null;
    rateRef.current.reset();
    latencyRef.current = null;
    setHasFrame(false);
    setStale(false);
    setStats(null);
    if (element === null || track === null) return;
    track.attach(element);
    return () => {
      track.detach(element);
    };
  }, [track]);

  // Frame liveness and statistics: note every presented frame, and judge staleness and the stat line once a
  // second. The stat line is refreshed on the timer, not per frame, so a 60 fps track does not re-render 60
  // times a second.
  useEffect(() => {
    const element = videoRef.current;
    if (element === null || track === null) return;
    let stopped = false;
    let callbackId: number | null = null;
    const markFrame = (metadata?: PresentedFrameMetadata): void => {
      const now = performance.now();
      lastFrameAtRef.current = now;
      rateRef.current.observe(metadata?.presentationTime ?? now, metadata?.presentedFrames);
      latencyRef.current = endToEndLatencyMs(metadata);
      setHasFrame(true);
      setStale(false);
    };
    const onTimeUpdate = (): void => markFrame();
    const watchFrames = (): void => {
      if (stopped) return;
      callbackId = element.requestVideoFrameCallback((_now, metadata) => {
        markFrame(metadata);
        watchFrames();
      });
    };
    // requestVideoFrameCallback reports presented frames exactly; without it, playback progress is the fallback.
    if (typeof element.requestVideoFrameCallback === "function") watchFrames();
    else element.addEventListener("timeupdate", onTimeUpdate);
    const timer = setInterval(() => {
      const last = lastFrameAtRef.current;
      const fresh = last !== null && performance.now() - last <= FRAME_STALE_AFTER_MS;
      if (last !== null && !fresh) setStale(true);
      // A rate is shown only for a picture that is current, the same rule a CaptureServer tile follows.
      setStats(fresh ? formatCameraStats(rateRef.current.rateHz(), advertisedFps, latencyRef.current) : null);
    }, 1000);
    return () => {
      stopped = true;
      clearInterval(timer);
      if (callbackId !== null) element.cancelVideoFrameCallback(callbackId);
      element.removeEventListener("timeupdate", onTimeUpdate);
    };
  }, [track, advertisedFps]);

  const placeholder = placeholderFor(phase, track, hasFrame, stale);
  return (
    <figure className={`camera-tile${primary ? " camera-tile--primary" : ""}${stale ? " tone-warning" : ""}`}>
      {/* The element stays mounted while a track exists so attach/detach never races a re-created node. */}
      {track !== null && <video ref={videoRef} muted autoPlay playsInline aria-label={name} />}
      {placeholder !== null && <div className="camera-placeholder">{placeholder}</div>}
      <figcaption>
        {name}
        {role !== null && <span className="camera-role">{ROLE_LABELS[role] ?? role}</span>}
        {stats !== null && (
          <span className="camera-stats" aria-label={`${name} frame rate and latency`}>
            {stats}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
