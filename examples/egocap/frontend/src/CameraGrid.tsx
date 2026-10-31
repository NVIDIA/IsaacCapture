/**
 * Selected-camera MJPEG tiles. Extracted from App.tsx into its own module;
 * App mounts this once as the one generic camera display -- the Ego Capture UI
 * recording panel (RecordingPanel.tsx) does not render its own copy.
 */

import { useEffect, useRef, useState } from "react";
import type { PreviewSnapshot } from "./captureProtocol";
import { TRACKING_STALE_AFTER_MS } from "./CaptureTree";
import { CameraRateTracker, formatCameraStats } from "./cameraRates";

const CAMERA_ERROR_RETRY_LIMIT = 3;

/** One selected camera's MJPEG preview tile, owning its own bounded
 * automatic-retry budget and manual retry action independently of every
 * other tile. */
function CameraTile({
  camera,
  live,
  stale,
  stats,
}: {
  camera: PreviewSnapshot["cameras"][number];
  live: boolean;
  /** Operator-facing rate/latency line, or null while nothing trustworthy can be shown
   * (no fresh sample): the tile then shows the name alone rather than a misleading figure. */
  stats: string | null;
  /** True only for a PREVIOUSLY received sample that's now too old or
   * disconnected -- never true for "no sample yet", which is a distinct,
   * unalarming initial state (see `neverHadSample` below). */
  stale: boolean;
}) {
  // currentRequestFailed/retryAttempts track the CURRENT image request's
  // own outcome, not a running total -- an automatic retry's own success
  // must clear the failed flag and reset the budget, and the manual Retry
  // button must reflect only a request that is actually currently failed
  // with its budget exhausted, not a stale count left over from a retry
  // whose outcome hasn't been observed yet.
  const [currentRequestFailed, setCurrentRequestFailed] = useState(false);
  const [retryAttempts, setRetryAttempts] = useState(0);
  const [retryOrdinal, setRetryOrdinal] = useState(0);
  const pendingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function clearPendingTimer(): void {
    if (pendingTimerRef.current !== null) {
      clearTimeout(pendingTimerRef.current);
      pendingTimerRef.current = null;
    }
  }

  useEffect(() => {
    // A change of identity/source/health invalidates any request this
    // tile was tracking -- cancel its timer and drop its failure state so
    // a stale automatic-retry budget never carries over to a new camera
    // association.
    return () => {
      clearPendingTimer();
      setCurrentRequestFailed(false);
      setRetryAttempts(0);
    };
  }, [camera.url, live, stale]);

  const neverHadSample = camera.sequence === null;
  // Only mounted for the current connected/running association with a
  // sample that has actually arrived and isn't stale.
  const hasFreshImage = live && !stale && !neverHadSample;
  const src = hasFreshImage ? `${camera.url}&r=${retryOrdinal}` : undefined;

  function handleLoad(): void {
    clearPendingTimer();
    setCurrentRequestFailed(false);
    setRetryAttempts(0);
  }

  function handleError(): void {
    setCurrentRequestFailed(true);
    if (retryAttempts >= CAMERA_ERROR_RETRY_LIMIT) return;
    // A second error for the same still-pending retry must not schedule a
    // competing timer -- one failure, at most one scheduled retry.
    if (pendingTimerRef.current !== null) return;
    pendingTimerRef.current = setTimeout(() => {
      pendingTimerRef.current = null;
      // Starting a new request: its own onLoad/onError establishes the
      // real outcome, so the PREVIOUS request's failure must not still be
      // reflected while this one is in flight -- that was exactly the bug
      // where a final automatic retry that succeeded still showed "Retry
      // image" because the stale failed flag combined with the just-
      // incremented attempt count.
      setCurrentRequestFailed(false);
      setRetryAttempts((n) => n + 1);
      setRetryOrdinal((n) => n + 1);
    }, 1000);
  }

  function manualRetry(): void {
    clearPendingTimer();
    setCurrentRequestFailed(false);
    setRetryAttempts(0);
    setRetryOrdinal((n) => n + 1);
  }

  const placeholderText = !live
    ? "Disconnected"
    : camera.diagnostic !== null
      ? camera.diagnostic
      : neverHadSample
        ? "Waiting for image"
        : stale
          ? "Preview stale"
          : "Waiting for image";
  const budgetExhausted = currentRequestFailed && retryAttempts >= CAMERA_ERROR_RETRY_LIMIT;

  return (
    <figure className={`camera-tile ${stale ? "tone-warning" : ""}`}>
      {src !== undefined ? (
        <img key={camera.url} src={src} alt={camera.name} onLoad={handleLoad} onError={handleError} />
      ) : (
        <div className="camera-placeholder">{placeholderText}</div>
      )}
      {budgetExhausted && hasFreshImage && (
        <button type="button" className="camera-retry-button" onClick={manualRetry}>
          Retry image
        </button>
      )}
      <figcaption>
        {camera.name}
        {stats !== null && (
          <span className="camera-stats" aria-label={`${camera.name} frame rate and latency`}>
            {stats}
          </span>
        )}
      </figcaption>
    </figure>
  );
}

/** The responsive grid of selected cameras' preview tiles, or a quiet
 * empty note when none are selected. */
export function CameraGrid({
  snapshot,
  live,
  receivedAtMonotonic,
}: {
  snapshot: PreviewSnapshot | null;
  live: boolean;
  receivedAtMonotonic: number | null;
}) {
  const [, forceTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => forceTick((n) => n + 1), 1_000);
    return () => clearInterval(id);
  }, []);

  // One tracker for the component's lifetime: a rate is a property of a history of snapshots,
  // so it must survive re-renders and only be fed when a genuinely new snapshot arrived (the
  // receipt time changes). Keys include the backend instance so a backend restart's sequence
  // reset starts a fresh history instead of a negative delta.
  const rateTracker = useRef(new CameraRateTracker());
  // Fed during render, not in an effect: an effect runs after paint, which would leave the
  // readout one heartbeat behind the tile it describes. This is safe because `observe` ignores
  // a repeated receipt time, so re-renders between snapshots change nothing.
  if (snapshot !== null && receivedAtMonotonic !== null) {
    const keys: string[] = [];
    for (const cam of snapshot.cameras) {
      const key = `${snapshot.backend_instance_id}:${cam.stream_id}`;
      keys.push(key);
      if (cam.sequence !== null) rateTracker.current.observe(key, receivedAtMonotonic, BigInt(cam.sequence));
    }
    rateTracker.current.retainOnly(keys);
  }

  const cameras = snapshot?.cameras ?? [];
  if (cameras.length === 0) {
    return (
      <section className="camera-grid">
        <h2 className="group-label">Cameras</h2>
        <p className="empty-note">No cameras selected.</p>
      </section>
    );
  }
  const elapsedSinceReceipt = receivedAtMonotonic === null ? 0 : performance.now() - receivedAtMonotonic;
  return (
    <section className="camera-grid">
      <h2 className="group-label">Cameras</h2>
      <div className="camera-grid-items">
        {cameras.map((cam) => {
          const effectiveAge = cam.age_ms === null ? null : cam.age_ms + elapsedSinceReceipt;
          // "Never had a sample" (sequence/age_ms both null) is a distinct,
          // unalarming initial state -- it must not count as stale, or
          // "Waiting for image" would be unreachable and every camera
          // would briefly flash "Preview stale" before its first frame.
          const neverHadSample = cam.sequence === null;
          const stale = !neverHadSample && (!live || effectiveAge === null || effectiveAge > TRACKING_STALE_AFTER_MS);
          // The advertised capture rate comes from the accepted session's stream descriptor;
          // showing "measured / advertised" lets an operator see a shortfall at a glance.
          const descriptor = snapshot?.started?.streams.find((s) => s.kind === "camera" && s.stream_id === cam.stream_id);
          const advertisedFps = descriptor !== undefined && descriptor.kind === "camera" ? descriptor.fps : null;
          // Stats are shown only for a live, fresh sample: a stale or disconnected tile must not
          // display a rate that looks current (same rule the tile's image follows).
          const stats =
            live && !stale && !neverHadSample
              ? formatCameraStats(
                  rateTracker.current.rateHz(`${snapshot?.backend_instance_id}:${cam.stream_id}`),
                  advertisedFps,
                  cam.age_ms,
                )
              : null;
          return (
            <CameraTile
              key={`${snapshot?.backend_instance_id}-${cam.name}`}
              camera={cam}
              live={live}
              stale={stale}
              stats={stats}
            />
          );
        })}
      </div>
    </section>
  );
}
