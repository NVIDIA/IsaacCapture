/**
 * Client-side camera frame-rate estimation from preview snapshots.
 *
 * The backend sends a snapshot roughly once a second (its heartbeat) or when the capture
 * lifecycle changes; each snapshot carries, per camera, the sequence number of the newest
 * frame the backend has relayed and that frame's producer-side age. Frames themselves are
 * not individually announced to the page, so the only rate the page can measure honestly
 * is the average over a window of snapshots: sequence delta divided by elapsed receipt time.
 * A per-frame jitter figure (p95 interval) is NOT derivable from one sample per second, so
 * this module deliberately exposes only the averaged rate and never fabricates a jitter.
 */

/** Sliding window over which the average rate is taken. Long enough to smooth the ~1 s
 * heartbeat quantisation (a 1 s window would flicker by whole frames), short enough that a
 * stall shows within a few seconds. */
export const RATE_WINDOW_MS = 4_000;

/** Minimum span between the oldest and newest observation before a rate is reported; one
 * heartbeat alone gives no delta, and a sub-second span would be dominated by send jitter. */
export const RATE_MIN_SPAN_MS = 1_500;

interface Observation {
  /** performance.now() at receipt of the snapshot that carried this sequence. */
  atMonotonic: number;
  sequence: bigint;
}

/**
 * Tracks the newest-frame sequence of each camera stream over time and estimates its rate.
 *
 * Keys are chosen by the caller (backend instance + stream id), so a backend restart, which
 * resets sequence numbers, naturally starts a fresh history rather than producing a negative
 * delta against the old one.
 */
export class CameraRateTracker {
  private readonly history = new Map<string, Observation[]>();

  /** Record that `sequence` was the newest frame for `key` in a snapshot received at `atMonotonic`. */
  observe(key: string, atMonotonic: number, sequence: bigint): void {
    const samples = this.history.get(key) ?? [];
    const last = samples[samples.length - 1];
    // A repeated heartbeat with the same receipt time (re-render, not a new snapshot) adds
    // nothing; a sequence that went backwards means the producer restarted, so the old
    // history would yield a nonsense rate and is dropped.
    if (last !== undefined && last.atMonotonic === atMonotonic) return;
    if (last !== undefined && sequence < last.sequence) samples.length = 0;
    samples.push({ atMonotonic, sequence });
    // Keep only the window, plus one older sample so the span can still reach back a full window.
    const cutoff = atMonotonic - RATE_WINDOW_MS;
    while (samples.length > 2 && samples[1].atMonotonic <= cutoff) samples.shift();
    this.history.set(key, samples);
  }

  /** Average frames per second over the window, or null when there is not yet enough span. */
  rateHz(key: string): number | null {
    const samples = this.history.get(key);
    if (samples === undefined || samples.length < 2) return null;
    const first = samples[0];
    const last = samples[samples.length - 1];
    const spanMs = last.atMonotonic - first.atMonotonic;
    if (spanMs < RATE_MIN_SPAN_MS) return null;
    // Sequence numbers are uint64 on the wire; the delta over a few seconds is tiny, so the
    // Number conversion is exact.
    return (Number(last.sequence - first.sequence) * 1000) / spanMs;
  }

  /** Drop every key not in `keep` -- cameras that left the snapshot must not pin stale history. */
  retainOnly(keep: Iterable<string>): void {
    const wanted = new Set(keep);
    for (const key of [...this.history.keys()]) if (!wanted.has(key)) this.history.delete(key);
  }
}

/**
 * Operator-facing text for one camera's stat line: measured rate against the advertised one,
 * then the producer-side latency of the newest frame.
 *
 * `measuredHz` null (not enough history yet) renders as a dash rather than a zero, because
 * "0 Hz" would be a false alarm during the first seconds after connecting.
 */
export function formatCameraStats(measuredHz: number | null, advertisedFps: number | null, ageMs: number | null): string {
  const rate = measuredHz === null ? "—" : measuredHz.toFixed(1);
  const target = advertisedFps === null ? "" : ` / ${Number.isInteger(advertisedFps) ? advertisedFps : advertisedFps.toFixed(1)}`;
  const latency = ageMs === null ? "—" : `${Math.round(ageMs)} ms`;
  return `${rate}${target} Hz · ${latency}`;
}
