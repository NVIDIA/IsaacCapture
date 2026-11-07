// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Frame-rate and latency figures for a LiveKit video tile, from the browser's own frame presentation.
 *
 * Unlike the CaptureServer tiles (one sequence number per heartbeat, see cameraRates.ts), a LiveKit tile sees
 * every presented frame through `requestVideoFrameCallback`, so its rate is the presented-frame count over a
 * sliding window. Latency is only reported when the browser's WebRTC stack supplies the sender's capture time
 * for a frame (`VideoFrameCallbackMetadata.captureTime`); without it there is no honest end-to-end figure, and
 * the tile shows a dash rather than a receive-to-display delay dressed up as one.
 */

/** Sliding window for the average rate: long enough that the figure does not flicker frame by frame. */
export const RATE_WINDOW_MS = 4_000;

/** Minimum span of observations before a rate is reported; a few frames give no trustworthy rate. */
export const RATE_MIN_SPAN_MS = 1_500;

interface Observation {
  /** Presentation time of the frame, on the `performance.now()` clock. */
  atMs: number;
  /** The browser's running presented-frame count, or this tracker's own count when the browser gives none. */
  presented: number;
}

/** Average presented frames per second over the window. */
export class PresentedFrameRate {
  private samples: Observation[] = [];
  private counted = 0;

  /** Record a presented frame. `presentedFrames` is the browser's cumulative counter when available; it also
   * covers frames the callback itself skipped, which a plain callback count would miss. */
  observe(atMs: number, presentedFrames?: number): void {
    this.counted += 1;
    const presented = presentedFrames ?? this.counted;
    // A counter that went backwards means the element restarted its count; the old history would give a
    // nonsense (even negative) rate, so it is dropped.
    const last = this.samples[this.samples.length - 1];
    if (last !== undefined && presented < last.presented) this.samples = [];
    this.samples.push({ atMs, presented });
    // Keep the window plus one older sample so the span can still reach back a full window.
    const cutoff = atMs - RATE_WINDOW_MS;
    while (this.samples.length > 2 && this.samples[1].atMs <= cutoff) this.samples.shift();
  }

  /** Frames per second over the window, or null until there is enough span to say. */
  rateHz(): number | null {
    if (this.samples.length < 2) return null;
    const first = this.samples[0];
    const last = this.samples[this.samples.length - 1];
    const spanMs = last.atMs - first.atMs;
    if (spanMs < RATE_MIN_SPAN_MS) return null;
    return ((last.presented - first.presented) * 1000) / spanMs;
  }

  /** Forget everything: a new track starts a new history. */
  reset(): void {
    this.samples = [];
    this.counted = 0;
  }
}

/** The subset of `VideoFrameCallbackMetadata` used here; `captureTime` is optional in the spec and absent in
 * browsers or paths that do not map the sender clock. */
export interface PresentedFrameMetadata {
  presentationTime: number;
  presentedFrames?: number;
  captureTime?: number;
}

/** End-to-end latency of a presented frame in milliseconds, or null when the browser did not report the
 * sender's capture time for it (or reported one in the future, which no clock mapping should produce). */
export function endToEndLatencyMs(metadata: PresentedFrameMetadata | undefined): number | null {
  if (metadata === undefined) return null;
  const { captureTime, presentationTime } = metadata;
  if (typeof captureTime !== "number" || !Number.isFinite(captureTime)) return null;
  const latency = presentationTime - captureTime;
  return latency >= 0 ? latency : null;
}
