// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Stamp-to-display latency of one presented video frame: the publisher's stamp for exactly that frame (frame-ready
 * or sensor capture, whichever the stream's timestamp basis declares), matched by the frame's RTP timestamp,
 * against the frame's expected display time, both expressed on the
 * publisher's clock through the four-timestamp mapping. A frame with no matching stamp, or with no current
 * mapping, has no latency: nothing is ever matched to the nearest frame.
 */

import { toRemoteNs } from "./clockExchange";
import type { ClockMapping } from "./clockExchange";

export interface LatencySample {
  ms: number;
  /** The clock mapping epoch the sample was made under; a window mixing epochs is not summarized. */
  epoch: number;
  /** Half the exchange round trip plus the browser's timer resolution, in ms. */
  uncertaintyMs: number;
}

export interface FrameStamp {
  userTimestamp: bigint;
}

/**
 * @param expectedDisplayTime The frame's expected display time (ms, the page's performance.now() timeline).
 * @param stamp The publisher's stamp for this frame in microseconds on the publisher's clock, if it has one.
 */
export function latencySample(
  expectedDisplayTime: number,
  stamp: FrameStamp | undefined,
  mapping: ClockMapping | null,
): LatencySample | null {
  if (stamp === undefined || mapping === null || !Number.isFinite(expectedDisplayTime)) return null;
  // The browser SDK reports a frame that carries no capture time (only a frame id) as a stamp of 0, not as absent.
  if (stamp.userTimestamp <= 0n) return null;
  const displayRemote = toRemoteNs(mapping, BigInt(Math.round(expectedDisplayTime * 1e6)));
  const ms = Number(displayRemote - stamp.userTimestamp * 1000n) / 1e6;
  const uncertaintyMs = Number(mapping.uncertaintyNs) / 1e6;
  // Display cannot precede capture; a value below zero by more than the uncertainty is not a measurement.
  if (!Number.isFinite(ms) || ms < -uncertaintyMs) return null;
  return { ms, epoch: mapping.epoch, uncertaintyMs };
}

/** Mean and 95th percentile of the samples made under `epoch`, and the largest uncertainty among them. */
export function summarizeLatency(
  samples: LatencySample[],
  epoch: number,
): { mean: number; p95: number; uncertaintyMs: number } | null {
  const own = samples.filter((s) => s.epoch === epoch);
  if (own.length === 0) return null;
  const sorted = own.map((s) => s.ms).sort((a, b) => a - b);
  const at = (q: number): number => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  return { mean, p95: at(0.95), uncertaintyMs: Math.max(...own.map((s) => s.uncertaintyMs)) };
}
