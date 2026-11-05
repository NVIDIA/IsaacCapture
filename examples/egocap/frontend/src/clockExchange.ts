// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Maps a remote machine's monotonic clock onto the browser's from four-timestamp exchanges, the same pattern and
 * policy as the backend's `clock_exchange.py` (see its docstring): t1 and t4 are read on this side, t2 and t3 on the
 * remote, and the offset is ((t2 - t1) + (t3 - t4)) / 2 with half the round trip as its uncertainty. The browser's
 * own timer is coarser than the remote's, so its resolution is added to the uncertainty. All times are integer
 * nanoseconds (BigInt); the offset is remote minus local.
 */

export interface Exchange {
  t1: bigint;
  t2: bigint;
  t3: bigint;
  t4: bigint;
}

export interface ClockMapping {
  /** Changes when the estimator starts over or finds a discontinuity. */
  epoch: number;
  /** Remote minus local, in ns. */
  offsetNs: bigint;
  rttNs: bigint;
  /** Half the round trip plus this browser's timer resolution: the largest error unknown asymmetry can cause. */
  uncertaintyNs: bigint;
  /** Local time at which the exchange behind this mapping completed. */
  acceptedAtNs: bigint;
}

const DEFAULT_MAX_RTT_NS = 250_000_000n;
const DEFAULT_STALE_AFTER_NS = 5_000_000_000n;
const DEFAULT_WINDOW = 8;
/** Relative rate difference tolerated between two mappings, in parts per million. */
const DEFAULT_DRIFT_ALLOWANCE_PPM = 200n;

export function rttNs(e: Exchange): bigint {
  return e.t4 - e.t1 - (e.t3 - e.t2);
}

export function offsetNs(e: Exchange): bigint {
  // Floor division like the backend's // so both sides agree on negative offsets.
  const sum = e.t2 - e.t1 + (e.t3 - e.t4);
  return sum >= 0n ? sum / 2n : -((-sum + 1n) / 2n);
}

export function wellFormed(e: Exchange): boolean {
  return e.t4 >= e.t1 && e.t3 >= e.t2 && rttNs(e) >= 0n;
}

/** The remote-clock time of a local-clock reading. */
export function toRemoteNs(mapping: ClockMapping, localNs: bigint): bigint {
  return localNs + mapping.offsetNs;
}

export class ClockEstimator {
  private recent: Exchange[] = [];
  private epoch = 1;
  private current: ClockMapping | null = null;
  /** Local time of the newest accepted exchange: staleness is about replies still arriving. */
  private newestNs: bigint | null = null;
  /** Set, and left set, when a new exchange disagrees with the current mapping by more than it can explain. */
  discontinuity = false;

  constructor(
    private readonly timerResolutionNs: bigint = 0n,
    private readonly maxRttNs: bigint = DEFAULT_MAX_RTT_NS,
    private readonly staleAfterNs: bigint = DEFAULT_STALE_AFTER_NS,
    private readonly window: number = DEFAULT_WINDOW,
  ) {}

  /** Starts over for a restarted remote: the old mapping no longer applies. */
  reset(): void {
    this.recent = [];
    this.current = null;
    this.newestNs = null;
    this.epoch += 1;
  }

  /** Takes one exchange into account; returns why it was rejected, or null when it was accepted. */
  add(exchange: Exchange): string | null {
    if (!wellFormed(exchange)) return "malformed exchange";
    if (rttNs(exchange) > this.maxRttNs) return "round trip too long";
    const previous = this.current;
    this.newestNs = exchange.t4;
    const newest = this.mappingOf(exchange);
    let candidate: ClockMapping;
    if (previous !== null && this.disagrees(previous, newest)) {
      // The clocks moved: judged on the newest exchange, so a jump is seen at once.
      this.discontinuity = true;
      this.epoch += 1;
      this.recent = [exchange];
      candidate = this.mappingOf(exchange);
    } else {
      this.recent.push(exchange);
      if (this.recent.length > this.window) this.recent.shift();
      // Lowest round trip wins; the newest on a tie, so a steady link stays fresh.
      let best = this.recent[0];
      for (const e of this.recent) {
        if (rttNs(e) < rttNs(best) || (rttNs(e) === rttNs(best) && e.t4 > best.t4)) best = e;
      }
      candidate = this.mappingOf(best);
    }
    this.current = candidate;
    return null;
  }

  /** The current mapping, or null when there is none yet or the last exchange is stale at `nowNs`. */
  mapping(nowNs: bigint): ClockMapping | null {
    const current = this.current;
    if (current === null || this.newestNs === null || nowNs - this.newestNs > this.staleAfterNs) return null;
    return current;
  }

  private mappingOf(e: Exchange): ClockMapping {
    return {
      epoch: this.epoch,
      offsetNs: offsetNs(e),
      rttNs: rttNs(e),
      uncertaintyNs: rttNs(e) / 2n + this.timerResolutionNs,
      acceptedAtNs: e.t4,
    };
  }

  private disagrees(previous: ClockMapping, candidate: ClockMapping): boolean {
    const elapsed = candidate.acceptedAtNs > previous.acceptedAtNs ? candidate.acceptedAtNs - previous.acceptedAtNs : 0n;
    const allowed = previous.uncertaintyNs + candidate.uncertaintyNs + (elapsed * DEFAULT_DRIFT_ALLOWANCE_PPM) / 1_000_000n;
    const delta = candidate.offsetNs - previous.offsetNs;
    return (delta < 0n ? -delta : delta) > allowed;
  }
}

/** Little-endian messages shared with the publisher: request `Q`, sequence, t1; reply `R`, sequence, t1, t2, t3. */
export const CLOCK_SYNC_TOPIC = "isaaccapture.clock-sync";

export function encodeRequest(sequence: number, t1: bigint): Uint8Array {
  const bytes = new Uint8Array(13);
  const view = new DataView(bytes.buffer);
  bytes[0] = "Q".charCodeAt(0);
  view.setUint32(1, sequence >>> 0, true);
  view.setBigInt64(5, t1, true);
  return bytes;
}

export function decodeReply(data: Uint8Array): { sequence: number; t1: bigint; t2: bigint; t3: bigint } | null {
  if (data.byteLength !== 29 || data[0] !== "R".charCodeAt(0)) return null;
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  return {
    sequence: view.getUint32(1, true),
    t1: view.getBigInt64(5, true),
    t2: view.getBigInt64(13, true),
    t3: view.getBigInt64(21, true),
  };
}
