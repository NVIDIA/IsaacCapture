// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  ClockEstimator,
  decodeReply,
  encodeRequest,
  offsetNs,
  rttNs,
  toRemoteNs,
  wellFormed,
} from "./clockExchange";
import type { Exchange } from "./clockExchange";

const MS = 1_000_000n;

/** An exchange with the given one-way delays and remote-minus-local offset, starting at local time t1. */
function exchange(t1: bigint, offset: bigint, out: bigint, back: bigint, hold = 0n): Exchange {
  const t2 = t1 + out + offset;
  const t3 = t2 + hold;
  return { t1, t2, t3, t4: t3 - offset + back };
}

describe("exchange arithmetic", () => {
  it("recovers a symmetric offset exactly", () => {
    const e = exchange(1000n * MS, 3_000n * MS, 5n * MS, 5n * MS, MS);
    expect(offsetNs(e)).toBe(3_000n * MS);
    expect(rttNs(e)).toBe(10n * MS);
    expect(toRemoteNs({ epoch: 1, offsetNs: offsetNs(e), rttNs: 0n, uncertaintyNs: 0n, acceptedAtNs: 0n }, 7n)).toBe(
      3_000n * MS + 7n,
    );
  });

  it("errs by less than half the round trip when the path is asymmetric", () => {
    const e = exchange(0n, 500n * MS, 2n * MS, 8n * MS);
    const error = offsetNs(e) - 500n * MS;
    expect(error < 0n ? -error : error).toBeLessThanOrEqual(rttNs(e) / 2n);
  });

  it("floors negative sums like Python's //", () => {
    // t2-t1 + t3-t4 = -3 -> -2, and -4 -> -2, -1 -> -1
    expect(offsetNs({ t1: 3n, t2: 0n, t3: 0n, t4: 0n })).toBe(-2n);
    expect(offsetNs({ t1: 4n, t2: 0n, t3: 0n, t4: 0n })).toBe(-2n);
    expect(offsetNs({ t1: 1n, t2: 0n, t3: 0n, t4: 0n })).toBe(-1n);
    expect(offsetNs({ t1: 0n, t2: 3n, t3: 0n, t4: 0n })).toBe(1n);
  });

  it("flags reversed clocks and negative round trips as malformed", () => {
    expect(wellFormed({ t1: 1n, t2: 5n, t3: 6n, t4: 2n })).toBe(true);
    expect(wellFormed({ t1: 5n, t2: 5n, t3: 6n, t4: 4n })).toBe(false);
    expect(wellFormed({ t1: 1n, t2: 6n, t3: 5n, t4: 9n })).toBe(false);
    expect(wellFormed({ t1: 1n, t2: 5n, t3: 9n, t4: 2n })).toBe(false); // held longer than the round trip
  });
});

describe("ClockEstimator", () => {
  it("has no mapping before an exchange", () => {
    expect(new ClockEstimator().mapping(0n)).toBeNull();
  });

  it("keeps the lowest round trip, the newest on a tie", () => {
    const est = new ClockEstimator();
    est.add(exchange(0n, 100n, 5n * MS, 5n * MS));
    est.add(exchange(1000n * MS, 100n, 2n * MS, 2n * MS));
    est.add(exchange(2000n * MS, 100n, 9n * MS, 9n * MS));
    expect(est.mapping(2100n * MS)?.rttNs).toBe(4n * MS);
    est.add(exchange(3000n * MS, 100n, 2n * MS, 2n * MS));
    expect(est.mapping(3100n * MS)?.acceptedAtNs).toBe(3000n * MS + 4n * MS);
  });

  it("rejects malformed and over-long exchanges without changing the mapping", () => {
    const est = new ClockEstimator(0n, 10n * MS);
    expect(est.add({ t1: 5n, t2: 5n, t3: 6n, t4: 4n })).toBe("malformed exchange");
    expect(est.add(exchange(0n, 0n, 20n * MS, 20n * MS))).toBe("round trip too long");
    expect(est.mapping(0n)).toBeNull();
  });

  it("flags a step as a discontinuity and bumps the epoch on that same-RTT exchange", () => {
    const est = new ClockEstimator();
    est.add(exchange(0n, 0n, MS, MS));
    const before = est.mapping(10n * MS)!;
    expect(est.discontinuity).toBe(false);
    est.add(exchange(1000n * MS, 500n * MS, MS, MS));
    const after = est.mapping(1000n * MS + 2n * MS)!;
    expect(est.discontinuity).toBe(true);
    expect(after.epoch).toBe(before.epoch + 1);
    expect(after.offsetNs).toBe(500n * MS);
  });

  it("reset starts over with a new epoch", () => {
    const est = new ClockEstimator();
    est.add(exchange(0n, 0n, MS, MS));
    const epoch = est.mapping(0n)!.epoch;
    est.reset();
    expect(est.mapping(0n)).toBeNull();
    est.add(exchange(0n, 0n, MS, MS));
    expect(est.mapping(0n)!.epoch).toBe(epoch + 1);
  });

  it("goes stale", () => {
    const est = new ClockEstimator(0n, 250n * MS, 5_000n * MS);
    est.add(exchange(0n, 0n, MS, MS));
    expect(est.mapping(5_000n * MS)).not.toBeNull();
    expect(est.mapping(2n * MS + 5_000n * MS + 1n)).toBeNull();
  });

  it("keeps an early low-round-trip exchange only while its age costs less than the round-trip saving", () => {
    // Early exchange rtt 2.2 ms vs later rtt 10 ms: the saving of 3.9 ms is spent by 200 ppm after about 19.5 s.
    const est = new ClockEstimator(0n, 250n * MS, 5_000n * MS, 64);
    const best = exchange(1000n * MS, 100n, MS, 1_200_000n);
    est.add(best);
    for (let i = 1; i <= 30; i++) {
      const e = exchange(1000n * MS + BigInt(i) * 1000n * MS, 100n, 5n * MS, 5n * MS);
      est.add(e);
      const m = est.mapping(e.t4)!;
      expect(m).not.toBeNull();
      if (i <= 19) {
        expect(m.acceptedAtNs).toBe(best.t4);
        expect(m.rttNs).toBe(2_200_000n);
      } else {
        expect(m.acceptedAtNs).toBe(e.t4);
        expect(m.uncertaintyNs).toBe(5n * MS);
      }
    }
  });

  it("does not let a high-round-trip outlier replace a better exchange, and ages the uncertainty", () => {
    const est = new ClockEstimator();
    const good = exchange(1000n * MS, 10n * MS, MS, MS);
    est.add(good);
    const slow = exchange(2000n * MS, 10n * MS, 20n * MS, 20n * MS);
    est.add(slow);
    const m = est.mapping(slow.t4)!;
    expect(m.acceptedAtNs).toBe(good.t4);
    expect(m.rttNs).toBe(2n * MS);
    expect(m.uncertaintyNs).toBe(MS + ((slow.t4 - good.t4) * 200n) / 1_000_000n);
  });

  it("includes the age of the chosen exchange in the uncertainty", () => {
    const est = new ClockEstimator();
    const old = exchange(1000n * MS, 0n, MS, MS);
    est.add(old);
    const fresh = exchange(11_000n * MS, 0n, 5n * MS, 5n * MS);
    est.add(fresh);
    const m = est.mapping(fresh.t4)!;
    expect(m.acceptedAtNs).toBe(old.t4);
    expect(m.uncertaintyNs).toBe(MS + ((fresh.t4 - old.t4) * 200n) / 1_000_000n);
  });

  it("picks the newest exchange when uncertainties tie", () => {
    // Same round trip and (via a window of one older exchange) the same age cost cannot tie, so tie at age zero:
    // two exchanges completing at the same local time.
    const est = new ClockEstimator();
    const a: Exchange = { t1: 1000n * MS, t2: 1000n * MS + 2n * MS, t3: 1000n * MS + 2n * MS, t4: 1000n * MS + 4n * MS };
    const b: Exchange = { t1: 1002n * MS, t2: 1002n * MS + 1_000n + 2n * MS, t3: 1002n * MS + 1_000n + 2n * MS, t4: 1000n * MS + 4n * MS };
    est.add(a);
    est.add(b);
    expect(est.mapping(a.t4)!.offsetNs).toBe(offsetNs(b));
  });

  it("does not grow the uncertainty with age while fresh exchanges of the same round trip keep arriving", () => {
    const est = new ClockEstimator();
    for (let i = 1; i <= 30; i++) {
      const e = exchange(BigInt(i) * 1000n * MS, MS, 2n * MS, 2n * MS);
      est.add(e);
      const m = est.mapping(e.t4)!;
      expect(m.acceptedAtNs).toBe(e.t4);
      expect(m.uncertaintyNs).toBe(2n * MS);
    }
  });

  it("adds the timer resolution to the uncertainty", () => {
    const est = new ClockEstimator(100n * MS);
    est.add(exchange(0n, 0n, 5n * MS, 5n * MS));
    expect(est.mapping(0n)!.uncertaintyNs).toBe(5n * MS + 100n * MS);
  });
});

describe("clock-sync messages", () => {
  it("encodes a 13-byte request", () => {
    const bytes = encodeRequest(7, -5n);
    expect(bytes.byteLength).toBe(13);
    expect(bytes[0]).toBe(0x51);
    expect(new DataView(bytes.buffer).getUint32(1, true)).toBe(7);
    expect(new DataView(bytes.buffer).getBigInt64(5, true)).toBe(-5n);
  });

  it("round-trips a reply", () => {
    const bytes = new Uint8Array(29);
    const view = new DataView(bytes.buffer);
    bytes[0] = 0x52;
    view.setUint32(1, 9, true);
    view.setBigInt64(5, 1n, true);
    view.setBigInt64(13, -2n, true);
    view.setBigInt64(21, 3n, true);
    expect(decodeReply(bytes)).toEqual({ sequence: 9, t1: 1n, t2: -2n, t3: 3n });
  });

  it("rejects a wrong length or kind", () => {
    const good = new Uint8Array(29);
    good[0] = 0x52;
    expect(decodeReply(good)).not.toBeNull();
    expect(decodeReply(good.slice(0, 28))).toBeNull();
    expect(decodeReply(new Uint8Array(30))).toBeNull();
    const wrong = good.slice();
    wrong[0] = 0x51;
    expect(decodeReply(wrong)).toBeNull();
  });
});
