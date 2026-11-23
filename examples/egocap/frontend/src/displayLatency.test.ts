// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { ClockMapping } from "./clockExchange";
import { latencySample, summarizeLatency } from "./displayLatency";

const mapping = (over: Partial<ClockMapping> = {}): ClockMapping => ({
  epoch: 1,
  offsetNs: 3_000_000_000n,
  rttNs: 4_000_000n,
  uncertaintyNs: 2_000_000n,
  acceptedAtNs: 0n,
  ...over,
});

describe("latencySample", () => {
  it("is display time on the publisher's clock minus its stamp", () => {
    // Display at local 997.072 s is 1000.072 s remote; the stamp is 1000.000 s.
    const s = latencySample(997_072, { userTimestamp: 1_000_000_000n }, mapping());
    expect(s).toEqual({ ms: 72, epoch: 1, uncertaintyMs: 2 });
  });

  it("is null without a stamp, a mapping, or a finite display time", () => {
    expect(latencySample(997_072, undefined, mapping())).toBeNull();
    expect(latencySample(997_072, { userTimestamp: 1_000_000_000n }, null)).toBeNull();
    expect(latencySample(Number.NaN, { userTimestamp: 1_000_000_000n }, mapping())).toBeNull();
  });

  it("is null for a stamp of zero, which the browser SDK reports for a frame that carries no capture time", () => {
    // Without this a frame with only a frame id gives the publisher's whole monotonic uptime as its latency.
    expect(latencySample(997_072, { userTimestamp: 0n }, mapping())).toBeNull();
    expect(latencySample(997_072, { userTimestamp: -1n }, mapping())).toBeNull();
  });

  it("is null when display precedes capture by more than the uncertainty, but keeps a small negative", () => {
    expect(latencySample(996_990, { userTimestamp: 1_000_000_000n }, mapping())).toBeNull(); // -10 ms
    expect(latencySample(996_999, { userTimestamp: 1_000_000_000n }, mapping())?.ms).toBeCloseTo(-1, 6);
  });
});

describe("summarizeLatency", () => {
  const sample = (ms: number, epoch: number, uncertaintyMs = 1) => ({ ms, epoch, uncertaintyMs });

  it("uses only samples from the given epoch", () => {
    const out = summarizeLatency([sample(10, 1), sample(30, 2, 3), sample(50, 2, 5), sample(70, 2, 4)], 2);
    expect(out).toEqual({ mean: 50, p95: 70, uncertaintyMs: 5 });
  });

  it("is null when no sample has that epoch", () => {
    expect(summarizeLatency([sample(10, 1)], 2)).toBeNull();
    expect(summarizeLatency([], 1)).toBeNull();
  });
});
