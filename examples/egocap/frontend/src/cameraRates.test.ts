// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { CameraRateTracker, RATE_MIN_SPAN_MS, RATE_WINDOW_MS, formatCameraStats } from "./cameraRates";

describe("CameraRateTracker", () => {
  it("reports no rate until the observations span the minimum window", () => {
    const t = new CameraRateTracker();
    t.observe("b1:1", 0, 0n);
    expect(t.rateHz("b1:1")).toBeNull();
    t.observe("b1:1", RATE_MIN_SPAN_MS - 1, 60n);
    expect(t.rateHz("b1:1")).toBeNull();
    t.observe("b1:1", RATE_MIN_SPAN_MS, 90n);
    expect(t.rateHz("b1:1")).toBeCloseTo(60, 5);
  });

  it("averages the sequence delta over the sliding window", () => {
    const t = new CameraRateTracker();
    for (let i = 0; i <= 8; i++) t.observe("k", i * 1000, BigInt(i * 48));
    const hz = t.rateHz("k");
    expect(hz).not.toBeNull();
    expect(hz as number).toBeCloseTo(48, 5);
  });

  it("forgets observations older than the window", () => {
    const t = new CameraRateTracker();
    t.observe("k", 0, 0n);
    t.observe("k", 1000, 60n);
    // A long stall, then the stream resumes at 30 fps: the old 60 fps samples must age out.
    for (let i = 0; i <= 6; i++) t.observe("k", 10_000 + i * 1000, BigInt(60 + i * 30));
    expect(t.rateHz("k") as number).toBeCloseTo(30, 5);
    expect(RATE_WINDOW_MS).toBeGreaterThan(RATE_MIN_SPAN_MS);
  });

  it("restarts the history when the sequence goes backwards", () => {
    const t = new CameraRateTracker();
    t.observe("k", 0, 1000n);
    t.observe("k", 2000, 1120n);
    expect(t.rateHz("k") as number).toBeCloseTo(60, 5);
    t.observe("k", 3000, 5n);
    expect(t.rateHz("k")).toBeNull();
    t.observe("k", 5000, 65n);
    expect(t.rateHz("k") as number).toBeCloseTo(30, 5);
  });

  it("ignores a re-observation with the same receipt time", () => {
    const t = new CameraRateTracker();
    t.observe("k", 0, 0n);
    t.observe("k", 2000, 120n);
    t.observe("k", 2000, 120n);
    expect(t.rateHz("k") as number).toBeCloseTo(60, 5);
  });

  it("drops keys that are no longer present", () => {
    const t = new CameraRateTracker();
    t.observe("a", 0, 0n);
    t.observe("a", 2000, 120n);
    t.observe("b", 0, 0n);
    t.observe("b", 2000, 60n);
    t.retainOnly(["b"]);
    expect(t.rateHz("a")).toBeNull();
    expect(t.rateHz("b") as number).toBeCloseTo(30, 5);
  });
});

describe("formatCameraStats", () => {
  it("shows measured against advertised rate and the latency", () => {
    expect(formatCameraStats(47.94, 60, 21.4)).toBe("47.9 / 60 Hz · 21 ms");
  });
  it("uses dashes, not zeros, while the rate or latency is unknown", () => {
    expect(formatCameraStats(null, 60, null)).toBe("— / 60 Hz · —");
    expect(formatCameraStats(59.6, null, 18)).toBe("59.6 Hz · 18 ms");
  });
});
