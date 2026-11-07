/** Presented-frame rate over a window, and latency only from a reported capture time. */
import { describe, expect, it } from "vitest";
import { PresentedFrameRate, RATE_MIN_SPAN_MS, RATE_WINDOW_MS, endToEndLatencyMs } from "./liveKitFrameStats";

describe("PresentedFrameRate", () => {
  it("reports nothing until the observations span the minimum", () => {
    const rate = new PresentedFrameRate();
    rate.observe(0);
    expect(rate.rateHz()).toBeNull();
    rate.observe(RATE_MIN_SPAN_MS - 1);
    expect(rate.rateHz()).toBeNull();
    rate.observe(RATE_MIN_SPAN_MS);
    expect(rate.rateHz()).not.toBeNull();
  });

  it("averages its own frame count when the browser gives no counter", () => {
    const rate = new PresentedFrameRate();
    for (let i = 0; i <= 60; i++) rate.observe(i * 50); // 20 fps for 3 s
    expect(rate.rateHz()).toBeCloseTo(20, 5);
  });

  it("prefers the browser's presented-frame counter, which covers skipped callbacks", () => {
    const rate = new PresentedFrameRate();
    rate.observe(0, 100);
    rate.observe(2000, 160); // 60 frames in 2 s, though only two callbacks ran
    expect(rate.rateHz()).toBeCloseTo(30, 5);
  });

  it("drops observations older than the window so a change in rate shows within it", () => {
    const rate = new PresentedFrameRate();
    for (let i = 0; i <= 100; i++) rate.observe(i * 100); // 10 fps for 10 s
    expect(rate.rateHz()).toBeCloseTo(10, 5);
    const start = 10_000;
    for (let i = 1; i <= 200; i++) rate.observe(start + i * 25); // then 40 fps
    expect(rate.rateHz()).toBeCloseTo(40, 0);
    // Nothing older than the window survives beyond the one sample kept for span.
    rate.observe(start + 5000 + RATE_WINDOW_MS);
    expect(rate.rateHz()).not.toBeNull();
  });

  it("drops its history when the browser's counter goes backwards", () => {
    const rate = new PresentedFrameRate();
    rate.observe(0, 500);
    rate.observe(2000, 560);
    rate.observe(2033, 1); // the element restarted its count
    expect(rate.rateHz()).toBeNull();
    rate.observe(4033, 61);
    expect(rate.rateHz()).toBeCloseTo(30, 5);
  });

  it("starts over on reset", () => {
    const rate = new PresentedFrameRate();
    rate.observe(0);
    rate.observe(3000);
    rate.reset();
    expect(rate.rateHz()).toBeNull();
  });
});

describe("endToEndLatencyMs", () => {
  it("is the presentation time minus the sender's capture time", () => {
    expect(endToEndLatencyMs({ presentationTime: 1500, captureTime: 1380 })).toBe(120);
  });

  it("is null without a capture time, without metadata, or for a capture time in the future", () => {
    expect(endToEndLatencyMs(undefined)).toBeNull();
    expect(endToEndLatencyMs({ presentationTime: 1500 })).toBeNull();
    expect(endToEndLatencyMs({ presentationTime: 1500, captureTime: Number.NaN })).toBeNull();
    expect(endToEndLatencyMs({ presentationTime: 1500, captureTime: 1600 })).toBeNull();
  });
});
