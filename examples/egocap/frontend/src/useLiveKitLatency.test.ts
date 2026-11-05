// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useLiveKitLatency } from "./useLiveKitLatency";

const ENTRY = {
  basis: "SENSOR_CAPTURE_UNIX_US",
  summary: { median_ms: 50, p95_ms: 60, samples: 3 },
  note: "",
};
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.useFakeTimers();
  fetchMock = vi.fn(async () => ({
    ok: true,
    json: async () => ({ streams: { Head: ENTRY } }),
  }));
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useLiveKitLatency", () => {
  it("polls once per second while active and returns the streams", async () => {
    const { result } = renderHook(() => useLiveKitLatency(true));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/livekit-latency");
    expect(result.current).toEqual({ Head: ENTRY });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("clears the value and stops polling when it becomes inactive", async () => {
    const { result, rerender } = renderHook(
      ({ active }) => useLiveKitLatency(active),
      { initialProps: { active: true } },
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current).toEqual({ Head: ENTRY });
    rerender({ active: false });
    expect(result.current).toEqual({});
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not fetch while inactive", async () => {
    renderHook(() => useLiveKitLatency(false));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
