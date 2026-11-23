// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The flow hook: loading on mount, on a changed key or token and on its timer, and what a failed load leaves. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { SetupStatus } from "./setupApi";
import { SETUP_HELD, SETUP_READY } from "./setupTestDoubles";
import { SETUP_REFRESH_MS, SETUP_RETRY_MS, useSetup } from "./useSetup";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

/** Let pending loads settle, then advance the fake clock. */
async function settle(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("useSetup", () => {
  it("loads on mount, again when the key changes, and again on its timer", async () => {
    const answers: Array<SetupStatus | Error> = [SETUP_HELD, SETUP_READY, new Error("offline"), SETUP_READY];
    const load = vi.fn(async () => {
      const next = answers.shift() as SetupStatus | Error;
      if (next instanceof Error) throw next;
      return next;
    });
    const { result, rerender } = renderHook(({ key }) => useSetup(key, { load }), { initialProps: { key: "a" } });
    expect(result.current).toBeNull();
    await settle();
    expect(result.current).toEqual(SETUP_HELD);
    rerender({ key: "b" });
    await settle();
    expect(result.current).toEqual(SETUP_READY);
    await settle(SETUP_REFRESH_MS);
    expect(result.current).toBeNull(); // a failed load shows no flow rather than a stale one
    // A failed read is tried again sooner than the timer, so one refusal is not ten seconds without a flow.
    await settle(SETUP_RETRY_MS);
    expect(result.current).toEqual(SETUP_READY);
    expect(load).toHaveBeenCalledTimes(4);
  });

  it("does not show one station's flow as another's while the new one is read", async () => {
    const pending: Array<(value: SetupStatus) => void> = [];
    const load = vi.fn(() => new Promise<SetupStatus>((resolve) => pending.push(resolve)));
    const { result, rerender } = renderHook(({ scope }) => useSetup("a", { scope, load }), { initialProps: { scope: "Bench A" } });
    pending[0](SETUP_HELD);
    await settle();
    expect(result.current).toEqual(SETUP_HELD);
    rerender({ scope: "Cart B" });
    expect(result.current).toBeNull();
    pending[1](SETUP_READY);
    await settle();
    expect(result.current).toEqual(SETUP_READY);
  });

  it("stops showing the headset address once the credential it was given for is gone", async () => {
    const withHeadset = { ...SETUP_READY, headset: { address: "192.0.2.10", saved: null, changed: false } };
    const pending: Array<(value: SetupStatus) => void> = [];
    const load = vi.fn(() => new Promise<SetupStatus>((resolve) => pending.push(resolve)));
    const { result, rerender } = renderHook(({ controlToken }) => useSetup("a", { controlToken, load }), { initialProps: { controlToken: "tok" as string | null } });
    pending[0](withHeadset);
    await settle();
    expect(result.current?.headset?.address).toBe("192.0.2.10");
    rerender({ controlToken: null });
    // The rest of the flow stays on screen while it is read again; only what the credential bought is withheld.
    expect(result.current).toEqual({ ...withHeadset, headset: null });
  });

  it("sends the control token with each read and reads again when it is entered", async () => {
    const load = vi.fn(async () => SETUP_READY);
    const { rerender } = renderHook(({ controlToken }) => useSetup("a", { controlToken, load }), { initialProps: { controlToken: null as string | null } });
    await settle();
    expect(load).toHaveBeenLastCalledWith(expect.any(AbortSignal), null);
    rerender({ controlToken: "tok" });
    await settle();
    expect(load).toHaveBeenCalledTimes(2);
    expect(load).toHaveBeenLastCalledWith(expect.any(AbortSignal), "tok");
  });

  it("reads on the interval it is given", async () => {
    const load = vi.fn(async () => SETUP_READY);
    renderHook(() => useSetup("a", { intervalMs: 2_000, load }));
    await settle();
    expect(load).toHaveBeenCalledTimes(1);
    await settle(1_999);
    expect(load).toHaveBeenCalledTimes(1);
    await settle(1);
    expect(load).toHaveBeenCalledTimes(2);
    await settle(2_000);
    expect(load).toHaveBeenCalledTimes(3);
  });

  it("waits for an answer slower than its interval instead of abandoning the read", async () => {
    const pending: Array<(value: SetupStatus) => void> = [];
    const signals: AbortSignal[] = [];
    const load = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return new Promise<SetupStatus>((resolve) => pending.push(resolve));
    });
    const { result } = renderHook(() => useSetup("a", { intervalMs: 2_000, load }));
    await settle(7_000); // several intervals pass while the backend is still answering
    expect(load).toHaveBeenCalledTimes(1);
    expect(signals[0].aborted).toBe(false);
    pending[0](SETUP_HELD);
    await settle();
    expect(result.current).toEqual(SETUP_HELD);
    // The next read is due one interval after the answer, not after the request.
    await settle(1_999);
    expect(load).toHaveBeenCalledTimes(1);
    await settle(1);
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("abandons its read and its timer when the page leaves", async () => {
    const signals: AbortSignal[] = [];
    const load = vi.fn(async (signal: AbortSignal) => {
      signals.push(signal);
      return SETUP_READY;
    });
    const { unmount } = renderHook(() => useSetup("a", { load }));
    await settle();
    unmount();
    expect(signals[0].aborted).toBe(true);
    await settle(SETUP_REFRESH_MS * 2);
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("discards an answer a newer request has overtaken", async () => {
    const pending: Array<(value: SetupStatus) => void> = [];
    const load = vi.fn(() => new Promise<SetupStatus>((resolve) => pending.push(resolve)));
    const { result, rerender } = renderHook(({ key }) => useSetup(key, { load }), { initialProps: { key: "a" } });
    rerender({ key: "b" });
    pending[1](SETUP_READY);
    await settle();
    pending[0](SETUP_HELD); // the first request answers late
    await settle();
    expect(result.current).toEqual(SETUP_READY);
  });
});
