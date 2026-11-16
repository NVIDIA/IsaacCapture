// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The flow hook: loading on mount, on a changed key and on its timer, and what a failed load leaves. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { SetupStatus } from "./setupApi";
import { SETUP_NOT_TESTED, SETUP_PASSED } from "./setupTestDoubles";
import { SETUP_REFRESH_MS, useSetup } from "./useSetup";

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
    const answers: Array<SetupStatus | Error> = [SETUP_NOT_TESTED, SETUP_PASSED, new Error("offline"), SETUP_PASSED];
    const load = vi.fn(async () => {
      const next = answers.shift() as SetupStatus | Error;
      if (next instanceof Error) throw next;
      return next;
    });
    const { result, rerender } = renderHook(({ key }) => useSetup(key, load), { initialProps: { key: "a" } });
    expect(result.current).toBeNull();
    await settle();
    expect(result.current).toEqual(SETUP_NOT_TESTED);
    rerender({ key: "b" });
    await settle();
    expect(result.current).toEqual(SETUP_PASSED);
    await settle(SETUP_REFRESH_MS);
    expect(result.current).toBeNull(); // a failed load shows no flow rather than a stale one
    await settle(SETUP_REFRESH_MS);
    expect(result.current).toEqual(SETUP_PASSED);
    expect(load).toHaveBeenCalledTimes(4);
  });

  it("discards an answer a newer request has overtaken", async () => {
    const pending: Array<(value: SetupStatus) => void> = [];
    const load = vi.fn(() => new Promise<SetupStatus>((resolve) => pending.push(resolve)));
    const { result, rerender } = renderHook(({ key }) => useSetup(key, load), { initialProps: { key: "a" } });
    rerender({ key: "b" });
    pending[1](SETUP_PASSED);
    await settle();
    pending[0](SETUP_NOT_TESTED); // the first request answers late
    await settle();
    expect(result.current).toEqual(SETUP_PASSED);
  });
});
