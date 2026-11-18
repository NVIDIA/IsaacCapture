// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The request bound: an answer in time, no answer, and a caller that abandons the request itself. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NO_ANSWER, withTimeout } from "./requestTimeout";
import { SettingsHttpError } from "./settingsApi";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

/** A request that only ever ends by being aborted. */
function silent(signal: AbortSignal): Promise<never> {
  return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))));
}

describe("withTimeout", () => {
  it("returns an answer that comes in time and leaves no timer behind", async () => {
    await expect(withTimeout(undefined, 1_000, async () => "ok")).resolves.toBe("ok");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("passes a failure of the request through unchanged", async () => {
    const failure = new SettingsHttpError("refused", 409);
    await expect(withTimeout(undefined, 1_000, () => Promise.reject(failure))).rejects.toBe(failure);
  });

  it("aborts a request that goes unanswered and says so", async () => {
    const outcome = expect(withTimeout(undefined, 1_000, silent)).rejects.toThrow(NO_ANSWER);
    await vi.advanceTimersByTimeAsync(1_000);
    await outcome;
  });

  it("lets the caller abandon the request, which stays an abort", async () => {
    const caller = new AbortController();
    const outcome = expect(withTimeout(caller.signal, 1_000, silent)).rejects.toMatchObject({ name: "AbortError" });
    caller.abort();
    await outcome;
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not start a request the caller had already abandoned without telling it so", async () => {
    const caller = new AbortController();
    caller.abort();
    const seen: boolean[] = [];
    await withTimeout(caller.signal, 1_000, async (signal) => seen.push(signal.aborted));
    expect(seen).toEqual([true]);
  });
});
