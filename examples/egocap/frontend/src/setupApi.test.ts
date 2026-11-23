// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The start-of-day read: what it sends, the backend's own reason on a refusal, and a malformed body. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { SettingsHttpError, SettingsParseError } from "./settingsApi";
import { fetchSetup, type SetupStatus } from "./setupApi";
import { SETUP_ADVISED, SETUP_HELD, SETUP_READY } from "./setupTestDoubles";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("the start-of-day flow", () => {
  it("is read without a credential", async () => {
    const fn = stubFetch(200, SETUP_HELD);
    await expect(fetchSetup()).resolves.toEqual(SETUP_HELD);
    expect(fn).toHaveBeenCalledWith("/api/setup", { signal: expect.any(AbortSignal), headers: {} });
  });

  it("is read with the control token once the page has it, and then says where the headset connects", async () => {
    const withHeadset: SetupStatus = { ...SETUP_READY, headset: { address: "10.0.0.42", saved: "10.0.0.7", changed: true } };
    const fn = stubFetch(200, withHeadset);
    const signal = new AbortController().signal;
    await expect(fetchSetup(signal, "tok")).resolves.toEqual(withHeadset);
    expect(fn).toHaveBeenCalledWith("/api/setup", { signal: expect.any(AbortSignal), headers: { "X-Control-Token": "tok" } });
  });

  it("surfaces the backend's reason, or the status when it gave none", async () => {
    stubFetch(503, { detail: "station setup is not available yet" });
    await expect(fetchSetup()).rejects.toThrow(new SettingsHttpError("station setup is not available yet"));
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gateway down", { status: 502 })));
    await expect(fetchSetup()).rejects.toThrow(new SettingsHttpError("HTTP 502"));
  });

  it("reads a check that only warns", async () => {
    stubFetch(200, SETUP_ADVISED);
    const setup = await fetchSetup();
    expect(setup.checks.find((check) => check.id === "calibration")?.status).toBe("warn");
    expect(setup.blocked_reason).toBeNull();
  });

  it("marks a check whose outcome it does not know as unknown, never as not needed, keeping the rest of the flow", async () => {
    stubFetch(200, { ...SETUP_READY, checks: [{ ...SETUP_READY.checks[0], status: "maybe" }, SETUP_READY.checks[1]] });
    const setup = await fetchSetup();
    expect(setup.checks.map((check) => check.status)).toEqual(["unknown", SETUP_READY.checks[1].status]);
    expect(setup.steps).toEqual(SETUP_READY.steps);
  });

  it("draws a step in a state it does not know as needing attention", async () => {
    stubFetch(200, { ...SETUP_READY, steps: [{ ...SETUP_READY.steps[0], status: "paused" }, ...SETUP_READY.steps.slice(1)] });
    const setup = await fetchSetup();
    expect(setup.steps.map((step) => step.status)).toEqual(["attention", ...SETUP_READY.steps.slice(1).map((step) => step.status)]);
  });

  it("gives up on a backend that never answers, so the page's timer is not left waiting", async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal(
        "fetch",
        vi.fn((_path: string, init: RequestInit) => new Promise((_resolve, reject) => init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))))),
      );
      const read = fetchSetup();
      const outcome = expect(read).rejects.toThrow("the backend did not answer in time");
      await vi.advanceTimersByTimeAsync(8_000);
      await outcome;
    } finally {
      vi.useRealTimers();
    }
  });

  it("rejects a body of the wrong shape", async () => {
    for (const body of [null, { ...SETUP_READY, steps: {} }, { ...SETUP_READY, checks: null }, { ...SETUP_READY, checks: [null] }, { ...SETUP_READY, steps: [null] }, { ...SETUP_READY, enforced: "yes" }]) {
      stubFetch(200, body);
      await expect(fetchSetup()).rejects.toThrow(SettingsParseError);
    }
  });
});
