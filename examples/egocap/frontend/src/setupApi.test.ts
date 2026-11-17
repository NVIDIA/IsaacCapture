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
    expect(fn).toHaveBeenCalledWith("/api/setup", { signal: undefined, headers: {} });
  });

  it("is read with the control token once the page has it, and then says where the headset connects", async () => {
    const withHeadset: SetupStatus = { ...SETUP_READY, headset: { address: "10.0.0.42", saved: "10.0.0.7", changed: true } };
    const fn = stubFetch(200, withHeadset);
    const signal = new AbortController().signal;
    await expect(fetchSetup(signal, "tok")).resolves.toEqual(withHeadset);
    expect(fn).toHaveBeenCalledWith("/api/setup", { signal, headers: { "X-Control-Token": "tok" } });
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

  it("draws a check whose status it does not know as not judged, keeping the rest of the flow", async () => {
    stubFetch(200, { ...SETUP_READY, checks: [{ ...SETUP_READY.checks[0], status: "maybe" }, SETUP_READY.checks[1]] });
    const setup = await fetchSetup();
    expect(setup.checks.map((check) => check.status)).toEqual(["skip", SETUP_READY.checks[1].status]);
    expect(setup.steps).toEqual(SETUP_READY.steps);
  });

  it("rejects a body of the wrong shape", async () => {
    for (const body of [null, { ...SETUP_READY, steps: {} }, { ...SETUP_READY, checks: null }, { ...SETUP_READY, checks: [null] }, { ...SETUP_READY, enforced: "yes" }]) {
      stubFetch(200, body);
      await expect(fetchSetup()).rejects.toThrow(SettingsParseError);
    }
  });
});
