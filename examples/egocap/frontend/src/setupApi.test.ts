// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The start-of-day calls: what each sends, the backend's own reason on a refusal, and a malformed body. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { SettingsHttpError, SettingsParseError } from "./settingsApi";
import { fetchSetup, runConnectionTest } from "./setupApi";
import { SETUP_NOT_TESTED, SETUP_PASSED } from "./setupTestDoubles";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("the start-of-day flow", () => {
  it("is read without a credential", async () => {
    const fn = stubFetch(200, SETUP_NOT_TESTED);
    await expect(fetchSetup()).resolves.toEqual(SETUP_NOT_TESTED);
    expect(fn).toHaveBeenCalledWith("/api/setup", { signal: undefined });
  });

  it("runs the connection test with the control token and returns the new flow", async () => {
    const fn = stubFetch(200, SETUP_PASSED);
    await expect(runConnectionTest("tok")).resolves.toEqual(SETUP_PASSED);
    expect(fn).toHaveBeenCalledWith("/api/setup/connection-test", { method: "POST", headers: { "X-Control-Token": "tok" } });
  });

  it("surfaces the backend's reason, or the status when it gave none", async () => {
    stubFetch(403, { detail: "control token rejected" });
    await expect(runConnectionTest("wrong")).rejects.toThrow(new SettingsHttpError("control token rejected"));
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gateway down", { status: 502 })));
    await expect(fetchSetup()).rejects.toThrow(new SettingsHttpError("HTTP 502"));
  });

  it("rejects a body of the wrong shape", async () => {
    for (const body of [null, { ...SETUP_PASSED, steps: {} }, { ...SETUP_PASSED, system: null }, { ...SETUP_PASSED, enforced: "yes" }]) {
      stubFetch(200, body);
      await expect(fetchSetup()).rejects.toThrow(SettingsParseError);
    }
  });
});
