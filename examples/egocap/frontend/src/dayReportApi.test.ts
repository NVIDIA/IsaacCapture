// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The end-of-day calls: what each sends, the backend's own reason on a refusal, and a malformed body. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { correctSession, fetchDayReport, saveDayReport } from "./dayReportApi";
import { DAY_REPORT_FIXTURE } from "./dayReportTestDoubles";
import { SettingsHttpError, SettingsParseError } from "./settingsApi";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

const JSON_AUTH = { "content-type": "application/json", "X-Control-Token": "tok" };

describe("the day report", () => {
  it("is read with the control token and no choice of day or station", async () => {
    const fn = stubFetch(200, DAY_REPORT_FIXTURE);
    const signal = new AbortController().signal;
    await expect(fetchDayReport("tok", signal)).resolves.toEqual(DAY_REPORT_FIXTURE);
    expect(fn).toHaveBeenLastCalledWith("/api/day-report", { signal, headers: { "X-Control-Token": "tok" } });
  });

  it("is saved with an empty body and answers with the files written", async () => {
    const saved = { ...DAY_REPORT_FIXTURE, saved: { report: "bench_a_2026-03-09_report.json", activity_log: "bench_a_2026-03-09_activity.json" } };
    const fn = stubFetch(200, saved);
    await expect(saveDayReport("tok")).resolves.toEqual(saved);
    expect(fn).toHaveBeenCalledWith("/api/day-report/save", { method: "POST", headers: JSON_AUTH, body: "{}" });
  });

  it("sends a correction of only the session, the takes and the task, and returns what it changed", async () => {
    const reply = { session: "session_a", updated: ["episode_0"], missing: ["episode_1"], config_updated: true };
    const fn = stubFetch(200, reply);
    const correction = { session: "session_a", episodes: "0-1", task: "Place the cup on the shelf." };
    await expect(correctSession("tok", correction)).resolves.toEqual(reply);
    expect(fn).toHaveBeenCalledWith("/api/day-report/correct", { method: "POST", headers: JSON_AUTH, body: JSON.stringify(correction) });
  });

  it("surfaces the backend's reason, or the status when it gave none", async () => {
    stubFetch(400, { detail: "the session is outside the recording output" });
    await expect(correctSession("tok", { session: "../x", episodes: "0", task: "x" })).rejects.toThrow(new SettingsHttpError("the session is outside the recording output"));
    stubFetch(403, { detail: "control token rejected" });
    await expect(fetchDayReport("wrong")).rejects.toThrow(new SettingsHttpError("control token rejected"));
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gateway down", { status: 502 })));
    await expect(saveDayReport("tok")).rejects.toThrow(new SettingsHttpError("HTTP 502"));
  });

  it("rejects bodies of the wrong shape", async () => {
    const { uptime: _dropped, ...withoutUptime } = DAY_REPORT_FIXTURE;
    for (const body of [null, {}, { report: { sessions: {}, totals: {} } }, { report: { sessions: [], totals: null } }, withoutUptime, { ...DAY_REPORT_FIXTURE, uptime: { state: "down" } }]) {
      stubFetch(200, body);
      await expect(fetchDayReport("tok")).rejects.toThrow(SettingsParseError);
    }
    stubFetch(200, { updated: "episode_0", missing: [] });
    await expect(correctSession("tok", { session: "s", episodes: "0", task: "x" })).rejects.toThrow(SettingsParseError);
  });
});
