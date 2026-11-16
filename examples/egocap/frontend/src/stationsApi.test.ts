// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The station and uptime calls: what each sends, the backend's own reason on a refusal, and a body of the wrong shape. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { SettingsHttpError, SettingsParseError } from "./settingsApi";
import { UPTIME_FIXTURE } from "./dayReportTestDoubles";
import { STATIONS_FIXTURE } from "./settingsTestDoubles";
import { addStation, deleteStation, fetchStations, fetchStationUptime, saveStationNotes, selectStation, updateStation } from "./stationsApi";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

const JSON_AUTH = { "content-type": "application/json", "X-Control-Token": "tok" };

describe("the station registry", () => {
  it("is read without a credential", async () => {
    const fn = stubFetch(200, STATIONS_FIXTURE);
    await expect(fetchStations()).resolves.toEqual(STATIONS_FIXTURE);
    expect(fn).toHaveBeenLastCalledWith("/api/stations", { signal: undefined });
  });

  it("sends every change with the control token and returns the new listing", async () => {
    const fn = stubFetch(200, STATIONS_FIXTURE);
    await addStation("tok", { name: "Bench A" });
    expect(fn).toHaveBeenLastCalledWith("/api/stations", { method: "POST", headers: JSON_AUTH, body: JSON.stringify({ name: "Bench A" }) });
    await updateStation("tok", "st 1", { camera_roles: { ego: "cam_head" } });
    expect(fn).toHaveBeenLastCalledWith("/api/stations/st%201", { method: "PUT", headers: JSON_AUTH, body: JSON.stringify({ camera_roles: { ego: "cam_head" } }) });
    await deleteStation("tok", "st_1");
    expect(fn).toHaveBeenLastCalledWith("/api/stations/st_1", { method: "DELETE", headers: { "X-Control-Token": "tok" } });
    await expect(selectStation("tok", null)).resolves.toEqual(STATIONS_FIXTURE);
    expect(fn).toHaveBeenLastCalledWith("/api/stations/select", { method: "POST", headers: JSON_AUTH, body: JSON.stringify({ id: null }) });
  });

  it("surfaces the backend's reason for a refusal, or the status when it gave none", async () => {
    stubFetch(400, { detail: "a station named 'Bench A' already exists" });
    await expect(addStation("tok", { name: "bench a" })).rejects.toThrow(new SettingsHttpError("a station named 'Bench A' already exists"));
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gateway down", { status: 502 })));
    await expect(fetchStations()).rejects.toThrow(new SettingsHttpError("HTTP 502"));
  });

  it("rejects a listing of the wrong shape", async () => {
    for (const body of [null, { stations: {} }, { ...STATIONS_FIXTURE, label: 7 }, { ...STATIONS_FIXTURE, roles: "ego" }]) {
      stubFetch(200, body);
      await expect(fetchStations()).rejects.toThrow(SettingsParseError);
    }
  });
});

describe("station uptime", () => {
  it("is read without a credential", async () => {
    const fn = stubFetch(200, UPTIME_FIXTURE);
    await expect(fetchStationUptime()).resolves.toEqual(UPTIME_FIXTURE);
    expect(fn).toHaveBeenLastCalledWith("/api/station-uptime", { signal: undefined });
  });

  it("saves the day's notes with the control token and sets nothing else", async () => {
    const saved = { ...UPTIME_FIXTURE, notes: "left glove battery; replaced at 14:10" };
    const fn = stubFetch(200, saved);
    await expect(saveStationNotes("tok", "left glove battery; replaced at 14:10")).resolves.toEqual(saved);
    expect(fn).toHaveBeenLastCalledWith("/api/station-uptime", { method: "POST", headers: JSON_AUTH, body: JSON.stringify({ notes: "left glove battery; replaced at 14:10" }) });
  });

  it("reports a refusal and a malformed body", async () => {
    stubFetch(400, { detail: "notes may be at most 4000 characters" });
    await expect(saveStationNotes("tok", "x")).rejects.toThrow(new SettingsHttpError("notes may be at most 4000 characters"));
    stubFetch(503, { detail: "stations are not available yet" });
    await expect(fetchStationUptime()).rejects.toThrow(new SettingsHttpError("stations are not available yet"));
    for (const body of [null, { ...UPTIME_FIXTURE, state: "active" }, { ...UPTIME_FIXTURE, down_seconds: "60" }, { ...UPTIME_FIXTURE, blockers: 2 }, { ...UPTIME_FIXTURE, notes: null }]) {
      stubFetch(200, body);
      await expect(fetchStationUptime()).rejects.toThrow(SettingsParseError);
    }
  });
});
