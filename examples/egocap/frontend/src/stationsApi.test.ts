// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The station calls: what each sends, the backend's own reason on a refusal, and a body of the wrong shape. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { SettingsHttpError, SettingsParseError } from "./settingsApi";
import { HEADSET_ADDRESS_FIXTURE, STATIONS_FIXTURE } from "./settingsTestDoubles";
import {
  addStation,
  changeStationActivity,
  deleteStation,
  fetchHeadsetAddress,
  fetchStationActivity,
  fetchStations,
  selectStation,
  updateStation,
  type StationActivity,
} from "./stationsApi";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

const ACTIVITY: StationActivity = {
  station: "Bench A",
  date: "2026-03-09",
  state: "active",
  since: 1_000,
  active_seconds: 60,
  inactive_seconds: 0,
  blocker_count: 0,
  notes: "",
  as_of: 1_060,
};

const JSON_AUTH = { "content-type": "application/json", "X-Control-Token": "tok" };

describe("the station registry", () => {
  it("is read without a credential, and with one when the page has it", async () => {
    const fn = stubFetch(200, STATIONS_FIXTURE);
    await expect(fetchStations()).resolves.toEqual(STATIONS_FIXTURE);
    expect(fn).toHaveBeenLastCalledWith("/api/stations", { signal: undefined, headers: {} });
    await fetchStations(undefined, "tok");
    expect(fn).toHaveBeenLastCalledWith("/api/stations", { signal: undefined, headers: { "X-Control-Token": "tok" } });
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

describe("the headset address", () => {
  it("is read with the control token", async () => {
    const fn = stubFetch(200, HEADSET_ADDRESS_FIXTURE);
    await expect(fetchHeadsetAddress("tok")).resolves.toEqual(HEADSET_ADDRESS_FIXTURE);
    expect(fn).toHaveBeenCalledWith("/api/headset-address", { signal: undefined, headers: { "X-Control-Token": "tok" } });
  });

  it("reports a refusal and a malformed body", async () => {
    stubFetch(403, { detail: "control token rejected" });
    await expect(fetchHeadsetAddress("wrong")).rejects.toThrow(new SettingsHttpError("control token rejected"));
    stubFetch(200, { candidates: "10.0.0.42" });
    await expect(fetchHeadsetAddress("tok")).rejects.toThrow(SettingsParseError);
  });
});

describe("station activity", () => {
  it("is read without a credential and changed with one", async () => {
    const fn = stubFetch(200, ACTIVITY);
    await expect(fetchStationActivity()).resolves.toEqual(ACTIVITY);
    expect(fn).toHaveBeenLastCalledWith("/api/station-activity", { signal: undefined });
    await expect(changeStationActivity("tok", { state: "inactive" })).resolves.toEqual(ACTIVITY);
    expect(fn).toHaveBeenLastCalledWith("/api/station-activity", { method: "POST", headers: JSON_AUTH, body: JSON.stringify({ state: "inactive" }) });
  });

  it("reports a refusal and a malformed body", async () => {
    stubFetch(400, { detail: "state must be one of active, inactive" });
    await expect(changeStationActivity("tok", { notes: "x" })).rejects.toThrow(new SettingsHttpError("state must be one of active, inactive"));
    stubFetch(503, { detail: "stations are not available yet" });
    await expect(fetchStationActivity()).rejects.toThrow(new SettingsHttpError("stations are not available yet"));
    for (const body of [null, { ...ACTIVITY, state: "paused" }, { ...ACTIVITY, active_seconds: "60" }, { ...ACTIVITY, station: null }]) {
      stubFetch(200, body);
      await expect(fetchStationActivity()).rejects.toThrow(SettingsParseError);
    }
  });
});
