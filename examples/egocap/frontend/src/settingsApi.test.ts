// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The settings fetch: the happy path, the backend's own error detail, and a body of the wrong shape. */
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchSettings,
  fetchTaskFile,
  fetchTaskFiles,
  importTaskFile,
  parseSettings,
  removeTaskFile,
  saveOperatorSettings,
  SettingsHttpError,
  SettingsParseError,
} from "./settingsApi";
import { SETTINGS_FIXTURE } from "./settingsTestDoubles";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("fetchSettings", () => {
  it("asks the backend once and returns the parsed body", async () => {
    const fn = stubFetch(200, SETTINGS_FIXTURE);
    await expect(fetchSettings(undefined, "tok")).resolves.toEqual(SETTINGS_FIXTURE);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith("/api/settings", { signal: undefined, headers: { "X-Control-Token": "tok" } });
  });

  it("surfaces the backend's detail when it is not ready", async () => {
    stubFetch(503, { detail: "settings are not resolved yet" });
    await expect(fetchSettings()).rejects.toThrow(new SettingsHttpError("settings are not resolved yet"));
  });

  it("falls back to the status code when the error body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gateway down", { status: 502 })));
    await expect(fetchSettings()).rejects.toThrow(new SettingsHttpError("HTTP 502"));
  });
});

describe("parseSettings", () => {
  it("rejects a non-object body and a body missing a group", () => {
    expect(() => parseSettings(null)).toThrow(SettingsParseError);
    expect(() => parseSettings("x")).toThrow(SettingsParseError);
    const { origins: _dropped, ...withoutOrigins } = SETTINGS_FIXTURE;
    expect(() => parseSettings(withoutOrigins)).toThrow(new SettingsParseError("settings.origins is missing"));
  });
});

describe("task catalog calls", () => {
  it("imports browser content with the control token, and reads the listing back", async () => {
    const fn = stubFetch(200, { files: ["Kitchen.csv", "Mid.csv"], imported: ["Mid.csv"], file: { name: "Mid.csv", format: "mid_train_csv", tasks: [] } });
    const result = await importTaskFile("tok", { name: "Mid.csv", content: "a,b" });
    expect(result.imported).toEqual(["Mid.csv"]);
    expect(result.file.name).toBe("Mid.csv");
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/task-files/import");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ control_token: "tok", name: "Mid.csv", content: "a,b" });
    await importTaskFile("tok", { name: "G.csv", content: "a,b" });
    expect(JSON.parse((fn.mock.calls[1] as unknown as [string, RequestInit])[1].body as string)).toEqual({ control_token: "tok", name: "G.csv", content: "a,b" });
  });

  it("removes an imported catalog with DELETE and the token in a header", async () => {
    const fn = stubFetch(200, { files: ["Kitchen.csv"], imported: [] });
    await expect(removeTaskFile("tok", "Mid.csv")).resolves.toEqual({ files: ["Kitchen.csv"], imported: [] });
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/task-files/Mid.csv");
    expect(init.method).toBe("DELETE");
    expect(init.headers).toEqual({ "X-Control-Token": "tok" });
    expect(init.body).toBeUndefined();
  });

  it("saves only the operator-setting fields it is given", async () => {
    const fn = stubFetch(200, {
      camera_type: "oak_d",
      camera_types: ["sensing_gmsl", "oak_d"],
      camera_roles: { ego: null, wrist_left: null, wrist_right: null },
      roles: ["ego", "wrist_left", "wrist_right"],
      task_file: "Kitchen.csv",
      preselected_task: null,
    });
    await saveOperatorSettings("tok", { task_file: "Kitchen.csv" });
    const [, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ control_token: "tok", task_file: "Kitchen.csv" });
  });

  it("rejects a malformed listing or catalog", async () => {
    stubFetch(200, { files: "nope" });
    await expect(fetchTaskFiles()).rejects.toThrow(SettingsParseError);
    stubFetch(200, { name: 3 });
    await expect(fetchTaskFile("Kitchen.csv")).rejects.toThrow(SettingsParseError);
  });
});
