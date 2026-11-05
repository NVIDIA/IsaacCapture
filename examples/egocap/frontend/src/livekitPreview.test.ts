// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchLiveKitPreview,
  LiveKitPreviewHttpError,
  LiveKitPreviewParseError,
  parseLiveKitPreview,
} from "./livekitPreview";

const GOOD = {
  url: "wss://lk",
  room: "r",
  publisher: "orin",
  token: "tok",
  tracks: [{ name: "Head left", track: "left" }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubFetch(status: number, body: unknown, jsonThrows = false): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async () => ({
    status,
    ok: status >= 200 && status < 300,
    json: async () => {
      if (jsonThrows) throw new Error("not json");
      return body;
    },
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("parseLiveKitPreview", () => {
  it("accepts a complete body", () => {
    expect(parseLiveKitPreview(GOOD)).toEqual(GOOD);
  });

  it.each([
    ["null body", null],
    ["string body", "x"],
    ["missing url", { ...GOOD, url: undefined }],
    ["empty room", { ...GOOD, room: "" }],
    ["non-string publisher", { ...GOOD, publisher: 3 }],
    ["missing token", { ...GOOD, token: undefined }],
    ["tracks not an array", { ...GOOD, tracks: "x" }],
    ["empty tracks", { ...GOOD, tracks: [] }],
    ["non-object track", { ...GOOD, tracks: ["x"] }],
    ["track missing name", { ...GOOD, tracks: [{ track: "a" }] }],
    ["track with empty track id", { ...GOOD, tracks: [{ name: "a", track: "" }] }],
  ])("rejects %s", (_label, body) => {
    expect(() => parseLiveKitPreview(body)).toThrow(LiveKitPreviewParseError);
  });
});

describe("fetchLiveKitPreview", () => {
  it("returns the parsed config on 200 and requests the preview route", async () => {
    const fn = stubFetch(200, GOOD);
    expect(await fetchLiveKitPreview()).toEqual(GOOD);
    expect(fn.mock.calls[0][0]).toBe("/api/livekit-preview");
  });

  it("returns null on 404 because the station has no LiveKit preview", async () => {
    stubFetch(404, { detail: "LiveKit preview is not configured" });
    expect(await fetchLiveKitPreview()).toBeNull();
  });

  it("throws an HTTP error carrying the backend detail on 503", async () => {
    stubFetch(503, { detail: "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set for the backend" });
    await expect(fetchLiveKitPreview()).rejects.toThrow(LiveKitPreviewHttpError);
    await expect(fetchLiveKitPreview()).rejects.toThrow("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set");
  });

  it("falls back to the status code when an error body is not JSON", async () => {
    stubFetch(500, null, true);
    await expect(fetchLiveKitPreview()).rejects.toThrow("HTTP 500");
  });

  it("throws a parse error when a 200 body is unusable", async () => {
    stubFetch(200, { url: "wss://lk" });
    await expect(fetchLiveKitPreview()).rejects.toThrow(LiveKitPreviewParseError);
  });
});
