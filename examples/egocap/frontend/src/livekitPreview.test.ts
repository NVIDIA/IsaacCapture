// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchLiveKitToken, LiveKitTokenHttpError, LiveKitTokenParseError } from "./livekitPreview";

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

describe("fetchLiveKitToken", () => {
  it("returns the token and requests the token route with the room URL-encoded", async () => {
    const fn = stubFetch(200, { token: "tok" });
    expect(await fetchLiveKitToken("room a/b")).toBe("tok");
    expect(fn.mock.calls[0][0]).toBe("/api/livekit-token?room=room%20a%2Fb");
  });

  it("passes the abort signal to fetch", async () => {
    const fn = stubFetch(200, { token: "tok" });
    const controller = new AbortController();
    await fetchLiveKitToken("r", controller.signal);
    expect((fn.mock.calls[0] as unknown[])[1]).toEqual({ signal: controller.signal });
  });

  it("throws an HTTP error carrying the backend detail on 503", async () => {
    stubFetch(503, { detail: "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set for the backend" });
    await expect(fetchLiveKitToken("r")).rejects.toThrow(LiveKitTokenHttpError);
    await expect(fetchLiveKitToken("r")).rejects.toThrow("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set");
  });

  it("treats 404 as an error too, since the room was offered", async () => {
    stubFetch(404, { detail: "room is not offered by CaptureServer" });
    await expect(fetchLiveKitToken("r")).rejects.toThrow("room is not offered");
  });

  it("falls back to the status code when an error body is not JSON", async () => {
    stubFetch(500, null, true);
    await expect(fetchLiveKitToken("r")).rejects.toThrow("HTTP 500");
  });

  it.each([
    ["missing token", {}],
    ["empty token", { token: "" }],
    ["non-string token", { token: 3 }],
  ])("throws a parse error for %s", async (_label, body) => {
    stubFetch(200, body);
    await expect(fetchLiveKitToken("r")).rejects.toThrow(LiveKitTokenParseError);
  });
});
