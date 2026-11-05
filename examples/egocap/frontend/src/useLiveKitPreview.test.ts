// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { CONFIG, FakeRoom, fakeTrack, participant, publication } from "./livekitTestDoubles";

vi.mock("livekit-client", async () => {
  const doubles = await import("./livekitTestDoubles");
  return {
    Room: doubles.FakeRoom,
    RoomEvent: {
      TrackSubscribed: "TrackSubscribed",
      TrackUnsubscribed: "TrackUnsubscribed",
      ParticipantDisconnected: "ParticipantDisconnected",
      Reconnecting: "Reconnecting",
      Reconnected: "Reconnected",
      Disconnected: "Disconnected",
    },
    Track: { Kind: { Video: "video", Audio: "audio" } },
  };
});

import { useLiveKitPreview } from "./useLiveKitPreview";

const RETRY_MS = 3000;

function stubFetch(handler: () => Promise<unknown>): ReturnType<typeof vi.fn> {
  const fn = vi.fn(handler);
  vi.stubGlobal("fetch", fn);
  return fn;
}

const ok = (body: unknown) => async () => ({ status: 200, ok: true, json: async () => body });
const notFound = async () => ({ status: 404, ok: false, json: async () => ({ detail: "nf" }) });

async function flush(): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

async function mounted() {
  const view = renderHook(() => useLiveKitPreview());
  await flush();
  return view;
}

beforeEach(() => {
  FakeRoom.reset();
  vi.useFakeTimers();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useLiveKitPreview", () => {
  it("is unconfigured and creates no Room when the backend answers 404", async () => {
    stubFetch(notFound);
    const { result } = await mounted();
    expect(result.current.phase).toBe("unconfigured");
    expect(result.current.tiles).toEqual([]);
    expect(FakeRoom.instances).toHaveLength(0);
  });

  it("connects with the fetched url and token and lists configured tiles empty", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    expect(FakeRoom.instances[0].connect).toHaveBeenCalledWith("wss://lk", "tok");
    expect(result.current.phase).toBe("connected");
    expect(result.current.tiles).toEqual([
      { name: "Head left", track: null },
      { name: "Head right", track: null },
    ]);
  });

  it("is connecting while the connect is still pending", async () => {
    stubFetch(ok(CONFIG));
    let release!: () => void;
    FakeRoom.connectGate = new Promise<void>((r) => (release = r));
    const { result } = await mounted();
    expect(result.current.phase).toBe("connecting");
    release();
    await flush();
    expect(result.current.phase).toBe("connected");
  });

  it("adopts already-published video tracks of the publisher after connect", async () => {
    stubFetch(ok(CONFIG));
    const left = fakeTrack();
    FakeRoom.seed = [
      [
        "orin",
        participant("orin", [
          { name: "left", track: left },
          { name: "right", track: fakeTrack(), kind: "audio" },
          { name: "right", track: null },
          { name: "unconfigured", track: fakeTrack() },
        ]),
      ],
    ];
    const { result } = await mounted();
    expect(result.current.phase).toBe("connected");
    expect(result.current.tiles[0].track).toBe(left);
    expect(result.current.tiles[1].track).toBeNull();
  });

  it("does not adopt same-named tracks from another participant after connect", async () => {
    stubFetch(ok(CONFIG));
    FakeRoom.seed = [["intruder", participant("intruder", [{ name: "left", track: fakeTrack() }])]];
    const { result } = await mounted();
    expect(result.current.tiles[0].track).toBeNull();
  });

  it("picks up tracks published while the SDK was reconnecting", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    const room = FakeRoom.instances[0];
    const left = fakeTrack();
    room.remoteParticipants.set("orin", participant("orin", [{ name: "left", track: left }]));
    act(() => room.emit("Reconnected"));
    expect(result.current.tiles[0].track).toBe(left);
  });

  it("shows a track on TrackSubscribed for the configured publisher and track name", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    const track = fakeTrack();
    act(() => FakeRoom.instances[0].emit("TrackSubscribed", track, publication("right"), { identity: "orin" }));
    expect(result.current.tiles[1].track).toBe(track);
    expect(result.current.tiles[0].track).toBeNull();
  });

  it("ignores a correctly named track from a different participant", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    act(() => FakeRoom.instances[0].emit("TrackSubscribed", fakeTrack(), publication("left"), { identity: "intruder" }));
    expect(result.current.tiles[0].track).toBeNull();
  });

  it("ignores a subscribed audio track with a configured name", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    const audio = { ...fakeTrack(), kind: "audio" };
    act(() => FakeRoom.instances[0].emit("TrackSubscribed", audio, publication("left"), { identity: "orin" }));
    expect(result.current.tiles[0].track).toBeNull();
  });

  it("clears a tile on TrackUnsubscribed", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    const track = fakeTrack();
    act(() => FakeRoom.instances[0].emit("TrackSubscribed", track, publication("left"), { identity: "orin" }));
    act(() => FakeRoom.instances[0].emit("TrackUnsubscribed", track, publication("left"), { identity: "orin" }));
    expect(result.current.tiles[0].track).toBeNull();
  });

  it("keeps a republished track when the old one is unsubscribed afterwards", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    const oldTrack = fakeTrack();
    const newTrack = fakeTrack();
    const room = FakeRoom.instances[0];
    act(() => room.emit("TrackSubscribed", oldTrack, publication("left"), { identity: "orin" }));
    act(() => room.emit("TrackSubscribed", newTrack, publication("left"), { identity: "orin" }));
    act(() => room.emit("TrackUnsubscribed", oldTrack, publication("left"), { identity: "orin" }));
    expect(result.current.tiles[0].track).toBe(newTrack);
  });

  it("clears every tile when the publisher disconnects but not for another participant", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    const room = FakeRoom.instances[0];
    act(() => room.emit("TrackSubscribed", fakeTrack(), publication("left"), { identity: "orin" }));
    act(() => room.emit("TrackSubscribed", fakeTrack(), publication("right"), { identity: "orin" }));
    act(() => room.emit("ParticipantDisconnected", { identity: "someone-else" }));
    expect(result.current.tiles[0].track).not.toBeNull();
    act(() => room.emit("ParticipantDisconnected", { identity: "orin" }));
    expect(result.current.tiles.map((t) => t.track)).toEqual([null, null]);
    expect(result.current.phase).toBe("connected");
  });

  it("reports reconnecting then connected, keeping tracks across the SDK reconnect", async () => {
    stubFetch(ok(CONFIG));
    const { result } = await mounted();
    const room = FakeRoom.instances[0];
    const track = fakeTrack();
    act(() => room.emit("TrackSubscribed", track, publication("left"), { identity: "orin" }));
    act(() => room.emit("Reconnecting"));
    expect(result.current.phase).toBe("reconnecting");
    act(() => room.emit("Reconnected"));
    expect(result.current.phase).toBe("connected");
    expect(result.current.tiles[0].track).toBe(track);
    expect(FakeRoom.instances).toHaveLength(1);
  });

  it("registers each listener once per Room", async () => {
    stubFetch(ok(CONFIG));
    await mounted();
    const room = FakeRoom.instances[0];
    for (const event of ["TrackSubscribed", "TrackUnsubscribed", "ParticipantDisconnected", "Reconnecting", "Reconnected", "Disconnected"]) {
      expect(room.listenerCount(event)).toBe(1);
    }
  });

  it("on full Disconnected shows an error, then refetches a fresh token and builds a new Room after the delay", async () => {
    let n = 0;
    const fetchFn = stubFetch(async () => ({ status: 200, ok: true, json: async () => ({ ...CONFIG, token: `tok${++n}` }) }));
    const { result } = await mounted();
    act(() => FakeRoom.instances[0].emit("Disconnected"));
    expect(result.current.phase).toBe("error");
    expect(result.current.message).toBe("Disconnected from the LiveKit room");
    expect(result.current.tiles.map((t) => t.track)).toEqual([null, null]);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RETRY_MS - 1);
    });
    expect(fetchFn).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(FakeRoom.instances).toHaveLength(2);
    expect(FakeRoom.instances[1].connect).toHaveBeenCalledWith("wss://lk", "tok2");
    expect(result.current.phase).toBe("connected");
  });

  it("retries after a failed connect with the SDK's message", async () => {
    stubFetch(ok(CONFIG));
    FakeRoom.connectError = new Error("ws refused");
    const { result } = await mounted();
    expect(result.current.phase).toBe("error");
    expect(result.current.message).toBe("ws refused");
    FakeRoom.connectError = null;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RETRY_MS);
    });
    expect(FakeRoom.instances).toHaveLength(2);
    expect(result.current.phase).toBe("connected");
  });

  it("reports the backend's detail on HTTP 503 and keeps retrying", async () => {
    const fetchFn = stubFetch(async () => ({ status: 503, ok: false, json: async () => ({ detail: "creds missing" }) }));
    const { result } = await mounted();
    expect(result.current.phase).toBe("error");
    expect(result.current.message).toBe("creds missing");
    expect(FakeRoom.instances).toHaveLength(0);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RETRY_MS);
    });
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("reports a parse error for an unusable body", async () => {
    stubFetch(ok({ url: "wss://lk" }));
    const { result } = await mounted();
    expect(result.current.phase).toBe("error");
    expect(result.current.message).toContain("must be");
  });

  it("reports 'Cannot reach the backend' on a network failure and retries after the delay", async () => {
    let fail = true;
    const fetchFn = stubFetch(async () => {
      if (fail) throw new TypeError("fetch failed");
      return { status: 200, ok: true, json: async () => CONFIG };
    });
    const { result } = await mounted();
    expect(result.current.phase).toBe("error");
    expect(result.current.message).toBe("Cannot reach the backend");
    expect(result.current.tiles).toEqual([]);
    fail = false;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RETRY_MS);
    });
    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(result.current.phase).toBe("connected");
  });

  it("disconnects on unmount and does not retry afterwards", async () => {
    const fetchFn = stubFetch(ok(CONFIG));
    const { unmount } = await mounted();
    const room = FakeRoom.instances[0];
    unmount();
    expect(room.disconnect).toHaveBeenCalledTimes(1);
    // The Disconnected the SDK emits on teardown must not schedule a retry.
    room.emit("Disconnected");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RETRY_MS * 3);
    });
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(FakeRoom.instances).toHaveLength(1);
  });

  it("cancels a pending retry on unmount", async () => {
    const fetchFn = stubFetch(async () => {
      throw new TypeError("down");
    });
    const { unmount } = await mounted();
    unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RETRY_MS * 3);
    });
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("aborts the in-flight config fetch on unmount and creates no Room", async () => {
    let signal: AbortSignal | undefined;
    let resolve!: (v: unknown) => void;
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, init: { signal: AbortSignal }) => {
        signal = init.signal;
        return new Promise((r) => (resolve = r));
      }),
    );
    const { unmount } = renderHook(() => useLiveKitPreview());
    unmount();
    expect(signal?.aborted).toBe(true);
    resolve({ status: 200, ok: true, json: async () => CONFIG });
    await flush();
    expect(FakeRoom.instances).toHaveLength(0);
  });

  it("disconnects a Room whose connect finishes after unmount", async () => {
    stubFetch(ok(CONFIG));
    let release!: () => void;
    FakeRoom.connectGate = new Promise<void>((r) => (release = r));
    const { unmount } = await mounted();
    const room = FakeRoom.instances[0];
    unmount();
    release();
    await flush();
    expect(room.disconnect).toHaveBeenCalled();
  });
});
