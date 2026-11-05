// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CLOCK_SYNC_TOPIC, encodeRequest } from "./clockExchange";
import { ASSUMED_TIMER_RESOLUTION_NS, LiveKitClockSync } from "./liveKitClockSync";

const DATA_RECEIVED = "DataReceived";

vi.mock("livekit-client", () => ({ RoomEvent: { DataReceived: "DataReceived" } }));

function fakeRoom() {
  const handlers: Array<(...a: any[]) => void> = [];
  return {
    handlers,
    on: vi.fn((event: string, h: (...a: any[]) => void) => {
      if (event === DATA_RECEIVED) handlers.push(h);
    }),
    off: vi.fn((event: string, h: (...a: any[]) => void) => {
      const i = handlers.indexOf(h);
      if (event === DATA_RECEIVED && i >= 0) handlers.splice(i, 1);
    }),
    localParticipant: { publishData: vi.fn(async () => {}) },
  };
}

function reply(sequence: number, t1: bigint, t2: bigint, t3: bigint, length = 29): Uint8Array {
  const bytes = new Uint8Array(29);
  const view = new DataView(bytes.buffer);
  bytes[0] = 0x52;
  view.setUint32(1, sequence, true);
  view.setBigInt64(5, t1, true);
  view.setBigInt64(13, t2, true);
  view.setBigInt64(21, t3, true);
  return bytes.slice(0, length);
}

const MS = 1_000_000n;
let now = 0n;
beforeEach(() => {
  vi.useFakeTimers();
  now = 1_000n * MS;
});
afterEach(() => vi.useRealTimers());

function started() {
  const room = fakeRoom();
  const sync = new LiveKitClockSync(room as any, "orin", () => now, 0n);
  sync.start();
  return { room, sync, deliver: (...a: any[]) => room.handlers.forEach((h) => h(...a)) };
}

describe("LiveKitClockSync", () => {
  it("sends one request to the publisher on start, with t1 read from the clock", () => {
    const { room } = started();
    expect(room.localParticipant.publishData).toHaveBeenCalledTimes(1);
    const [bytes, options] = (room.localParticipant.publishData.mock.calls[0] as unknown) as [Uint8Array, unknown];
    expect(bytes).toEqual(encodeRequest(1, 1_000n * MS));
    expect(options).toEqual({ reliable: false, destinationIdentities: ["orin"], topic: CLOCK_SYNC_TOPIC });
  });

  it("sends another request each second", () => {
    const { room } = started();
    vi.advanceTimersByTime(2000);
    expect(room.localParticipant.publishData).toHaveBeenCalledTimes(3);
  });

  it("turns a matching reply into a mapping", () => {
    const { sync, deliver } = started();
    expect(sync.mapping()).toBeNull();
    now = 1_010n * MS;
    deliver(reply(1, 1_000n * MS, 4_005n * MS, 4_005n * MS), { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC);
    const m = sync.mapping()!;
    expect(m.offsetNs).toBe(3_000n * MS);
    expect(m.rttNs).toBe(10n * MS);
  });

  it("ignores wrong topic, identity, length, sequence and t1 and keeps the outstanding request", () => {
    const { sync, deliver } = started();
    now = 1_010n * MS;
    const good = reply(1, 1_000n * MS, 4_005n * MS, 4_005n * MS);
    deliver(good, { identity: "orin" }, undefined, "other");
    deliver(good, { identity: "viewer" }, undefined, CLOCK_SYNC_TOPIC);
    deliver(good, undefined, undefined, CLOCK_SYNC_TOPIC);
    deliver(reply(1, 1_000n * MS, 0n, 0n, 28), { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC);
    deliver(reply(99, 1_000n * MS, 4_005n * MS, 4_005n * MS), { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC);
    deliver(reply(1, 999n * MS, 4_005n * MS, 4_005n * MS), { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC);
    expect(sync.mapping()).toBeNull();
    deliver(good, { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC);
    expect(sync.mapping()).not.toBeNull();
  });

  it("does not accept the same reply twice", () => {
    const { sync, deliver } = started();
    now = 1_010n * MS;
    const good = reply(1, 1_000n * MS, 4_005n * MS, 4_005n * MS);
    deliver(good, { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC);
    now = 1_020n * MS;
    deliver(good, { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC);
    expect(sync.mapping()!.acceptedAtNs).toBe(1_010n * MS);
  });

  it("adds the assumed timer resolution to the mapping uncertainty by default", () => {
    const room = fakeRoom();
    const sync = new LiveKitClockSync(room as any, "orin", () => now);
    sync.start();
    now = 1_010n * MS;
    room.handlers.forEach((h) => h(reply(1, 1_000n * MS, 4_005n * MS, 4_005n * MS), { identity: "orin" }, undefined, CLOCK_SYNC_TOPIC));
    expect(ASSUMED_TIMER_RESOLUTION_NS).toBe(1n * MS);
    expect(sync.mapping()!.uncertaintyNs).toBe(5n * MS + ASSUMED_TIMER_RESOLUTION_NS);
  });

  it("stop removes the handler and the timer", () => {
    const { room, sync } = started();
    sync.stop();
    expect(room.handlers).toHaveLength(0);
    vi.advanceTimersByTime(3000);
    expect(room.localParticipant.publishData).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
});
