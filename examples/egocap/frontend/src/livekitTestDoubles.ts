// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** Fake livekit-client Room and tracks shared by the LiveKit hook and tile tests. */
import { vi } from "vitest";
import type { LiveKitStreamDescriptor } from "./captureProtocol";

type Handler = (...args: any[]) => void;

export class FakeRoom {
  static instances: FakeRoom[] = [];
  handlers = new Map<string, Handler[]>();
  remoteParticipants = new Map<string, any>();
  connect = vi.fn(async (_url: string, _token: string) => {
    if (FakeRoom.connectGate !== null) await FakeRoom.connectGate;
    if (FakeRoom.connectError !== null) throw FakeRoom.connectError;
  });
  disconnect = vi.fn(async () => {});
  /** Participants already in the room when a new Room is created, i.e. published before we joined. */
  static seed: Array<[string, any]> = [];
  static connectError: Error | null = null;
  static connectGate: Promise<void> | null = null;

  constructor() {
    FakeRoom.instances.push(this);
    for (const [identity, p] of FakeRoom.seed) this.remoteParticipants.set(identity, p);
  }
  on(event: string, handler: Handler): this {
    this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
    return this;
  }
  emit(event: string, ...args: any[]): void {
    for (const h of this.handlers.get(event) ?? []) h(...args);
  }
  listenerCount(event: string): number {
    return (this.handlers.get(event) ?? []).length;
  }
  static reset(): void {
    FakeRoom.instances = [];
    FakeRoom.connectError = null;
    FakeRoom.connectGate = null;
    FakeRoom.seed = [];
  }
}

export function fakeTrack(): any {
  return { kind: "video", attach: vi.fn(), detach: vi.fn() };
}

export function participant(identity: string, published: Array<{ name: string; track: any | null; kind?: string }> = []): any {
  const trackPublications = new Map(
    published.map((p, i) => [`s${i}`, { kind: p.kind ?? "video", trackName: p.name, track: p.track }]),
  );
  return { identity, trackPublications };
}

export function publication(name: string): any {
  return { trackName: name };
}

export const STREAMS: LiveKitStreamDescriptor[] = [
  { kind: "livekit_camera", stream_id: 16, name: "Head left", url: "wss://lk", room: "r", publisher: "orin", track: "left" },
  { kind: "livekit_camera", stream_id: 17, name: "Head right", url: "wss://lk", room: "r", publisher: "orin", track: "right" },
];
