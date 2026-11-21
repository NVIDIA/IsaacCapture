// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Keeps a clock mapping between this browser and a LiveKit publisher, from four-timestamp exchanges sent as small
 * data messages addressed to the publisher only, once a second. It rides the room connection the preview already
 * has and never the video path. A reply that does not match an outstanding request (sequence and t1) is ignored.
 */

import { RoomEvent } from "livekit-client";
import type { Room } from "livekit-client";
import { CLOCK_SYNC_TOPIC, ClockEstimator, decodeReply, encodeRequest } from "./clockExchange";
import type { ClockMapping } from "./clockExchange";

const INTERVAL_MS = 1000;
const MAX_OUTSTANDING = 16;

/** This browser's monotonic clock in integer nanoseconds. */
export function browserNowNs(): bigint {
  return BigInt(Math.round(performance.now() * 1e6));
}

/**
 * A conservative bound on this browser's timer resolution, in ns. Browsers coarsen performance.now() (1 ms by
 * default in Firefox, more with fingerprinting resistance, 0.1 ms or finer in others), and that is an error source
 * in every exchange. It is a declared bound, not a measurement: measuring it would mean spinning on the UI thread.
 * A browser coarser than this under-reports the uncertainty.
 */
export const ASSUMED_TIMER_RESOLUTION_NS = 1_000_000n;

export class LiveKitClockSync {
  readonly estimator: ClockEstimator;
  private outstanding = new Map<number, bigint>();
  private sequence = 0;
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private readonly room: Room,
    private readonly publisher: string,
    private readonly nowNs: () => bigint = browserNowNs,
    timerResolutionNs: bigint = ASSUMED_TIMER_RESOLUTION_NS,
  ) {
    this.estimator = new ClockEstimator(timerResolutionNs);
  }

  start(): void {
    this.room.on(RoomEvent.DataReceived, this.onData);
    void this.sendRequest();
    this.timer = setInterval(() => void this.sendRequest(), INTERVAL_MS);
  }

  stop(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    this.room.off(RoomEvent.DataReceived, this.onData);
  }

  /** The publisher-to-browser clock mapping, or null when there is none yet or it is stale. */
  mapping(): ClockMapping | null {
    return this.estimator.mapping(this.nowNs());
  }

  get discontinuity(): boolean {
    return this.estimator.discontinuity;
  }

  private readonly onData = (payload: Uint8Array, participant?: { identity: string }, _kind?: unknown, topic?: string): void => {
    const t4 = this.nowNs(); // first, so it is as close to the receipt as software allows
    if (topic !== CLOCK_SYNC_TOPIC || participant?.identity !== this.publisher) return;
    const reply = decodeReply(payload);
    if (reply === null) return;
    if (this.outstanding.get(reply.sequence) !== reply.t1) return; // not an answer to a request of ours
    this.outstanding.delete(reply.sequence);
    this.estimator.add({ t1: reply.t1, t2: reply.t2, t3: reply.t3, t4 });
  };

  private async sendRequest(): Promise<void> {
    this.sequence = (this.sequence + 1) >>> 0;
    const sequence = this.sequence;
    while (this.outstanding.size >= MAX_OUTSTANDING) {
      const oldest = this.outstanding.keys().next().value as number;
      this.outstanding.delete(oldest);
    }
    try {
      // t1 is read as late as possible, immediately before the send.
      const t1 = this.nowNs();
      this.outstanding.set(sequence, t1);
      await this.room.localParticipant.publishData(encodeRequest(sequence, t1), {
        reliable: false,
        destinationIdentities: [this.publisher],
        topic: CLOCK_SYNC_TOPIC,
      });
    } catch {
      // A failed send just means no reply; the mapping goes stale, which the consumer treats as unavailable.
    }
  }
}
