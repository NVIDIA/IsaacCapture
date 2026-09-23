/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HeadsetControlChannel } from './controlChannel';

class FakeWebSocket {
  static readonly OPEN = 1;
  readonly sent: Array<{ type: string; payload: Record<string, unknown> }> = [];
  readyState = FakeWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(readonly url: string) {}

  send(data: string): void {
    this.sent.push(JSON.parse(data));
  }

  close(): void {
    this.readyState = 3;
  }
}

it('answers a generation probe with cached stream and metric metadata', () => {
  jest.useFakeTimers();
  const originalWebSocket = globalThis.WebSocket;
  const sockets: FakeWebSocket[] = [];
  const constructor = class extends FakeWebSocket {
    constructor(url: string) {
      super(url);
      sockets.push(this);
    }
  };
  globalThis.WebSocket = constructor as unknown as typeof WebSocket;
  let channel: HeadsetControlChannel | undefined;
  try {
    channel = new HeadsetControlChannel({
      url: 'wss://example.test/oob/v1/ws',
      onConfig: () => {},
      getMetricsSnapshot: () => [{ cadence: 'network', metrics: { rtt: 2 } }],
    });
    channel.sendStreamStatus(true);
    channel.connect();
    const socket = sockets[0];
    socket.onopen?.();
    jest.advanceTimersByTime(500);
    socket.onmessage?.({
      data: JSON.stringify({
        type: 'healthProbe',
        payload: { probeId: 'probe-1', lifecycleGeneration: 3 },
      }),
    });
    const report = socket.sent.find(message => message.type === 'healthReport');
    expect(report?.payload).toMatchObject({
      probeId: 'probe-1',
      lifecycleGeneration: 3,
      streamStatus: true,
      metricCadences: ['network'],
    });
    expect(report?.payload.lastMetricsAt).toEqual(expect.any(Number));
  } finally {
    channel?.dispose();
    globalThis.WebSocket = originalWebSocket;
    jest.useRealTimers();
  }
});
