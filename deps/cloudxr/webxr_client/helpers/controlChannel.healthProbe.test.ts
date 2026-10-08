/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HeadsetControlChannel, streamPhaseForStatus } from './controlChannel';

class FakeWebSocket {
  static readonly OPEN = 1;
  readonly sent: Array<{ type: string; payload: Record<string, unknown> }> = [];
  readyState = FakeWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(readonly url: string) {}

  send(data: string): void {
    this.sent.push(JSON.parse(data));
  }

  close(): void {
    this.readyState = 3;
  }
}

it('classifies browser-local retry separately from terminal exhaustion', () => {
  expect(streamPhaseForStatus(false, 'Reconnecting (2/3)')).toBe('retrying');
  expect(streamPhaseForStatus(false, 'Error')).toBe('terminal');
  expect(streamPhaseForStatus(false, 'Disconnected')).toBe('idle');
  expect(streamPhaseForStatus(true, 'Connected')).toBe('streaming');
});

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

it('keeps one terminal event stable across idle and replay, then clears it for retry', () => {
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
    });
    channel.connect();
    sockets[0].onopen?.();
    channel.sendStreamStatus(false, 'terminal', 'Error');
    const terminal = sockets[0].sent.at(-1)?.payload;
    expect(terminal?.terminalEventId).toEqual(expect.any(String));

    channel.sendStreamStatus(false, 'idle', 'Disconnected');
    expect(sockets[0].sent.at(-1)?.payload.terminalEventId).toBe(terminal?.terminalEventId);

    sockets[0].onclose?.({ code: 1006 } as CloseEvent);
    jest.advanceTimersByTime(3000);
    sockets[1].onopen?.();
    expect(sockets[1].sent.at(-1)?.payload.terminalEventId).toBe(terminal?.terminalEventId);

    channel.sendStreamStatus(false, 'retrying', 'Reconnecting (1/3)');
    expect(sockets[1].sent.at(-1)?.payload).not.toHaveProperty('terminalEventId');
    expect(sockets[1].sent.at(-1)?.payload.phase).toBe('retrying');
  } finally {
    channel?.dispose();
    globalThis.WebSocket = originalWebSocket;
    jest.useRealTimers();
  }
});
