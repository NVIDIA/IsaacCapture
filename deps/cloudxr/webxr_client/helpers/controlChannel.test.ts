/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Drives a real HeadsetControlChannel against a real OOBControlHub - both full,
 * unmocked implementations talking over an actual loopback WebSocket - by spawning
 * tests/python/core/cloudxr/oob_hub_test_server.py as a child process. Node's native
 * `WebSocket` global (stable since Node 22, matches this repo's Node version) is exactly
 * what HeadsetControlChannel needs, so no browser and no WebSocket polyfill are required.
 */

import { type ChildProcess, spawn } from 'child_process';
import * as path from 'path';
import { createInterface } from 'readline';

import { HeadsetControlChannel, type StreamConfig } from './controlChannel';

const REPO_ROOT = path.resolve(__dirname, '../../../..');
// tests/python/core/cloudxr/ is its own uv project (see its pyproject.toml - the same one
// CTest invokes via `uv run --extra dev pytest`), not the repo root's ad-hoc dev .venv,
// which is gitignored and doesn't exist in CI.
const SERVER_DIR = path.join(REPO_ROOT, 'tests', 'python', 'core', 'cloudxr');

/** Spawns the real hub server and resolves once it reports the port it's listening on. */
function startHub(): Promise<{
  proc: ChildProcess;
  port: number;
  nextLine: () => Promise<string>;
}> {
  return new Promise((resolve, reject) => {
    // `uv run` is a wrapper that spawns its own `python3` child rather than exec'ing into
    // it, so killing just this process leaves that child running. `detached: true` makes
    // this process its own process group leader (on Linux); killing the whole group
    // (negative pid) in the caller's cleanup takes the python3 child down with it.
    const proc = spawn('uv', ['run', '--extra', 'dev', 'python3', 'oob_hub_test_server.py'], {
      cwd: SERVER_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: true,
    });
    const rl = createInterface({ input: proc.stdout! });
    // Lines from the Python process arrive on their own schedule, but a test calls
    // nextLine() at its own pace - `pending` buffers lines nobody's asked for yet,
    // `waiters` buffers a nextLine() call that arrived before its line did. Each line
    // resolves at most one waiter (FIFO), so this never double-delivers a line.
    const pending: string[] = [];
    const waiters: Array<(line: string) => void> = [];

    rl.on('line', line => {
      const waiter = waiters.shift();
      if (waiter) waiter(line);
      else pending.push(line);
    });

    const nextLine = (): Promise<string> =>
      new Promise(res => {
        const line = pending.shift();
        if (line !== undefined) res(line);
        else waiters.push(res);
      });

    let stderr = '';
    proc.stderr!.on('data', chunk => {
      stderr += String(chunk);
    });
    proc.on('error', reject);

    nextLine().then(readyLine => {
      const match = /^READY (\d+)$/.exec(readyLine);
      if (!match) {
        reject(new Error(`Expected "READY <port>", got: ${readyLine}\nstderr: ${stderr}`));
        return;
      }
      resolve({ proc, port: Number(match[1]), nextLine });
    }, reject);
  });
}

describe('HeadsetControlChannel <-> OOBControlHub (real instances, real WebSocket)', () => {
  let hub: Awaited<ReturnType<typeof startHub>>;
  let channel: HeadsetControlChannel;

  afterEach(async () => {
    channel?.dispose();
    // Wait for the process group to actually exit rather than firing SIGTERM and hoping -
    // otherwise a slow-to-die process from one test could still be shutting down when the
    // next test (or the suite) starts.
    if (hub?.proc.pid) {
      const exited = new Promise(resolve => hub.proc.once('exit', resolve));
      process.kill(-hub.proc.pid, 'SIGTERM');
      await exited;
    }
  });

  test('registers, receives hello, and the hub sees it as connected', async () => {
    hub = await startHub();

    const configs: Array<{ config: StreamConfig; version: number }> = [];
    const connectionChanges: boolean[] = [];

    channel = new HeadsetControlChannel({
      url: `ws://127.0.0.1:${hub.port}`,
      deviceLabel: 'jest-test',
      onConfig: (config, configVersion) => configs.push({ config, version: configVersion }),
      onConnectionChange: connected => connectionChanges.push(connected),
    });
    channel.connect();

    // The hub only prints SNAPSHOT once handle_connection observes a registered client -
    // this line arriving at all proves the register -> hello round trip succeeded for real.
    const snapshotLine = await hub.nextLine();
    expect(snapshotLine.startsWith('SNAPSHOT ')).toBe(true);
    const snapshot = JSON.parse(snapshotLine.slice('SNAPSHOT '.length));

    expect(snapshot.headsets).toHaveLength(1);
    expect(snapshot.headsets[0]).toMatchObject({
      connected: true,
      streaming: false,
      streamingSince: null,
      deviceLabel: 'jest-test',
    });

    // The client side agrees: onConnectionChange(true) fired (WebSocket opened) and
    // onConfig fired once with the hub's `hello` payload (empty config, version 0 - the
    // hub was just constructed with no initial_config).
    expect(connectionChanges).toEqual([true]);
    expect(configs).toEqual([{ config: {}, version: 0 }]);
  });

  test('sendStreamStatus is reflected in the hub snapshot', async () => {
    hub = await startHub();

    channel = new HeadsetControlChannel({
      url: `ws://127.0.0.1:${hub.port}`,
      onConfig: () => {},
    });
    channel.connect();

    const registerSnapshot = JSON.parse((await hub.nextLine()).slice('SNAPSHOT '.length));
    expect(registerSnapshot.headsets[0].streaming).toBe(false);

    channel.sendStreamStatus(true);

    // The hub prints a fresh SNAPSHOT whenever any headset's `streaming` flag changes
    // (see oob_hub_test_server.py's fingerprint diff) - this line arriving with
    // streaming: true proves the real streamStatus message round-tripped over the wire
    // and OOBControlHub's own state machine updated.
    const streamingSnapshot = JSON.parse((await hub.nextLine()).slice('SNAPSHOT '.length));
    expect(streamingSnapshot.headsets[0]).toMatchObject({ connected: true, streaming: true });
    expect(streamingSnapshot.headsets[0].streamingSince).not.toBeNull();
  });

  test('clientMetrics is reflected in the hub snapshot', async () => {
    hub = await startHub();

    channel = new HeadsetControlChannel({
      url: `ws://127.0.0.1:${hub.port}`,
      onConfig: () => {},
      metricsIntervalMs: 20,
      getMetricsSnapshot: () => [{ cadence: 'frame', metrics: { StreamingFramerate: 72.5 } }],
    });
    channel.connect();

    // The hub prints a fresh SNAPSHOT whenever any headset's metricsByCadence changes
    // (see oob_hub_test_server.py's fingerprint) - proving a real clientMetrics message
    // round-tripped over the wire and OOBControlHub stored it. The metrics timer (20ms)
    // can race the registration snapshot - the first printed line may already include
    // metrics, or metrics may land in a later line - so poll lines until one has them
    // rather than assuming a fixed ordering.
    let metricsByCadence: { frame?: { at: number; metrics: Record<string, number> } } = {};
    for (let i = 0; i < 20 && !metricsByCadence.frame; i++) {
      const snapshot = JSON.parse((await hub.nextLine()).slice('SNAPSHOT '.length));
      metricsByCadence = snapshot.headsets[0].metricsByCadence;
    }

    expect(metricsByCadence.frame).toMatchObject({ metrics: { StreamingFramerate: 72.5 } });
    expect(metricsByCadence.frame?.at).toEqual(expect.any(Number));
  });
});
