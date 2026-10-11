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
 * StreamAttachTimeoutTest - mounts the real CloudXRComponent against MockCloudXR (see
 * CloudXRComponentTest.tsx's header for the general approach) to exercise its three
 * connection-phase deadlines back-to-back in one continuous 4-attempt retry sequence:
 * streamAttachTimeoutMs (attempt 1: stream never attaches), warmupBeginTimeoutMs (attempt 2:
 * attaches but no warm-up log ever arrives), warmupEndTimeoutMs (attempt 3: warm-up begins but
 * never completes), then a normal attempt 4 proving none of the timers fire once attach and
 * warm-up genuinely succeed. Each attempt drives MockCloudXR via setWarmupTotalFrames() +
 * videoFrameReceived() (test-driven, no background clock) on a schedule relative to Connected.
 *
 * Also asserts a real diagnostic signal: streaming/render metrics (MetricsCadence.PerFrame)
 * must not fire during a stalled warm-up (attempt 3) and must fire once warm-up completes
 * (attempt 4) - a frozen frame count while the session reads Connected is what a stuck warm-up
 * looks like from the outside, and MockCloudXR.render() reproduces that.
 *
 * Click the button (or press "S") to run it.
 */

import { Canvas } from '@react-three/fiber';
import { createXRStore, noEvents, PointerEvents, XR, XROrigin } from '@react-three/xr';
import { useCallback, useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';

import { loadIWERIfNeeded } from '@helpers/LoadIWER';
import CloudXRComponent, { WarmupStatus } from '@helpers/react/CloudXRComponent';
import type { CloudXRConfig } from '@helpers/utils';

import type { MockCloudXR } from './MockCloudXR';

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitUntil(predicate: () => boolean, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) {
      return true;
    }
    await sleep(50);
  }
  return predicate();
}

const STREAM_ATTACH_TIMEOUT_MS = 1000;
const WARMUP_BEGIN_TIMEOUT_MS = 500;
const WARMUP_END_TIMEOUT_MS = 500;

// Delays (ms after Connected) used for each attempt's videoFrameReceived() calls - only the
// first frameCallCount entries are used for a given attempt.
const FRAME_CALL_DELAYS_MS = [50, 100, 150];

interface AttemptConfig {
  connectWaitMs: number;
  /** Passed to setWarmupTotalFrames() right after connect(). Omitted = leave the mock default (0). */
  warmupTotalFrames?: number | null;
  /** How many no-arg videoFrameReceived() calls to make, at FRAME_CALL_DELAYS_MS after Connected. */
  frameCallCount: number;
}

// Per-attempt mock configuration - see the module doc comment for what each attempt tests.
const ATTEMPT_CONFIGS: AttemptConfig[] = [
  { connectWaitMs: 1250, frameCallCount: 0 }, // 1: attach timeout (over 1x budget)
  { connectWaitMs: 100, frameCallCount: 0 }, // 2: attaches fine, no frame ever -> warmup-begin timeout
  { connectWaitMs: 100, warmupTotalFrames: null, frameCallCount: 1 }, // 3: warm-up begins, then stalls -> warmup-end timeout
  { connectWaitMs: 100, warmupTotalFrames: 2, frameCallCount: 3 }, // 4: warm-up begins and completes fast -> success
];

const config: CloudXRConfig = {
  serverIP: 'mock',
  port: 0,
  useSecureConnection: false,
  perEyeWidth: 1024,
  perEyeHeight: 1024,
  deviceFrameRate: 72,
  maxStreamingBitrateMbps: 100,
  immersiveMode: 'vr',
  serverType: 'mock',
  proxyUrl: '',
  referenceSpaceType: 'local-floor',
};

const store = createXRStore({
  emulate: false,
  hand: { model: false },
  controller: { model: false },
  offerSession: false,
});

function appendLog(message: string): void {
  console.info(`[StreamAttachTimeoutTest] ${message}`);
  const logEl = document.getElementById('log');
  if (!logEl) {
    return;
  }
  const line = document.createElement('div');
  line.textContent = message;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

// Mutable script state, module-level rather than React state - same rationale as
// CloudXRComponentTest.tsx: Scene/App never need to re-render on these changing.
let activeSession: MockCloudXR | null = null;
// Consumed one entry per new MockCloudXR instance (each retry creates a fresh one, which starts
// at mock defaults until connectWait()/setWarmupTotalFrames() are called on it again - see the
// module doc). The current attempt's frame-call count is stashed here until Connected actually
// fires (see onStatusChange).
let attemptConfigQueue: AttemptConfig[] = [];
let pendingFrameCallCount = 0;
let statusMessages: string[] = [];
let warmupStatuses: WarmupStatus[] = [];
// Set the moment an onStreamingPerformanceMetrics event is observed to fire while warm-up
// hadn't completed yet (checked against warmupStatuses at that instant - see onStatusChange's
// use of the same pattern) - a real CloudXR.js bug repro, not something this test should ever
// see. See the module doc comment.
let streamingMetricsFiredDuringWarmup = false;
let streamingMetricsFiredAfterWarmupComplete = false;

function Scene() {
  return (
    <XR store={store}>
      <XROrigin />
      <CloudXRComponent
        config={config}
        applicationName="StreamAttachTimeoutTest"
        iceServers={{ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] }}
        reconnect={{ maxAttempts: 3, delayMs: 300 }}
        streamAttachTimeoutMs={STREAM_ATTACH_TIMEOUT_MS}
        warmupBeginTimeoutMs={WARMUP_BEGIN_TIMEOUT_MS}
        warmupEndTimeoutMs={WARMUP_END_TIMEOUT_MS}
        onStatusChange={(isConnected, status) => {
          statusMessages.push(status);
          appendLog(`[status] connected=${isConnected} ${status}`);
          if (status === 'Connected') {
            // Capture the session in this attempt's closure - by the time a later timeout fires
            // (500ms+), a retry may already have replaced the module-level `activeSession` with
            // a fresh instance, and these calls must land on the one that actually connected.
            const sessionForThisAttempt = activeSession;
            for (let i = 0; i < pendingFrameCallCount; i++) {
              setTimeout(
                () => sessionForThisAttempt?.videoFrameReceived(),
                FRAME_CALL_DELAYS_MS[i]
              );
            }
            pendingFrameCallCount = 0;
          }
        }}
        onWarmupStatus={status => {
          warmupStatuses.push(status);
          appendLog(`[warmup] framesSeen=${status.framesSeen} completed=${status.completed}`);
        }}
        onStreamingPerformanceMetrics={() => {
          // warmupStatuses already reflects this instant: onWarmupStatus (pushed synchronously
          // from MockCloudXR's onLog handling inside render()) always fires before this same
          // render() call reaches its own onMetrics(PerFrame) emission - see
          // MockCloudXR.render()'s doc comment.
          if (warmupStatuses.some(s => s.completed)) {
            streamingMetricsFiredAfterWarmupComplete = true;
          } else {
            streamingMetricsFiredDuringWarmup = true;
          }
          appendLog('[event] onStreamingPerformanceMetrics');
        }}
        onError={error => appendLog(`[error] ${error}`)}
        onExitImmersiveXR={() => {
          appendLog('[event] onExitImmersiveXR');
          store.getState().session?.end();
        }}
        onSessionReady={session => {
          activeSession = session as MockCloudXR | null;
          // Only consume a queue entry for a real new session - a null onSessionReady is not
          // the start of an attempt, and shifting on one would silently drain the queue before
          // the next real attempt ever sees its config.
          if (activeSession) {
            const attemptConfig = attemptConfigQueue.shift();
            if (attemptConfig) {
              activeSession.connectWait(attemptConfig.connectWaitMs);
              if (attemptConfig.warmupTotalFrames !== undefined) {
                activeSession.setWarmupTotalFrames(attemptConfig.warmupTotalFrames);
              }
              // Without this, the default threshold (10) means attempt 3's single warm-up call
              // would never log at all - CloudXRComponent would never see it, so
              // armWarmupBeginTimer would never clear and warmupEndTimeoutMs would never get
              // armed in the first place.
              activeSession.setWarmupFirstStatusFrame(1);
              pendingFrameCallCount = attemptConfig.frameCallCount;
            }
          }
          appendLog(`[event] onSessionReady ${session ? 'session' : 'null'}`);
        }}
      />
    </XR>
  );
}

async function startSession(): Promise<void> {
  const { supportsImmersive } = await loadIWERIfNeeded();
  if (!supportsImmersive) {
    appendLog('[error] No immersive WebXR support and IWER emulation failed to load.');
    return;
  }
  try {
    await store.enterVR();
  } catch (error) {
    appendLog(
      `[error] Failed to start XR session: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

async function runTest(): Promise<void> {
  appendLog('=== Attempt 1 (1.25s wait, 1s attach budget) should time out on attach, ===');
  appendLog('=== attempt 2 (attaches, no frame ever) should time out on warmup begin, ===');
  appendLog('=== attempt 3 (warm-up begins, then stalls) should time out on warmup end, ===');
  appendLog('=== attempt 4 (warm-up begins and completes fast) should succeed ===');
  statusMessages = [];
  warmupStatuses = [];
  streamingMetricsFiredDuringWarmup = false;
  streamingMetricsFiredAfterWarmupComplete = false;
  attemptConfigQueue = [...ATTEMPT_CONFIGS];
  pendingFrameCallCount = 0;
  await startSession();

  // Staged waits mirroring the component's own three-phase progression: Connected (attach done)
  // -> warm-up begun (first onWarmupStatus event) -> warm-up finished (completed: true). Each
  // condition is monotonic over the whole run's accumulated statusMessages/warmupStatuses, so
  // this naturally spans attempts 2-3 (which each reach a subset of the stages, then fail and
  // retry) and attempt 4 (which reaches all three). Total timeout budget: 1s attach budget fail
  // + 300ms retry + 500ms warmup-begin budget fail + 300ms retry + 500ms warmup-end budget fail
  // + 300ms retry + fast success, plus margin.
  const reachedConnected = await waitUntil(() => statusMessages.includes('Connected'), 3000);
  const warmupBegan = await waitUntil(() => warmupStatuses.length > 0, 3000);
  const warmupCompleted = await waitUntil(() => warmupStatuses.some(s => s.completed), 3000);
  // A little extra time after warm-up completes for attempt 4's render loop to actually emit at
  // least one PerFrame metrics tick - proving the pipe resumes once warm-up ends, not just that
  // it stayed silent throughout (which sawStalledWarmupProgress alone wouldn't distinguish from
  // a metrics pipe that's permanently broken).
  await waitUntil(() => streamingMetricsFiredAfterWarmupComplete, 1000);

  const retryCount = statusMessages.filter(s => s.startsWith('Reconnecting')).length;
  const sawStalledWarmupProgress = warmupStatuses.some(s => !s.completed);
  const pass =
    reachedConnected &&
    warmupBegan &&
    warmupCompleted &&
    retryCount === 3 &&
    sawStalledWarmupProgress &&
    !streamingMetricsFiredDuringWarmup &&
    streamingMetricsFiredAfterWarmupComplete;
  appendLog(
    pass
      ? '[result] PASS: attempt 1 attach-timed-out, attempt 2 warmup-begin-timed-out, ' +
          'attempt 3 warmup-end-timed-out, attempt 4 attached and warmed up, streaming metrics ' +
          'stayed frozen throughout warm-up and resumed once it completed'
      : `[result] FAIL: reachedConnected=${reachedConnected} warmupBegan=${warmupBegan} ` +
          `warmupCompleted=${warmupCompleted} retryCount=${retryCount} ` +
          `sawStalledWarmupProgress=${sawStalledWarmupProgress} ` +
          `streamingMetricsFiredDuringWarmup=${streamingMetricsFiredDuringWarmup} ` +
          `streamingMetricsFiredAfterWarmupComplete=${streamingMetricsFiredAfterWarmupComplete} ` +
          `statusMessages=${JSON.stringify(statusMessages)} warmupStatuses=${JSON.stringify(warmupStatuses)}`
  );

  await sleep(500);
  store.getState().session?.end();
}

function App() {
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);

  const run = useCallback(async () => {
    if (runningRef.current) {
      return;
    }
    runningRef.current = true;
    setRunning(true);
    try {
      await runTest();
    } finally {
      runningRef.current = false;
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === 's') {
        void run();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [run]);

  return (
    <>
      <div id="panel">
        {!running && (
          <button id="startButton" type="button" onClick={() => void run()}>
            Start StreamAttachTimeoutTest (Mock) [S]
          </button>
        )}
        <div id="log" />
      </div>
      <Canvas events={noEvents} style={{ position: 'fixed', inset: 0, zIndex: -1 }}>
        <PointerEvents batchEvents={false} />
        <Scene />
      </Canvas>
    </>
  );
}

const container = document.getElementById('root');
if (container) {
  ReactDOM.createRoot(container).render(<App />);
} else {
  console.error('StreamAttachTimeoutTest: #root container not found');
}
