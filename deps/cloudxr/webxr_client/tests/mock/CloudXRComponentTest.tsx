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
 * CloudXRComponentTest - minimal harness mounting the real CloudXRComponent (not App.tsx's full
 * production UI: no CloudXR2DUI/CloudXR3DUI, settings panel, or recorder) inside a bare
 * Canvas/XR tree, against MockCloudXR via the build-time '@nvidia/cloudxr' alias (see
 * cloudxr-mock-alias.ts / webpack.component-mock.js). Exercises the real CloudXRComponent.tsx
 * code path - unlike MockCloudXRTests.ts, which hand-drives MockCloudXR directly with no React
 * involved at all.
 *
 * Every prop on CloudXRComponentProps is wired: all callbacks are passed (and logged, see
 * appendLog below - every line also goes through console.info, so a Playwright test can assert
 * against captured console output instead of querying the DOM), and every non-callback prop
 * (metricsSettings, trackingFrameAdapter, iceServers, streamTest, reconnect) is exercised with a
 * real value. headless is the one prop left at its default (false): this page's other steps
 * depend on MockCloudXR actually rendering, and headless=true would only prove the frame gets
 * skipped, which is better covered at the MockCloudXR level than here.
 *
 * streamTest needs its own step (6) rather than a static prop on every session: the effect that
 * reads CloudXRComponent's props only depends on [threeRenderer, config] (see
 * CloudXRComponent.tsx), so changing `streamTest` on a later render has no effect until the
 * component actually remounts. Step 6 forces that remount via a `key` change instead.
 *
 * Click the button (or press "S") to run a scripted six-step sequence covering the full retry API
 * surface too (reconnect prop, retry progress via onStatusChange's "Reconnecting (n/maxAttempts)"
 * status text, isRecoverable() classification, attempt counter reset, and cancellation on session
 * end):
 *   1. Start normally, then close cleanly - expect zero error events.
 *   2. Start with a 2s connect delay, trigger an unrecoverable failure 1s in (still Connecting) -
 *      expect an immediate give-up (onError + onExitImmersiveXR), no retry attempted.
 *   3. Start with no connect delay, wait 2s (now Connected), trigger a recoverable failure -
 *      expect an automatic bounded retry ("Reconnecting (1/maxAttempts)", then reconnected) with
 *      no onExitImmersiveXR; then trigger a second, unrelated failure and confirm it also reports
 *      attempt 1 (not 2) - proving a successful reconnect resets the attempt counter.
 *   4. Force every retried session to stay Connecting well past the reconnect delay, then trigger
 *      four consecutive recoverable failures - expect exactly maxAttempts (3) retries
 *      ("Reconnecting (1/3)", "(2/3)", "(3/3)") before the fourth falls back to give-up.
 *   5. Trigger a recoverable failure (scheduling a retry), then end the WebXR session before the
 *      retry's delay elapses - expect the pending retry to be cancelled, not to create a new
 *      session afterward.
 *   6. Remount with a real streamTest config and let it run to completion.
 * Unrecoverable/recoverable here means CloudXRComponent's isRecoverable() classification
 * (streamingErrorClassification.ts): server-disconnect-range codes (0xC0F223xx) give up
 * immediately, everything else gets a bounded retry.
 */

import * as CloudXR from '@nvidia/cloudxr';
import { Canvas } from '@react-three/fiber';
import { createXRStore, noEvents, PointerEvents, XR, XROrigin } from '@react-three/xr';
import { useCallback, useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';

import { loadIWERIfNeeded } from '@helpers/LoadIWER';
import CloudXRComponent from '@helpers/react/CloudXRComponent';
import type { CloudXRConfig } from '@helpers/utils';

import type { MockCloudXR } from './MockCloudXR';

/** Identity passthrough; just proves the prop is wired and invoked (logged once, see below). */
const trackingFrameAdapter = (frame: XRFrame): XRFrame => frame;

const iceServers: CloudXR.SessionOptions['iceServers'] = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

const metricsSettings = { renderFpsWindow: 10, streamingFpsWindow: 5, poseToRenderWindow: 5 };

const NON_RETRYABLE_CODE = 0xc0f22300; // server-disconnect range
const RETRYABLE_CODE = 0xc0f22204; // NetworkInterrupted

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

// IWER is loaded manually below (same as MockCloudXRTests.ts), so disable the store's own
// emulation; controller/hand models are skipped since this harness never renders them.
const store = createXRStore({
  emulate: false,
  hand: { model: false },
  controller: { model: false },
  offerSession: false,
});

/** Also logs to console (prefixed) so a Playwright test can assert on captured console output. */
function appendLog(message: string): void {
  console.info(`[CloudXRComponentTest] ${message}`);
  const logEl = document.getElementById('log');
  if (!logEl) {
    return;
  }
  const line = document.createElement('div');
  line.textContent = message;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

// Mutable script state, module-level rather than React state: Scene/App never need to re-render
// on these changing, and the step functions below need to read/write them outside any component.
let activeSession: MockCloudXR | null = null;
let pendingConnectWaitMs: number | null = null;
let hadError = false;
let reconnectingAttempts: number[] = [];
let exitImmersiveCount = 0;

// Step 4 only: overrides pendingConnectWaitMs for *every* session (not just the next one), so a
// retried session stays Connecting well past RECONNECT_DELAY_MS - giving the script a wide,
// reliable window to trigger the next failure before that retry would otherwise succeed.
let alwaysConnectWaitMs: number | null = null;

// Step 5 only: while true, a new non-null session reaching onSessionReady means a retry fired
// after it should have been cancelled.
let watchForSessionAfterCancel = false;
let sawSessionAfterCancel = false;
// Step 5 only: cancelling a pending retry (no live session to disconnect()) must still emit a
// terminal status itself, or callers are stuck seeing "Reconnecting (n/maxAttempts)" forever.
let sawDisconnectedAfterCancel = false;

// Render/streaming/network metrics and trackingFrameAdapter all fire every frame (or close to
// it) once connected - logging every occurrence would flood the console, so each just logs once
// per session to prove it's wired, reset whenever a fresh session appears.
let loggedOnce = new Set<string>();

/** Logs `message` the first time it's called with a given `key`; a no-op on later calls. */
function logOnce(key: string, message: string): void {
  if (!loggedOnce.has(key)) {
    loggedOnce.add(key);
    appendLog(message);
  }
}

// Shorter than CloudXRComponent's own default (3000ms) so the scripted sequence can observe a
// full retry-and-reconnect cycle without a long wait; MockCloudXR's own connect delay for the
// retried session falls back to its default (pendingConnectWaitMs is only consumed once, by the
// session that failed) unless alwaysConnectWaitMs overrides it (step 4).
const RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_ATTEMPTS = 3;

function Scene({ streamTestEnabled }: { streamTestEnabled: boolean }) {
  return (
    <XR store={store}>
      <XROrigin />
      <CloudXRComponent
        config={config}
        applicationName="CloudXRComponentTest"
        metricsSettings={metricsSettings}
        trackingFrameAdapter={frame => {
          logOnce('trackingFrameAdapter', '[prop] trackingFrameAdapter called');
          return trackingFrameAdapter(frame);
        }}
        iceServers={iceServers}
        streamTest={streamTestEnabled ? { durationSeconds: 3, mode: 'warn' } : undefined}
        reconnect={{ maxAttempts: MAX_RECONNECT_ATTEMPTS, delayMs: RECONNECT_DELAY_MS }}
        onStatusChange={(isConnected, status) => {
          // No dedicated retry-progress callback - CloudXRComponent reports it through
          // onStatusChange's status text, "Reconnecting (n/maxAttempts)", matching
          // HeadsetControlChannel's single-callback (onConnectionChange) shape.
          const match = /^Reconnecting \((\d+)\/(\d+)\)$/.exec(status);
          if (match) {
            reconnectingAttempts.push(Number(match[1]));
          }
          if (watchForSessionAfterCancel && !isConnected && status === 'Disconnected') {
            sawDisconnectedAfterCancel = true;
          }
          appendLog(`[status] connected=${isConnected} ${status}`);
        }}
        onError={error => {
          hadError = true;
          appendLog(`[error] ${error}`);
        }}
        onExitImmersiveXR={() => {
          exitImmersiveCount += 1;
          appendLog('[event] onExitImmersiveXR');
          // Mirrors App.tsx's handleDisconnect: exiting immersive XR means ending the WebXR
          // session, which is what actually drives CloudXRComponent's own cleanup/disconnect.
          store.getState().session?.end();
        }}
        onSessionReady={session => {
          activeSession = session as MockCloudXR | null;
          if (session) {
            loggedOnce = new Set();
          }
          if (activeSession) {
            if (alwaysConnectWaitMs !== null) {
              activeSession.connectWait(alwaysConnectWaitMs);
            } else if (pendingConnectWaitMs !== null) {
              activeSession.connectWait(pendingConnectWaitMs);
              pendingConnectWaitMs = null;
            }
          }
          if (watchForSessionAfterCancel && session) {
            sawSessionAfterCancel = true;
          }
          appendLog(`[event] onSessionReady ${session ? 'session' : 'null'}`);
        }}
        onServerAddress={address => appendLog(`[event] onServerAddress ${address}`)}
        onLog={entries =>
          appendLog(`[event] onLog ${entries.length} entr${entries.length === 1 ? 'y' : 'ies'}`)
        }
        onRenderPerformanceMetrics={() =>
          logOnce('renderMetrics', '[event] onRenderPerformanceMetrics (first occurrence)')
        }
        onStreamingPerformanceMetrics={() =>
          logOnce('streamingMetrics', '[event] onStreamingPerformanceMetrics (first occurrence)')
        }
        onNetworkPerformanceMetrics={() =>
          logOnce('networkMetrics', '[event] onNetworkPerformanceMetrics (first occurrence)')
        }
        onStreamTestStarted={() => appendLog('[event] onStreamTestStarted')}
        onStreamTestStopped={result =>
          appendLog(`[event] onStreamTestStopped passed=${result.passed}`)
        }
      />
    </XR>
  );
}

/** Sets the next session's connect delay (applied in onSessionReady, see above), then enters VR. */
async function startSession(connectWaitMs: number | null): Promise<void> {
  pendingConnectWaitMs = connectWaitMs;
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

/** startSession(), then waits for the resulting MockCloudXR session to reach Connected. */
async function startAndWaitConnected(
  connectWaitMs: number | null,
  timeoutMs = 3000
): Promise<boolean> {
  await startSession(connectWaitMs);
  return waitUntil(() => activeSession?.state === CloudXR.SessionState.Connected, timeoutMs);
}

async function runStep1(): Promise<void> {
  appendLog('=== Step 1: start, then close cleanly - expect no errors ===');
  hadError = false;
  await startAndWaitConnected(null);
  store.getState().session?.end();
  await sleep(500);
  appendLog(hadError ? '[step1] FAIL: saw an error event' : '[step1] PASS: no errors');
}

async function runStep2(): Promise<void> {
  appendLog('=== Step 2: fail while still connecting (unrecoverable) - expect no retry ===');
  reconnectingAttempts = [];
  await startSession(2000);
  await sleep(1000);
  activeSession?.triggerFailure({
    name: 'StreamingError',
    message: 'Mock unrecoverable failure (server-disconnect range)',
    code: NON_RETRYABLE_CODE,
  });
  await sleep(500);
  appendLog(
    reconnectingAttempts.length === 0
      ? '[step2] PASS: no retry attempted'
      : `[step2] FAIL: unexpected retry attempts=${JSON.stringify(reconnectingAttempts)}`
  );
}

async function runStep3(): Promise<void> {
  appendLog(
    '=== Step 3: fail once connected (recoverable) - expect automatic retry, then counter reset ==='
  );
  reconnectingAttempts = [];
  await startSession(0);
  await sleep(2000);
  activeSession?.triggerFailure({
    name: 'StreamingError',
    message: 'Mock recoverable failure (network interrupted)',
    code: RETRYABLE_CODE,
  });
  // Past RECONNECT_DELAY_MS plus MockCloudXR's own default connect delay for the retried session.
  const reconnected = await waitUntil(
    () => activeSession?.state === CloudXR.SessionState.Connected,
    RECONNECT_DELAY_MS + 3000
  );
  const firstAttemptOk = reconnectingAttempts.length === 1 && reconnectingAttempts[0] === 1;

  // A second, unrelated failure after the successful reconnect: if onStreamStarted actually reset
  // the attempt counter, this also reports attempt 1 (not 2).
  reconnectingAttempts = [];
  activeSession?.triggerFailure({
    name: 'StreamingError',
    message: 'Mock recoverable failure #2 (network interrupted)',
    code: RETRYABLE_CODE,
  });
  const reconnectedAgain = await waitUntil(
    () => activeSession?.state === CloudXR.SessionState.Connected,
    RECONNECT_DELAY_MS + 3000
  );
  const resetOk = reconnectingAttempts.length === 1 && reconnectingAttempts[0] === 1;

  appendLog(
    firstAttemptOk && reconnected && resetOk && reconnectedAgain
      ? '[step3] PASS: retried, reconnected, and the attempt counter reset for a later failure'
      : `[step3] FAIL: firstAttemptOk=${firstAttemptOk} reconnected=${reconnected} ` +
          `resetOk=${resetOk} reconnectedAgain=${reconnectedAgain}`
  );
  store.getState().session?.end();
  await sleep(500);
}

async function runStep4(): Promise<void> {
  appendLog(
    `=== Step 4: exhaust retry attempts - expect give-up after ${MAX_RECONNECT_ATTEMPTS} ===`
  );
  reconnectingAttempts = [];
  const exitCountBefore = exitImmersiveCount;
  const initiallyConnected = await startAndWaitConnected(0);
  if (!initiallyConnected) {
    appendLog('[step4] FAIL: initial session did not connect');
    return;
  }
  // Keep every retried session Connecting well past RECONNECT_DELAY_MS, so each failure below
  // lands on a session that hasn't reconnected yet (a reconnect would otherwise reset the
  // attempt counter before the next failure, per step 3's own PASS case). Set only now, after
  // the initial connection succeeds - onSessionReady checks this before pendingConnectWaitMs, so
  // setting it earlier would also delay (and starve the 3000ms wait for) the initial session.
  alwaysConnectWaitMs = 5000;

  for (let i = 0; i < MAX_RECONNECT_ATTEMPTS + 1; i++) {
    activeSession?.triggerFailure({
      name: 'StreamingError',
      message: `Mock recoverable failure #${i + 1} (network interrupted)`,
      code: RETRYABLE_CODE,
    });
    await sleep(RECONNECT_DELAY_MS + 300);
  }
  alwaysConnectWaitMs = null;

  const gaveUp = exitImmersiveCount > exitCountBefore;
  const attemptsOk =
    reconnectingAttempts.length === MAX_RECONNECT_ATTEMPTS &&
    reconnectingAttempts.every((attempt, i) => attempt === i + 1);
  appendLog(
    attemptsOk && gaveUp
      ? `[step4] PASS: retried ${MAX_RECONNECT_ATTEMPTS} times then gave up`
      : `[step4] FAIL: attempts=${JSON.stringify(reconnectingAttempts)} gaveUp=${gaveUp}`
  );
  await sleep(500);
}

async function runStep5(): Promise<void> {
  appendLog(
    '=== Step 5: end session while a retry is pending - expect the retry to be cancelled ==='
  );
  await startAndWaitConnected(0);

  activeSession?.triggerFailure({
    name: 'StreamingError',
    message: 'Mock recoverable failure (network interrupted)',
    code: RETRYABLE_CODE,
  });
  // Retry is now scheduled ~RECONNECT_DELAY_MS out; end the session well before it fires.
  await sleep(200);
  watchForSessionAfterCancel = true;
  sawSessionAfterCancel = false;
  sawDisconnectedAfterCancel = false;
  store.getState().session?.end();

  await sleep(RECONNECT_DELAY_MS + 1000);
  watchForSessionAfterCancel = false;
  appendLog(
    sawSessionAfterCancel
      ? '[step5] FAIL: a new session was created after the session ended'
      : sawDisconnectedAfterCancel
        ? '[step5] PASS: pending retry was cancelled'
        : '[step5] FAIL: cancelling the pending retry never emitted a terminal status'
  );
}

/**
 * Remounts CloudXRComponent (via the `key` change setStreamTestEnabled drives, see App()) with a
 * real streamTest config, since that prop is frozen at mount by CloudXRComponent's own effect
 * dependency array - toggling it on an already-mounted instance would otherwise do nothing.
 */
async function runStep6(setStreamTestEnabled: (enabled: boolean) => void): Promise<void> {
  appendLog('=== Step 6: stream test (remounts with streamTest enabled) ===');
  setStreamTestEnabled(true);
  await sleep(100); // let React apply the remount before entering VR
  await startAndWaitConnected(0, 8000);
  store.getState().session?.end();
  await sleep(500);
  setStreamTestEnabled(false);
  await sleep(100); // remount back to the normal (no streamTest) instance
}

function App() {
  const [running, setRunning] = useState(false);
  const [streamTestEnabled, setStreamTestEnabled] = useState(false);
  const runningRef = useRef(false);

  const runSequence = useCallback(async () => {
    if (runningRef.current) {
      return;
    }
    runningRef.current = true;
    setRunning(true);
    try {
      await runStep1();
      await runStep2();
      await runStep3();
      await runStep4();
      await runStep5();
      await runStep6(setStreamTestEnabled);
      appendLog('=== Test sequence complete ===');
    } finally {
      runningRef.current = false;
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === 's') {
        void runSequence();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [runSequence]);

  return (
    <>
      <div id="panel">
        {!running && (
          <button id="startButton" type="button" onClick={() => void runSequence()}>
            Start CloudXRComponent (Mock) [S]
          </button>
        )}
        <div id="log" />
      </div>
      <Canvas events={noEvents} style={{ position: 'fixed', inset: 0, zIndex: -1 }}>
        <PointerEvents batchEvents={false} />
        <Scene
          key={streamTestEnabled ? 'streamtest' : 'normal'}
          streamTestEnabled={streamTestEnabled}
        />
      </Canvas>
    </>
  );
}

const container = document.getElementById('root');
if (container) {
  ReactDOM.createRoot(container).render(<App />);
} else {
  console.error('CloudXRComponentTest: #root container not found');
}
