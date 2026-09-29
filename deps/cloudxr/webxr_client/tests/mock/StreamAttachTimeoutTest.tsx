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
 * StreamAttachTimeoutTest - minimal harness mounting the real CloudXRComponent against
 * MockCloudXR (same pattern as CloudXRComponentTest.tsx - see that file's header for the
 * general approach), focused specifically on the passthrough-only stream-attach timeout
 * (CloudXRComponent.tsx's streamAttachTimeoutMs prop): a session that enters XR and calls
 * connect() but whose stream never attaches within the deadline.
 *
 * streamAttachTimeoutMs is set to 1000ms and reconnect is enabled (1 attempt, 300ms delay).
 * MockCloudXR's connectWait() is set per-attempt via connectWaitQueue (reapplied in
 * onSessionReady, since each retry creates a brand-new MockCloudXR instance - connectWait()
 * doesn't carry over from one instance to the next):
 *
 *   1. First attempt: connectWait(1250) - 1.25s, over the 1s (1x) budget. Our timer fires
 *      first, synthesizes a recoverable error, and the existing bounded-retry path (PR #1122)
 *      schedules a retry.
 *   2. Second attempt: connectWait(2000) - 2s, under the 2s (2x, doubled) budget the second
 *      attempt gets. This one should reach Connected before its timer fires.
 *
 * Click the button (or press "S") to run it.
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
const FIRST_ATTEMPT_CONNECT_WAIT_MS = 1250; // over the 1x (1000ms) budget - expected to fail
const SECOND_ATTEMPT_CONNECT_WAIT_MS = 2000; // under the 2x (2000ms) budget - expected to succeed

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
// at the default connect delay until connectWait() is called on it again - see the module doc).
let connectWaitQueue: number[] = [];
let statusMessages: string[] = [];

function Scene() {
  return (
    <XR store={store}>
      <XROrigin />
      <CloudXRComponent
        config={config}
        applicationName="StreamAttachTimeoutTest"
        iceServers={{ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] }}
        reconnect={{ maxAttempts: 1, delayMs: 300 }}
        streamAttachTimeoutMs={STREAM_ATTACH_TIMEOUT_MS}
        onStatusChange={(isConnected, status) => {
          statusMessages.push(status);
          appendLog(`[status] connected=${isConnected} ${status}`);
        }}
        onError={error => appendLog(`[error] ${error}`)}
        onExitImmersiveXR={() => {
          appendLog('[event] onExitImmersiveXR');
          store.getState().session?.end();
        }}
        onSessionReady={session => {
          activeSession = session as MockCloudXR | null;
          if (activeSession && connectWaitQueue.length > 0) {
            activeSession.connectWait(connectWaitQueue.shift()!);
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
  appendLog('=== Stream-attach timeout: attempt 1 (1.25s wait, 1s budget) should fail, ===');
  appendLog('=== attempt 2 (2s wait, 2s doubled budget) should succeed ===');
  statusMessages = [];
  connectWaitQueue = [FIRST_ATTEMPT_CONNECT_WAIT_MS, SECOND_ATTEMPT_CONNECT_WAIT_MS];
  await startSession();

  // Long enough for: 1s budget fail + 300ms retry delay + 2s budget success, plus margin.
  const reachedConnected = await waitUntil(
    () => activeSession?.state === CloudXR.SessionState.Connected,
    6000
  );

  const sawRetryStatus = statusMessages.some(s => s.startsWith('Reconnecting'));
  const pass = reachedConnected && sawRetryStatus;
  appendLog(
    pass
      ? '[result] PASS: attempt 1 timed out and retried, attempt 2 reached Connected'
      : `[result] FAIL: reachedConnected=${reachedConnected} sawRetryStatus=${sawRetryStatus} ` +
          `statusMessages=${JSON.stringify(statusMessages)}`
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
