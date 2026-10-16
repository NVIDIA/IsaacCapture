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
 * MockCloudXRTests - standalone demo page that drives {@link MockCloudXR} through a real
 * (IWER-emulated, if no headset is attached) WebXR session, so a developer can visually
 * confirm the stereo mock render without a CloudXR server. Open via `dev-server:mock` /
 * `build:mock` (see package.json); not part of the production app bundle.
 */

import * as CloudXR from '@nvidia/cloudxr';

import { loadIWERIfNeeded } from '@helpers/LoadIWER';

import { createMockCloudXRSession, MockCloudXRController } from './MockCloudXR';

declare global {
  interface Window {
    /** Live mock controls, for poking at from the devtools console: `mockCloudXR.triggerFailure()`. */
    mockCloudXR?: MockCloudXRController;
  }
}

const startButton = document.getElementById('startButton') as HTMLButtonElement;
const logEl = document.getElementById('log') as HTMLDivElement;
const canvas = document.getElementById('gl') as HTMLCanvasElement;

function appendLog(message: string): void {
  const line = document.createElement('div');
  line.textContent = message;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

async function startMockSession(): Promise<void> {
  startButton.disabled = true;

  const { supportsImmersive } = await loadIWERIfNeeded();
  if (!supportsImmersive || !navigator.xr) {
    appendLog('No immersive WebXR support and IWER emulation failed to load.');
    startButton.disabled = false;
    return;
  }

  const gl = canvas.getContext('webgl2', { xrCompatible: true }) as WebGL2RenderingContext;
  await gl.makeXRCompatible();

  const xrSession = await navigator.xr.requestSession('immersive-vr', {
    requiredFeatures: ['local-floor'],
  });
  await xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
  const referenceSpace = await xrSession.requestReferenceSpace('local-floor');

  const delegates: CloudXR.SessionDelegates = {
    onLog: entries => entries.forEach(entry => appendLog(`[log] ${entry.message}`)),
    onMetrics: (metrics, cadence) => {
      if (cadence === CloudXR.MetricsCadence.PerNetwork) {
        appendLog(`[metrics/network] ${JSON.stringify(metrics)}`);
      }
    },
    onStreamStarted: () => appendLog('[event] onStreamStarted'),
    onStreamStopped: error =>
      appendLog(`[event] onStreamStopped ${error ? error.message : '(clean)'}`),
    onStreamTestStarted: () => appendLog('[event] onStreamTestStarted'),
    onStreamTestStopped: result => appendLog(`[event] onStreamTestStopped passed=${result.passed}`),
    onMicStateUpdate: state => appendLog(`[event] onMicStateUpdate ${CloudXR.MicState[state]}`),
    onServerMessageReceived: data => appendLog(`[event] onServerMessageReceived ${data.length}B`),
  };

  const cxrSession = createMockCloudXRSession(
    {
      serverAddress: 'mock',
      serverPort: 0,
      useSecureConnection: false,
      gl,
      perEyeWidth: 1024,
      perEyeHeight: 1024,
      referenceSpace,
    },
    delegates
  );

  window.mockCloudXR = new MockCloudXRController(cxrSession);
  appendLog(
    '[info] window.mockCloudXR is live: try mockCloudXR.triggerFailure(), ' +
      'mockCloudXR.connectWait(3000), mockCloudXR.setNetworkQuality(1) (Unsustainable..4 Excellent), ' +
      'mockCloudXR.videoFrameReceived(1, 100) (real frame, scene time = 100ms), ' +
      'mockCloudXR.setWarmupTotalFrames(3) (or null to never complete), ' +
      'mockCloudXR.setWarmupFirstStatusFrame(1), mockCloudXR.getCurrentFrameID()'
  );

  cxrSession.connect();

  const onXRFrame = (timestamp: DOMHighResTimeStamp, frame: XRFrame): void => {
    xrSession.requestAnimationFrame(onXRFrame);
    try {
      cxrSession.sendTrackingStateToServer(timestamp, frame);
      if (cxrSession.state === CloudXR.SessionState.Connected) {
        cxrSession.render(timestamp, frame, xrSession.renderState.baseLayer as XRWebGLLayer);
      }
    } catch (error) {
      console.error('MockCloudXRTests render loop error:', error);
    }
  };
  xrSession.requestAnimationFrame(onXRFrame);

  xrSession.addEventListener('end', () => {
    cxrSession.disconnect();
    startButton.disabled = false;
  });
}

startButton.addEventListener('click', () => {
  startMockSession().catch(error => {
    console.error('MockCloudXRTests failed to start:', error);
    appendLog(`[error] ${error instanceof Error ? error.message : String(error)}`);
    startButton.disabled = false;
  });
});
