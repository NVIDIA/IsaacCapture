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
 * App.tsx - Main CloudXR React Application
 *
 * This is the root component of the CloudXR React example application. It sets up:
 * - WebXR session management and XR store configuration
 * - CloudXR server configuration (IP, port, stream settings)
 * - UI state management (connection status, session state)
 * - Integration between CloudXR rendering component and UI components
 * - Entry point for AR/VR experiences with CloudXR streaming
 *
 * The app integrates with the HTML interface which provides a "CONNECT" button
 * to enter AR mode and displays the CloudXR UI with controls for teleop actions
 * and disconnect when in XR mode.
 */

import * as CloudXR from '@nvidia/cloudxr';
import { getResolutionValidationError } from '@nvidia/cloudxr';
import { computed, signal } from '@preact/signals-react';
import { Canvas } from '@react-three/fiber';
import { setPreferredColorScheme } from '@react-three/uikit';
import { createXRStore, noEvents, PointerEvents, useXR, XR, XROrigin } from '@react-three/xr';
import type { XRDevice } from 'iwer';
import { useEffect, useMemo, useRef, useState } from 'react';
import { v5 } from 'uuid';

import { checkCapabilities } from '@helpers/BrowserCapabilities';
import { HeadsetControlChannel } from '@helpers/controlChannel';
import { getDeviceProfile, resolveDeviceProfileId } from '@helpers/DeviceProfiles';
import { loadIWERIfNeeded } from '@helpers/LoadIWER';
import { MetricsAccumulator } from '@helpers/metricsAccumulator';
import type {
  FrameMetricsUpdate,
  NetworkMetricsUpdate,
  RenderMetricsUpdate,
} from '@helpers/metricsUpdates';
import { overridePressureObserver } from '@helpers/overridePressureObserver';
import { kPerformanceOptions } from '@helpers/PerformanceProfiles';
import CloudXRComponent from '@helpers/react/CloudXRComponent';
import { SimpleEnvironment } from '@helpers/react/SimpleEnvironment';
import { getControlPanelPositionVector } from '@helpers/react/utils';
import {
  DEFAULT_TELEOP_PATH,
  loadStoredTeleopPath,
  parseTeleopPathFromHash,
  saveStoredTeleopPath,
} from '@helpers/TeleopProjects';
import { logImmersiveXRSessionToConsole } from '@helpers/webxrModeDebugText';

import { CloudXR2DUI, COUNTDOWN_STORAGE_KEY } from './CloudXR2DUI';
import CloudXR3DUI from './CloudXRUI';
import { readUrlParam } from './config/resolve';
import { RecorderComponent } from './RecorderComponent';
import { RecorderProvider, useRecorder } from './RecorderContext';
import { SuppressWebGLRendererWhenHeadless } from './SuppressWebGLRendererWhenHeadless';
import { TraceVisualization } from './TraceVisualization';
import {
  SystemNotice,
  formatSystemNotice,
  formatSystemNoticeBody,
  isSystemNoticeMessage,
} from './types/serverMessages';

// Performance metrics signals - raw numeric data backing the in-XR HUD.
// Signals update their value without triggering React re-renders.
// See: https://pmndrs.github.io/uikit/docs/advanced/performance
//
// Only the metrics the HUD draws live here. The full set the SDK reports goes to the
// OOB teleop hub via metricsAccumulator, which is a superset of these.
const renderFps = signal<number | null>(null);
const poseSendFps = signal<number | null>(null);
const streamingMetrics = signal<{ fps: number; latencyMs: number } | null>(null);

// Live session quality 0-4; see CloudXR.QualityScore. 0 is NoData, which is also the
// resting state between sessions.
const sessionQuality = signal<number>(0);

// Network test status: text plus a traffic-light color for the in-XR panel. Module-scoped
// to match the metric signals above; persists last known value across sessions.
const streamTest = signal<{ text: string; color: string } | null>(null);

// Computed signals derive formatted text from raw data.
// When a source signal changes, computed() automatically recalculates the text.
// The @react-three/uikit Text component subscribes to these computed signals
// and updates the displayed text directly in Three.js - bypassing React entirely.
const renderFpsText = computed(() => (renderFps.value !== null ? renderFps.value.toFixed(1) : '-'));
const poseSendFpsText = computed(() =>
  poseSendFps.value !== null ? poseSendFps.value.toFixed(1) : '-'
);
const streamingFpsText = computed(() =>
  streamingMetrics.value ? streamingMetrics.value.fps.toFixed(1) : '-'
);
const poseToRenderText = computed(() =>
  streamingMetrics.value ? `${streamingMetrics.value.latencyMs.toFixed(1)}ms` : '-'
);
const streamTestText = computed(() => streamTest.value?.text ?? '');
const streamTestColor = computed(() => streamTest.value?.color ?? 'white');

/** Accumulates every metric the SDK reports, for periodic upload to the OOB teleop hub. */
const metricsAccumulator = new MetricsAccumulator();

/**
 * Fallback network-test window when the config omits a duration, and the bounds the SDK
 * clamps `SessionOptions.streamTest.durationSeconds` to. Clamping here too keeps the
 * on-panel countdown honest when a config or URL param asks for something out of range.
 */
const DEFAULT_STREAM_TEST_SECONDS = 5;
const MIN_STREAM_TEST_SECONDS = 5;
const MAX_STREAM_TEST_SECONDS = 30;

/** Clamps a configured network-test duration into the range the SDK accepts. */
function resolveStreamTestSeconds(configured: number | undefined): number {
  const seconds = configured ?? DEFAULT_STREAM_TEST_SECONDS;
  return Math.min(MAX_STREAM_TEST_SECONDS, Math.max(MIN_STREAM_TEST_SECONDS, seconds));
}

// Advisory pushed by the host when its workstation is below the recommended
// teleop spec. Held in a signal so the in-XR banner updates without a React
// re-render, matching the metrics signals above.
const systemNotice = signal<SystemNotice | null>(null);
const systemNoticeTitleText = computed(() => systemNotice.value?.title ?? '');
// Shares formatSystemNoticeBody with the 2D banner rather than formatting here,
// so the in-headset text cannot drift from what a desktop tester sees -- notably
// the per-item remediation hint, which is the most actionable part of a notice.
const systemNoticeBodyText = computed(() =>
  systemNotice.value ? formatSystemNoticeBody(systemNotice.value) : ''
);

/** How long the in-XR notice stays up before dismissing itself [ms]. */
const SYSTEM_NOTICE_AUTO_DISMISS_MS = 20000;

const CONTROL_PANEL_LAYOUT = {
  distance: 1.8,
  height: 1.85,
  angleDegrees: 70,
} as const;

// Override PressureObserver early to catch errors from buggy browser implementations
overridePressureObserver();

setPreferredColorScheme('dark');

const TELEOP_CHANNEL_UUID: Uint8Array = v5('teleop_command', v5.DNS, new Uint8Array(16));

type AvailableChannel = CloudXR.Session['availableMessageChannels'][number];

function findChannelByUuid(
  channels: AvailableChannel[],
  targetUuid: Uint8Array
): AvailableChannel | undefined {
  return channels.find(
    ch =>
      ch.uuid.length === targetUuid.length &&
      ch.uuid.every((b: number, i: number) => b === targetUuid[i])
  );
}

const START_TELEOP_COMMAND = {
  type: 'teleop_command',
  message: {
    command: 'start teleop',
  },
} as const;

/** When set with ``serverIP`` + ``port``, WebXR builds ``wss://{serverIP}:{port}/oob/v1/ws``. */
function isOobEnabled(searchParams: URLSearchParams): boolean {
  const v = readUrlParam(searchParams, 'oobEnable');
  return v === '1' || v?.toLowerCase() === 'true';
}

function buildOobHubWsUrlFromQuery(searchParams: URLSearchParams): string | null {
  if (!isOobEnabled(searchParams)) return null;
  const serverIP = readUrlParam(searchParams, 'serverIP')?.trim();
  const portStr = readUrlParam(searchParams, 'port')?.trim();
  if (!serverIP || portStr === undefined || portStr === '') return null;
  if (!/^\d{1,5}$/.test(portStr)) return null;
  const host = serverIP.includes(':') && !serverIP.startsWith('[') ? `[${serverIP}]` : serverIP;
  return `wss://${host}:${portStr}/oob/v1/ws`;
}

function AppContent() {
  const { recorder, onLoadRecording, setReplayPacing } = useRecorder();
  const COUNTDOWN_MAX_SECONDS = 9;
  // 2D UI management
  const [cloudXR2DUI, setCloudXR2DUI] = useState<CloudXR2DUI | null>(null);
  // IWER loading state
  const [iwerLoaded, setIwerLoaded] = useState(false);
  // Capability state management
  const [capabilitiesValid, setCapabilitiesValid] = useState(false);
  const capabilitiesCheckedRef = useRef(false);
  // Connection state management
  const [isConnected, setIsConnected] = useState(false);
  // Session status management
  const [sessionStatus, setSessionStatus] = useState('Disconnected');
  // Error message management
  const [errorMessage, setErrorMessage] = useState('');
  // CloudXR session reference
  const [cloudXRSession, setCloudXRSession] = useState<CloudXR.Session | null>(null);
  // XR mode state for UI visibility
  const [isXRMode, setIsXRMode] = useState(false);
  // Server address being used for connection
  const [serverAddress, setServerAddress] = useState<string>('');
  // Teleop countdown and state
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdownRemaining, setCountdownRemaining] = useState(0);
  const [isTeleopRunning, setIsTeleopRunning] = useState(false);
  const countdownTimerRef = useRef<number | null>(null);
  /** App-owned countdown for the pre-stream network test window. */
  const streamTestCountdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Clear the countdown if the component unmounts mid-test (e.g. the user exits XR).
  useEffect(
    () => () => {
      if (streamTestCountdownRef.current) {
        clearInterval(streamTestCountdownRef.current);
        streamTestCountdownRef.current = null;
      }
    },
    []
  );
  /** Avoid repeating immersive session dumps on every XR store tick. */
  const immersiveSessionDumpLoggedRef = useRef(false);
  const [countdownDuration, setCountdownDuration] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(COUNTDOWN_STORAGE_KEY);
      if (saved != null) {
        const value = parseInt(saved, 10);
        if (!isNaN(value)) {
          return Math.min(COUNTDOWN_MAX_SECONDS, Math.max(0, value));
        }
      }
    } catch (_) {}
    return 3;
  });

  // Persist countdown duration on change
  useEffect(() => {
    try {
      localStorage.setItem(COUNTDOWN_STORAGE_KEY, String(countdownDuration));
    } catch (_) {}
  }, [countdownDuration]);

  // Load IWER first (must happen before anything else)
  // Note: React Three Fiber's emulation is disabled (emulate: false) to avoid conflicts
  useEffect(() => {
    const loadIWER = async () => {
      const { supportsImmersive, iwerLoaded: wasIwerLoaded } = await loadIWERIfNeeded();
      if (!supportsImmersive) {
        setErrorMessage('Immersive mode not supported');
        setIwerLoaded(false);
        setCapabilitiesValid(false);
        capabilitiesCheckedRef.current = false; // Reset check flag on failure
        return;
      }
      // IWER loaded successfully, now we can proceed with capability checks
      setIwerLoaded(true);
      // Store whether IWER was loaded for status message display later
      if (wasIwerLoaded) {
        sessionStorage.setItem('iwerWasLoaded', 'true');
      }
    };

    loadIWER();
  }, []);

  // Update button state when IWER fails and UI becomes ready
  useEffect(() => {
    if (cloudXR2DUI && !iwerLoaded && !capabilitiesValid) {
      cloudXR2DUI.setStartButtonState(true, 'CONNECT (immersive mode not supported)');
    }
  }, [cloudXR2DUI, iwerLoaded, capabilitiesValid]);

  // Check capabilities once CloudXR2DUI is ready and IWER is loaded
  useEffect(() => {
    const checkCapabilitiesOnce = async () => {
      if (!cloudXR2DUI || !iwerLoaded) {
        return;
      }

      // Guard: only check capabilities once
      if (capabilitiesCheckedRef.current) {
        return;
      }
      capabilitiesCheckedRef.current = true;

      // Disable button and show checking status
      cloudXR2DUI.setStartButtonState(true, 'CONNECT (checking capabilities)');

      // Set by the IWER load effect above; passed to checkCapabilities to skip browser
      // version checks that don't apply when running under a desktop XR emulator.
      const iwerWasLoaded = sessionStorage.getItem('iwerWasLoaded') === 'true';
      let result: { success: boolean; failures: string[]; warnings: string[] } = {
        success: false,
        failures: [],
        warnings: [],
      };
      try {
        result = await checkCapabilities(iwerWasLoaded);
      } catch (error) {
        cloudXR2DUI.showStatus(`Capability check error: ${error}`, 'error');
        setCapabilitiesValid(false);
        cloudXR2DUI.setStartButtonState(true, 'CONNECT (capability check failed)');
        capabilitiesCheckedRef.current = false; // Reset on error for potential retry
        return;
      }
      if (!result.success) {
        cloudXR2DUI.showStatus(
          'Browser does not meet required capabilities:\n' + result.failures.join('\n'),
          'error'
        );
        setCapabilitiesValid(false);
        cloudXR2DUI.setStartButtonState(true, 'CONNECT (capability check failed)');
        capabilitiesCheckedRef.current = false; // Reset on failure for potential retry
        return;
      }

      // Show final status message with IWER info if applicable
      if (result.warnings.length > 0) {
        cloudXR2DUI.showStatus('Performance notice:\n' + result.warnings.join('\n'), 'info');
      } else if (iwerWasLoaded) {
        // Include IWER status in the final success message
        cloudXR2DUI.showStatus(
          'CloudXR.js SDK is supported.\nUsing IWER (Immersive Web Emulator Runtime) - Emulating Meta Quest 3.',
          'info'
        );
      } else {
        cloudXR2DUI.showStatus('CloudXR.js SDK is supported.', 'success');
      }

      setCapabilitiesValid(true);
      cloudXR2DUI.setStartButtonState(false, 'CONNECT');
      cloudXR2DUI.updateConnectButtonState();
    };

    checkCapabilitiesOnce();
  }, [cloudXR2DUI, iwerLoaded]);

  // Track config changes to trigger re-renders when form values change
  const [configVersion, setConfigVersion] = useState(0);

  // Derive the active device profile from the UI. This drives XR store defaults.
  // The UI can change these values, so we need to recompute when config changes.
  const deviceProfile = useMemo(
    () => getDeviceProfile(resolveDeviceProfileId(cloudXR2DUI?.getConfiguration().deviceProfileId)),
    [cloudXR2DUI, configVersion]
  );
  const xrFoveation =
    deviceProfile.web?.foveation ?? kPerformanceOptions.xrWebGLLayer_fixedFoveationLevel;
  const xrFrameBufferScaling =
    deviceProfile.web?.frameBufferScaling ??
    kPerformanceOptions.xrWebGLLayer_framebufferScaleFactor;
  const hideControllerModel = cloudXR2DUI?.getConfiguration().hideControllerModel ?? false;

  // XR store must be created after we know which device profile is active.
  // useMemo prevents re-creating the store for unrelated UI changes.
  const store = useMemo(
    () =>
      createXRStore({
        emulate: false, // Disable IWER emulation from react in favor of custom iwer loading function
        foveation: xrFoveation,
        // CloudXRComponent applies the configured rate before CloudXR negotiates the stream.
        // Disabling the store's automatic "high" preference avoids racing that negotiation.
        frameRate: false,
        frameBufferScaling: xrFrameBufferScaling,
        // Use local WebXR input profile assets only when bundled (optional build without assets)
        ...(process.env.WEBXR_ASSETS_VERSION && {
          baseAssetPath: `${new URL('.', window.location.href).href}npm/@webxr-input-profiles/assets@${process.env.WEBXR_ASSETS_VERSION}/dist/profiles/`,
        }),
        hand: {
          model: false, // Disable hand models but keep pointer functionality
        },
        controller: {
          model: !hideControllerModel, // Allow UI to hide controller models while keeping input active
        },
        // Request optional WebXR features - use property names, not optionalFeatures array!
        handTracking: true,
        bodyTracking: true,
        // Explicitly disable environment/scene feature requests to avoid extra headset prompts.
        anchors: false,
        layers: false,
        meshDetection: false,
        planeDetection: false,
        depthSensing: false,
        domOverlay: false,
        hitTest: false,
        // Explicitly enable session offer flows; keep session entry on explicit button action.
        offerSession: true,
      }),
    // hideControllerModel omitted: changing it must not recreate the store or the session would be lost
    [xrFoveation, xrFrameBufferScaling]
  );

  // Apply controller model visibility when the option changes. store.setController() updates
  // at runtime without recreating the store, so the change takes effect immediately (including in XR).
  useEffect(() => {
    store.setController({ model: !hideControllerModel });
  }, [store, hideControllerModel]);

  // Initialize CloudXR2DUI
  useEffect(() => {
    // Create and initialize the 2D UI manager.
    const ui = new CloudXR2DUI(() => {
      setConfigVersion(v => v + 1);
    });
    // Teleop path: URL hash -> last-used (localStorage) -> DEFAULT_TELEOP_PATH.
    let resolvedPath = parseTeleopPathFromHash(window.location.hash);
    if (!resolvedPath) {
      resolvedPath =
        parseTeleopPathFromHash(`#/${loadStoredTeleopPath() ?? ''}`) ?? DEFAULT_TELEOP_PATH;
    }
    // Reflect canonical form (parse may have lowercased/truncated). `#/…` is a
    // fragment-relative URL so replaceState preserves path and search.
    const canonicalHash = `#/${resolvedPath}`;
    if (window.location.hash !== canonicalHash) {
      window.history.replaceState(null, '', canonicalHash);
    }
    saveStoredTeleopPath(resolvedPath);

    // URL query params (URL_PARAMS) are applied inside initialize() and win over stored values.
    ui.initialize(resolvedPath);
    const doConnect = async () => {
      const config = ui.getConfiguration();
      const resolutionError = getResolutionValidationError(config.perEyeWidth, config.perEyeHeight);
      if (resolutionError) {
        ui.updateConnectButtonState();
        return;
      }
      // CloudXR2DUI.updateConfiguration already sets immersiveMode to 'vr' when headless is on.
      // Repeat the rule here so session entry stays correct even if config were stale or built
      // elsewhere; immersive-ar is wrong for headless (no passthrough blit path).
      const immersiveMode: 'ar' | 'vr' = config.headless ? 'vr' : config.immersiveMode;
      if (immersiveMode === 'ar') {
        await store.enterAR();
      } else if (immersiveMode === 'vr') {
        await store.enterVR();
      } else {
        setErrorMessage('Unrecognized immersive mode');
      }
    };

    ui.setupConnectButtonHandler(doConnect, (error: Error) => {
      setErrorMessage(`Failed to start XR session: ${error}`);
    });

    setCloudXR2DUI(ui);

    // Cleanup function
    return () => {
      if (ui) {
        ui.cleanup();
      }
    };
  }, [store]);

  // Address-bar hash edits need a reload to re-run init.
  useEffect(() => {
    const onHashChange = () => window.location.reload();
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    const button = document.getElementById('loadRecordingBtn');
    button?.addEventListener('click', onLoadRecording);
    return () => button?.removeEventListener('click', onLoadRecording);
  }, [onLoadRecording]);

  // Update HTML error message display when error state changes
  useEffect(() => {
    if (cloudXR2DUI) {
      if (errorMessage) {
        cloudXR2DUI.showError(errorMessage);
      } else {
        cloudXR2DUI.hideError();
      }
    }
  }, [errorMessage, cloudXR2DUI]);

  // Listen for XR session state changes to update button and UI visibility
  useEffect(() => {
    const handleXRStateChange = () => {
      const xrState = store.getState();

      if (xrState.mode === 'immersive-ar' || xrState.mode === 'immersive-vr') {
        // XR session is active
        setIsXRMode(true);

        // Check if body tracking is supported in the XR session
        const session = xrState.session;
        if (session) {
          const enabledFeatures = session.enabledFeatures || [];
          const hasBodyTracking = enabledFeatures.includes('body-tracking');
          console.warn(
            `[Body Tracking] XR Session started. Body tracking enabled: ${hasBodyTracking}`
          );
          console.warn(`[Body Tracking] Enabled features: ${enabledFeatures.join(', ')}`);
        }

        // One dump per immersive session: session.mode is authoritative (immersive-vr vs immersive-ar).
        if (session && !immersiveSessionDumpLoggedRef.current) {
          immersiveSessionDumpLoggedRef.current = true;
          logImmersiveXRSessionToConsole(session, xrState.mode);
        }

        if (cloudXR2DUI) {
          cloudXR2DUI.setStartButtonState(true, 'CONNECT (XR session active)');
        }
      } else {
        immersiveSessionDumpLoggedRef.current = false;
        // XR session ended
        setIsXRMode(false);
        if (cloudXR2DUI) {
          cloudXR2DUI.setStartButtonState(false, 'CONNECT');
          cloudXR2DUI.updateConnectButtonState();
        }

        if (xrState.error) {
          setErrorMessage(`XR session error: ${xrState.error}`);
        }
      }
    };

    // Subscribe to XR state changes
    const unsubscribe = store.subscribe(handleXRStateChange);

    // Cleanup
    return () => {
      unsubscribe();
      setIsXRMode(false);
    };
  }, [cloudXR2DUI, store]);

  // Held in a ref so handleStatusChange can forward streaming state without
  // re-rendering when the channel attaches.
  const controlChannelRef = useRef<HeadsetControlChannel | null>(null);

  // Tracks the exact "Reconnecting (n/maxAttempts)" text last shown, so it can be retracted via
  // hideStatusIfShowing once the status moves on - without erasing a different message (e.g. the
  // real error onError shows right after a give-up) that may have already replaced it.
  const lastReconnectingStatusRef = useRef<string | null>(null);

  // Only ``(true, 'Connected')`` corresponds to onStreamStarted; everything
  // else is "not streaming" — pre-stream errors as well as stop/disconnect.
  const handleStatusChange = (connected: boolean, status: string) => {
    setIsConnected(connected);
    setSessionStatus(status);
    controlChannelRef.current?.sendStreamStatus(connected && status === 'Connected');

    // CloudXRComponent reports retry progress through this same status text (see
    // streamingErrorClassification.ts / CloudXRComponent's reconnect handling) rather than a
    // dedicated callback - surface it as an info banner, not an error, and retract it once the
    // status moves past reconnecting (succeeded, or gave up and onError shows the real message).
    if (status.startsWith('Reconnecting')) {
      // A countdown in progress captured the pre-reconnect sendMessage closure; left running,
      // it would fire START_TELEOP_COMMAND through the stopped session instead of the
      // replacement one once the timer elapses.
      if (countdownTimerRef.current !== null) {
        clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
      setIsCountingDown(false);
      setCountdownRemaining(0);
      cloudXR2DUI?.showStatus(status, 'info');
      lastReconnectingStatusRef.current = status;
    } else if (lastReconnectingStatusRef.current) {
      cloudXR2DUI?.hideStatusIfShowing(lastReconnectingStatusRef.current);
      lastReconnectingStatusRef.current = null;
    }

    // Drop the previous session's quality reading rather than leaving a stale green
    // indicator on a dead stream. Safe on every non-Connected status, including the
    // 'Testing network' transition, since quality is only sampled while streaming.
    if (!connected || status !== 'Connected') {
      sessionQuality.value = 0;
    }
    // Reset metrics only on an actual session end. Deliberately not on 'Testing network':
    // that status is emitted by onStreamTestStarted, i.e. on the way *into* a session, and
    // clearing there would discard any samples taken during the measurement window.
    if (status === 'Disconnected' || status === 'Error') {
      metricsAccumulator.reset();
      // Clear the HUD cards too, so a dead session cannot leave its last FPS and latency
      // readings on the panel looking live. The computed texts fall back to '-' on null.
      renderFps.value = null;
      poseSendFps.value = null;
      streamingMetrics.value = null;
    }

    // Reload on session end per mode; read live off the stable 2D UI to avoid a stale closure.
    const autoRefreshMode = cloudXR2DUI?.getConfiguration().autoRefreshMode;
    if (
      (status === 'Disconnected' && (autoRefreshMode === 'clean' || autoRefreshMode === 'any')) ||
      (status === 'Error' && autoRefreshMode === 'any')
    ) {
      window.location.reload();
    }
  };

  // Metrics callbacks. Each payload is partial, so HUD signals are only overwritten for
  // fields actually present, and everything is merged into the accumulator for the hub.
  const handleRenderPerformanceMetrics = (metrics: RenderMetricsUpdate) => {
    if (metrics.framerate !== undefined) renderFps.value = metrics.framerate;
    if (metrics.sendFramerate !== undefined) poseSendFps.value = metrics.sendFramerate;
    metricsAccumulator.recordRender(metrics);
  };

  const handleStreamingPerformanceMetrics = (metrics: FrameMetricsUpdate) => {
    if (metrics.framerate !== undefined || metrics.poseToRenderTimeMs !== undefined) {
      // Carry forward the fields this tick did not carry so the HUD does not flicker.
      const prev = streamingMetrics.value ?? { fps: 0, latencyMs: 0 };
      streamingMetrics.value = {
        fps: metrics.framerate ?? prev.fps,
        latencyMs: metrics.poseToRenderTimeMs ?? prev.latencyMs,
      };
    }
    metricsAccumulator.recordFrame(metrics);
  };

  const handleNetworkPerformanceMetrics = (metrics: NetworkMetricsUpdate) => {
    if (metrics.sessionQuality !== undefined) {
      sessionQuality.value = metrics.sessionQuality;
    }
    metricsAccumulator.recordNetwork(metrics);
  };

  const stopStreamTestCountdown = () => {
    if (streamTestCountdownRef.current) {
      clearInterval(streamTestCountdownRef.current);
      streamTestCountdownRef.current = null;
    }
  };

  // The SDK only signals the start and end of the measurement window, so the countdown
  // is driven here from the configured duration.
  const handleStreamTestStarted = () => {
    stopStreamTestCountdown();
    let remaining = resolveStreamTestSeconds(config?.streamTestDurationSeconds);
    const tick = () => {
      // Hold at 0 until the result arrives rather than counting into negatives.
      if (remaining <= 0) {
        streamTest.value = { text: 'Testing network… 0s', color: '#ffd24d' };
        stopStreamTestCountdown();
        return;
      }
      streamTest.value = { text: `Testing network… ${remaining}s`, color: '#ffd24d' };
      remaining -= 1;
    };
    tick();
    streamTestCountdownRef.current = setInterval(tick, 1000);
  };

  const handleStreamTestStopped = (result: CloudXR.StreamTestResult) => {
    stopStreamTestCountdown();
    // Red matches the gate exactly: red iff the test failed. On a pass, green only when
    // every measured dimension is Good or better, otherwise yellow as a caution.
    const measured = [
      result.latencyScore,
      result.jitterScore,
      result.bandwidthScore,
      result.devicePerformanceScore,
    ].filter(score => score !== CloudXR.QualityScore.NoData);
    let color: string;
    let label: string;
    if (!result.passed) {
      color = '#ff5d5d';
      label = 'failed';
    } else {
      const worst = measured.length ? Math.min(...measured) : CloudXR.QualityScore.NoData;
      color = worst >= CloudXR.QualityScore.Good ? '#5dd35d' : '#ffd24d';
      label = 'passed';
    }
    // QualityScore is a numeric enum; reverse-map to the name so the panel reads as a
    // rating ("Good") rather than an ambiguous number.
    const scoreName = (score: CloudXR.QualityScore) => CloudXR.QualityScore[score];
    streamTest.value = {
      text: `Network test ${label} — latency: ${scoreName(result.latencyScore)}, jitter: ${scoreName(result.jitterScore)}, ${result.serverFps ?? '-'} fps`,
      color,
    };
  };

  // Clear the network-test indicator on both a new session and a teardown.
  // onStreamTestStopped does not fire when the user disconnects mid-test (an abort is not
  // a result), so this is what stops a countdown that would otherwise keep ticking.
  const handleSessionReady = (session: CloudXR.Session | null) => {
    stopStreamTestCountdown();
    streamTest.value = null;
    setCloudXRSession(session);
  };

  const systemNoticeTimerRef = useRef<number | null>(null);
  // Signal writes deliberately bypass React, so the banner's *text* updates
  // without a re-render -- but its presence must be React state, or mounting and
  // unmounting it would never happen.
  const [systemNoticeVisible, setSystemNoticeVisible] = useState(false);
  // Level drives the XR banner palette. Kept as React state next to the
  // visibility flag rather than read off the signal, because the banner picks
  // static colors at render time rather than subscribing.
  const [systemNoticeLevel, setSystemNoticeLevel] = useState<'warning' | 'info'>('warning');

  /** Exact text this component last wrote to the shared 2D status box. */
  const systemNoticeTextRef = useRef<string | null>(null);

  /** Take the notice down, in both surfaces, and cancel any pending auto-dismiss. */
  const dismissSystemNotice = () => {
    systemNotice.value = null;
    setSystemNoticeVisible(false);
    // showStatus() sets .show and never clears itself, so the 2D box needs an
    // explicit retraction. Clear it only while it still holds our notice: the
    // box is shared, and handleDisconnect() runs this on the way out of a
    // failed session -- moments after CloudXR reported the failure into that
    // same box. Blanking unconditionally erased that error, and because it is
    // reported imperatively there is no React state to bring it back.
    if (systemNoticeTextRef.current !== null) {
      cloudXR2DUI?.hideStatusIfShowing(systemNoticeTextRef.current);
      systemNoticeTextRef.current = null;
    }
    if (systemNoticeTimerRef.current !== null) {
      clearTimeout(systemNoticeTimerRef.current);
      systemNoticeTimerRef.current = null;
    }
  };

  // [4] Cancel a pending auto-dismiss if the app tears down first; otherwise the
  // timeout fires setSystemNoticeVisible on an unmounted component.
  useEffect(
    () => () => {
      if (systemNoticeTimerRef.current !== null) {
        clearTimeout(systemNoticeTimerRef.current);
      }
    },
    []
  );

  /**
   * Dispatch a message received from the server on the teleop channel.
   *
   * Unknown `type` values are logged and ignored rather than treated as errors:
   * the host and this client are versioned independently, so a newer host may
   * send message kinds this build does not know about.
   */
  const handleServerMessage = (message: unknown) => {
    if (isSystemNoticeMessage(message)) {
      const notice = message.message;
      // No unmet requirements: treat it as an all-clear so a host can retract a
      // notice it raised earlier, rather than leaving a stale banner up.
      if (notice.items.length === 0) {
        dismissSystemNotice();
        return;
      }

      systemNotice.value = notice;
      setSystemNoticeVisible(true);
      setSystemNoticeLevel(notice.level);
      // Restart the countdown so a second notice gets its full dwell time.
      if (systemNoticeTimerRef.current !== null) {
        clearTimeout(systemNoticeTimerRef.current);
      }
      systemNoticeTimerRef.current = window.setTimeout(() => {
        systemNoticeTimerRef.current = null;
        dismissSystemNotice();
      }, SYSTEM_NOTICE_AUTO_DISMISS_MS);

      // Mirror to the 2D banner so the notice is visible when testing from a
      // desktop browser, where the in-XR panel never renders. Remember the exact
      // text so dismissal can retract it without clobbering a later message.
      const mirroredText = formatSystemNotice(notice);
      systemNoticeTextRef.current = mirroredText;
      cloudXR2DUI?.showStatus(mirroredText, notice.level === 'warning' ? 'error' : 'info');
      return;
    }

    const type = (message as { type?: unknown })?.type;
    console.info(`Ignoring server message of unhandled type: ${String(type)}`);
  };

  // The receive loop below is created once per session and lives for its whole
  // duration, so it must not capture this handler directly -- that would pin the
  // first render's `cloudXR2DUI`. Route through a ref that each render refreshes.
  const handleServerMessageRef = useRef(handleServerMessage);
  handleServerMessageRef.current = handleServerMessage;

  /**
   * Helper to send a message using MessageChannel API (new) or legacy API (fallback).
   * Looks for the teleop_command channel by UUID, then falls back to legacy API.
   */
  const sendMessage = async (message: any) => {
    if (!cloudXRSession) {
      console.error('CloudXR session not available');
      return false;
    }

    // Try new MessageChannel API first - find the teleop channel by UUID
    const channels = cloudXRSession.availableMessageChannels;
    const channel = findChannelByUuid(channels, TELEOP_CHANNEL_UUID);
    if (channel) {
      console.info(`Using teleop MessageChannel (${channels.length} channel(s) available)`);

      try {
        const encoder = new TextEncoder();
        const data = encoder.encode(JSON.stringify(message));
        const success = channel.sendServerMessage(data);
        if (success) {
          console.info('Message sent via MessageChannel:', message);
        } else {
          console.error('Failed to send message via MessageChannel');
        }
        return success;
      } catch (error) {
        console.error('Error sending via MessageChannel:', error);
        return false;
      }
    }

    // Fallback to legacy API
    console.info('Using legacy sendServerMessage API');
    try {
      cloudXRSession.sendServerMessage(message);
      console.info('Message sent via legacy API:', message);
      return true;
    } catch (error) {
      console.error('Error sending via legacy API:', error);
      return false;
    }
  };

  // UI Event Handlers
  const handleStartTeleop = async () => {
    console.info('Start Teleop pressed');

    if (!cloudXRSession) {
      console.error('CloudXR session not available');
      return;
    }

    if (isCountingDown || isTeleopRunning) {
      return;
    }

    // Begin countdown before starting teleop (immediately if 0)
    if (countdownDuration <= 0) {
      setIsCountingDown(false);
      setCountdownRemaining(0);

      const success = await sendMessage(START_TELEOP_COMMAND);
      if (success) {
        setIsTeleopRunning(true);
      } else {
        setIsTeleopRunning(false);
      }
      return;
    }

    setIsCountingDown(true);
    setCountdownRemaining(countdownDuration);

    countdownTimerRef.current = window.setInterval(() => {
      setCountdownRemaining(prev => {
        if (prev <= 1) {
          // Countdown finished
          if (countdownTimerRef.current !== null) {
            clearInterval(countdownTimerRef.current);
            countdownTimerRef.current = null;
          }
          setIsCountingDown(false);

          // Send start teleop command
          sendMessage(START_TELEOP_COMMAND).then(success => {
            if (success) {
              setIsTeleopRunning(true);
            } else {
              setIsTeleopRunning(false);
            }
          });

          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleResetTeleop = async () => {
    console.info('Reset Teleop pressed');

    // Cancel any active countdown
    if (countdownTimerRef.current !== null) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    setIsCountingDown(false);
    setCountdownRemaining(0);

    if (!cloudXRSession) {
      console.error('CloudXR session not available');
      return;
    }

    // Send stop teleop command first
    const stopCommand = {
      type: 'teleop_command',
      message: {
        command: 'stop teleop',
      },
    };

    // Send reset teleop command
    const resetCommand = {
      type: 'teleop_command',
      message: {
        command: 'reset teleop',
      },
    };

    const stopSuccess = await sendMessage(stopCommand);
    if (stopSuccess) {
      const resetSuccess = await sendMessage(resetCommand);
      if (resetSuccess) {
        setIsTeleopRunning(false);
      }
    }
  };

  const handleDisconnect = () => {
    console.info('Disconnect pressed');

    // Cleanup countdown state on disconnect
    if (countdownTimerRef.current !== null) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    setIsCountingDown(false);
    setCountdownRemaining(0);
    setIsTeleopRunning(false);
    // The notice describes the host we are leaving; it must not persist into a
    // later connection to a different one.
    dismissSystemNotice();

    // Close message channels before ending XR session to avoid
    // "Cannot send control message" errors during SDK cleanup.
    if (cloudXRSession) {
      for (const ch of cloudXRSession.availableMessageChannels) {
        ch.disconnect();
      }
    }

    // Auto-refresh is handled centrally in handleStatusChange on the resulting stream-stop.
    const xrState = store.getState();
    const session = xrState.session;
    if (session) {
      session.end().catch((err: unknown) => {
        setErrorMessage(
          `Failed to end XR session: ${err instanceof Error ? err.message : String(err)}`
        );
      });
    }
  };

  // OOB WebSocket: only when oobEnable=1 and query has valid serverIP + port → wss://{serverIP}:{port}/oob/v1/ws.
  useEffect(() => {
    if (!cloudXR2DUI) return;
    const p = new URLSearchParams(window.location.search);
    const hubWsUrl = buildOobHubWsUrlFromQuery(p);
    if (!hubWsUrl) {
      return;
    }

    console.info('[Teleop] OOB control WebSocket:', hubWsUrl);

    const channel = new HeadsetControlChannel({
      url: hubWsUrl,
      token: readUrlParam(p, 'controlToken') ?? undefined,
      onConfig: () => {
        // Config push handling deferred to phase 2.
      },
      // Reports every metric the SDK emits, keyed by CloudXR.MetricsName, across the
      // render / frame / network cadences. The hub stores metrics as an arbitrary
      // {name: value} map per cadence, so new SDK metrics need no hub-side change.
      getMetricsSnapshot: () => metricsAccumulator.takeSnapshot(),
    });
    channel.connect();
    controlChannelRef.current = channel;

    return () => {
      controlChannelRef.current = null;
      channel.dispose();
    };
  }, [cloudXR2DUI]);

  // Countdown configuration handlers (0-5 seconds)
  const handleIncreaseCountdown = () => {
    if (isCountingDown) return;
    setCountdownDuration(prev => Math.min(COUNTDOWN_MAX_SECONDS, prev + 1));
  };

  const handleDecreaseCountdown = () => {
    if (isCountingDown) return;
    setCountdownDuration(prev => Math.max(0, prev - 1));
  };

  // Memo config based on configVersion (manual dependency tracker incremented on config changes)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const config = useMemo(
    () => (cloudXR2DUI ? cloudXR2DUI.getConfiguration() : null),
    [cloudXR2DUI, configVersion]
  );

  useEffect(() => {
    setReplayPacing(config?.replayPacing ?? 'time');
  }, [config?.replayPacing, setReplayPacing]);

  // Build ICE server config from URL params (set in USB-local mode by oob_teleop_env.py).
  // turnServer e.g. "turn:127.0.0.1:3478?transport=tcp", iceRelayOnly=1 forces relay-only ICE.
  const iceServersConfig = useMemo<CloudXR.SessionOptions['iceServers'] | undefined>(() => {
    const p = new URLSearchParams(window.location.search);
    const turnServer = readUrlParam(p, 'turnServer');
    if (!turnServer) return undefined;
    const turnUsername = readUrlParam(p, 'turnUsername') ?? undefined;
    const turnCredential = readUrlParam(p, 'turnCredential') ?? undefined;
    const iceRelayOnly = readUrlParam(p, 'iceRelayOnly') === '1';
    return {
      iceServers: [
        {
          urls: turnServer,
          ...(turnUsername !== undefined && { username: turnUsername }),
          ...(turnCredential !== undefined && { credential: turnCredential }),
        },
      ],
      ...(iceRelayOnly && { iceTransportPolicy: 'relay' as RTCIceTransportPolicy }),
    };
  }, []);

  // Calculate panel position from config and memoize it as the vector used in CloudXR3DUI.
  const controlPanelPositionVector = useMemo(
    () =>
      getControlPanelPositionVector(config?.controlPanelPosition ?? 'center', CONTROL_PANEL_LAYOUT),
    [config?.controlPanelPosition]
  );

  // Sync XR mode state to body class for CSS styling
  useEffect(() => {
    if (isXRMode) {
      document.body.classList.add('xr-mode');
    } else {
      document.body.classList.remove('xr-mode');
    }

    return () => {
      document.body.classList.remove('xr-mode');
    };
  }, [isXRMode]);

  // Set up message receiving from MessageChannel (new API) or legacy callback
  // Poll for channel availability since channels can be announced at any time
  useEffect(() => {
    // Wait for Connected, not just a non-null session: opening a channel sends a control
    // message that requires a connected session. With the pre-stream network test enabled
    // the session sits in Connecting for the whole test window, and opening the teleop
    // channel there would fail.
    if (!cloudXRSession || !isConnected) {
      return;
    }

    let active = true;
    let receiverActive = false;

    const checkAndSetupReceiver = () => {
      if (!active || receiverActive) return;

      const channels = cloudXRSession.availableMessageChannels;
      if (channels.length > 0) {
        console.info(`[MessageChannel] ${channels.length} channel(s) available:`);
        channels.forEach((ch, i) => {
          const uuidHex = Array.from(ch.uuid as Uint8Array)
            .map((b: number) => b.toString(16).padStart(2, '0'))
            .join('');
          console.info(`  [${i}] uuid=${uuidHex} status=${ch.status}`);
        });

        const channel = findChannelByUuid(channels, TELEOP_CHANNEL_UUID);
        if (!channel) {
          console.info('[MessageChannel] Teleop channel not found yet, will retry...');
          return;
        }
        console.info('[MessageChannel] Found teleop channel, setting up receiver');
        receiverActive = true;

        const receiveMessages = async () => {
          while (active) {
            try {
              const data = await channel.receiveMessage();
              if (data === null) {
                console.info('MessageChannel closed');
                break;
              }

              // Decode and handle message
              const decoder = new TextDecoder();
              const messageText = decoder.decode(data);
              console.info('Received message via MessageChannel:', messageText);

              // Parse if JSON
              try {
                const message = JSON.parse(messageText);
                console.info('Parsed message:', message);
                handleServerMessageRef.current(message);
              } catch {
                console.info('Non-JSON message:', messageText);
              }
            } catch (error) {
              console.error('Error receiving message:', error);
              break;
            }
          }
        };

        receiveMessages().finally(() => {
          receiverActive = false;
        });
      }
    };

    // Check immediately
    checkAndSetupReceiver();

    // Poll every 1 second to check if channels become available
    const pollInterval = setInterval(checkAndSetupReceiver, 1000);

    return () => {
      active = false;
      clearInterval(pollInterval);
    };
  }, [cloudXRSession, isConnected]);

  return (
    <>
      <Canvas
        events={noEvents}
        style={{
          background: '#000',
          width: '100vw',
          height: '100vh',
          position: 'fixed',
          top: 0,
          left: 0,
          zIndex: -1,
        }}
        gl={{
          alpha: true, // R3F default, but being explicit
          depth: true,
          stencil: false,
          antialias:
            deviceProfile.web?.webglAntialias ?? kPerformanceOptions.webglContext_antialias,
          failIfMajorPerformanceCaveat: true,
          powerPreference: deviceProfile.web?.powerPreference ?? 'high-performance', // R3F default, but being explicit
          premultipliedAlpha: false,
          preserveDrawingBuffer: true, // Keep buffer for custom rendering
        }}
        camera={{ position: [0, 0, 0.65] }}
        onWheel={e => {
          e.preventDefault();
        }}
      >
        <SuppressWebGLRendererWhenHeadless headless={!!config?.headless} />
        <PointerEvents batchEvents={false} />
        <XR store={store}>
          <SimpleEnvironment />
          <XROrigin />
          {cloudXR2DUI && config && (
            <>
              <RecorderComponent isConnected={isConnected} showTrace={config.showTrace ?? false} />
              <TraceVisualization showTrace={config.showTrace ?? false} />
              <CloudXRComponent
                config={config}
                applicationName={`Isaac Teleop Web Client (${config.teleopPath})`}
                trackingFrameAdapter={recorder.adaptTrackingFrame}
                iceServers={iceServersConfig}
                onStatusChange={handleStatusChange}
                onError={error => {
                  if (cloudXR2DUI) {
                    cloudXR2DUI.showError(error);
                  }
                }}
                onExitImmersiveXR={handleDisconnect}
                onSessionReady={handleSessionReady}
                onServerAddress={setServerAddress}
                onRenderPerformanceMetrics={handleRenderPerformanceMetrics}
                onStreamingPerformanceMetrics={handleStreamingPerformanceMetrics}
                onNetworkPerformanceMetrics={handleNetworkPerformanceMetrics}
                streamTest={
                  config.streamTestMode && config.streamTestMode !== 'off'
                    ? {
                        durationSeconds: resolveStreamTestSeconds(config.streamTestDurationSeconds),
                        mode: config.streamTestMode,
                      }
                    : undefined
                }
                onStreamTestStarted={handleStreamTestStarted}
                onStreamTestStopped={handleStreamTestStopped}
                headless={!!config.headless}
              />
              {!config.headless && (
                <CloudXR3DUI
                  onStartTeleop={handleStartTeleop}
                  onDisconnect={handleDisconnect}
                  onResetTeleop={handleResetTeleop}
                  isXRMode={isXRMode}
                  panelHiddenAtStart={config.panelHiddenAtStart ?? false}
                  serverAddress={serverAddress || config.serverIP}
                  sessionStatus={sessionStatus}
                  playLabel={
                    isTeleopRunning
                      ? 'Running'
                      : isCountingDown
                        ? `Starting in ${countdownRemaining} sec...`
                        : 'Play'
                  }
                  playInProgress={isCountingDown || isTeleopRunning}
                  countdownSeconds={countdownDuration}
                  onCountdownIncrease={handleIncreaseCountdown}
                  onCountdownDecrease={handleDecreaseCountdown}
                  countdownDisabled={isCountingDown}
                  position={controlPanelPositionVector}
                  rotation={[0, 0, 0]}
                  renderFpsText={renderFpsText}
                  poseSendFpsText={poseSendFpsText}
                  streamingFpsText={streamingFpsText}
                  poseToRenderText={poseToRenderText}
                  sessionQuality={sessionQuality}
                  streamTestText={streamTestText}
                  streamTestColor={streamTestColor}
                  showRecordingControls={config.showRecordingControls}
                  systemNoticeTitleText={systemNoticeTitleText}
                  systemNoticeBodyText={systemNoticeBodyText}
                  systemNoticeVisible={systemNoticeVisible}
                  systemNoticeLevel={systemNoticeLevel}
                  onDismissSystemNotice={dismissSystemNotice}
                />
              )}
            </>
          )}
        </XR>
      </Canvas>
    </>
  );
}

function App() {
  return (
    <RecorderProvider>
      <AppContent />
    </RecorderProvider>
  );
}

export default App;
