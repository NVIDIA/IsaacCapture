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
 * CloudXRComponent.tsx - CloudXR WebXR Integration Component
 *
 * This component handles the core CloudXR streaming functionality and WebXR integration.
 * It manages:
 * - CloudXR session lifecycle (creation, connection, disconnection, cleanup)
 * - WebXR session event handling (sessionstart, sessionend)
 * - WebGL state management and render target preservation
 * - Frame-by-frame rendering loop with pose tracking and stream rendering
 * - Server configuration and connection parameters
 * - Status reporting back to parent components
 *
 * The component accepts configuration via props and communicates status changes
 * and disconnect requests through callback props. It integrates with Three.js
 * and React Three Fiber for WebXR rendering while preserving WebGL state
 * for CloudXR's custom rendering pipeline.
 */

import * as CloudXR from '@nvidia/cloudxr';
import { useFrame, useThree } from '@react-three/fiber';
import { useXR } from '@react-three/xr';
import { useCallback, useEffect, useRef } from 'react';
import type { WebGLRenderer } from 'three';
import { Color } from 'three';

import { MetricsTracker } from '@helpers/Metrics';
import type {
  FrameMetricsUpdate,
  NetworkMetricsUpdate,
  RenderMetricsUpdate,
} from '@helpers/metricsUpdates';
import { isRecoverable } from '@helpers/streamingErrorClassification';
import { CloudXRConfig, ConnectionConfiguration, getConnectionConfig } from '@helpers/utils';
import { bindGL } from '@helpers/WebGLStateBinding';
import { clearPendingGLErrors } from '@helpers/WebGlUtils';

import { applyTargetFrameRate } from '../../src/config/frameRate';

/** Clear color shown in headless mode so it's visually obvious the client is running with rendering suppressed. */
const HEADLESS_CLEAR_COLOR = 0x00194d;

// Matches the SDK's own onLog message reporting decoder warm-up progress (server encoder
// priming before any pose-backed frame is renderable), periodically while it's in progress.
// Parsed from onLog rather than a dedicated delegate because the SDK doesn't expose warm-up
// progress any other way.
const WARMUP_PROGRESS_LOG_PATTERN =
  /^Decoder warm-up in progress: (\d+) initialization frames received$/;
// Matches the SDK's own onLog message reporting that decoder warm-up has finished and the first
// pose-backed frame has rendered, emitted exactly once, with count >= 1.
const WARMUP_COMPLETE_LOG_PATTERN =
  /^Stream rendering started after (\d+) decoder warm-up frames?$/;

// Default for the streamAttachTimeoutMs prop below - see its doc comment. Deliberately generous:
// a real CloudXR server/network can legitimately take much longer than a mock ever would to
// attach a stream, and a false positive here means an otherwise-fine session gets torn down and
// retried for no reason. Callers who want faster passthrough-only detection (e.g. tests) should
// override via the streamAttachTimeoutMs prop, not by lowering this default.
const STREAM_ATTACH_BASE_TIMEOUT_MS = 120000; // 2 minutes

// Defaults for the warmupBeginTimeoutMs/warmupEndTimeoutMs props below. Both shorter than
// STREAM_ATTACH_BASE_TIMEOUT_MS and, unlike it, neither grows per reconnect attempt (see
// armWarmupBeginTimer/armWarmupEndTimer): once the stream has attached at all, decoder warm-up
// is a fast, bounded startup step (server encoder priming), not something that legitimately
// takes longer on a slow network the way the initial attach can. Split into two deadlines
// because "no warm-up signal at all since attaching" and "warm-up started but never finishes"
// are different failure modes worth distinguishing in the synthetic error message.
const WARMUP_BEGIN_BASE_TIMEOUT_MS = 10000; // 10 seconds - first warm-up log since onStreamStarted
const WARMUP_END_BASE_TIMEOUT_MS = 30000; // 30 seconds - completion since warm-up began

// setTimeout's delay is stored as a 32-bit signed int internally; a value above this overflows
// and fires (almost) immediately instead of waiting - the opposite of what a long configured
// timeout is asking for. Both the base value (user/URL-configurable, no upper bound today) and
// its doubled-per-attempt growth need to stay under this.
const MAX_SET_TIMEOUT_MS = 2147483647;

/**
 * Decoder warm-up status for the current connection attempt, derived by parsing the SDK's own
 * `onLog` messages (see WARMUP_PROGRESS_LOG_PATTERN / WARMUP_COMPLETE_LOG_PATTERN) - the SDK
 * doesn't expose this as a dedicated delegate. Reset to framesSeen: 0 at the start of each
 * connection attempt (including retries), since warm-up is a per-attempt counter in the SDK too.
 */
export interface WarmupStatus {
  /** Decoder warm-up (RTP-timestamp-0) frames received so far this connection attempt. */
  framesSeen: number;
  /** `Date.now()` when this status last changed. */
  timestamp: number;
  /** True once the first pose-backed frame has rendered after warm-up. */
  completed: boolean;
}

/**
 * Props for the CloudXRComponent.
 */
interface CloudXRComponentProps {
  /** CloudXR configuration including server address, resolution, and XR settings. */
  config: CloudXRConfig;

  /** Application name used for telemetry. */
  applicationName: string;

  /** Callback fired when connection status changes. Receives connection state and human-readable status message. */
  onStatusChange?: (isConnected: boolean, status: string) => void;

  /** Callback fired when an error occurs. Receives error message string. */
  onError?: (error: string) => void;

  /**
   * Called when CloudXR fails to connect or streaming stops with an error.
   * Use this to end the immersive WebXR session (same as the user pressing Disconnect).
   */
  onExitImmersiveXR?: () => void;

  /** Callback fired when CloudXR session is created or destroyed. Receives session instance or null. */
  onSessionReady?: (session: CloudXR.Session | null) => void;

  /**
   * Opt-in bounded retry policy for a mid-stream error (see streamingErrorClassification.ts's
   * isRecoverable() for which errors qualify). Omit entirely to keep today's behavior (onError +
   * onExitImmersiveXR on any stream error, no retry) - retry only ever runs when this prop is
   * provided, even as `{}`. When provided, defaults: 3 attempts, 3000ms delay - matching
   * controlChannel.ts's HeadsetControlChannel reconnect, which this mirrors, including its
   * single-callback shape: retry progress is reported through onStatusChange (status text
   * "Reconnecting (n/maxAttempts)"), the same channel HeadsetControlChannel's onConnectionChange
   * uses, rather than a dedicated callback. An unrecoverable error, or exhausting maxAttempts,
   * falls back to the no-retry behavior: onError + onExitImmersiveXR.
   */
  reconnect?: { maxAttempts?: number; delayMs?: number };

  /**
   * Base timeout (ms) for detecting "passthrough-only" - a session that entered XR and called
   * connect() but whose stream never attached (see the stream-attach timer in establishSession).
   * Independent of `reconnect`: detection always runs, with or without that prop, so a stream
   * that never attaches is never silently left hanging even with no reconnect configured at all -
   * only whether the resulting synthetic error gets retried depends on `reconnect`. Doubles on
   * every reconnect attempt (this value, then x2, then x4, ...) so a connection that's genuinely
   * just slow, not stuck, gets more time on each retry instead of being cut off at the same fixed
   * threshold every attempt. Defaults to STREAM_ATTACH_BASE_TIMEOUT_MS (2 minutes).
   */
  streamAttachTimeoutMs?: number;

  /**
   * Timeout (ms) for detecting a stream that attached (onStreamStarted fired) but produced no
   * decoder warm-up signal at all - not even a single WARMUP_PROGRESS_LOG_PATTERN/
   * WARMUP_COMPLETE_LOG_PATTERN log (see armWarmupBeginTimer, armed in onStreamStarted).
   * A separate deadline from streamAttachTimeoutMs because it covers a later, disjoint window:
   * onStreamStarted already cleared the attach timer by the time this one is armed. Distinct
   * from warmupEndTimeoutMs because "nothing happened since attaching" and "warm-up started but
   * never finishes" are different failure modes worth distinguishing in the synthetic error
   * message. Like warmupEndTimeoutMs, does NOT grow per reconnect attempt - warm-up is a fast,
   * bounded server-side priming step once the stream has attached at all, not something that
   * legitimately takes longer under retry the way the initial attach can. Defaults to
   * WARMUP_BEGIN_BASE_TIMEOUT_MS (10 seconds).
   */
  warmupBeginTimeoutMs?: number;

  /**
   * Timeout (ms) for detecting decoder warm-up that began (at least one warm-up log arrived)
   * but never completed - no pose-backed frame ever rendered (see armWarmupEndTimer, armed once
   * the first warm-up log arrives and cleared by WARMUP_COMPLETE_LOG_PATTERN). See
   * warmupBeginTimeoutMs's doc comment for why this is a separate deadline. Does NOT grow per
   * reconnect attempt, for the same reason warmupBeginTimeoutMs doesn't. Defaults to
   * WARMUP_END_BASE_TIMEOUT_MS (30 seconds).
   */
  warmupEndTimeoutMs?: number;

  /** Callback fired with the resolved server address after proxy configuration is applied. */
  onServerAddress?: (address: string) => void;

  /**
   * Callback fired with render performance metrics ({@link CloudXR.MetricsCadence.PerRender}).
   * Payloads are partial - only the fields sampled on this tick are present.
   */
  onRenderPerformanceMetrics?: (metrics: RenderMetricsUpdate) => void;

  /**
   * Callback fired with streaming performance metrics ({@link CloudXR.MetricsCadence.PerFrame}).
   * Payloads are partial - the render and decode paths each report their own subset.
   */
  onStreamingPerformanceMetrics?: (metrics: FrameMetricsUpdate) => void;

  /**
   * Callback fired with network performance metrics ({@link CloudXR.MetricsCadence.PerNetwork}),
   * including session quality. Payloads are partial.
   */
  onNetworkPerformanceMetrics?: (metrics: NetworkMetricsUpdate) => void;

  /**
   * Callback fired with batched SDK log entries, in addition to the browser console.
   * Entries are already mirrored to the console; use this only to route them elsewhere.
   */
  onLog?: (entries: CloudXR.LogEntry[]) => void;

  /**
   * Callback fired whenever decoder warm-up status changes: on each warm-up progress log
   * (roughly every 10 warm-up frames) and once more when warm-up completes (first pose-backed
   * frame rendered). Parsed from SDK log messages - see {@link WarmupStatus}.
   */
  onWarmupStatus?: (status: WarmupStatus) => void;

  /**
   * Pre-stream network test. When set, the SDK measures link quality before streaming
   * starts. Leave undefined to skip the test entirely (the IsaacTeleop default): a teleop
   * operator connecting to a robot should not be held behind a measurement window.
   */
  streamTest?: { durationSeconds: number; mode: 'off' | 'warn' | 'block' };

  /** Callback fired once when the network test's measurement window begins. */
  onStreamTestStarted?: () => void;

  /** Callback fired when the network test finishes, with its result. */
  onStreamTestStopped?: (result: CloudXR.StreamTestResult) => void;

  /**
   * Settings for the performance metrics reported via onRenderPerformanceMetrics and onStreamingPerformanceMetrics callbacks.
   * Each window size controls how many samples are averaged before reporting.
   */
  metricsSettings?: {
    /** Window size for render FPS rolling average. Default: 100 */
    renderFpsWindow?: number;
    /** Window size for streaming FPS rolling average. Default: 20 */
    streamingFpsWindow?: number;
    /** Window size for pose-to-render latency rolling average. Default: 20 */
    poseToRenderWindow?: number;
  };

  /** When true, skip WebGL rendering. */
  headless?: boolean;

  /**
   * Optionally adapt the frame used for tracking submission. The original
   * frame is always retained for client-side rendering and reprojection.
   */
  trackingFrameAdapter?: (frame: XRFrame) => XRFrame;

  /**
   * Custom ICE server configuration (STUN/TURN) for WebRTC.
   * In USB-local mode, set this to a TURN server accessible via adb reverse
   * with iceTransportPolicy 'relay' so WebRTC can gather candidates without WiFi.
   */
  iceServers?: CloudXR.SessionOptions['iceServers'];
}

// React component that integrates CloudXR with Three.js/WebXR
// This component handles the CloudXR session lifecycle and render loop
export default function CloudXRComponent({
  config,
  applicationName,
  onStatusChange,
  onError,
  onExitImmersiveXR,
  onSessionReady,
  reconnect,
  streamAttachTimeoutMs,
  warmupBeginTimeoutMs,
  warmupEndTimeoutMs,
  onServerAddress,
  onRenderPerformanceMetrics,
  onStreamingPerformanceMetrics,
  onNetworkPerformanceMetrics,
  onLog,
  onWarmupStatus,
  streamTest,
  onStreamTestStarted,
  onStreamTestStopped,
  metricsSettings = {},
  headless = false,
  trackingFrameAdapter,
  iceServers,
}: CloudXRComponentProps) {
  const threeRenderer: WebGLRenderer = useThree().gl;
  const { session } = useXR();
  // React reference to the CloudXR session that persists across re-renders.
  const cxrSessionRef = useRef<CloudXR.Session | null>(null);
  const onExitImmersiveXRRef = useRef(onExitImmersiveXR);
  onExitImmersiveXRRef.current = onExitImmersiveXR;

  // No `reconnect` prop at all means opt-out (0 attempts), not "use the defaults" - only a
  // caller that explicitly asks for retry (even as `reconnect={{}}`) gets it.
  const maxReconnectAttempts = reconnect ? (reconnect.maxAttempts ?? 3) : 0;
  const reconnectDelayMs = reconnect?.delayMs ?? 3000;
  const streamAttachBaseTimeoutMs = streamAttachTimeoutMs ?? STREAM_ATTACH_BASE_TIMEOUT_MS;
  const warmupBeginBaseTimeoutMs = warmupBeginTimeoutMs ?? WARMUP_BEGIN_BASE_TIMEOUT_MS;
  const warmupEndBaseTimeoutMs = warmupEndTimeoutMs ?? WARMUP_END_BASE_TIMEOUT_MS;
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamAttachTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const warmupBeginTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const warmupEndTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // False from session creation until a real onStreamStarted fires; lets the effect below tell
  // "still waiting to attach" apart from "already connected" when it re-registers mid-connection
  // (see the config-change re-arm logic further down).
  const hasStreamStartedRef = useRef(false);
  // The delegates object currently wired to cxrSessionRef.current - only ever reassigned inside
  // handleSessionStart, but read from outside its closure (the config-change re-arm check below)
  // to redispatch through the exact same callbacks a real stream event would use.
  const cloudXRDelegatesRef = useRef<CloudXR.SessionDelegates | null>(null);
  // Reset at the top of establishSession() (each connection attempt, including retries) since
  // the SDK's own warm-up frame counter is per-attempt.
  const warmupStatusRef = useRef<WarmupStatus>({ framesSeen: 0, timestamp: 0, completed: false });
  // True once at least one warm-up log (progress or completion) has arrived this attempt - lets
  // the effect's re-arm logic below tell "waiting for warm-up to begin" apart from "warm-up
  // began, waiting for it to end".
  const warmupBegunRef = useRef(false);

  /**
   * Arms (or re-arms) the passthrough-only stream-attach timer for *cxrSession*: if
   * onStreamStarted hasn't fired by the deadline, disconnects the session (so it can never later
   * fire a stray onStreamStarted reporting a false Connected once we've moved on to a retry) and
   * dispatches a synthetic recoverable StreamingError through onStreamStopped, so it gets the
   * same bounded-retry treatment a real stream error does. Deadline doubles per reconnect attempt
   * and is capped at MAX_SET_TIMEOUT_MS (setTimeout's own 32-bit-int delay limit - above it, the
   * browser fires (almost) immediately instead of waiting, the opposite of what a long configured
   * timeout is asking for).
   */
  const armStreamAttachTimer = useCallback(
    (cxrSession: CloudXR.Session) => {
      const attemptForThisTimer = reconnectAttemptRef.current;
      const attachTimeoutMs = Math.min(
        streamAttachBaseTimeoutMs * 2 ** attemptForThisTimer,
        MAX_SET_TIMEOUT_MS
      );
      streamAttachTimerRef.current = setTimeout(() => {
        streamAttachTimerRef.current = null;
        console.warn(
          `CloudXR stream did not attach within ${attachTimeoutMs}ms ` +
            `(attempt ${attemptForThisTimer + 1})`
        );
        try {
          cxrSession.disconnect();
        } catch {
          // Ignore errors from disconnect() - best effort, matching the connect()-catch block.
        }
        cloudXRDelegatesRef.current?.onStreamStopped?.({
          message: `Stream did not attach within ${attachTimeoutMs}ms`,
        } as CloudXR.StreamingError);
      }, attachTimeoutMs);
    },
    [streamAttachBaseTimeoutMs]
  );

  /**
   * Arms the decoder-warm-up-begin timer for *cxrSession*: if no warm-up log (progress or
   * completion) has arrived by the deadline, disconnects the session and dispatches a synthetic
   * recoverable StreamingError through onStreamStopped, the same treatment armStreamAttachTimer
   * gives a stuck attach. Called from onStreamStarted, i.e. after the attach timer has already
   * been cleared - this covers the next, disjoint window. Cleared (see the onLog handler below)
   * as soon as any warm-up log arrives, at which point armWarmupEndTimer takes over. Like
   * armWarmupEndTimer, the deadline is fixed: it does not grow per reconnect attempt, since a
   * stream that has already attached should start warming up in roughly constant time regardless
   * of which attempt this is.
   */
  const armWarmupBeginTimer = useCallback(
    (cxrSession: CloudXR.Session) => {
      warmupBeginTimerRef.current = setTimeout(() => {
        warmupBeginTimerRef.current = null;
        console.warn(`CloudXR decoder warm-up did not begin within ${warmupBeginBaseTimeoutMs}ms`);
        try {
          cxrSession.disconnect();
        } catch {
          // Ignore errors from disconnect() - best effort, matching the connect()-catch block.
        }
        cloudXRDelegatesRef.current?.onStreamStopped?.({
          message: `Decoder warm-up did not begin within ${warmupBeginBaseTimeoutMs}ms`,
        } as CloudXR.StreamingError);
      }, warmupBeginBaseTimeoutMs);
    },
    [warmupBeginBaseTimeoutMs]
  );

  /**
   * Arms the decoder-warm-up-end timer for *cxrSession*: if warm-up hasn't completed (no
   * pose-backed frame rendered - see WARMUP_COMPLETE_LOG_PATTERN) by the deadline since it began,
   * disconnects the session and dispatches a synthetic recoverable StreamingError through
   * onStreamStopped. Called from the onLog handler once the first warm-up log arrives, i.e. after
   * armWarmupBeginTimer has already been cleared - this covers the next, disjoint window. Fixed
   * deadline, same rationale as armWarmupBeginTimer.
   */
  const armWarmupEndTimer = useCallback(
    (cxrSession: CloudXR.Session) => {
      warmupEndTimerRef.current = setTimeout(() => {
        warmupEndTimerRef.current = null;
        console.warn(`CloudXR decoder warm-up did not complete within ${warmupEndBaseTimeoutMs}ms`);
        try {
          cxrSession.disconnect();
        } catch {
          // Ignore errors from disconnect() - best effort, matching the connect()-catch block.
        }
        cloudXRDelegatesRef.current?.onStreamStopped?.({
          message: `Decoder warm-up did not complete within ${warmupEndBaseTimeoutMs}ms`,
        } as CloudXR.StreamingError);
      }, warmupEndBaseTimeoutMs);
    },
    [warmupEndBaseTimeoutMs]
  );

  // Metrics trackers for averaging performance metrics
  // Use prop values if provided, otherwise use defaults
  const renderFpsTrackerRef = useRef<MetricsTracker>(
    new MetricsTracker(metricsSettings.renderFpsWindow ?? 100)
  );
  // Pose send rate is sampled on the same PerRender cadence as render FPS, so it uses
  // the same window to keep the two HUD cards directly comparable.
  const poseSendFpsTrackerRef = useRef<MetricsTracker>(
    new MetricsTracker(metricsSettings.renderFpsWindow ?? 100)
  );
  const streamingFpsTrackerRef = useRef<MetricsTracker>(
    new MetricsTracker(metricsSettings.streamingFpsWindow ?? 20)
  );
  const poseToRenderTrackerRef = useRef<MetricsTracker>(
    new MetricsTracker(metricsSettings.poseToRenderWindow ?? 20)
  );

  // Disable Three.js so it doesn't clear the framebuffer after CloudXR renders.
  threeRenderer.autoClear = false;

  // In headless mode the CloudXR frame blit is skipped, so mark the connected-but-suppressed
  // case with a distinct dark blue instead of leaving whatever was last in the framebuffer -
  // that's the one case where, non-headless, the client would actually be rendering a scene.
  // Captures the renderer's original clear color/alpha once so toggling headless off (without
  // remounting) restores it instead of leaving the dark blue marker color applied.
  const originalClearColorRef = useRef<{ color: Color; alpha: number } | null>(null);
  useEffect(() => {
    if (!originalClearColorRef.current) {
      originalClearColorRef.current = {
        color: threeRenderer.getClearColor(new Color()),
        alpha: threeRenderer.getClearAlpha(),
      };
    }
    if (headless) {
      threeRenderer.setClearColor(HEADLESS_CLEAR_COLOR, 1);
    } else {
      const original = originalClearColorRef.current;
      threeRenderer.setClearColor(original.color, original.alpha);
    }
  }, [headless, threeRenderer]);

  // Access Three.js WebXRManager and WebGL context.
  const gl: WebGL2RenderingContext = threeRenderer.getContext() as WebGL2RenderingContext;

  const trackedGL = bindGL(gl);

  // Set up event listeners in useEffect to add them only once
  useEffect(() => {
    const webXRManager = threeRenderer.xr;

    // This effect re-registers on every `config` change (e.g. the operator edits a setting
    // while a stream is still connecting). The cleanup below already cleared every timer ref,
    // but a fresh sessionstart event won't fire just because the effect re-ran - the WebXR
    // session is already live. Without this, the in-flight connection would keep running with
    // no deadline for the rest of this attempt.
    if (cxrSessionRef.current && !hasStreamStartedRef.current) {
      armStreamAttachTimer(cxrSessionRef.current);
    } else if (cxrSessionRef.current && !warmupBegunRef.current) {
      armWarmupBeginTimer(cxrSessionRef.current);
    } else if (cxrSessionRef.current && !warmupStatusRef.current.completed) {
      armWarmupEndTimer(cxrSessionRef.current);
    }

    if (webXRManager) {
      const handleSessionStart = async () => {
        reconnectAttemptRef.current = 0;
        const xrSession = webXRManager.getSession();

        // CloudXR must advertise the rate actually used by the headset. Wait for WebXR to
        // apply the configured rate before creating the CloudXR session to avoid a pacing race.
        const effectiveDeviceFrameRate = xrSession
          ? await applyTargetFrameRate(xrSession, config.deviceFrameRate)
          : config.deviceFrameRate;

        // Explicitly request the desired reference space from the XRSession to avoid
        // inheriting a default 'local-floor' space that could stack with UI offsets.
        let referenceSpace: XRReferenceSpace | null = null;
        try {
          if (xrSession) {
            if (config.referenceSpaceType === 'auto') {
              const fallbacks: XRReferenceSpaceType[] = [
                'local-floor',
                'local',
                'viewer',
                'unbounded',
              ];
              for (const t of fallbacks) {
                try {
                  referenceSpace = await xrSession.requestReferenceSpace(t);
                  if (referenceSpace) break;
                } catch (_) {}
              }
            } else {
              try {
                referenceSpace = await xrSession.requestReferenceSpace(
                  config.referenceSpaceType as XRReferenceSpaceType
                );
              } catch (error) {
                console.error(
                  `Failed to request reference space '${config.referenceSpaceType}':`,
                  error
                );
              }
            }
          }
        } catch (error) {
          console.error('Failed to request XR reference space:', error);
          referenceSpace = null;
        }

        if (!referenceSpace) {
          // As a last resort, fall back to WebXRManager's current reference space
          referenceSpace = webXRManager.getReferenceSpace();
        }

        if (referenceSpace) {
          // Ensure that the session is not already created.
          if (cxrSessionRef.current) {
            console.error('CloudXR session already exists');
            return;
          }

          const glBinding = webXRManager.getBinding();
          if (!glBinding) {
            console.warn('No WebGL binding found');
          }

          // Apply proxy configuration logic
          let connectionConfig: ConnectionConfiguration;
          try {
            connectionConfig = getConnectionConfig(config.serverIP, config.port, config.proxyUrl);
            onServerAddress?.(connectionConfig.serverIP);
          } catch (error) {
            onStatusChange?.(false, 'Configuration Error');
            onError?.(`Proxy configuration failed: ${error}`);
            onExitImmersiveXRRef.current?.();
            return;
          }

          // Apply XR offset if provided in config (meters)
          const offsetX = config.xrOffsetX || 0;
          const offsetY = config.xrOffsetY || 0;
          const offsetZ = config.xrOffsetZ || 0;
          if (offsetX !== 0 || offsetY !== 0 || offsetZ !== 0) {
            const offsetTransform = new XRRigidTransform(
              { x: offsetX, y: offsetY, z: offsetZ },
              { x: 0, y: 0, z: 0, w: 1 }
            );
            referenceSpace = referenceSpace.getOffsetReferenceSpace(offsetTransform);
          }

          // Fill in CloudXR session options.
          const cloudXROptions: CloudXR.SessionOptions = {
            serverAddress: connectionConfig.serverIP,
            serverPort: connectionConfig.port,
            useSecureConnection: connectionConfig.useSecureConnection,
            signalingResourcePath: connectionConfig.resourcePath,
            perEyeWidth: config.perEyeWidth,
            perEyeHeight: config.perEyeHeight,
            reprojectionGridCols: config.reprojectionGridCols,
            reprojectionGridRows: config.reprojectionGridRows,
            codec: config.codec,
            gl: gl,
            referenceSpace: referenceSpace,
            deviceFrameRate: effectiveDeviceFrameRate,
            maxStreamingBitrateKbps: config.maxStreamingBitrateMbps * 1000, // Convert Mbps to Kbps
            enablePoseSmoothing: config.enablePoseSmoothing,
            posePredictionFactor: config.posePredictionFactor,
            enableTexSubImage2D: config.enableTexSubImage2D,
            useQuestColorWorkaround: config.useQuestColorWorkaround,
            mediaAddress: config.mediaAddress,
            mediaPort: config.mediaPort,
            iceServers: iceServers,
            glBinding: glBinding,
            // Undefined when the network test is off, which is the IsaacTeleop default.
            streamTest,
            logLevel: CloudXR.LogLevel.Info, // Deliver Info-and-above SDK logs to onLog
            telemetry: {
              enabled: true,
              appInfo: {
                version: '6.3.0',
                product: applicationName,
              },
            },
          };

          // Store the render target and key GL bindings to restore after CloudXR rendering
          const cloudXRDelegates: CloudXR.SessionDelegates = {
            onLog: (entries: CloudXR.LogEntry[]) => {
              for (const { level, message } of entries) {
                // Pick the console method matching this severity so DevTools filtering works.
                const printFn =
                  level === CloudXR.LogLevel.Error
                    ? console.error
                    : level === CloudXR.LogLevel.Warning
                      ? console.warn
                      : level === CloudXR.LogLevel.Debug
                        ? console.debug
                        : console.info;
                printFn(`[CloudXR] ${message}`);

                const progressMatch = message.match(WARMUP_PROGRESS_LOG_PATTERN);
                const completeMatch = message.match(WARMUP_COMPLETE_LOG_PATTERN);
                if ((progressMatch || completeMatch) && !warmupBegunRef.current) {
                  // First warm-up log this attempt: hand off from "waiting for warm-up to
                  // begin" to "waiting for it to end" (unless it already ended in this same
                  // log line, in which case there's nothing left to wait for).
                  warmupBegunRef.current = true;
                  if (warmupBeginTimerRef.current !== null) {
                    clearTimeout(warmupBeginTimerRef.current);
                    warmupBeginTimerRef.current = null;
                  }
                  if (!completeMatch && cxrSessionRef.current) {
                    armWarmupEndTimer(cxrSessionRef.current);
                  }
                }
                if (progressMatch) {
                  warmupStatusRef.current = {
                    framesSeen: Number(progressMatch[1]),
                    timestamp: Date.now(),
                    completed: false,
                  };
                  onWarmupStatus?.(warmupStatusRef.current);
                } else if (completeMatch) {
                  warmupStatusRef.current = {
                    framesSeen: Number(completeMatch[1]),
                    timestamp: Date.now(),
                    completed: true,
                  };
                  if (warmupEndTimerRef.current !== null) {
                    clearTimeout(warmupEndTimerRef.current);
                    warmupEndTimerRef.current = null;
                  }
                  onWarmupStatus?.(warmupStatusRef.current);
                }
              }
              onLog?.(entries);
            },
            onStreamTestStarted: () => {
              onStatusChange?.(false, 'Testing network');
              onStreamTestStarted?.();
            },
            onStreamTestStopped: (result: CloudXR.StreamTestResult) => {
              console.debug('Network test complete:', result);
              onStreamTestStopped?.(result);
            },
            onWebGLStateChangeBegin: () => {
              // Save the current render target before CloudXR changes state
              trackedGL.save();
            },
            onWebGLStateChangeEnd: () => {
              // Restore the tracked GL state to the state before CloudXR rendering.
              trackedGL.restore();
            },
            onStreamStarted: () => {
              // A successful (re)connect clears the counter, so an unrelated later failure
              // gets its own full budget of attempts rather than inheriting this one's count.
              reconnectAttemptRef.current = 0;
              hasStreamStartedRef.current = true;
              if (streamAttachTimerRef.current !== null) {
                clearTimeout(streamAttachTimerRef.current);
                streamAttachTimerRef.current = null;
              }
              // The stream has attached, but decoder warm-up (no pose-backed frame rendered yet)
              // starts a new, disjoint window that the attach timer above never covered. Reset
              // per-attempt warm-up-begun tracking before arming, matching warmupStatusRef's own
              // reset in establishSession().
              warmupBegunRef.current = false;
              if (cxrSessionRef.current) {
                armWarmupBeginTimer(cxrSessionRef.current);
              }
              console.debug('CloudXR stream started');
              onStatusChange?.(true, 'Connected');
            },
            onStreamStopped: (error?: CloudXR.StreamingError) => {
              // A real stream-stop (including the synthetic ones the attach/warmup timers below
              // dispatch through this same delegate) means we're no longer "waiting to attach" or
              // "warming up" - clear any pending timer so it can't fire again after this attempt
              // has already been handled.
              if (streamAttachTimerRef.current !== null) {
                clearTimeout(streamAttachTimerRef.current);
                streamAttachTimerRef.current = null;
              }
              if (warmupBeginTimerRef.current !== null) {
                clearTimeout(warmupBeginTimerRef.current);
                warmupBeginTimerRef.current = null;
              }
              if (warmupEndTimerRef.current !== null) {
                clearTimeout(warmupEndTimerRef.current);
                warmupEndTimerRef.current = null;
              }
              if (error) {
                // Display user-friendly error message with error code if available
                const errorMsg = error.code
                  ? `${error.message} (Error code: 0x${error.code.toString(16).toUpperCase()})`
                  : error.message;

                if (isRecoverable(error) && reconnectAttemptRef.current < maxReconnectAttempts) {
                  reconnectAttemptRef.current += 1;
                  console.warn(
                    `CloudXR stream error, retry ${reconnectAttemptRef.current}/${maxReconnectAttempts} ` +
                      `in ${reconnectDelayMs}ms:`,
                    errorMsg
                  );
                  onStatusChange?.(
                    false,
                    `Reconnecting (${reconnectAttemptRef.current}/${maxReconnectAttempts})`
                  );
                  cxrSessionRef.current = null;
                  onSessionReady?.(null);
                  reconnectTimerRef.current = setTimeout(() => {
                    reconnectTimerRef.current = null;
                    establishSession();
                  }, reconnectDelayMs);
                  return;
                }

                console.error('Stream stopped with error:', errorMsg);
                onStatusChange?.(false, 'Error');
                onError?.(`CloudXR session stopped: ${errorMsg}`);

                // Log additional debug info if available
                if (error.reasonCode !== undefined) {
                  console.debug('Stop reason code:', error.reasonCode);
                }
                onExitImmersiveXRRef.current?.();
              } else {
                console.debug('CloudXR session stopped');
                onStatusChange?.(false, 'Disconnected');
              }
              // Clear the session reference
              cxrSessionRef.current = null;
              onSessionReady?.(null);
            },
            onMetrics: (metrics: CloudXR.Metrics, cadence: CloudXR.MetricsCadence) => {
              // Every cadence delivers a *partial* record: only the metrics sampled on this
              // tick are present. Forward just those keys and let the consumer merge, so an
              // absent metric is never reported downstream as a zero.
              if (onRenderPerformanceMetrics && cadence === CloudXR.MetricsCadence.PerRender) {
                const update: RenderMetricsUpdate = {};
                const renderFps = metrics[CloudXR.MetricsName.RenderFramerate];
                if (renderFps !== undefined) {
                  update.framerate = renderFpsTrackerRef.current.add(renderFps);
                }
                const poseSendFps = metrics[CloudXR.MetricsName.PoseSendFramerate];
                if (poseSendFps !== undefined) {
                  update.sendFramerate = poseSendFpsTrackerRef.current.add(poseSendFps);
                }
                const xrPoseAgeMs = metrics[CloudXR.MetricsName.LatencyXrPoseAgeMs];
                if (xrPoseAgeMs !== undefined) {
                  update.xrPoseAgeMs = xrPoseAgeMs;
                }
                // The send-begin and render-begin paths each emit their own PerRender tick,
                // so skip ticks that carried none of the keys we track.
                if (Object.keys(update).length > 0) {
                  onRenderPerformanceMetrics(update);
                }
              }

              if (onNetworkPerformanceMetrics && cadence === CloudXR.MetricsCadence.PerNetwork) {
                const update: NetworkMetricsUpdate = {};
                const networkFields: Array<[keyof NetworkMetricsUpdate, string]> = [
                  ['streamingRateMbps', CloudXR.MetricsName.NetworkStreamingRateMbps],
                  ['availableBandwidthMbps', CloudXR.MetricsName.NetworkAvailableBandwidthMbps],
                  ['rttMs', CloudXR.MetricsName.NetworkRttMs],
                  ['packetLoss', CloudXR.MetricsName.NetworkPacketLoss],
                  ['avgDecodeTimeMs', CloudXR.MetricsName.NetworkAvgDecodeTimeMs],
                  ['qualityScore', CloudXR.MetricsName.NetworkQualityScore],
                  ['bandwidthScore', CloudXR.MetricsName.NetworkBandwidthScore],
                  ['lossScore', CloudXR.MetricsName.NetworkLossScore],
                  ['latencyScore', CloudXR.MetricsName.NetworkLatencyScore],
                  ['sessionQuality', CloudXR.MetricsName.SessionQuality],
                ];
                for (const [field, name] of networkFields) {
                  const value = metrics[name as CloudXR.MetricsName];
                  if (value !== undefined) {
                    update[field] = value;
                  }
                }
                if (Object.keys(update).length > 0) {
                  onNetworkPerformanceMetrics(update);
                }
              }

              // Handle streaming performance metrics (PerFrame cadence)
              if (onStreamingPerformanceMetrics && cadence === CloudXR.MetricsCadence.PerFrame) {
                const update: FrameMetricsUpdate = {};
                const streamingFps = metrics[CloudXR.MetricsName.StreamingFramerate];
                if (streamingFps !== undefined) {
                  update.framerate = streamingFpsTrackerRef.current.add(streamingFps);
                }
                const poseToRenderMs = metrics[CloudXR.MetricsName.PoseToRenderTime];
                if (poseToRenderMs !== undefined) {
                  update.poseToRenderTimeMs = poseToRenderTrackerRef.current.add(poseToRenderMs);
                }
                // Reported raw: these are diagnostic counters for the hub, not HUD values,
                // so averaging them would only obscure spikes.
                const frameFields: Array<[keyof FrameMetricsUpdate, string]> = [
                  ['frameCount', CloudXR.MetricsName.StreamingFrameCount],
                  ['poseUploadMs', CloudXR.MetricsName.LatencyPoseUploadMs],
                  ['poseToFrameReceivedMs', CloudXR.MetricsName.LatencyPoseToFrameReceivedMs],
                  [
                    'compositorSkippedPercent',
                    CloudXR.MetricsName.FramePipelineCompositorSkippedPercent,
                  ],
                  ['outOfOrderPercent', CloudXR.MetricsName.FramePipelineOutOfOrderPercent],
                  ['mismatchedPercent', CloudXR.MetricsName.FramePipelineMismatchedPercent],
                ];
                for (const [field, name] of frameFields) {
                  const value = metrics[name as CloudXR.MetricsName];
                  if (value !== undefined) {
                    update[field] = value;
                  }
                }
                if (Object.keys(update).length > 0) {
                  onStreamingPerformanceMetrics(update);
                }
              }
            },
          };
          cloudXRDelegatesRef.current = cloudXRDelegates;

          // Creates and connects a CloudXR session against the options/delegates above. Called
          // once here for the initial connect, and again (unchanged) by onStreamStopped's retry
          // path below - cloudXROptions/cloudXRDelegates don't vary per attempt, only whether a
          // session currently exists.
          const establishSession = (): void => {
            if (cxrSessionRef.current) {
              console.error('CloudXR session already exists');
              return;
            }

            // Shared GL contexts may still have queued errors from the host app (e.g. R3F/XR).
            // Clear them before createSession so CloudXR setup only sees errors from its own path.
            clearPendingGLErrors(gl);

            // Create the CloudXR session.
            let cxrSession: CloudXR.Session;
            try {
              cxrSession = CloudXR.createSession(cloudXROptions, cloudXRDelegates);
            } catch (error) {
              onStatusChange?.(false, 'Session Creation Failed');
              onError?.(`Failed to create CloudXR session: ${error}`);
              onExitImmersiveXRRef.current?.();
              return;
            }

            // Store the session in the ref so it persists across re-renders
            cxrSessionRef.current = cxrSession;
            hasStreamStartedRef.current = false;
            warmupStatusRef.current = { framesSeen: 0, timestamp: 0, completed: false };
            warmupBegunRef.current = false;

            // Notify parent that session is ready
            onSessionReady?.(cxrSession);

            // Start session (synchronous call that initiates connection)
            try {
              cxrSession.connect();
              console.log('CloudXR session connect initiated');
              // Note: The session will transition to Connected state via the onStreamStarted callback
              // Use cxrSession.state to check if streaming has actually started

              // Passthrough-only detection: the session can enter XR and call connect()
              // successfully while the stream never actually attaches, with no error callback to
              // signal it - isXRMode alone can't tell that state apart from a slow-but-fine
              // connect. See armStreamAttachTimer's doc comment.
              armStreamAttachTimer(cxrSession);
            } catch (error) {
              onStatusChange?.(false, 'Connection Failed');
              // Report error via callback
              onError?.('Failed to connect CloudXR session');
              // Best-effort: release SDK resources if connect() threw after createSession(); ignore disconnect failures.
              try {
                cxrSession.disconnect();
              } catch {
                // Ignore errors from disconnect().
              }
              cxrSessionRef.current = null;
              onSessionReady?.(null);
              onExitImmersiveXRRef.current?.();
            }
          };

          establishSession();
        } else {
          onStatusChange?.(false, 'Reference Space Unavailable');
          onError?.('Could not obtain an XR reference space for CloudXR');
          onExitImmersiveXRRef.current?.();
        }
      };

      const handleSessionEnd = () => {
        // A pending retry belongs to this WebXR session; cancel it before it can create a new
        // CloudXR session against a gl/referenceSpace that's about to become stale. Checked
        // unconditionally (not inside the cxrSessionRef.current branch below): while a retry is
        // pending, cxrSessionRef.current is already null.
        const hadPendingRetry = reconnectTimerRef.current !== null;
        if (reconnectTimerRef.current !== null) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        if (streamAttachTimerRef.current !== null) {
          clearTimeout(streamAttachTimerRef.current);
          streamAttachTimerRef.current = null;
        }
        if (warmupBeginTimerRef.current !== null) {
          clearTimeout(warmupBeginTimerRef.current);
          warmupBeginTimerRef.current = null;
        }
        if (warmupEndTimerRef.current !== null) {
          clearTimeout(warmupEndTimerRef.current);
          warmupEndTimerRef.current = null;
        }
        if (cxrSessionRef.current) {
          cxrSessionRef.current.disconnect();
          cxrSessionRef.current = null;
          onSessionReady?.(null);
        } else if (hadPendingRetry) {
          // No live session to disconnect() - one was never re-established - so nothing else
          // will emit onStreamStopped -> onStatusChange here. Without this, the last status a
          // caller ever sees stays "Reconnecting (n/maxAttempts)" forever.
          onStatusChange?.(false, 'Disconnected');
        }
      };

      // Add start+end session event listeners to the WebXRManager.
      webXRManager.addEventListener('sessionstart', handleSessionStart);
      webXRManager.addEventListener('sessionend', handleSessionEnd);

      // Cleanup function to remove listeners
      return () => {
        webXRManager.removeEventListener('sessionstart', handleSessionStart);
        webXRManager.removeEventListener('sessionend', handleSessionEnd);
        if (reconnectTimerRef.current !== null) {
          clearTimeout(reconnectTimerRef.current);
          reconnectTimerRef.current = null;
        }
        if (streamAttachTimerRef.current !== null) {
          clearTimeout(streamAttachTimerRef.current);
          streamAttachTimerRef.current = null;
        }
        if (warmupBeginTimerRef.current !== null) {
          clearTimeout(warmupBeginTimerRef.current);
          warmupBeginTimerRef.current = null;
        }
        if (warmupEndTimerRef.current !== null) {
          clearTimeout(warmupEndTimerRef.current);
          warmupEndTimerRef.current = null;
        }
      };
    }
  }, [threeRenderer, config]); // Re-register handlers when renderer or config changes

  // Custom render loop - runs every frame
  useFrame((state, delta) => {
    const webXRManager = threeRenderer.xr;

    if (webXRManager.isPresenting && session) {
      // Access the current WebXR XRFrame
      const xrFrame = state.gl.xr.getFrame();
      if (xrFrame) {
        // Get THREE WebXRManager from the useFrame state.
        const webXRManager = state.gl.xr;

        if (!cxrSessionRef || !cxrSessionRef.current) {
          console.debug('Skipping frame, no session yet');
          if (!headless) {
            // Clear the framebuffer as we've set autoClear to false.
            threeRenderer.clear();
          }
          return;
        }

        // Get session from reference.
        const cxrSession: CloudXR.Session = cxrSessionRef.current;

        // Pump tracking during Connecting so the SDK can read XRSession.enabledFeatures from
        // these early frames and negotiate optional capabilities - notably body tracking -
        // before the stream starts. Without this the SDK never sees a frame pre-Connected and
        // full-body teleop silently degrades to upper body. Returns false while Connecting,
        // but may throw deferred exceptions from callbacks.
        //
        // Deliberately the raw xrFrame, not trackingFrameAdapter's: during replay the adapter
        // returns a Proxy with a substituted session, and capability detection must reflect
        // the real XRSession. The adapter still applies on the Connected path below.
        if (cxrSession.state === CloudXR.SessionState.Connecting) {
          try {
            cxrSession.sendTrackingStateToServer(state.clock.elapsedTime * 1000, xrFrame);
          } catch (error) {
            console.error('CloudXR render loop error:', error);
            return;
          }
        }

        // If the CloudXR session is not connected, skip the frame.
        if (cxrSession.state !== CloudXR.SessionState.Connected) {
          console.debug('Skipping frame, session not connected, state:', cxrSession.state);
          if (!headless) {
            // Clear the framebuffer as we've set autoClear to false.
            threeRenderer.clear();
          }
          return;
        }

        // Get timestamp from useFrame state and convert to milliseconds.
        const timestamp: DOMHighResTimeStamp = state.clock.elapsedTime * 1000;

        try {
          // Send the tracking state (including viewer pose and hand/controller data) to the server;
          // that triggers server-side rendering for the frame.
          const trackingFrame = trackingFrameAdapter?.(xrFrame) ?? xrFrame;
          cxrSession.sendTrackingStateToServer(timestamp, trackingFrame);

          if (!headless) {
            const layer: XRWebGLLayer = webXRManager.getBaseLayer() as XRWebGLLayer;
            cxrSession.render(timestamp, xrFrame, layer);
          } else {
            // Connected but the frame blit is suppressed - this is the one case where,
            // non-headless, the client would actually be rendering a scene. Clear to the dark
            // blue marker color every frame so it's visually obvious this is headless.
            threeRenderer.clear();
          }
        } catch (error) {
          // Handle deferred exceptions from callbacks or render errors
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error('CloudXR render loop error:', error);
          // Disconnect session on error
          cxrSession.disconnect();
          onError?.(`CloudXR error: ${errorMessage}`);
        }
      }
    }
  }, -1000);

  return null;
}
