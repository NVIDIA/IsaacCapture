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
 * MockCloudXR - a `CloudXR.Session` stand-in for manual/visual testing of the WebXR client
 * without a real CloudXR server.
 *
 * Drives `CloudXR.SessionDelegates` through a plausible lifecycle (connecting -> connected,
 * decoder warm-up, periodic onLog/onMetrics, onStreamStopped on disconnect) and, in place of
 * decoding a real video stream, renders a small placeholder scene (see webglMockScene.ts) into
 * each eye's viewport of the XRWebGLLayer handed to {@link MockCloudXR.render}, using plain
 * WebGL2 rather than three.js - see webglMockScene.ts's header comment for why. Decoder warm-up
 * (render() shows black until it completes) advances only when a test calls
 * {@link MockCloudXR.videoFrameReceived} to queue a frame - render() (run by
 * CloudXRComponent's own loop, not a background clock here) is what actually consumes it and
 * does the counting/logging, same relationship as the real SDK's decoded-video callback and
 * render(). Zero warm-up frames by default, configurable via
 * setWarmupTotalFrames()/setWarmupFirstStatusFrame().
 */

// Imports the real SDK's concrete entry file, not the bare '@nvidia/cloudxr' specifier: webpack
// builds that alias '@nvidia/cloudxr' to tests/mock/cloudxr-mock-alias.ts (see that file) would
// otherwise resolve this import back to the alias itself, circularly - the alias shim imports
// this module, and this module would import the (still-mid-load) alias back.
import * as CloudXR from '@nvidia/cloudxr/build/cloudxr.js';

import { mat4InvertRigid, rotateVectorByQuaternion, WebGLMockScene } from './webglMockScene';

const DEFAULT_CONNECT_DELAY_MS = 500;
const NETWORK_METRICS_INTERVAL_MS = 1000;
// No simulated warm-up delay by default - render() shows the scene from the first frame, like
// pre-warmup-simulation mock behavior. Set via setWarmupTotalFrames() for tests that need it.
const DEFAULT_WARMUP_TOTAL_FRAMES = 0;
// Matches the real SDK's own warm-up progress cadence: it logs every 10th warm-up frame, not
// every frame.
const WARMUP_LOG_INTERVAL = 10;
// Frame the *first* warm-up progress log fires on - matches WARMUP_LOG_INTERVAL by default (the
// real SDK's first log is also its 10th warm-up frame), but settable separately via
// setWarmupFirstStatusFrame() so a test can get fast confirmation that warm-up has begun without
// waiting for 10 real rendered frames.
const DEFAULT_WARMUP_FIRST_STATUS_FRAME = WARMUP_LOG_INTERVAL;

/**
 * Sentinel `gl` value for {@link CloudXR.SessionOptions.gl}: a MockCloudXR constructed with this
 * performs no WebGL operations at all in render() (no WebGLMockScene, no gl/layer calls, no
 * onWebGLStateChangeBegin/End) - useful for tests that only care about the session/delegate
 * lifecycle and have no real WebGL context to give it.
 */
export const NullWebGLContext: WebGL2RenderingContext = null as unknown as WebGL2RenderingContext;

/** Per-quality-band sample ranges used to synthesize plausible {@link CloudXR.MetricsName} values. */
const NETWORK_QUALITY_PROFILES: Record<
  CloudXR.QualityScore,
  {
    streamingRateMbps: [number, number];
    availableBandwidthMbps: [number, number];
    rttMs: [number, number];
  }
> = {
  [CloudXR.QualityScore.NoData]: {
    streamingRateMbps: [0, 0],
    availableBandwidthMbps: [0, 0],
    rttMs: [0, 0],
  },
  [CloudXR.QualityScore.Excellent]: {
    streamingRateMbps: [150, 170],
    availableBandwidthMbps: [180, 200],
    rttMs: [5, 15],
  },
  [CloudXR.QualityScore.Good]: {
    streamingRateMbps: [80, 100],
    availableBandwidthMbps: [90, 110],
    rttMs: [30, 50],
  },
  [CloudXR.QualityScore.Degraded]: {
    streamingRateMbps: [40, 55],
    availableBandwidthMbps: [50, 65],
    rttMs: [100, 150],
  },
  [CloudXR.QualityScore.Unsustainable]: {
    streamingRateMbps: [10, 20],
    availableBandwidthMbps: [20, 30],
    rttMs: [250, 300],
  },
};

function randomInRange([min, max]: [number, number]): number {
  return min + Math.random() * (max - min);
}

/**
 * Implements `CloudXR.Session` by rendering a mock three.js scene instead of decoded video.
 * Construct via {@link createMockCloudXRSession} rather than directly.
 */
export class MockCloudXR implements CloudXR.Session {
  private sessionState: CloudXR.SessionState = CloudXR.SessionState.Initialized;
  private sessionMicState: CloudXR.MicState = CloudXR.MicState.UNINITIALIZED;
  // Indexed-access type rather than naming CloudXR.MessageChannel directly: the SDK's shipped
  // .d.ts re-exports that type from a MessageChannel.d.ts file missing from the npm package.
  readonly availableMessageChannels: CloudXR.Session['availableMessageChannels'] = [];

  private connectTimer: ReturnType<typeof setTimeout> | null = null;
  private streamTestTimer: ReturnType<typeof setTimeout> | null = null;
  private networkMetricsTimer: ReturnType<typeof setInterval> | null = null;
  private frameCount = 0;
  private lastRenderTimestamp: DOMHighResTimeStamp | null = null;
  // Monotonic per-render() counter, reset to 0 at the top of each connect()'s Connected
  // transition (see finishConnecting()). Exposed read-only via getCurrentFrameID() - a stand-in
  // for the real client's own outgoing per-frame ID sequence, for tests that want a frame number
  // to hand to videoFrameReceived().
  private currentFrameId = 0;

  private connectDelayMs = DEFAULT_CONNECT_DELAY_MS;
  private networkQuality: CloudXR.QualityScore = CloudXR.QualityScore.Excellent;
  // Mailbox written by videoFrameReceived(), consumed (once) by the next render() call - same
  // relationship as the real SDK's own per-frame state, written by its decoded-video handling
  // and read by render(). hasPendingFrame distinguishes "nothing received yet" from a pending
  // frameId of literally `undefined` (fall back to the internal warm-up counter - see
  // videoFrameReceived's doc comment).
  private hasPendingFrame = false;
  private pendingFrameId: number | undefined = undefined;
  // Set via setWarmupTotalFrames(); persists across connect()s on this instance the same way
  // connectDelayMs does. `null` simulates warm-up that never completes - drives
  // CloudXRComponent's warmupEndTimeoutMs in a test. Only consulted when a pending frame's
  // frameId is `undefined` - see videoFrameReceived's doc comment.
  private warmupTotalFrames: number | null = DEFAULT_WARMUP_TOTAL_FRAMES;
  private warmupFramesSeen = 0;
  // Set via setWarmupFirstStatusFrame(); see DEFAULT_WARMUP_FIRST_STATUS_FRAME's doc comment.
  private warmupFirstStatusFrame = DEFAULT_WARMUP_FIRST_STATUS_FRAME;
  // True once this connect()'s warm-up has completed - render() renders black (no scene) until
  // then, matching the real SDK: decoder warm-up frames carry no pose-backed video to show.
  private warmupComplete = false;

  // Scene time is explicit rather than timestamp-driven, so the render is static/reproducible by
  // default, set only via videoFrameReceived()'s optional applicationTime.
  private sceneTime = 0;

  private scene: WebGLMockScene | null = null;

  constructor(
    private readonly options: CloudXR.SessionOptions,
    private readonly delegates: CloudXR.SessionDelegates
  ) {}

  get state(): CloudXR.SessionState {
    return this.sessionState;
  }

  get micState(): CloudXR.MicState {
    return this.sessionMicState;
  }

  connect(): void {
    if (
      this.sessionState !== CloudXR.SessionState.Initialized &&
      this.sessionState !== CloudXR.SessionState.Disconnected
    ) {
      throw new Error(`MockCloudXR.connect() called while in state ${this.sessionState}`);
    }
    this.sessionState = CloudXR.SessionState.Connecting;
    this.log(
      CloudXR.LogLevel.Info,
      `Mock connecting to ${this.options.serverAddress}:${this.options.serverPort} ` +
        `(codec=${this.options.codec ?? 'auto'}, perEye=${this.options.perEyeWidth}x${this.options.perEyeHeight})`
    );

    const streamTest = this.options.streamTest;
    if (streamTest && streamTest.mode !== 'off') {
      this.delegates.onStreamTestStarted?.();
      this.log(CloudXR.LogLevel.Info, `Mock stream test running (${streamTest.durationSeconds}s)`);
      this.streamTestTimer = setTimeout(() => {
        this.streamTestTimer = null;
        const result = this.buildStreamTestResult();
        this.delegates.onStreamTestStopped?.(result);
        if (streamTest.mode === 'block' && !result.passed) {
          this.log(CloudXR.LogLevel.Error, 'Mock stream test failed; blocking connect');
          this.triggerFailure({
            name: 'StreamingError',
            message: 'Mock stream test did not pass (blocking mode)',
          });
          return;
        }
        this.finishConnecting();
      }, streamTest.durationSeconds * 1000);
      return;
    }

    this.finishConnecting();
  }

  private finishConnecting(): void {
    this.connectTimer = setTimeout(() => {
      this.connectTimer = null;
      this.sessionState = CloudXR.SessionState.Connected;
      // Reset per-attempt warm-up state *before* onStreamStarted, not after: that delegate call
      // is synchronous, and a caller reacting to it (e.g. onStatusChange('Connected') calling
      // videoFrameReceived() - see CloudXRComponentTest.tsx) can set hasPendingFrame within this
      // same call stack. Resetting afterward would silently wipe out a pending frame queued that
      // way.
      this.warmupComplete = false;
      this.warmupFramesSeen = 0;
      this.currentFrameId = 0;
      this.hasPendingFrame = false;
      this.pendingFrameId = undefined;
      this.log(CloudXR.LogLevel.Info, 'Mock stream started');
      this.delegates.onStreamStarted?.();
      this.networkMetricsTimer = setInterval(
        () => this.emitNetworkMetrics(),
        NETWORK_METRICS_INTERVAL_MS
      );
    }, this.connectDelayMs);
  }

  private buildStreamTestResult(): CloudXR.StreamTestResult {
    const profile = NETWORK_QUALITY_PROFILES[this.networkQuality];
    const passed = this.networkQuality >= CloudXR.QualityScore.Degraded;
    return {
      passed,
      latencyScore: this.networkQuality,
      jitterScore: this.networkQuality,
      bandwidthScore: this.networkQuality,
      devicePerformanceScore: this.networkQuality,
      rttMs: randomInRange(profile.rttMs),
      serverFps: passed ? 90 : 30,
    };
  }

  disconnect(): void {
    // Also idempotent from Error: CloudXRComponent.tsx's handleSessionEnd calls disconnect()
    // unconditionally on WebXR sessionend, which can race a just-failed session's own teardown
    // (onExitImmersiveXR ending the WebXR session before cxrSessionRef is cleared) - without this,
    // that second call re-fires onStreamStopped(undefined) on top of the original error.
    if (
      this.sessionState === CloudXR.SessionState.Initialized ||
      this.sessionState === CloudXR.SessionState.Disconnected ||
      this.sessionState === CloudXR.SessionState.Error
    ) {
      return;
    }
    this.clearTimers();
    this.sessionState = CloudXR.SessionState.Disconnecting;
    this.log(CloudXR.LogLevel.Info, 'Mock disconnecting');
    this.sessionState = CloudXR.SessionState.Disconnected;
    this.delegates.onStreamStopped?.(undefined);
  }

  sendTrackingStateToServer(timestamp: DOMHighResTimeStamp, frame: XRFrame): boolean {
    if (this.sessionState === CloudXR.SessionState.Connecting) {
      return false;
    }
    if (this.sessionState !== CloudXR.SessionState.Connected) {
      throw new Error(
        `MockCloudXR.sendTrackingStateToServer() called while in state ${this.sessionState}`
      );
    }
    const pose = frame.getViewerPose(this.options.referenceSpace);
    // Controller poses are part of the tracking state submitted here, same as a real CloudXR
    // session: render() only consumes the decoded stream, it doesn't read XRFrame input state.
    this.trackControllers(frame);
    this.delegates.onMetrics?.(
      { [CloudXR.MetricsName.PoseSendFramerate]: this.estimateFps(timestamp) },
      CloudXR.MetricsCadence.PerRender
    );
    return pose !== null;
  }

  /**
   * Also consumes any frame videoFrameReceived() queued (see processPendingFrame's doc
   * comment) - matching the real SDK's own render(), which reads whatever per-frame state its
   * decoded-video handling last wrote, rather than doing any counting/logging itself.
   */
  render(timestamp: DOMHighResTimeStamp, frame: XRFrame, layer: XRWebGLLayer): void {
    if (this.sessionState !== CloudXR.SessionState.Connected) {
      return;
    }
    const pose = frame.getViewerPose(this.options.referenceSpace);
    if (!pose) {
      return;
    }
    this.currentFrameId++;
    if (this.hasPendingFrame) {
      this.processPendingFrame();
    }

    if (!this.warmupComplete) {
      // Decoder warm-up frames carry no pose-backed video to show - see processPendingFrame,
      // which counts warm-up frames (from a frame videoFrameReceived() queued) and flips this
      // flag.
      if (this.options.gl !== NullWebGLContext) {
        const gl = this.options.gl;
        this.delegates.onWebGLStateChangeBegin?.();
        gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        this.delegates.onWebGLStateChangeEnd?.();
      }
      return;
    }

    if (this.options.gl !== NullWebGLContext) {
      const gl = this.options.gl;
      const scene = this.ensureScene(gl);
      scene.setSceneTime(this.sceneTime);
      this.delegates.onWebGLStateChangeBegin?.();
      gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
      for (const view of pose.views) {
        const viewport = layer.getViewport(view);
        if (!viewport) {
          continue;
        }
        gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
        gl.scissor(viewport.x, viewport.y, viewport.width, viewport.height);
        gl.enable(gl.SCISSOR_TEST);
        scene.renderEye(mat4InvertRigid(view.transform.matrix), view.projectionMatrix);
      }
      // Real GL state we touched above is captured/restored by the caller's own
      // onWebGLStateChangeBegin/End save/restore (see webglMockScene.ts's header comment) - no
      // separate cached-renderer cleanup needed here, unlike the earlier three.js version.
      this.delegates.onWebGLStateChangeEnd?.();
    }

    // Deliberately unreachable while still warming up (the early return above): in the real SDK,
    // StreamingFramerate/StreamingFrameCount only update on a real, pose-correlated frame - its
    // warm-up handling never reaches the code path that feeds these into the PerFrame metrics
    // batch. A frozen StreamingFrameCount/StreamingFramerate while the session otherwise reads
    // Connected is exactly the customer-visible symptom of a stuck decoder warm-up - don't move
    // this above the warm-up check, even for convenience.
    this.frameCount++;
    this.delegates.onMetrics?.(
      {
        [CloudXR.MetricsName.StreamingFramerate]: this.estimateFps(timestamp),
        [CloudXR.MetricsName.StreamingFrameCount]: this.frameCount,
      },
      CloudXR.MetricsCadence.PerFrame
    );
  }

  sendServerMessage(): void {
    throw new Error('MockCloudXR does not support sendServerMessage(); use a real session');
  }

  setMicEnabled(enabled: boolean): boolean {
    if (this.sessionState !== CloudXR.SessionState.Connected) {
      return false;
    }
    this.sessionMicState = enabled ? CloudXR.MicState.STARTED : CloudXR.MicState.STOPPED;
    this.delegates.onMicStateUpdate?.(this.sessionMicState);
    return true;
  }

  // --- Mock controls (not part of CloudXR.Session; used by MockCloudXRController) ---

  /** Sets how long the *next* connect() takes to reach Connected. Does not affect a connect() already in flight. */
  connectWait(ms: number): void {
    this.connectDelayMs = ms;
  }

  /** Changes the quality band used to synthesize network metrics and stream-test results. */
  setNetworkQuality(quality: CloudXR.QualityScore): void {
    this.networkQuality = quality;
  }

  /**
   * Sets how many warm-up frames the *next* connect() simulates before completing.
   * `null` simulates warm-up that never completes - drives CloudXRComponent's
   * warmupEndTimeoutMs.
   */
  setWarmupTotalFrames(totalFrames: number | null): void {
    this.warmupTotalFrames = totalFrames;
  }

  /**
   * Sets which warm-up frame the *next* connect()'s first progress log fires on (subsequent
   * logs still follow WARMUP_LOG_INTERVAL). See DEFAULT_WARMUP_FIRST_STATUS_FRAME's doc comment.
   */
  setWarmupFirstStatusFrame(frame: number): void {
    this.warmupFirstStatusFrame = frame;
  }

  /**
   * Returns the current render() frame counter for this connect() (see currentFrameId's field
   * comment) - a convenience for a test that wants a real, monotonically increasing frame number
   * to pass as videoFrameReceived()'s frameId, rather than picking an arbitrary one.
   */
  getCurrentFrameID(): number {
    return this.currentFrameId;
  }

  /**
   * Simulates one decoded video frame arriving from the server - but, matching the real SDK's
   * architecture, this only writes the mailbox (pendingFrameId) that the *next* render() call
   * consumes; it does no counting or logging itself. A test calls this explicitly to drive
   * warm-up progress and/or animation deterministically - there is no background clock doing
   * either automatically, but render() itself keeps running on every real XR frame (driven by
   * CloudXRComponent's own loop), so a pending frame set here is consumed on the very next one.
   *
   * `frameId` mirrors the real SDK's own warm-up/real-frame decision: a client-sent pose gets
   * correlated with the decoded frame it produced via an RTP-derived frame ID, and a value of
   * `0` is the sentinel for "server encoder warm-up frame, not tied to any client pose."
   * `frameId === 0` simulates that sentinel here; any other `frameId` simulates a real,
   * pose-correlated frame. Omitting `frameId` entirely (undefined) falls back to the internal
   * warm-up counter instead of a real ID - for a test that only cares about warm-up progress/
   * timing, not the ID protocol.
   *
   * @param frameId - Simulated RTP-derived frame ID. `0` = warm-up frame (matches the real
   *   warm-up sentinel); any other number = a real, pose-correlated frame. Omit entirely to
   *   fall back to the internal warm-up counter instead of a real ID.
   * @param applicationTime - Sets the scene's animation time to this many milliseconds (absolute,
   *   not a delta). Omit to leave it unchanged. The scene is otherwise static between calls:
   *   render() poses animated objects from this value rather than the real render timestamp, so
   *   a given sequence of calls always renders the same frames.
   */
  videoFrameReceived(frameId?: number, applicationTime?: number): void {
    if (applicationTime !== undefined) {
      this.sceneTime = applicationTime / 1000;
    }
    this.pendingFrameId = frameId;
    this.hasPendingFrame = true;
  }

  /**
   * Consumes the mailbox videoFrameReceived() writes (see that method's doc comment) - called
   * once per render() call, only when a frame is actually pending, so repeated render() calls
   * between two videoFrameReceived() calls don't double-count the same frame. Does the counting
   * and logging that method used to do directly: while still warming up, logs progress on
   * warmupFirstStatusFrame and every WARMUP_LOG_INTERVALth frame after that (matching the real
   * SDK's own warm-up progress cadence, but with the first log's frame number separately
   * configurable via setWarmupFirstStatusFrame() for faster test feedback). Once warm-up
   * completes, sets warmupComplete so render() switches from black to the scene on its next
   * call.
   */
  private processPendingFrame(): void {
    this.hasPendingFrame = false;
    if (this.warmupComplete) {
      return;
    }
    const frameId = this.pendingFrameId;
    const isWarmupFrame =
      frameId === undefined
        ? this.warmupTotalFrames === null || this.warmupFramesSeen < this.warmupTotalFrames
        : frameId === 0;
    if (isWarmupFrame) {
      this.warmupFramesSeen++;
      if (
        this.warmupFramesSeen === this.warmupFirstStatusFrame ||
        this.warmupFramesSeen % WARMUP_LOG_INTERVAL === 0
      ) {
        this.log(
          CloudXR.LogLevel.Info,
          `Decoder warm-up in progress: ${this.warmupFramesSeen} initialization frames received`
        );
      }
      return;
    }
    this.warmupComplete = true;
    // Unlike the real SDK (which only logs this when at least one warm-up frame occurred), the
    // mock always logs completion here - including the framesSeen === 0 default - so
    // CloudXRComponent's onWarmupStatus/completed signal is always observable.
    this.log(
      CloudXR.LogLevel.Info,
      `Stream rendering started after ${this.warmupFramesSeen} decoder warm-up frame${this.warmupFramesSeen === 1 ? '' : 's'}`
    );
  }

  /** Delivers `data` through `onServerMessageReceived`, as if the server had sent it. */
  sendFakeServerMessage(data: Uint8Array): void {
    this.delegates.onServerMessageReceived?.(data);
  }

  /**
   * Immediately fails the session, as a dropped connection or server-side error would.
   * No-op outside Connecting/Connected (mirrors real SDK behavior: there is nothing to fail).
   */
  triggerFailure(
    error: CloudXR.StreamingError = { name: 'StreamingError', message: 'Mock-triggered failure' }
  ): void {
    if (
      this.sessionState !== CloudXR.SessionState.Connecting &&
      this.sessionState !== CloudXR.SessionState.Connected
    ) {
      return;
    }
    this.clearTimers();
    this.sessionState = CloudXR.SessionState.Error;
    this.log(CloudXR.LogLevel.Error, `Mock failure: ${error.message}`);
    this.delegates.onStreamStopped?.(error);
  }

  /** Moves the torus/sphere to 1 unit in front of the left/right controller, hiding each when
   * that hand isn't tracked (e.g. hand-tracking only, or no controller connected). No-op until
   * the scene exists (it's created lazily in render(), on the first frame with a real gl). */
  private trackControllers(frame: XRFrame): void {
    if (!this.scene) {
      return;
    }
    this.scene.setTorusTarget(this.positionInFrontOfController(frame, 'left'));
    this.scene.setSphereTarget(this.positionInFrontOfController(frame, 'right'));
  }

  private positionInFrontOfController(
    frame: XRFrame,
    handedness: XRHandedness
  ): { x: number; y: number; z: number } | null {
    const inputSource = Array.from(frame.session.inputSources).find(
      source => source.handedness === handedness
    );
    // targetRaySpace, not gripSpace: the WebXR spec only guarantees -Z is the pointing direction
    // for targetRaySpace. gripSpace is oriented for holding a virtual object in the hand and its
    // -Z can point elsewhere, which would otherwise put the tracked object well off the
    // controller's height even when it's held level.
    const space = inputSource?.targetRaySpace ?? inputSource?.gripSpace;
    const pose = space ? frame.getPose(space, this.options.referenceSpace) : undefined;
    if (!pose) {
      return null;
    }
    const { position, orientation } = pose.transform;
    const [fx, fy, fz] = rotateVectorByQuaternion(
      orientation.x,
      orientation.y,
      orientation.z,
      orientation.w,
      0,
      0,
      -1
    );
    return { x: position.x + fx, y: position.y + fy, z: position.z + fz };
  }

  private ensureScene(gl: WebGL2RenderingContext): WebGLMockScene {
    if (!this.scene) {
      this.scene = new WebGLMockScene(gl);
    }
    return this.scene;
  }

  private log(level: CloudXR.LogLevel, message: string): void {
    this.delegates.onLog?.([{ timestamp: performance.now(), level, message }]);
  }

  /** Clears and nulls out every pending timer; shared by disconnect() and triggerFailure(). */
  private clearTimers(): void {
    if (this.connectTimer !== null) {
      clearTimeout(this.connectTimer);
      this.connectTimer = null;
    }
    if (this.streamTestTimer !== null) {
      clearTimeout(this.streamTestTimer);
      this.streamTestTimer = null;
    }
    if (this.networkMetricsTimer !== null) {
      clearInterval(this.networkMetricsTimer);
      this.networkMetricsTimer = null;
    }
  }

  private emitNetworkMetrics(): void {
    const profile = NETWORK_QUALITY_PROFILES[this.networkQuality];
    this.delegates.onMetrics?.(
      {
        [CloudXR.MetricsName.NetworkStreamingRateMbps]: randomInRange(profile.streamingRateMbps),
        [CloudXR.MetricsName.NetworkAvailableBandwidthMbps]: randomInRange(
          profile.availableBandwidthMbps
        ),
        [CloudXR.MetricsName.NetworkRttMs]: randomInRange(profile.rttMs),
        [CloudXR.MetricsName.SessionQuality]: this.networkQuality,
      },
      CloudXR.MetricsCadence.PerNetwork
    );
  }

  private estimateFps(timestamp: DOMHighResTimeStamp): number {
    const fps =
      this.lastRenderTimestamp !== null && timestamp > this.lastRenderTimestamp
        ? 1000 / (timestamp - this.lastRenderTimestamp)
        : 0;
    this.lastRenderTimestamp = timestamp;
    return fps;
  }
}

/** Creates a {@link MockCloudXR} session; drop-in replacement for `CloudXR.createSession`. */
export function createMockCloudXRSession(
  options: CloudXR.SessionOptions,
  delegates: CloudXR.SessionDelegates = {}
): MockCloudXR {
  return new MockCloudXR(options, delegates);
}

/**
 * Thin wrapper exposing {@link MockCloudXR}'s mock-only controls under names meant to read well
 * from a devtools console (e.g. `mockCloudXR.triggerFailure()`, `mockCloudXR.connectWait(3000)`).
 * The mock demo page assigns one of these to `window.mockCloudXR`; see MockCloudXRTests.ts.
 */
export class MockCloudXRController {
  constructor(private readonly mock: MockCloudXR) {}

  /** Delays the *next* connect() by `ms` before it reaches Connected. */
  connectWait(ms: number): void {
    this.mock.connectWait(ms);
  }

  /** Fails the session right now, as a dropped connection or server error would. */
  triggerFailure(message?: string): void {
    this.mock.triggerFailure(message ? { name: 'StreamingError', message } : undefined);
  }

  /** Switches the simulated network/stream-test quality band. */
  setNetworkQuality(quality: CloudXR.QualityScore): void {
    this.mock.setNetworkQuality(quality);
  }

  /** Sets how many warm-up frames to simulate before completing (null = never) for the next connect(). */
  setWarmupTotalFrames(totalFrames: number | null): void {
    this.mock.setWarmupTotalFrames(totalFrames);
  }

  /** Sets which warm-up frame the first progress log fires on for the next connect(). */
  setWarmupFirstStatusFrame(frame: number): void {
    this.mock.setWarmupFirstStatusFrame(frame);
  }

  /** Returns the current render() frame counter for this connect() (0 until the first render()). */
  getCurrentFrameID(): number {
    return this.mock.getCurrentFrameID();
  }

  /**
   * Simulates one decoded video frame arriving. `frameId` 0 = warm-up frame, other = real frame,
   * omitted = fall back to the internal warm-up counter; see MockCloudXR.videoFrameReceived.
   */
  videoFrameReceived(frameId?: number, applicationTime?: number): void {
    this.mock.videoFrameReceived(frameId, applicationTime);
  }

  /** Delivers a fake inbound server message. */
  sendFakeServerMessage(data: Uint8Array): void {
    this.mock.sendFakeServerMessage(data);
  }
}
