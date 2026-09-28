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
 * Single source of truth for every URL query parameter the web client accepts.
 *
 * Two kinds of param live here:
 *  - form-backed (has `elementId`): seeded into a control in index.html, then read
 *    back through the form into the CloudXR config.
 *  - direct (no `elementId`): consumed straight from the URL by app logic (transport
 *    and session bootstrap, e.g. TURN/ICE and the OOB hub). Never seeded or stored.
 *
 * See resolve.ts for how each kind is read.
 */
export interface UrlParam {
  /** Canonical identifier for the parameter. */
  key: string;
  /** URL query parameter name. Defaults to `key` when omitted. */
  url?: string;
  /** id of the bound form control in index.html. Present only for form-backed params. */
  elementId?: string;
  /** How a form-backed value is applied to the control. Defaults to 'value'. */
  kind?: 'value' | 'checked';
  /** Optional check for the raw URL string; invalid values are ignored. */
  isValid?: (raw: string) => boolean;
  /**
   * One-line, user-facing description. Opt-in: only params with a `description`
   * are listed in the in-app "URL parameters" help panel, so secrets/transport
   * internals (tokens, ICE credentials) stay out by simply omitting it.
   */
  description?: string;
}

const oneOf =
  (...allowed: string[]) =>
  (raw: string): boolean =>
    allowed.includes(raw);
const isBool = oneOf('true', 'false');
const isNumber = (raw: string): boolean => raw.trim() !== '' && Number.isFinite(Number(raw));

export const URL_PARAMS: UrlParam[] = [
  // --- Form-backed settings (seeded into a control, then read through the form) ---
  {
    key: 'serverIP',
    elementId: 'serverIpInput',
    description: 'CloudXR server IP/hostname (default: page URL hostname).',
  },
  { key: 'port', elementId: 'portInput', isValid: isNumber, description: 'CloudXR server port.' },
  {
    key: 'serverType',
    elementId: 'serverType',
    isValid: oneOf('manual', 'nvcf'),
    description: 'Server backend: manual or nvcf.',
  },
  {
    key: 'codec',
    elementId: 'codec',
    isValid: oneOf('h264', 'h265', 'av1'),
    description: 'Preferred video codec: h264, h265, or av1.',
  },
  {
    key: 'immersiveMode',
    elementId: 'immersive',
    isValid: oneOf('ar', 'vr'),
    description: 'WebXR session mode: ar or vr.',
  },
  // deviceProfile is intentionally omitted: it is a preset that bulk-fills the fields below via a
  // change handler, which programmatic seeding does not trigger. Set the individual fields instead.
  {
    key: 'deviceFrameRate',
    elementId: 'deviceFrameRate',
    isValid: isNumber,
    description: 'Target device frame rate in FPS (e.g. 72, 90, 120).',
  },
  {
    key: 'maxStreamingBitrateMbps',
    elementId: 'maxStreamingBitrateMbps',
    isValid: isNumber,
    description: 'Maximum streaming bitrate in Mbps.',
  },
  {
    key: 'perEyeWidth',
    elementId: 'perEyeWidth',
    isValid: isNumber,
    description: 'Per-eye render width in pixels (multiple of 16, min 128).',
  },
  {
    key: 'perEyeHeight',
    elementId: 'perEyeHeight',
    isValid: isNumber,
    description: 'Per-eye render height in pixels (multiple of 64, min 128).',
  },
  {
    key: 'reprojectionGridCols',
    elementId: 'reprojectionGridCols',
    isValid: isNumber,
    description: 'Depth reprojection mesh columns (>= 2, with rows; blank = factor mode).',
  },
  {
    key: 'reprojectionGridRows',
    elementId: 'reprojectionGridRows',
    isValid: isNumber,
    description: 'Depth reprojection mesh rows (>= 2, with columns; blank = factor mode).',
  },
  {
    key: 'enablePoseSmoothing',
    elementId: 'enablePoseSmoothing',
    isValid: isBool,
    description: 'Smooth predicted positions to reduce jitter (true/false).',
  },
  {
    key: 'posePredictionFactor',
    elementId: 'posePredictionFactor',
    isValid: isNumber,
    description: 'Pose prediction horizon scale, 0.0 (off) to 1.0 (full).',
  },
  {
    key: 'enableTexSubImage2D',
    elementId: 'enableTexSubImage2D',
    isValid: isBool,
    description: 'Use texSubImage2D texture updates; faster on Quest (true/false).',
  },
  {
    key: 'useQuestColorWorkaround',
    elementId: 'useQuestColorWorkaround',
    isValid: isBool,
    description: 'Display P3 color workaround for Quest 3 Browser (true/false).',
  },
  {
    key: 'referenceSpace',
    elementId: 'referenceSpace',
    isValid: oneOf('auto', 'local-floor', 'local', 'viewer', 'unbounded'),
    description: 'XR reference space: auto, local-floor, local, viewer, or unbounded.',
  },
  {
    key: 'xrOffsetX',
    elementId: 'xrOffsetX',
    isValid: isNumber,
    description: 'Reference-space X offset (horizontal) in centimeters.',
  },
  {
    key: 'xrOffsetY',
    elementId: 'xrOffsetY',
    isValid: isNumber,
    description: 'Reference-space Y offset (vertical) in centimeters.',
  },
  {
    key: 'xrOffsetZ',
    elementId: 'xrOffsetZ',
    isValid: isNumber,
    description: 'Reference-space Z offset (depth) in centimeters.',
  },
  {
    key: 'controlPanelPosition',
    elementId: 'controlPanelPosition',
    isValid: oneOf('left', 'center', 'right'),
    description: 'In-XR control panel start position: left, center, or right.',
  },
  {
    key: 'controlPanelDistance',
    elementId: 'controlPanelDistance',
    isValid: isNumber,
    description: 'In-XR control panel distance from the viewer, in meters.',
  },
  {
    key: 'controlPanelHeight',
    elementId: 'controlPanelHeight',
    isValid: isNumber,
    description: 'In-XR control panel height, in meters (floor-relative).',
  },
  {
    key: 'controlPanelAngleDegrees',
    elementId: 'controlPanelAngleDegrees',
    isValid: isNumber,
    description: 'In-XR control panel angle from center for the left/right positions, in degrees.',
  },
  {
    key: 'controlPanelTrackHeadset',
    elementId: 'controlPanelTrackHeadset',
    kind: 'checked',
    isValid: isBool,
    description:
      'Keep the in-XR control panel at a fixed position/rotation relative to the headset instead of the room (true/false).',
  },
  {
    key: 'controllerModelVisibility',
    elementId: 'controllerModelVisibility',
    isValid: oneOf('show', 'hide'),
    description: 'Show or hide controller model meshes in XR.',
  },
  {
    key: 'showTraceInXR',
    elementId: 'showTraceInXR',
    isValid: isBool,
    description: 'Show rolling hand/controller traces in XR (true/false).',
  },
  {
    key: 'showRecordingControls',
    elementId: 'showRecordingControls',
    isValid: isBool,
    description: 'Show recording controls in XR (true/false).',
  },
  {
    key: 'replayPacing',
    elementId: 'replayPacing',
    isValid: oneOf('frame', 'time'),
    description: 'Input replay pacing: time (default) or frame.',
  },
  {
    key: 'panelHiddenAtStart',
    elementId: 'panelHiddenAtStart',
    isValid: isBool,
    description: 'Start with the in-XR control panel hidden (true/false).',
  },
  {
    key: 'headless',
    elementId: 'cloudxrHeadless',
    kind: 'checked',
    isValid: isBool,
    description: 'Headless: skip all client render code, keep tracking (true/false).',
  },
  {
    key: 'autoRefreshMode',
    elementId: 'cloudxrAutoRefreshMode',
    isValid: oneOf('never', 'clean', 'any'),
    description: 'Reload page after session ends: never, clean, or any.',
  },
  {
    key: 'reconnectEnabled',
    elementId: 'cloudxrReconnectEnabled',
    kind: 'checked',
    isValid: isBool,
    description: 'Retry a recoverable stream error instead of ending the session (true/false).',
  },
  {
    key: 'reconnectMaxAttempts',
    elementId: 'cloudxrReconnectMaxAttempts',
    isValid: isNumber,
    description: 'Max retry attempts before giving up.',
  },
  {
    key: 'reconnectDelayMs',
    elementId: 'cloudxrReconnectDelayMs',
    isValid: isNumber,
    description: 'Delay before each retry attempt, in milliseconds.',
  },
  {
    key: 'streamAttachTimeoutMs',
    elementId: 'cloudxrStreamAttachTimeoutMs',
    isValid: isNumber,
    description:
      'Base timeout for detecting a stream that never attaches ("passthrough-only"), in milliseconds. Independent of retry.',
  },
  {
    key: 'warmupBeginTimeoutMs',
    elementId: 'cloudxrWarmupBeginTimeoutMs',
    isValid: isNumber,
    description:
      'Timeout for detecting a stream that attached but produced no decoder warm-up signal at all, in milliseconds. Does not grow on retry.',
  },
  {
    key: 'warmupEndTimeoutMs',
    elementId: 'cloudxrWarmupEndTimeoutMs',
    isValid: isNumber,
    description:
      'Timeout for detecting decoder warm-up that began but never finished, in milliseconds. Does not grow on retry.',
  },
  {
    key: 'proxyUrl',
    elementId: 'proxyUrl',
    description: 'Proxy URL for routing (HTTPS); leave empty for direct WSS.',
  },
  {
    key: 'mediaAddress',
    elementId: 'mediaAddress',
    description: 'WebRTC media server address for NAT traversal (optional).',
  },
  {
    key: 'mediaPort',
    elementId: 'mediaPort',
    isValid: isNumber,
    description: 'WebRTC media server port (0 = auto).',
  },
  {
    key: 'streamTestMode',
    elementId: 'streamTestMode',
    isValid: oneOf('off', 'warn', 'block'),
    description: 'Pre-stream network test: off (default), warn, or block.',
  },
  {
    key: 'streamTestDurationSeconds',
    elementId: 'streamTestDurationSeconds',
    isValid: isNumber,
    description: 'Network test measurement window in seconds (ignored when off).',
  },

  // --- Direct params: read straight from the URL by app logic (no control, never stored) ---
  // TURN/ICE for NAT traversal and OOB hub, typically set in USB-local mode by oob_teleop_env.py.
  // No `isValid` here: the consumers apply their own interpretation (exact matching, regex, etc.).
  // No `description`: these are set by tooling, not hand-edited, and some are secrets — keep them
  // out of the user-facing help panel.
  { key: 'turnServer' },
  { key: 'turnUsername' },
  { key: 'turnCredential' },
  { key: 'iceRelayOnly' },
  { key: 'oobEnable' },
  { key: 'controlToken' },
];
