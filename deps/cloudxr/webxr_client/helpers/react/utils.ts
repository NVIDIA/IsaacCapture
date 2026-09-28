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
 * Shared utilities for React examples (e.g. control panel position).
 */

export type ControlPanelPosition = 'left' | 'center' | 'right';

/**
 * Loads a per-project-path setting from localStorage (key `cxr.isaac.<key>|<teleopPath>`).
 * `parse` should return `undefined` for unrecognized input so `fallback` wins.
 */
export function loadPerProject<T>(
  key: string,
  teleopPath: string,
  parse: (raw: string) => T | undefined,
  fallback: T
): T {
  try {
    const stored = localStorage.getItem(`cxr.isaac.${key}|${teleopPath}`);
    if (stored !== null) {
      const parsed = parse(stored);
      if (parsed !== undefined) return parsed;
    }
  } catch {
    /* localStorage unavailable */
  }
  return fallback;
}

/** Generic per-project-path localStorage save. See {@link loadPerProject}. */
export function savePerProject<T>(
  key: string,
  teleopPath: string,
  value: T,
  serialize: (v: T) => string = String
): void {
  try {
    localStorage.setItem(`cxr.isaac.${key}|${teleopPath}`, serialize(value));
  } catch {
    /* localStorage unavailable */
  }
}

/** React UI options (e.g. in-XR control panel position). */
export interface ReactUIConfig {
  controlPanelPosition?: ControlPanelPosition;
  /** When true, the control panel is hidden at immersive XR enter (small “show control panel” control only). */
  panelHiddenAtStart?: boolean;
  /** When true, all WebGL rendering is skipped. */
  headless?: boolean;
  /** Page-refresh behavior when the XR session ends. See {@link AutoRefreshMode}. */
  autoRefreshMode?: AutoRefreshMode;
  /** When true, CloudXRComponent retries a recoverable stream error instead of giving up immediately. */
  reconnectEnabled?: boolean;
  /** Max retry attempts before giving up. Only used when reconnectEnabled is true. */
  reconnectMaxAttempts?: number;
  /** Delay between retry attempts in milliseconds. Only used when reconnectEnabled is true. */
  reconnectDelayMs?: number;
  /** Active teleop project path (a key path in `TELEOP_PROJECTS`). */
  teleopPath: string;
}

/** When to reload the page once the XR session ends: never, only on a clean end, or on any end. */
export type AutoRefreshMode = 'never' | 'clean' | 'any';

const AUTO_REFRESH_MODES: readonly AutoRefreshMode[] = ['never', 'clean', 'any'];

/**
 * Parses a string into a valid auto-refresh mode.
 * @param unvalidatedValue - String to validate (e.g. from URL, config, or form). May be invalid or empty.
 * @param fallback - Value to return when unvalidatedValue is not valid.
 */
export function parseAutoRefreshMode(
  unvalidatedValue: string,
  fallback: AutoRefreshMode
): AutoRefreshMode {
  if (AUTO_REFRESH_MODES.includes(unvalidatedValue as AutoRefreshMode)) {
    return unvalidatedValue as AutoRefreshMode;
  }
  return fallback;
}

/** Pre-stream network test mode: skip it, measure and report, or measure and gate on it. */
export type StreamTestMode = 'off' | 'warn' | 'block';

const STREAM_TEST_MODES: readonly StreamTestMode[] = ['off', 'warn', 'block'];

/**
 * Parses a string into a valid network test mode.
 * @param unvalidatedValue - String to validate (e.g. from URL, config, or form). May be invalid or empty.
 * @param fallback - Value to return when unvalidatedValue is not valid.
 */
export function parseStreamTestMode(
  unvalidatedValue: string,
  fallback: StreamTestMode
): StreamTestMode {
  if (STREAM_TEST_MODES.includes(unvalidatedValue as StreamTestMode)) {
    return unvalidatedValue as StreamTestMode;
  }
  return fallback;
}

const CONTROL_PANEL_POSITIONS: readonly ControlPanelPosition[] = ['left', 'center', 'right'];

/**
 * Parses a string into a valid control panel position.
 * @param unvalidatedValue - String to validate (e.g. from URL, config, or form). May be invalid or empty.
 * @param fallback - Value to return when unvalidatedValue is not valid.
 */
export function parseControlPanelPosition(
  unvalidatedValue: string,
  fallback: ControlPanelPosition
): ControlPanelPosition {
  if (CONTROL_PANEL_POSITIONS.includes(unvalidatedValue as ControlPanelPosition)) {
    return unvalidatedValue as ControlPanelPosition;
  }
  return fallback;
}

export interface ControlPanelLayoutOptions {
  /** Distance from viewer to panel (meters). */
  distance: number;
  /** Height of panel (meters). */
  height: number;
  /** Angle in degrees for left/right positions from center. */
  angleDegrees: number;
}

/**
 * Returns [x, y, z] for the in-XR control panel. Center is in front; left/right at the given angle.
 */
export function getControlPanelPositionVector(
  pos: ControlPanelPosition,
  layout: ControlPanelLayoutOptions
): [number, number, number] {
  if (pos === 'center') {
    return [0, layout.height, -layout.distance];
  }
  const rad = (layout.angleDegrees * Math.PI) / 180;
  const x = layout.distance * Math.sin(rad);
  const z = -layout.distance * Math.cos(rad);
  return [pos === 'left' ? -x : x, layout.height, z];
}
