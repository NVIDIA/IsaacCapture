// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Wire types and runtime validation for GET /api/capture-status.
 *
 * Field names, enum numeric values and the decimal-string timestamp
 * convention match backend/egocap_web/capture_status.py exactly -- this is
 * a hand-written mirror of that module, not generated, so any backend
 * field change must be mirrored here by hand too.
 */

// Must match backend/egocap_web/capture_status.py's SCHEMA_VERSION.
export const SCHEMA_VERSION = 1;

export const CaptureMode = { MOCK: 0, LIVE: 1 } as const;
export type CaptureModeValue = (typeof CaptureMode)[keyof typeof CaptureMode];

export const CaptureStatusType = { STARTING: 1, BEGUN: 2, STATUS_CHANGE: 3 } as const;
export type CaptureStatusTypeValue = (typeof CaptureStatusType)[keyof typeof CaptureStatusType];

export const StartupStage = {
  INITIALIZING: 1,
  WAITING_FOR_HEADSET: 2,
  CONNECTING: 3,
  STARTING_SOURCES: 4,
} as const;
export type StartupStageValue = (typeof StartupStage)[keyof typeof StartupStage];

export const StopReason = {
  REQUESTED: 1,
  OWNER_DISCONNECTED: 4,
  SERVER_SHUTDOWN: 6,
  IC_ERROR: 7,
  IC_EXITED: 8,
  CAMERA_ERROR: 9,
} as const;
export type StopReasonValue = (typeof StopReason)[keyof typeof StopReason];

export const ProviderState = { INITIALIZING: 1, AVAILABLE: 2, FAILED: 3, STOPPED: 4 } as const;
export type ProviderStateValue = (typeof ProviderState)[keyof typeof ProviderState];

export const DeviceState = {
  INITIALIZING: 1,
  UNKNOWN: 2,
  CONNECTED: 3,
  DISCONNECTED: 4,
  DEGRADED: 5,
  FAILED: 6,
  UNAVAILABLE: 7,
  DISABLED: 8,
} as const;
export type DeviceStateValue = (typeof DeviceState)[keyof typeof DeviceState];

/** The server's camera delivery check; STALE is a pass that has aged out and no longer counts. */
export const NetworkCheckState = { NOT_TESTED: 0, CHECKING: 1, READY: 2, UNSUITABLE: 3, STALE: 4 } as const;
export type NetworkCheckStateValue = (typeof NetworkCheckState)[keyof typeof NetworkCheckState];

export const ProviderType = { PLUGIN: 1, OPENXR_RUNTIME: 2 } as const;
export type ProviderTypeValue = (typeof ProviderType)[keyof typeof ProviderType];

export const PluginType = { MANUS: 1 } as const;

export const StatusReason = {
  NONE: 0,
  PROVIDER_INITIALIZING: 1,
  PROVIDER_FAILED: 2,
  PROVIDER_STOPPED: 3,
  NO_REPORT: 4,
  STALE_REPORT: 5,
  MALFORMED_REPORT: 6,
  UNSUPPORTED_SCHEMA_VERSION: 7,
  NO_HARDWARE_SIGNAL: 8,
  CALIBRATION_FAILED: 9,
  NO_CURRENT_DATA: 10,
  DEVICE_ERROR: 11,
  DISABLED_BY_CONFIGURATION: 12,
  PROCESS_EXITED: 13,
  PROCESS_SIGNALED: 14,
  PROCESS_OBSERVATION_ERROR: 15,
  STARTUP_FAILED: 16,
  RUNTIME_UPDATE_FAILED: 17,
  OPENXR_SESSION_READY: 18,
  EXTERNAL_OPENXR_HEALTH_UNKNOWN: 19,
  OPENXR_FORM_FACTOR_UNAVAILABLE: 20,
  OPENXR_SESSION_LOST: 21,
  OPENXR_INSTANCE_LOST: 22,
  OPENXR_POLL_ERROR: 23,
} as const;
export type StatusReasonValue = (typeof StatusReason)[keyof typeof StatusReason];

/** `SCREAMING_SNAKE_CASE` -> `Screaming snake case`, driven by the actual
 * enum member names above, never a separately hand-written label list that
 * could silently drift from the numeric values it describes. */
function deriveLabel(enumObj: Record<string, number>, value: number | null): string {
  if (value === null) return "Unknown";
  const name = Object.keys(enumObj).find((k) => enumObj[k] === value);
  if (name === undefined) return `Unrecognized (${value})`;
  const lower = name.toLowerCase().replaceAll("_", " ");
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

export const deviceStateLabel = (v: DeviceStateValue | null): string => deriveLabel(DeviceState, v);
export const providerStateLabel = (v: ProviderStateValue | null): string =>
  deriveLabel(ProviderState, v);
export const statusReasonLabel = (v: StatusReasonValue | null): string =>
  deriveLabel(StatusReason, v);
export const stopReasonLabel = (v: StopReasonValue | null): string => deriveLabel(StopReason, v);
export const startupStageLabel = (v: StartupStageValue | null): string =>
  deriveLabel(StartupStage, v);

export interface ProviderStatus {
  id: string;
  type: ProviderTypeValue;
  plugin: number | null;
  name: string;
  status: ProviderStateValue;
  reason: StatusReasonValue;
  error: string;
  pid: number | null;
  exit_code: number | null;
  term_signal: number | null;
}

export interface DeviceStatus {
  id: string;
  path: string | null;
  type: string;
  description: string;
  provider_id: string | null;
  status: DeviceStateValue;
  reason: StatusReasonValue;
  error: string;
}

export interface CameraStatus {
  name: string;
  stream_id: number;
  status: DeviceStateValue;
  reason: StatusReasonValue;
  error: string;
  intrinsics: number[][] | null;
}

/** What the check measured for one camera. A null measurement is unknown, never zero. */
export interface NetworkCheckCamera {
  stream_id: number;
  name: string;
  /** null while the camera has not been judged. */
  passed: boolean | null;
  expected_fps: number;
  usable_frames: number | null;
  delivered_fps: number | null;
  missing_frames: number | null;
  max_capture_gap_ms: number | null;
  max_arrival_gap_ms: number | null;
  start_silence_ms: number | null;
  end_silence_ms: number | null;
  age_median_ms: number | null;
  age_max_ms: number | null;
  age_growth_ms: number | null;
  reasons: string[];
}

/** CaptureServer's verdict on whether its cameras' frames arrive well enough to record: media delivery only, apart
 * from recorder and disk health. `generation` names the cameras and settings that were tested. */
export interface NetworkCheckStatus {
  state: NetworkCheckStateValue;
  generation: string;
  endpoint: string;
  legs_tested: string[];
  legs_not_tested: string[];
  window_s: number;
  valid_for_s: number;
  phase: string | null;
  progress_s: number | null;
  completed_at_unix_ns: number | null;
  reason: string | null;
  cameras: NetworkCheckCamera[];
}

export interface StatusSnapshot {
  /** Capture-host nanosecond clock, as a decimal string -- see captureStatus.ts's
   * module docstring and capture_status.py's to_json() for why. Not a browser
   * freshness clock; do not subtract it from Date.now(). */
  updated_at_ns: string;
  providers: ProviderStatus[];
  devices: DeviceStatus[];
  cameras: CameraStatus[];
  /** null when the server runs no network check. */
  network_check: NetworkCheckStatus | null;
}

export interface CaptureStatusResponse {
  schema_version: number;
  backend_instance_id: string;
  station_label: string;
  profile_label: string;
  mode: CaptureModeValue | null;
  session_id: string | null;
  connecting: boolean;
  status_type: CaptureStatusTypeValue | null;
  running: boolean;
  startup_stage: StartupStageValue | null;
  message: string | null;
  terminal: boolean;
  stop_reason: StopReasonValue | null;
  error_message: string | null;
  status: StatusSnapshot | null;
}

/** Thrown by parseCaptureStatusResponse on a malformed body -- the caller
 * treats this the same as a network failure (stale/unavailable), never as
 * if it had received a half-valid status. */
export class CaptureStatusParseError extends Error {}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function expectString(value: unknown, field: string): string {
  if (typeof value !== "string") throw new CaptureStatusParseError(`${field} must be a string`);
  return value;
}

function expectNullableString(value: unknown, field: string): string | null {
  if (value === null) return null;
  return expectString(value, field);
}

function expectBoolean(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") throw new CaptureStatusParseError(`${field} must be a boolean`);
  return value;
}

function expectNumber(value: unknown, field: string): number {
  if (typeof value !== "number") throw new CaptureStatusParseError(`${field} must be a number`);
  return value;
}

function expectNullableNumber(value: unknown, field: string): number | null {
  if (value === null) return null;
  return expectNumber(value, field);
}

/** Validates that `value` is one of `enumObj`'s own numeric values -- the
 * wire contract is a closed set of integers, so anything else (including a
 * value some *future* server version might add) must not silently become a
 * fabricated label; see e.g. modeLabel() in App.tsx. */
function expectEnum<T extends Record<string, number>>(enumObj: T, value: unknown, field: string): T[keyof T] {
  const n = expectNumber(value, field);
  if (!Object.values(enumObj).includes(n)) {
    throw new CaptureStatusParseError(`${field} ${n} is not a recognized value`);
  }
  return n as T[keyof T];
}

function expectNullableEnum<T extends Record<string, number>>(
  enumObj: T,
  value: unknown,
  field: string,
): T[keyof T] | null {
  if (value === null) return null;
  return expectEnum(enumObj, value, field);
}

function parseProviderStatus(value: unknown, field: string): ProviderStatus {
  if (!isObject(value)) throw new CaptureStatusParseError(`${field} must be an object`);
  return {
    id: expectString(value.id, `${field}.id`),
    type: expectEnum(ProviderType, value.type, `${field}.type`),
    plugin: expectNullableEnum(PluginType, value.plugin, `${field}.plugin`),
    name: expectString(value.name, `${field}.name`),
    status: expectEnum(ProviderState, value.status, `${field}.status`),
    reason: expectEnum(StatusReason, value.reason, `${field}.reason`),
    error: expectString(value.error, `${field}.error`),
    pid: expectNullableNumber(value.pid, `${field}.pid`),
    exit_code: expectNullableNumber(value.exit_code, `${field}.exit_code`),
    term_signal: expectNullableNumber(value.term_signal, `${field}.term_signal`),
  };
}

function parseDeviceStatus(value: unknown, field: string): DeviceStatus {
  if (!isObject(value)) throw new CaptureStatusParseError(`${field} must be an object`);
  return {
    id: expectString(value.id, `${field}.id`),
    path: expectNullableString(value.path, `${field}.path`),
    type: expectString(value.type, `${field}.type`),
    description: expectString(value.description, `${field}.description`),
    provider_id: expectNullableString(value.provider_id, `${field}.provider_id`),
    status: expectEnum(DeviceState, value.status, `${field}.status`),
    reason: expectEnum(StatusReason, value.reason, `${field}.reason`),
    error: expectString(value.error, `${field}.error`),
  };
}

function parseCameraStatus(value: unknown, field: string): CameraStatus {
  if (!isObject(value)) throw new CaptureStatusParseError(`${field} must be an object`);
  return {
    name: expectString(value.name, `${field}.name`),
    stream_id: expectNumber(value.stream_id, `${field}.stream_id`),
    status: expectEnum(DeviceState, value.status, `${field}.status`),
    reason: expectEnum(StatusReason, value.reason, `${field}.reason`),
    error: expectString(value.error, `${field}.error`),
    // Not rendered by this page and not validated -- an unchecked cast,
    // left explicitly unvalidated rather than claimed-safe. Only the
    // null-vs-present distinction is real; the shape inside a present
    // value is never checked. Validate it for real before ever reading it.
    intrinsics: value.intrinsics === null ? null : (value.intrinsics as number[][]),
  };
}

function expectStringArray(value: unknown, field: string): string[] {
  if (!Array.isArray(value)) throw new CaptureStatusParseError(`${field} must be an array`);
  return value.map((v, i) => expectString(v, `${field}[${i}]`));
}

function parseNetworkCheckCamera(value: unknown, field: string): NetworkCheckCamera {
  if (!isObject(value)) throw new CaptureStatusParseError(`${field} must be an object`);
  const passed = value.passed;
  if (passed !== null && typeof passed !== "boolean") {
    throw new CaptureStatusParseError(`${field}.passed must be a boolean or null`);
  }
  const num = (key: string) => expectNullableNumber(value[key], `${field}.${key}`);
  return {
    stream_id: expectNumber(value.stream_id, `${field}.stream_id`),
    name: expectString(value.name, `${field}.name`),
    passed,
    expected_fps: expectNumber(value.expected_fps, `${field}.expected_fps`),
    usable_frames: num("usable_frames"),
    delivered_fps: num("delivered_fps"),
    missing_frames: num("missing_frames"),
    max_capture_gap_ms: num("max_capture_gap_ms"),
    max_arrival_gap_ms: num("max_arrival_gap_ms"),
    start_silence_ms: num("start_silence_ms"),
    end_silence_ms: num("end_silence_ms"),
    age_median_ms: num("age_median_ms"),
    age_max_ms: num("age_max_ms"),
    age_growth_ms: num("age_growth_ms"),
    reasons: expectStringArray(value.reasons, `${field}.reasons`),
  };
}

function parseNetworkCheck(value: unknown, field: string): NetworkCheckStatus {
  if (!isObject(value)) throw new CaptureStatusParseError(`${field} must be an object`);
  if (!Array.isArray(value.cameras)) throw new CaptureStatusParseError(`${field}.cameras must be an array`);
  return {
    state: expectEnum(NetworkCheckState, value.state, `${field}.state`),
    generation: expectString(value.generation, `${field}.generation`),
    endpoint: expectString(value.endpoint, `${field}.endpoint`),
    legs_tested: expectStringArray(value.legs_tested, `${field}.legs_tested`),
    legs_not_tested: expectStringArray(value.legs_not_tested, `${field}.legs_not_tested`),
    window_s: expectNumber(value.window_s, `${field}.window_s`),
    valid_for_s: expectNumber(value.valid_for_s, `${field}.valid_for_s`),
    phase: expectNullableString(value.phase, `${field}.phase`),
    progress_s: expectNullableNumber(value.progress_s, `${field}.progress_s`),
    completed_at_unix_ns: expectNullableNumber(value.completed_at_unix_ns, `${field}.completed_at_unix_ns`),
    reason: expectNullableString(value.reason, `${field}.reason`),
    cameras: value.cameras.map((v, i) => parseNetworkCheckCamera(v, `${field}.cameras[${i}]`)),
  };
}

/** One camera CaptureServer is configured with, merged with what a running session has observed of it. A null
 * `state` means no session has observed it; a LiveKit camera is offered, never opened, by the server. */
export interface ServerCamera {
  name: string;
  transport: "capture_server" | "livekit";
  selected: boolean;
  device_id: string | null;
  room: string | null;
  publisher: string | null;
  track: string | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  stream_id: number | null;
  state: DeviceStateValue | null;
  error: string | null;
}

/** CaptureServer's own general status (its read-only status query), not the recording page's view of it. */
export interface ServerStatus {
  mode: CaptureModeValue;
  session_id: string | null;
  status: StatusSnapshot | null;
  network_check: NetworkCheckStatus | null;
  cameras: ServerCamera[];
}

function parseServerCamera(value: unknown, field: string): ServerCamera {
  if (!isObject(value)) throw new CaptureStatusParseError(`${field} must be an object`);
  const transport = value.transport;
  if (transport !== "capture_server" && transport !== "livekit") {
    throw new CaptureStatusParseError(`${field}.transport must be capture_server or livekit`);
  }
  const num = (key: string) => expectNullableNumber(value[key], `${field}.${key}`);
  const text = (key: string) => expectNullableString(value[key], `${field}.${key}`);
  return {
    name: expectString(value.name, `${field}.name`),
    transport,
    selected: expectBoolean(value.selected, `${field}.selected`),
    device_id: text("device_id"),
    room: text("room"),
    publisher: text("publisher"),
    track: text("track"),
    width: num("width"),
    height: num("height"),
    fps: num("fps"),
    stream_id: num("stream_id"),
    state: expectNullableEnum(DeviceState, value.state, `${field}.state`),
    error: text("error"),
  };
}

export function parseServerStatus(body: unknown): ServerStatus {
  if (!isObject(body)) throw new CaptureStatusParseError("server status must be an object");
  if (!Array.isArray(body.cameras)) throw new CaptureStatusParseError("cameras must be an array");
  return {
    mode: expectEnum(CaptureMode, body.mode, "mode"),
    session_id: expectNullableString(body.session_id, "session_id"),
    status: body.status === null ? null : parseStatusSnapshot(body.status),
    network_check: body.network_check === null ? null : parseNetworkCheck(body.network_check, "network_check"),
    cameras: body.cameras.map((c, i) => parseServerCamera(c, `cameras[${i}]`)),
  };
}

export function parseStatusSnapshot(value: unknown): StatusSnapshot {
  if (!isObject(value)) throw new CaptureStatusParseError("status must be an object");
  const updated = value.updated_at_ns;
  if (typeof updated !== "string" || !/^\d+$/.test(updated)) {
    throw new CaptureStatusParseError("status.updated_at_ns must be a decimal string");
  }
  const providers = value.providers;
  const devices = value.devices;
  const cameras = value.cameras;
  if (!Array.isArray(providers) || !Array.isArray(devices) || !Array.isArray(cameras)) {
    throw new CaptureStatusParseError("status.providers/devices/cameras must be arrays");
  }
  return {
    updated_at_ns: updated,
    providers: providers.map((v, i) => parseProviderStatus(v, `status.providers[${i}]`)),
    devices: devices.map((v, i) => parseDeviceStatus(v, `status.devices[${i}]`)),
    cameras: cameras.map((v, i) => parseCameraStatus(v, `status.cameras[${i}]`)),
    // Absent from a server that runs no check, which is not the same as a check that has not run.
    network_check:
      value.network_check === undefined || value.network_check === null
        ? null
        : parseNetworkCheck(value.network_check, "status.network_check"),
  };
}

/** Validates a parsed JSON body against the documented response shape.
 * Throws CaptureStatusParseError rather than returning a half-trusted
 * value -- the hook treats a parse failure the same as a fetch failure. */
export function parseCaptureStatusResponse(body: unknown): CaptureStatusResponse {
  if (!isObject(body)) throw new CaptureStatusParseError("response body must be an object");
  const schemaVersion = expectNumber(body.schema_version, "schema_version");
  if (schemaVersion !== SCHEMA_VERSION) {
    throw new CaptureStatusParseError(
      `schema_version ${schemaVersion} is not supported (expected ${SCHEMA_VERSION})`,
    );
  }
  return {
    schema_version: schemaVersion,
    backend_instance_id: expectString(body.backend_instance_id, "backend_instance_id"),
    station_label: expectString(body.station_label, "station_label"),
    profile_label: expectString(body.profile_label, "profile_label"),
    mode: expectNullableEnum(CaptureMode, body.mode, "mode"),
    session_id: expectNullableString(body.session_id, "session_id"),
    connecting: expectBoolean(body.connecting, "connecting"),
    status_type: expectNullableEnum(CaptureStatusType, body.status_type, "status_type"),
    running: expectBoolean(body.running, "running"),
    startup_stage: expectNullableEnum(StartupStage, body.startup_stage, "startup_stage"),
    message: expectNullableString(body.message, "message"),
    terminal: expectBoolean(body.terminal, "terminal"),
    stop_reason: expectNullableEnum(StopReason, body.stop_reason, "stop_reason"),
    error_message: expectNullableString(body.error_message, "error_message"),
    status: body.status === null ? null : parseStatusSnapshot(body.status),
  };
}
