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

export interface StatusSnapshot {
  /** Capture-host nanosecond clock, as a decimal string -- see captureStatus.ts's
   * module docstring and capture_status.py's to_json() for why. Not a browser
   * freshness clock; do not subtract it from Date.now(). */
  updated_at_ns: string;
  providers: ProviderStatus[];
  devices: DeviceStatus[];
  cameras: CameraStatus[];
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
