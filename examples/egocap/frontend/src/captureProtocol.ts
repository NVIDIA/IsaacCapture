/**
 * WS /api/session SNAPSHOT=3 types, runtime validation, and the 32-byte
 * browser binary tracking header. Mirrors backend/egocap_web/preview_protocol.py
 * by hand -- any backend field change must be mirrored here too. This is a
 * browser-facing envelope distinct from CaptureServer's own wire protocol;
 * the binary header's bytes are not the upstream GTC1 header.
 */

import {
  CaptureMode,
  CaptureStatusType,
  StartupStage,
  StopReason,
  parseStatusSnapshot,
  type CaptureModeValue,
  type CaptureStatusTypeValue,
  type StartupStageValue,
  type StopReasonValue,
  type StatusSnapshot,
} from "./captureStatus";

export const PREVIEW_VERSION = 1;
export const SNAPSHOT_TYPE = 3;

// Mirrors the backend's MAX_RECORDING_COMMAND_BYTES (preview_protocol.py) so an oversized recording command is
// rejected locally, before it reaches the socket, instead of relying on the backend to close the connection for it.
export const MAX_RECORDING_COMMAND_BYTES = 8 * 1024;

export const BINARY_HEADER_SIZE = 32;
export const BINARY_KIND_TRACKING = 1;
export const BINARY_KIND_IMAGE = 2;

export const WireDType = { UINT8: 1, FLOAT32_LE: 2, FLOAT64_LE: 3 } as const;
export type WireDTypeValue = (typeof WireDType)[keyof typeof WireDType];

// The native enum values this page supports, mirrored from
// capture_server/protocol.py -- not guessed widths or representations.
// A descriptor advertising anything else is an unsupported-protocol error,
// not something to silently ignore and keep showing "connected" for.
export const StreamKind = { TRACKING: 1, IMAGE: 2 } as const;
export const TrackingSchema = { IC_TRACKING_V1: 1 } as const;
export const PoseRepresentation = { POSITION_QUATERNION: 1 } as const;
export const QuaternionOrder = { XYZW: 1 } as const;
export const PositionUnit = { METERS: 1 } as const;
export const Axes = { RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD: 1 } as const;
export const HandJointOrder = { OPENXR_26: 1 } as const;

// Bounds on a descriptor's own numbers -- initial application limits for
// this preview relay (not a hardware-capacity claim), checked so a
// corrupt/hostile descriptor can't request an unreasonable allocation.
const MAX_SAFE_OFFSET = 1_000_000;
const MAX_PAYLOAD_SIZE = 1_000_000;
const MAX_FIELD_COUNT = 128;
const MAX_SHAPE_DIMS = 8;
const MAX_SHAPE_PRODUCT = 100_000;
const MAX_CAMERA_DIMENSION = 16_384;

/** Validates a positive integer no greater than `max`. */
function expectBoundedPositiveInt(value: unknown, field: string, max: number): number {
  const n = expectNumber(value, field);
  if (!Number.isInteger(n) || n <= 0 || n > max) {
    throw new CapturePreviewParseError(`${field} must be a positive integer at most ${max}, got ${n}`);
  }
  return n;
}

/** Validates a nonnegative integer no greater than `max`. */
function expectBoundedNonnegativeInt(value: unknown, field: string, max: number): number {
  const n = expectNumber(value, field);
  if (!Number.isInteger(n) || n < 0 || n > max) {
    throw new CapturePreviewParseError(`${field} must be a nonnegative integer at most ${max}, got ${n}`);
  }
  return n;
}

const DTYPE_ITEM_BYTES: Record<number, number> = {
  [WireDType.UINT8]: 1,
  [WireDType.FLOAT32_LE]: 4,
  [WireDType.FLOAT64_LE]: 8,
};

/** The exact ic_tracking_v1 field contract (name -> dtype/shape), mirrored
 * by hand from capture_server/protocol.py's _build_ic_tracking_v1_fields.
 * A field whose name matches one of these must have exactly this
 * dtype/shape; a mismatch is a hard protocol error, not a guessed-width
 * fallback. An advertised field name NOT in this table is tolerated
 * (ignored at decode time) as forward-compatible extra data, matching the
 * plan's "decode the advertised field names" scope -- it is not itself
 * evidence of an unsupported schema when TrackingSchema is already
 * IC_TRACKING_V1. */
const KNOWN_TRACKING_FIELDS: Record<string, { dtype: WireDTypeValue; shape: number[] }> = (() => {
  const fields: Record<string, { dtype: WireDTypeValue; shape: number[] }> = {
    "head.present": { dtype: WireDType.UINT8, shape: [] },
    "head.position": { dtype: WireDType.FLOAT32_LE, shape: [3] },
    "head.orientation": { dtype: WireDType.FLOAT32_LE, shape: [4] },
    "head.is_valid": { dtype: WireDType.UINT8, shape: [] },
    "head.is_tracked": { dtype: WireDType.UINT8, shape: [] },
  };
  for (const prefix of ["controller_left", "controller_right"]) {
    fields[`${prefix}.present`] = { dtype: WireDType.UINT8, shape: [] };
    fields[`${prefix}.grip_position`] = { dtype: WireDType.FLOAT32_LE, shape: [3] };
    fields[`${prefix}.grip_orientation`] = { dtype: WireDType.FLOAT32_LE, shape: [4] };
    fields[`${prefix}.grip_is_valid`] = { dtype: WireDType.UINT8, shape: [] };
    fields[`${prefix}.aim_position`] = { dtype: WireDType.FLOAT32_LE, shape: [3] };
    fields[`${prefix}.aim_orientation`] = { dtype: WireDType.FLOAT32_LE, shape: [4] };
    fields[`${prefix}.aim_is_valid`] = { dtype: WireDType.UINT8, shape: [] };
    for (const input of [
      "primary_click",
      "secondary_click",
      "thumbstick_x",
      "thumbstick_y",
      "thumbstick_click",
      "menu_click",
      "squeeze_value",
      "trigger_value",
    ]) {
      fields[`${prefix}.${input}`] = { dtype: WireDType.FLOAT64_LE, shape: [] };
    }
  }
  for (const prefix of ["hand_left", "hand_right"]) {
    fields[`${prefix}.present`] = { dtype: WireDType.UINT8, shape: [] };
    fields[`${prefix}.joint_positions`] = { dtype: WireDType.FLOAT32_LE, shape: [26, 3] };
    fields[`${prefix}.joint_orientations`] = { dtype: WireDType.FLOAT32_LE, shape: [26, 4] };
    fields[`${prefix}.joint_radii`] = { dtype: WireDType.FLOAT32_LE, shape: [26] };
    fields[`${prefix}.joint_valid`] = { dtype: WireDType.UINT8, shape: [26] };
  }
  return fields;
})();

export class CapturePreviewParseError extends Error {}

/** Narrows to a non-null object. */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Validates a string field. */
function expectString(value: unknown, field: string): string {
  if (typeof value !== "string") throw new CapturePreviewParseError(`${field} must be a string`);
  return value;
}

/** Validates a string-or-null field. */
function expectNullableString(value: unknown, field: string): string | null {
  return value === null ? null : expectString(value, field);
}

/** Validates a boolean field. */
function expectBoolean(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") throw new CapturePreviewParseError(`${field} must be a boolean`);
  return value;
}

/** Validates a number field. */
function expectNumber(value: unknown, field: string): number {
  if (typeof value !== "number") throw new CapturePreviewParseError(`${field} must be a number`);
  return value;
}

/** Validates a number-or-null field. */
function expectNullableNumber(value: unknown, field: string): number | null {
  return value === null ? null : expectNumber(value, field);
}

/** Validates a decimal-string-encoded uint64 (e.g. a sequence number) or null. */
function expectDecimalStringOrNull(value: unknown, field: string): string | null {
  if (value === null) return null;
  if (typeof value !== "string" || !/^\d+$/.test(value)) {
    throw new CapturePreviewParseError(`${field} must be a decimal string`);
  }
  return value;
}

/** Validates a number field is one of `enumObj`'s recognized values. */
function expectEnum<T extends Record<string, number>>(enumObj: T, value: unknown, field: string): T[keyof T] {
  const n = expectNumber(value, field);
  if (!Object.values(enumObj).includes(n)) {
    throw new CapturePreviewParseError(`${field} ${n} is not a recognized value`);
  }
  return n as T[keyof T];
}

/** Validates an enum-or-null field. */
function expectNullableEnum<T extends Record<string, number>>(
  enumObj: T,
  value: unknown,
  field: string,
): T[keyof T] | null {
  return value === null ? null : expectEnum(enumObj, value, field);
}

export interface FieldDescriptor {
  name: string;
  dtype: WireDTypeValue;
  shape: number[];
  offset: number;
}

/** Validates one tracking field descriptor, including bounded shape/offset
 * extents and an exact dtype/shape match against KNOWN_TRACKING_FIELDS for
 * any recognized field name. */
function parseFieldDescriptor(value: unknown, field: string): FieldDescriptor {
  if (!isObject(value)) throw new CapturePreviewParseError(`${field} must be an object`);
  const name = expectString(value.name, `${field}.name`);
  if (name === "") throw new CapturePreviewParseError(`${field}.name must be nonempty`);
  const dtype = expectEnum(WireDType, value.dtype, `${field}.dtype`);
  const shapeRaw = value.shape;
  if (!Array.isArray(shapeRaw) || shapeRaw.length > MAX_SHAPE_DIMS) {
    throw new CapturePreviewParseError(`${field}.shape must be an array of at most ${MAX_SHAPE_DIMS} dimensions`);
  }
  let product = 1;
  const shape = shapeRaw.map((d, i) => {
    const dim = expectBoundedPositiveInt(d, `${field}.shape[${i}]`, MAX_SHAPE_PRODUCT);
    product *= dim;
    if (product > MAX_SHAPE_PRODUCT) {
      throw new CapturePreviewParseError(`${field}.shape product exceeds ${MAX_SHAPE_PRODUCT}`);
    }
    return dim;
  });
  const offset = expectBoundedNonnegativeInt(value.offset, `${field}.offset`, MAX_SAFE_OFFSET);
  const known = KNOWN_TRACKING_FIELDS[name];
  if (known !== undefined && (known.dtype !== dtype || JSON.stringify(known.shape) !== JSON.stringify(shape))) {
    throw new CapturePreviewParseError(
      `${field} (${name}) has dtype/shape [${dtype}]/[${shape.join(",")}], expected ` +
        `[${known.dtype}]/[${known.shape.join(",")}]`,
    );
  }
  return { name, dtype, shape, offset };
}

export interface TrackingStreamDescriptor {
  kind: "tracking";
  stream_id: number;
  payload_size: number;
  fields: FieldDescriptor[];
  /** Metadata needed to label observations -- all validated against the
   * one supported native contract below, not guessed. */
  positionUnit: typeof PositionUnit.METERS;
  quaternionOrder: typeof QuaternionOrder.XYZW;
  axes: typeof Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD;
  handJointOrder: typeof HandJointOrder.OPENXR_26;
}

export interface CameraStreamDescriptor {
  kind: "camera";
  stream_id: number;
  name: string;
  width: number;
  height: number;
  fps: number;
}

export type StreamDescriptor = TrackingStreamDescriptor | CameraStreamDescriptor;

/** Validates every field's extent is within payload_size and that no two
 * fields overlap. This is the one authoritative check for a descriptor's
 * byte layout -- a failure here fails the whole snapshot parse (-> a
 * visible incompatible-protocol transition) rather than quietly producing
 * an empty preview while still "connected". trackingDecoder.decodeTracking
 * relies on this having already run and does not re-check bounds per
 * sample. */
function validateFieldExtents(fields: FieldDescriptor[], payloadSize: number, field: string): void {
  const spans = fields
    .map((f) => {
      const itemBytes = DTYPE_ITEM_BYTES[f.dtype];
      const count = f.shape.reduce((a, b) => a * b, 1);
      return { name: f.name, start: f.offset, end: f.offset + count * itemBytes };
    })
    .sort((a, b) => a.start - b.start);
  let prevEnd = 0;
  for (const span of spans) {
    if (span.end > payloadSize) {
      throw new CapturePreviewParseError(
        `${field}: field ${span.name} extent [${span.start},${span.end}) exceeds payload_size ${payloadSize}`,
      );
    }
    if (span.start < prevEnd) {
      throw new CapturePreviewParseError(`${field}: field ${span.name} overlaps a preceding field`);
    }
    prevEnd = span.end;
  }
}

/** Validates one advertised stream descriptor (tracking or camera), including
 * the supported native contract enums for a tracking stream. */
function parseStreamDescriptor(value: unknown, field: string): StreamDescriptor {
  if (!isObject(value)) throw new CapturePreviewParseError(`${field} must be an object`);
  const kind = expectEnum(StreamKind, value.kind, `${field}.kind`);
  if (kind === StreamKind.TRACKING) {
    const streamId = value.stream_id;
    if (streamId !== 0) throw new CapturePreviewParseError(`${field}.stream_id must be 0 for a tracking stream`);
    expectEnum(TrackingSchema, value.schema, `${field}.schema`);
    expectEnum(PoseRepresentation, value.pose_representation, `${field}.pose_representation`);
    expectEnum(QuaternionOrder, value.quaternion_order, `${field}.quaternion_order`);
    expectEnum(PositionUnit, value.position_unit, `${field}.position_unit`);
    expectEnum(Axes, value.axes, `${field}.axes`);
    expectEnum(HandJointOrder, value.hand_joint_order, `${field}.hand_joint_order`);
    const layout = value.layout;
    if (!isObject(layout)) throw new CapturePreviewParseError(`${field}.layout must be an object`);
    const payloadSize = expectBoundedPositiveInt(layout.payload_size, `${field}.layout.payload_size`, MAX_PAYLOAD_SIZE);
    const fieldsRaw = layout.fields;
    if (!Array.isArray(fieldsRaw) || fieldsRaw.length > MAX_FIELD_COUNT) {
      throw new CapturePreviewParseError(`${field}.layout.fields must be an array of at most ${MAX_FIELD_COUNT} entries`);
    }
    const seen = new Set<string>();
    const fields = fieldsRaw.map((f, i) => {
      const parsed = parseFieldDescriptor(f, `${field}.layout.fields[${i}]`);
      if (seen.has(parsed.name)) {
        throw new CapturePreviewParseError(`${field}.layout.fields has duplicate name ${parsed.name}`);
      }
      seen.add(parsed.name);
      return parsed;
    });
    validateFieldExtents(fields, payloadSize, field);
    return {
      kind: "tracking",
      stream_id: 0,
      payload_size: payloadSize,
      fields,
      positionUnit: PositionUnit.METERS,
      quaternionOrder: QuaternionOrder.XYZW,
      axes: Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD,
      handJointOrder: HandJointOrder.OPENXR_26,
    };
  }
  // kind === StreamKind.IMAGE (the only other value expectEnum(StreamKind, ...) allows)
  const streamId = expectBoundedPositiveInt(value.stream_id, `${field}.stream_id`, 3);
  return {
    kind: "camera",
    stream_id: streamId,
    name: expectString(value.name, `${field}.name`),
    width: expectBoundedPositiveInt(value.width, `${field}.width`, MAX_CAMERA_DIMENSION),
    height: expectBoundedPositiveInt(value.height, `${field}.height`, MAX_CAMERA_DIMENSION),
    fps: expectNumber(value.fps, `${field}.fps`),
  };
}

export interface CaptureStarted {
  session_id: string;
  running: boolean;
  mode: CaptureModeValue;
  streams: StreamDescriptor[];
}

/** Validates the `capture.started` snapshot member. */
function parseCaptureStarted(value: unknown): CaptureStarted {
  if (!isObject(value)) throw new CapturePreviewParseError("capture.started must be an object");
  const streamsRaw = value.streams;
  if (!Array.isArray(streamsRaw)) {
    throw new CapturePreviewParseError("capture.started.streams must be an array");
  }
  return {
    session_id: expectString(value.session_id, "capture.started.session_id"),
    running: expectBoolean(value.running, "capture.started.running"),
    mode: expectEnum(CaptureMode, value.mode, "capture.started.mode"),
    streams: streamsRaw.map((s, i) => parseStreamDescriptor(s, `capture.started.streams[${i}]`)),
  };
}

export interface CaptureStatusFacts {
  session_id: string;
  type: CaptureStatusTypeValue;
  running: boolean;
  startup_stage: StartupStageValue | null;
  message: string | null;
  status: StatusSnapshot | null;
}

/** Validates the `capture.status` snapshot member. */
function parseCaptureStatusFacts(value: unknown): CaptureStatusFacts {
  if (!isObject(value)) throw new CapturePreviewParseError("capture.status must be an object");
  return {
    session_id: expectString(value.session_id, "capture.status.session_id"),
    type: expectEnum(CaptureStatusType, value.type, "capture.status.type"),
    running: expectBoolean(value.running, "capture.status.running"),
    startup_stage: expectNullableEnum(StartupStage, value.startup_stage, "capture.status.startup_stage"),
    message: expectNullableString(value.message, "capture.status.message"),
    status: value.status === null ? null : parseStatusSnapshot(value.status),
  };
}

export interface CaptureStoppedFacts {
  session_id: string;
  running: boolean;
  reason: StopReasonValue | null;
  error_message: string | null;
  status: StatusSnapshot | null;
}

/** Validates the `capture.stopped` snapshot member, preserving the final
 * native status rather than deriving it from an earlier observation. */
function parseCaptureStoppedFacts(value: unknown): CaptureStoppedFacts {
  if (!isObject(value)) throw new CapturePreviewParseError("capture.stopped must be an object");
  return {
    session_id: expectString(value.session_id, "capture.stopped.session_id"),
    running: expectBoolean(value.running, "capture.stopped.running"),
    reason: expectNullableEnum(StopReason, value.reason, "capture.stopped.reason"),
    error_message: expectNullableString(value.error_message, "capture.stopped.error_message"),
    status: value.status === null ? null : parseStatusSnapshot(value.status),
  };
}

export interface TrackingSample {
  sequence: string | null;
  age_ms: number | null;
  /** Why no/an old sample is being served, e.g. an oversized sample the
   * backend dropped -- distinct from "no sample received yet". */
  diagnostic: string | null;
}

export interface CameraSample {
  stream_id: number;
  name: string;
  url: string;
  sequence: string | null;
  age_ms: number | null;
  /** Why no/an old sample is being served, e.g. an oversized sample the
   * backend dropped -- distinct from "no sample received yet". */
  diagnostic: string | null;
}

/** The backend's 3D tracking view, advertised in each snapshot. `ready` means a usable Viser server, not fresh
 * tracking or a connected browser; `url` is what the page embeds. */
export interface VisualizationInfo {
  status: "disabled" | "ready" | "unavailable";
  url: string | null;
  diagnostic: string | null;
}

export interface PreviewSnapshot {
  version: number;
  type: number;
  backend_instance_id: string;
  revision: number;
  station_label: string;
  profile_label: string;
  connecting: boolean;
  started: CaptureStarted | null;
  status: CaptureStatusFacts | null;
  stopped: CaptureStoppedFacts | null;
  error_message: string | null;
  binary_generation: number | null;
  tracking: TrackingSample | null;
  cameras: CameraSample[];
  /** Null when the backend sends no `visualization` member (a backend without the 3D view). */
  visualization: VisualizationInfo | null;
}

/** Validates the optional top-level `visualization` snapshot member. */
function parseVisualization(value: unknown): VisualizationInfo {
  if (!isObject(value)) throw new CapturePreviewParseError("visualization must be an object");
  const status = value.status;
  if (status !== "disabled" && status !== "ready" && status !== "unavailable") {
    throw new CapturePreviewParseError("visualization.status must be disabled, ready or unavailable");
  }
  return {
    status,
    url: expectNullableString(value.url, "visualization.url"),
    diagnostic: expectNullableString(value.diagnostic, "visualization.diagnostic"),
  };
}

/** Validates the `samples.tracking` snapshot member. */
function parseTrackingSample(value: unknown): TrackingSample {
  if (!isObject(value)) throw new CapturePreviewParseError("samples.tracking must be an object");
  return {
    sequence: expectDecimalStringOrNull(value.sequence, "samples.tracking.sequence"),
    age_ms: expectNullableNumber(value.age_ms, "samples.tracking.age_ms"),
    diagnostic: expectNullableString(value.diagnostic, "samples.tracking.diagnostic"),
  };
}

/** Validates one `samples.cameras` entry. */
function parseCameraSample(value: unknown, field: string): CameraSample {
  if (!isObject(value)) throw new CapturePreviewParseError(`${field} must be an object`);
  return {
    stream_id: expectNumber(value.stream_id, `${field}.stream_id`),
    name: expectString(value.name, `${field}.name`),
    url: expectString(value.url, `${field}.url`),
    sequence: expectDecimalStringOrNull(value.sequence, `${field}.sequence`),
    age_ms: expectNullableNumber(value.age_ms, `${field}.age_ms`),
    diagnostic: expectNullableString(value.diagnostic, `${field}.diagnostic`),
  };
}

/** Validates one parsed WS JSON message as a SNAPSHOT=3 message. Throws
 * CapturePreviewParseError for anything else -- an unsupported version/type
 * is a protocol error, not a value to coerce or ignore. */
export function parseSnapshot(body: unknown): PreviewSnapshot {
  if (!isObject(body)) throw new CapturePreviewParseError("snapshot body must be an object");
  const version = expectNumber(body.version, "version");
  const type = expectNumber(body.type, "type");
  if (version !== PREVIEW_VERSION || type !== SNAPSHOT_TYPE) {
    throw new CapturePreviewParseError(`unsupported snapshot version=${version} type=${type}`);
  }
  const capture = body.capture;
  if (!isObject(capture)) throw new CapturePreviewParseError("capture must be an object");
  const samples = body.samples;
  if (!isObject(samples)) throw new CapturePreviewParseError("samples must be an object");
  const camerasRaw = samples.cameras;
  if (!Array.isArray(camerasRaw)) throw new CapturePreviewParseError("samples.cameras must be an array");
  return {
    version,
    type,
    backend_instance_id: expectString(body.backend_instance_id, "backend_instance_id"),
    revision: expectNumber(body.revision, "revision"),
    station_label: expectString(body.station_label, "station_label"),
    profile_label: expectString(body.profile_label, "profile_label"),
    connecting: expectBoolean(capture.connecting, "capture.connecting"),
    started: capture.started === null ? null : parseCaptureStarted(capture.started),
    status: capture.status === null ? null : parseCaptureStatusFacts(capture.status),
    stopped: capture.stopped === null ? null : parseCaptureStoppedFacts(capture.stopped),
    error_message: expectNullableString(capture.error_message, "capture.error_message"),
    binary_generation: expectNullableNumber(capture.binary_generation, "capture.binary_generation"),
    tracking: samples.tracking === null ? null : parseTrackingSample(samples.tracking),
    cameras: camerasRaw.map((c, i) => parseCameraSample(c, `samples.cameras[${i}]`)),
    visualization: body.visualization === undefined ? null : parseVisualization(body.visualization),
  };
}

export interface ParsedBinaryHeader {
  generation: number;
  kind: number;
  streamId: number;
  sequence: bigint;
  captureTimeNs: bigint;
  receiveTimeNs: bigint;
  payloadOffset: number;
}

export class BinaryHeaderError extends Error {}

/** Parses the fixed 32-byte browser envelope prefix. Does not validate
 * generation/stream/kind against current connection state -- that is the
 * caller's job (useCaptureSession), since this module has no connection
 * context. */
export function parseBinaryHeader(buffer: ArrayBuffer): ParsedBinaryHeader {
  if (buffer.byteLength < BINARY_HEADER_SIZE) {
    throw new BinaryHeaderError(`binary message is ${buffer.byteLength} bytes, expected at least ${BINARY_HEADER_SIZE}`);
  }
  const view = new DataView(buffer);
  const generation = view.getUint32(0, true);
  const kind = view.getUint8(4);
  const reserved = view.getUint8(5);
  if (reserved !== 0) throw new BinaryHeaderError("binary header reserved byte must be zero");
  const streamId = view.getUint16(6, true);
  const sequence = view.getBigUint64(8, true);
  const captureTimeNs = view.getBigUint64(16, true);
  const receiveTimeNs = view.getBigUint64(24, true);
  return { generation, kind, streamId, sequence, captureTimeNs, receiveTimeNs, payloadOffset: BINARY_HEADER_SIZE };
}
