// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Descriptor-driven DataView decoding of one ic_tracking_v1 payload into
 * display values. Bounds, shapes and dtypes come entirely from the
 * TrackingStreamDescriptor advertised in the snapshot -- nothing here
 * hardcodes field offsets from the total byte count. Packed field offsets
 * are not necessarily aligned, so Float32Array/Float64Array views are never
 * constructed directly at a field offset; every read goes through DataView.
 */

import { WireDType, type FieldDescriptor, type TrackingStreamDescriptor } from "./captureProtocol";

export class TrackingDecodeError extends Error {}

const DTYPE_ITEM_BYTES: Record<number, number> = {
  [WireDType.UINT8]: 1,
  [WireDType.FLOAT32_LE]: 4,
  [WireDType.FLOAT64_LE]: 8,
};

/** Total element count for a field's shape (empty shape = scalar = 1). */
function shapeCount(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}


/** Reads one dtype-typed scalar at an arbitrary (possibly unaligned) byte offset. */
function readScalar(view: DataView, byteOffset: number, dtype: number): number {
  switch (dtype) {
    case WireDType.UINT8:
      return view.getUint8(byteOffset);
    case WireDType.FLOAT32_LE:
      return view.getFloat32(byteOffset, true);
    case WireDType.FLOAT64_LE:
      return view.getFloat64(byteOffset, true);
    default:
      throw new TrackingDecodeError(`unsupported dtype ${dtype}`);
  }
}

/** Reads one field's full value (scalar, or flat array for a shaped field). */
function readField(view: DataView, payloadOffset: number, field: FieldDescriptor): number | number[] {
  const itemBytes = DTYPE_ITEM_BYTES[field.dtype];
  if (itemBytes === undefined) throw new TrackingDecodeError(`field ${field.name} has unsupported dtype ${field.dtype}`);
  const count = shapeCount(field.shape);
  if (field.shape.length === 0) {
    return readScalar(view, payloadOffset + field.offset, field.dtype);
  }
  const out: number[] = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = readScalar(view, payloadOffset + field.offset + i * itemBytes, field.dtype);
  }
  return out;
}

/** Rejects a nonfinite value used for display -- NaN/Infinity must never render as a number. */
function isFinitely(value: number): boolean {
  return Number.isFinite(value);
}

/** Reads a known scalar field as a boolean; absent/wrong-shape reads as false. */
function boolField(fields: Map<string, FieldDescriptor>, view: DataView, payloadOffset: number, name: string): boolean {
  const field = fields.get(name);
  if (field === undefined || field.shape.length !== 0) return false;
  const v = readField(view, payloadOffset, field);
  return typeof v === "number" && v !== 0;
}

/** Reads a known fixed-length numeric vector field; absent/wrong-shape/nonfinite reads as null. */
function vecField(fields: Map<string, FieldDescriptor>, view: DataView, payloadOffset: number, name: string, expectedLen: number): number[] | null {
  const field = fields.get(name);
  if (field === undefined || field.shape.length !== 1 || field.shape[0] !== expectedLen) return null;
  const v = readField(view, payloadOffset, field);
  if (!Array.isArray(v)) return null;
  if (!v.every(isFinitely)) return null;
  return v;
}

export interface ControllerDisplay {
  present: boolean;
  gripPosition: number[] | null;
  gripOrientation: number[] | null;
  gripValid: boolean;
  aimPosition: number[] | null;
  aimOrientation: number[] | null;
  aimValid: boolean;
  primaryClick: number | null;
  secondaryClick: number | null;
  thumbstickX: number | null;
  thumbstickY: number | null;
  thumbstickClick: number | null;
  menuClick: number | null;
  squeezeValue: number | null;
  triggerValue: number | null;
}

export interface HandJoint {
  position: number[] | null;
  orientation: number[] | null;
  radius: number | null;
  valid: boolean;
}

export interface HandDisplay {
  present: boolean;
  joints: HandJoint[];
}

export interface HeadDisplay {
  present: boolean;
  position: number[] | null;
  orientation: number[] | null;
  isValid: boolean;
  isTracked: boolean;
}

export interface DecodedTracking {
  head: HeadDisplay;
  controllerLeft: ControllerDisplay;
  controllerRight: ControllerDisplay;
  handLeft: HandDisplay;
  handRight: HandDisplay;
}

/** Reads a known scalar field as a finite number or null if absent/invalid. */
function scalarOrNull(fields: Map<string, FieldDescriptor>, view: DataView, payloadOffset: number, name: string): number | null {
  const field = fields.get(name);
  if (field === undefined || field.shape.length !== 0) return null;
  const v = readField(view, payloadOffset, field);
  return typeof v === "number" && isFinitely(v) ? v : null;
}

/** Decodes one controller's (left or right) present/pose/input fields. */
function decodeController(fields: Map<string, FieldDescriptor>, view: DataView, payloadOffset: number, prefix: string): ControllerDisplay {
  return {
    present: boolField(fields, view, payloadOffset, `${prefix}.present`),
    gripPosition: vecField(fields, view, payloadOffset, `${prefix}.grip_position`, 3),
    gripOrientation: vecField(fields, view, payloadOffset, `${prefix}.grip_orientation`, 4),
    gripValid: boolField(fields, view, payloadOffset, `${prefix}.grip_is_valid`),
    aimPosition: vecField(fields, view, payloadOffset, `${prefix}.aim_position`, 3),
    aimOrientation: vecField(fields, view, payloadOffset, `${prefix}.aim_orientation`, 4),
    aimValid: boolField(fields, view, payloadOffset, `${prefix}.aim_is_valid`),
    primaryClick: scalarOrNull(fields, view, payloadOffset, `${prefix}.primary_click`),
    secondaryClick: scalarOrNull(fields, view, payloadOffset, `${prefix}.secondary_click`),
    thumbstickX: scalarOrNull(fields, view, payloadOffset, `${prefix}.thumbstick_x`),
    thumbstickY: scalarOrNull(fields, view, payloadOffset, `${prefix}.thumbstick_y`),
    thumbstickClick: scalarOrNull(fields, view, payloadOffset, `${prefix}.thumbstick_click`),
    menuClick: scalarOrNull(fields, view, payloadOffset, `${prefix}.menu_click`),
    squeezeValue: scalarOrNull(fields, view, payloadOffset, `${prefix}.squeeze_value`),
    triggerValue: scalarOrNull(fields, view, payloadOffset, `${prefix}.trigger_value`),
  };
}

/** Decodes one hand's (left or right) present flag and all 26 native OpenXR
 * joints' position/orientation/radius/validity. */
function decodeHand(fields: Map<string, FieldDescriptor>, view: DataView, payloadOffset: number, prefix: string): HandDisplay {
  const present = boolField(fields, view, payloadOffset, `${prefix}.present`);
  const positions = fields.get(`${prefix}.joint_positions`);
  const orientations = fields.get(`${prefix}.joint_orientations`);
  const radii = fields.get(`${prefix}.joint_radii`);
  const valids = fields.get(`${prefix}.joint_valid`);
  const joints: HandJoint[] = [];
  for (let i = 0; i < 26; i++) {
    let position: number[] | null = null;
    let orientation: number[] | null = null;
    let radius: number | null = null;
    let valid = false;
    if (positions !== undefined && positions.shape.length === 2 && positions.shape[0] === 26 && positions.shape[1] === 3) {
      const itemBytes = DTYPE_ITEM_BYTES[positions.dtype] ?? 4;
      const v = [0, 1, 2].map((j) => readScalar(view, payloadOffset + positions.offset + (i * 3 + j) * itemBytes, positions.dtype));
      if (v.every(isFinitely)) position = v;
    }
    if (orientations !== undefined && orientations.shape.length === 2 && orientations.shape[0] === 26 && orientations.shape[1] === 4) {
      const itemBytes = DTYPE_ITEM_BYTES[orientations.dtype] ?? 4;
      const v = [0, 1, 2, 3].map((j) => readScalar(view, payloadOffset + orientations.offset + (i * 4 + j) * itemBytes, orientations.dtype));
      if (v.every(isFinitely)) orientation = v;
    }
    if (radii !== undefined && radii.shape.length === 1 && radii.shape[0] === 26) {
      const itemBytes = DTYPE_ITEM_BYTES[radii.dtype] ?? 4;
      const v = readScalar(view, payloadOffset + radii.offset + i * itemBytes, radii.dtype);
      if (isFinitely(v)) radius = v;
    }
    if (valids !== undefined && valids.shape.length === 1 && valids.shape[0] === 26) {
      const itemBytes = DTYPE_ITEM_BYTES[valids.dtype] ?? 1;
      valid = readScalar(view, payloadOffset + valids.offset + i * itemBytes, valids.dtype) !== 0;
    }
    joints.push({ position, orientation, radius, valid });
  }
  return { present, joints };
}

/** Decodes one ic_tracking_v1 payload. The descriptor's extent/overlap
 * bounds are already enforced by captureProtocol.parseSnapshot (the one
 * authoritative boundary validator) before a descriptor ever reaches this
 * function -- it does not re-validate bounds per sample for performance,
 * only reads what the descriptor advertised. */
export function decodeTracking(descriptor: TrackingStreamDescriptor, buffer: ArrayBuffer, payloadOffset: number): DecodedTracking {
  if (buffer.byteLength - payloadOffset < descriptor.payload_size) {
    throw new TrackingDecodeError(
      `payload is ${buffer.byteLength - payloadOffset} bytes, expected at least ${descriptor.payload_size}`,
    );
  }
  const view = new DataView(buffer);
  const fields = new Map(descriptor.fields.map((f) => [f.name, f]));
  return {
    head: {
      present: boolField(fields, view, payloadOffset, "head.present"),
      position: vecField(fields, view, payloadOffset, "head.position", 3),
      orientation: vecField(fields, view, payloadOffset, "head.orientation", 4),
      isValid: boolField(fields, view, payloadOffset, "head.is_valid"),
      isTracked: boolField(fields, view, payloadOffset, "head.is_tracked"),
    },
    controllerLeft: decodeController(fields, view, payloadOffset, "controller_left"),
    controllerRight: decodeController(fields, view, payloadOffset, "controller_right"),
    handLeft: decodeHand(fields, view, payloadOffset, "hand_left"),
    handRight: decodeHand(fields, view, payloadOffset, "hand_right"),
  };
}
