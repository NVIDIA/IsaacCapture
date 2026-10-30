import { describe, expect, it } from "vitest";
import type { FieldDescriptor, TrackingStreamDescriptor, WireDTypeValue } from "./captureProtocol";
import { TrackingDecodeError, decodeTracking } from "./trackingDecoder";

function field(name: string, dtype: WireDTypeValue, shape: number[], offset: number): FieldDescriptor {
  return { name, dtype, shape, offset };
}

// The validated-metadata fields every TrackingStreamDescriptor must now
// carry -- fixed at their one supported value in these fixtures too.
const META = { positionUnit: 1, quaternionOrder: 1, axes: 1, handJointOrder: 1 } as const;

function minimalDescriptor(): TrackingStreamDescriptor {
  // head.present (u1@0), head.position (f4x3@1..13)
  return {
    kind: "tracking",
    stream_id: 0,
    payload_size: 13,
    fields: [field("head.present", 1, [], 0), field("head.position", 2, [3], 1)],
    ...META,
  };
}

describe("decodeTracking", () => {
  it("decodes head present/position with validity preserved", () => {
    const descriptor = minimalDescriptor();
    const buf = new ArrayBuffer(32 + descriptor.payload_size);
    const view = new DataView(buf);
    view.setUint8(32 + 0, 1); // head.present = true
    view.setFloat32(32 + 1, 1.5, true);
    view.setFloat32(32 + 5, 2.5, true);
    view.setFloat32(32 + 9, 3.5, true);
    const decoded = decodeTracking(descriptor, buf, 32);
    expect(decoded.head.present).toBe(true);
    expect(decoded.head.position).toEqual([1.5, 2.5, 3.5]);
    expect(decoded.head.isValid).toBe(false); // field not advertised -> false, not fabricated
  });

  it("decodes a 26-joint hand with per-joint validity", () => {
    const fields: FieldDescriptor[] = [
      field("hand_left.present", 1, [], 0),
      field("hand_left.joint_positions", 2, [26, 3], 1),
      field("hand_left.joint_orientations", 2, [26, 4], 313),
      field("hand_left.joint_radii", 2, [26], 729),
      field("hand_left.joint_valid", 1, [26], 833),
    ];
    const payloadSize = 859;
    const descriptor: TrackingStreamDescriptor = { kind: "tracking", stream_id: 0, payload_size: payloadSize, fields, ...META };
    const buf = new ArrayBuffer(32 + payloadSize);
    const view = new DataView(buf);
    view.setUint8(32, 1);
    // joint 0 valid, joint 1 invalid
    view.setFloat32(32 + 1, 0.1, true);
    view.setFloat32(32 + 5, 0.2, true);
    view.setFloat32(32 + 9, 0.3, true);
    view.setUint8(32 + 833, 1);
    view.setUint8(32 + 834, 0);
    const decoded = decodeTracking(descriptor, buf, 32);
    expect(decoded.handLeft.present).toBe(true);
    expect(decoded.handLeft.joints[0].position?.[0]).toBeCloseTo(0.1, 5);
    expect(decoded.handLeft.joints[0].position?.[1]).toBeCloseTo(0.2, 5);
    expect(decoded.handLeft.joints[0].position?.[2]).toBeCloseTo(0.3, 5);
    expect(decoded.handLeft.joints[0].valid).toBe(true);
    expect(decoded.handLeft.joints[1].valid).toBe(false);
  });

  it("decodes float64 controller input fields", () => {
    const fields: FieldDescriptor[] = [field("controller_left.trigger_value", 3, [], 0)];
    const descriptor: TrackingStreamDescriptor = { kind: "tracking", stream_id: 0, payload_size: 8, fields, ...META };
    const buf = new ArrayBuffer(32 + 8);
    new DataView(buf).setFloat64(32, 0.75, true);
    const decoded = decodeTracking(descriptor, buf, 32);
    expect(decoded.controllerLeft.triggerValue).toBe(0.75);
  });

  it("rejects a payload shorter than payload_size", () => {
    const descriptor = minimalDescriptor();
    const buf = new ArrayBuffer(32 + 2);
    expect(() => decodeTracking(descriptor, buf, 32)).toThrow(TrackingDecodeError);
  });

  it("reads unaligned packed field offsets correctly", () => {
    // u1 at 0, f4x3 at 1 (unaligned for Float32Array), f8 at 13 (unaligned for Float64Array)
    const fields: FieldDescriptor[] = [
      field("head.present", 1, [], 0),
      field("head.position", 2, [3], 1),
      field("controller_left.trigger_value", 3, [], 13),
    ];
    const descriptor: TrackingStreamDescriptor = { kind: "tracking", stream_id: 0, payload_size: 21, fields, ...META };
    const buf = new ArrayBuffer(32 + 21);
    const view = new DataView(buf);
    view.setFloat32(32 + 1, 9.5, true);
    view.setFloat32(32 + 5, 1.5, true);
    view.setFloat32(32 + 9, 2.5, true);
    view.setFloat64(32 + 13, 0.25, true);
    const decoded = decodeTracking(descriptor, buf, 32);
    expect(decoded.head.position).toEqual([9.5, 1.5, 2.5]);
    expect(decoded.controllerLeft.triggerValue).toBe(0.25);
  });
});
