// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Validates against the ACTUAL complete descriptor CaptureServer publishes
 * for ic_tracking_v1 -- not a partial/synthetic one. Generated once via:
 *
 *   uv run python -c "
 *   import json
 *   from capture_server import protocol as p
 *   d = p.tracking_stream_descriptor(
 *       (p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS), 60.0
 *   )
 *   print(json.dumps(d.to_json()))
 *   "
 *
 * (run from egocap/, using capture_server.protocol's own
 * tracking_stream_descriptor()/to_json() -- the same function the real
 * CaptureServer calls, not a hand-written approximation). Regenerate this
 * fixture with that command if IC_TRACKING_V1_FIELDS ever changes.
 *
 * This exists because a partial/head-only synthetic descriptor in the
 * other test files can't catch a mismatch between this table and a field
 * that only appears in the full real layout -- e.g. a wrong shape on a
 * hand-only field like joint_valid.
 */
import { describe, expect, it } from "vitest";
import { parseSnapshot, PREVIEW_VERSION, SNAPSHOT_TYPE, type TrackingStreamDescriptor } from "./captureProtocol";
import { decodeTracking } from "./trackingDecoder";

const REAL_TRACKING_STREAM_JSON = {
  stream_id: 0,
  kind: 1,
  name: "tracking",
  schema: 1,
  sources: [1, 2, 3],
  rate_hz: 60.0,
  pose_representation: 1,
  quaternion_order: 1,
  position_unit: 1,
  axes: 1,
  reference_space: "session_floor",
  hand_joint_order: 1,
  layout: {
    payload_size: 1995,
    fields: [
      { name: "head.present", dtype: 1, shape: [], offset: 0 },
      { name: "head.position", dtype: 2, shape: [3], offset: 1 },
      { name: "head.orientation", dtype: 2, shape: [4], offset: 13 },
      { name: "head.is_valid", dtype: 1, shape: [], offset: 29 },
      { name: "head.is_tracked", dtype: 1, shape: [], offset: 30 },
      { name: "controller_left.present", dtype: 1, shape: [], offset: 31 },
      { name: "controller_left.grip_position", dtype: 2, shape: [3], offset: 32 },
      { name: "controller_left.grip_orientation", dtype: 2, shape: [4], offset: 44 },
      { name: "controller_left.grip_is_valid", dtype: 1, shape: [], offset: 60 },
      { name: "controller_left.aim_position", dtype: 2, shape: [3], offset: 61 },
      { name: "controller_left.aim_orientation", dtype: 2, shape: [4], offset: 73 },
      { name: "controller_left.aim_is_valid", dtype: 1, shape: [], offset: 89 },
      { name: "controller_left.primary_click", dtype: 3, shape: [], offset: 90 },
      { name: "controller_left.secondary_click", dtype: 3, shape: [], offset: 98 },
      { name: "controller_left.thumbstick_x", dtype: 3, shape: [], offset: 106 },
      { name: "controller_left.thumbstick_y", dtype: 3, shape: [], offset: 114 },
      { name: "controller_left.thumbstick_click", dtype: 3, shape: [], offset: 122 },
      { name: "controller_left.menu_click", dtype: 3, shape: [], offset: 130 },
      { name: "controller_left.squeeze_value", dtype: 3, shape: [], offset: 138 },
      { name: "controller_left.trigger_value", dtype: 3, shape: [], offset: 146 },
      { name: "controller_right.present", dtype: 1, shape: [], offset: 154 },
      { name: "controller_right.grip_position", dtype: 2, shape: [3], offset: 155 },
      { name: "controller_right.grip_orientation", dtype: 2, shape: [4], offset: 167 },
      { name: "controller_right.grip_is_valid", dtype: 1, shape: [], offset: 183 },
      { name: "controller_right.aim_position", dtype: 2, shape: [3], offset: 184 },
      { name: "controller_right.aim_orientation", dtype: 2, shape: [4], offset: 196 },
      { name: "controller_right.aim_is_valid", dtype: 1, shape: [], offset: 212 },
      { name: "controller_right.primary_click", dtype: 3, shape: [], offset: 213 },
      { name: "controller_right.secondary_click", dtype: 3, shape: [], offset: 221 },
      { name: "controller_right.thumbstick_x", dtype: 3, shape: [], offset: 229 },
      { name: "controller_right.thumbstick_y", dtype: 3, shape: [], offset: 237 },
      { name: "controller_right.thumbstick_click", dtype: 3, shape: [], offset: 245 },
      { name: "controller_right.menu_click", dtype: 3, shape: [], offset: 253 },
      { name: "controller_right.squeeze_value", dtype: 3, shape: [], offset: 261 },
      { name: "controller_right.trigger_value", dtype: 3, shape: [], offset: 269 },
      { name: "hand_left.present", dtype: 1, shape: [], offset: 277 },
      { name: "hand_left.joint_positions", dtype: 2, shape: [26, 3], offset: 278 },
      { name: "hand_left.joint_orientations", dtype: 2, shape: [26, 4], offset: 590 },
      { name: "hand_left.joint_radii", dtype: 2, shape: [26], offset: 1006 },
      { name: "hand_left.joint_valid", dtype: 1, shape: [26], offset: 1110 },
      { name: "hand_right.present", dtype: 1, shape: [], offset: 1136 },
      { name: "hand_right.joint_positions", dtype: 2, shape: [26, 3], offset: 1137 },
      { name: "hand_right.joint_orientations", dtype: 2, shape: [26, 4], offset: 1449 },
      { name: "hand_right.joint_radii", dtype: 2, shape: [26], offset: 1865 },
      { name: "hand_right.joint_valid", dtype: 1, shape: [26], offset: 1969 },
    ],
  },
};

function snapshotBodyWithRealDescriptor(): Record<string, unknown> {
  return {
    version: PREVIEW_VERSION,
    type: SNAPSHOT_TYPE,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station",
    profile_label: "Profile",
    capture: {
      connecting: false,
      started: {
        session_id: "s1",
        running: false,
        mode: 0,
        streams: [REAL_TRACKING_STREAM_JSON],
      },
      status: { session_id: "s1", type: 2, running: true, startup_stage: null, message: null, status: null },
      stopped: null,
      error_message: null,
      binary_generation: 1,
    },
    samples: { tracking: { sequence: "1", age_ms: 5, diagnostic: null }, cameras: [] },
  };
}

describe("the real, complete ic_tracking_v1 descriptor CaptureServer publishes", () => {
  it("parses successfully -- 45 fields, 1995-byte payload, all head/controller/hand fields present", () => {
    const parsed = parseSnapshot(snapshotBodyWithRealDescriptor());
    const stream = parsed.started?.streams[0];
    expect(stream?.kind).toBe("tracking");
    const tracking = stream as TrackingStreamDescriptor;
    expect(tracking.payload_size).toBe(1995);
    expect(tracking.fields).toHaveLength(45);
    expect(tracking.fields.map((f) => f.name)).toContain("hand_left.joint_valid");
    expect(tracking.fields.map((f) => f.name)).toContain("hand_right.joint_valid");
    // parseSnapshot succeeding above already proves the real layout passes
    // the one authoritative extent/overlap boundary check.
  });

  it("decodes a real 1995-byte frame built against the actual complete layout", () => {
    const parsed = parseSnapshot(snapshotBodyWithRealDescriptor());
    const tracking = parsed.started?.streams[0] as TrackingStreamDescriptor;

    const buf = new ArrayBuffer(32 + tracking.payload_size);
    const view = new DataView(buf);
    const fieldByName = new Map(tracking.fields.map((f) => [f.name, f]));

    // Head present + position.
    view.setUint8(32 + fieldByName.get("head.present")!.offset, 1);
    const headPos = fieldByName.get("head.position")!.offset;
    view.setFloat32(32 + headPos, 1, true);
    view.setFloat32(32 + headPos + 4, 2, true);
    view.setFloat32(32 + headPos + 8, 3, true);

    // Left hand present + all 26 joints valid.
    view.setUint8(32 + fieldByName.get("hand_left.present")!.offset, 1);
    const validOffset = fieldByName.get("hand_left.joint_valid")!.offset;
    for (let i = 0; i < 26; i++) {
      view.setUint8(32 + validOffset + i, 1);
    }

    const decoded = decodeTracking(tracking, buf, 32);
    expect(decoded.head.present).toBe(true);
    expect(decoded.head.position).toEqual([1, 2, 3]);
    expect(decoded.handLeft.present).toBe(true);
    expect(decoded.handLeft.joints.every((j) => j.valid)).toBe(true);
    expect(decoded.handLeft.joints).toHaveLength(26);
  });
});
