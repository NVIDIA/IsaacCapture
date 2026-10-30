import { describe, expect, it } from "vitest";
import {
  BINARY_HEADER_SIZE,
  BinaryHeaderError,
  CapturePreviewParseError,
  PREVIEW_VERSION,
  SNAPSHOT_TYPE,
  parseBinaryHeader,
  parseSnapshot,
} from "./captureProtocol";

function validSnapshot(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    version: PREVIEW_VERSION,
    type: SNAPSHOT_TYPE,
    backend_instance_id: "b1",
    revision: 1,
    station_label: "Station 04",
    profile_label: "Headset and hands",
    capture: {
      connecting: false,
      started: null,
      status: null,
      stopped: null,
      error_message: null,
      binary_generation: null,
    },
    samples: { tracking: null, cameras: [] },
    ...overrides,
  };
}

function trackingStreamJson(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    kind: 1, // StreamKind.TRACKING
    stream_id: 0,
    schema: 1, // TrackingSchema.IC_TRACKING_V1
    pose_representation: 1, // POSITION_QUATERNION
    quaternion_order: 1, // XYZW
    position_unit: 1, // METERS
    axes: 1, // RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD
    hand_joint_order: 1, // OPENXR_26
    layout: {
      payload_size: 13,
      fields: [
        { name: "head.present", dtype: 1, shape: [], offset: 0 },
        { name: "head.position", dtype: 2, shape: [3], offset: 1 },
      ],
    },
    ...overrides,
  };
}

function cameraStreamJson(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return { kind: 2, stream_id: 1, name: "head_camera", width: 640, height: 480, fps: 30, ...overrides };
}

function startedWith(streams: unknown[]): Record<string, unknown> {
  return { session_id: "s1", running: false, mode: 0, streams };
}

describe("parseSnapshot", () => {
  it("accepts a minimal well-formed snapshot", () => {
    const parsed = parseSnapshot(validSnapshot());
    expect(parsed.revision).toBe(1);
    expect(parsed.started).toBeNull();
    expect(parsed.cameras).toEqual([]);
  });

  it("accepts accepted metadata, tracking/camera descriptors and samples", () => {
    const body = validSnapshot({
      capture: {
        connecting: false,
        started: startedWith([trackingStreamJson(), cameraStreamJson()]),
        status: { session_id: "s1", type: 2, running: true, startup_stage: null, message: null, status: null },
        stopped: null,
        error_message: null,
        binary_generation: 1,
      },
      samples: {
        tracking: { sequence: "5", age_ms: 10, diagnostic: null },
        cameras: [
          { stream_id: 1, name: "head_camera", url: "/stream1?x=1", sequence: "7", age_ms: 20, diagnostic: null },
        ],
      },
    });
    const parsed = parseSnapshot(body);
    expect(parsed.started?.streams).toHaveLength(2);
    expect(parsed.started?.streams[0]).toMatchObject({
      kind: "tracking",
      payload_size: 13,
      positionUnit: 1,
      quaternionOrder: 1,
      axes: 1,
      handJointOrder: 1,
    });
    expect(parsed.started?.streams[1]).toMatchObject({ kind: "camera", name: "head_camera" });
    expect(parsed.tracking).toEqual({ sequence: "5", age_ms: 10, diagnostic: null });
    expect(parsed.cameras[0].name).toBe("head_camera");
  });

  it("rejects an unsupported version", () => {
    expect(() => parseSnapshot(validSnapshot({ version: 2 }))).toThrow(CapturePreviewParseError);
  });

  it("rejects an unsupported type", () => {
    expect(() => parseSnapshot(validSnapshot({ type: 99 }))).toThrow(CapturePreviewParseError);
  });

  it("reads the optional visualization member, and treats an absent one as no 3D view", () => {
    expect(parseSnapshot(validSnapshot()).visualization).toBeNull();
    const parsed = parseSnapshot(
      validSnapshot({ visualization: { status: "ready", url: "http://h:8085/", diagnostic: "tracking is stale" } }),
    );
    expect(parsed.visualization).toEqual({ status: "ready", url: "http://h:8085/", diagnostic: "tracking is stale" });
  });

  it("rejects a visualization member with an unknown status or a non-object value", () => {
    expect(() => parseSnapshot(validSnapshot({ visualization: { status: "maybe", url: null, diagnostic: null } }))).toThrow(
      CapturePreviewParseError,
    );
    expect(() => parseSnapshot(validSnapshot({ visualization: "ready" }))).toThrow(CapturePreviewParseError);
  });

  it("rejects a non-decimal sequence string", () => {
    const body = validSnapshot({ samples: { tracking: { sequence: "not-a-number", age_ms: 1 }, cameras: [] } });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("rejects a duplicate field name in a tracking descriptor", () => {
    const stream = trackingStreamJson();
    (stream.layout as { fields: unknown[] }).fields.push({ name: "head.present", dtype: 1, shape: [], offset: 5 });
    const body = validSnapshot({
      capture: {
        connecting: false,
        started: startedWith([stream]),
        status: null,
        stopped: null,
        error_message: null,
        binary_generation: null,
      },
    });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("rejects an unsupported tracking schema -- the whole snapshot fails to parse, not just that stream", () => {
    const stream = trackingStreamJson({ schema: 99 });
    const body = validSnapshot({
      capture: {
        connecting: false,
        started: startedWith([stream]),
        status: null,
        stopped: null,
        error_message: null,
        binary_generation: null,
      },
    });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("rejects an unsupported quaternion_order/axes/position_unit/hand_joint_order", () => {
    for (const key of ["quaternion_order", "axes", "position_unit", "hand_joint_order", "pose_representation"]) {
      const stream = trackingStreamJson({ [key]: 99 });
      const body = validSnapshot({
        capture: {
          connecting: false,
          started: startedWith([stream]),
          status: null,
          stopped: null,
          error_message: null,
          binary_generation: null,
        },
      });
      expect(() => parseSnapshot(body), `key=${key}`).toThrow(CapturePreviewParseError);
    }
  });

  it("rejects a field whose extent exceeds payload_size", () => {
    const stream = trackingStreamJson({ layout: { payload_size: 2, fields: [{ name: "head.present", dtype: 1, shape: [], offset: 0 }, { name: "head.position", dtype: 2, shape: [3], offset: 1 }] } });
    const body = validSnapshot({
      capture: { connecting: false, started: startedWith([stream]), status: null, stopped: null, error_message: null, binary_generation: null },
    });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("rejects two fields whose extents overlap", () => {
    const stream = trackingStreamJson({
      layout: {
        payload_size: 2,
        fields: [
          { name: "custom.a", dtype: 1, shape: [], offset: 0 },
          { name: "custom.b", dtype: 1, shape: [2], offset: 0 },
        ],
      },
    });
    const body = validSnapshot({
      capture: { connecting: false, started: startedWith([stream]), status: null, stopped: null, error_message: null, binary_generation: null },
    });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("rejects a known field with a mismatched dtype/shape (not a guessed width)", () => {
    const stream = trackingStreamJson({
      layout: { payload_size: 20, fields: [{ name: "head.position", dtype: 3 /* FLOAT64_LE, wrong */, shape: [3], offset: 0 }] },
    });
    const body = validSnapshot({
      capture: { connecting: false, started: startedWith([stream]), status: null, stopped: null, error_message: null, binary_generation: null },
    });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("rejects a fractional shape dimension", () => {
    const stream = trackingStreamJson({
      layout: { payload_size: 20, fields: [{ name: "custom.field", dtype: 2, shape: [3.5], offset: 0 }] },
    });
    const body = validSnapshot({
      capture: { connecting: false, started: startedWith([stream]), status: null, stopped: null, error_message: null, binary_generation: null },
    });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("rejects an oversized/unbounded shape product", () => {
    const stream = trackingStreamJson({
      layout: { payload_size: 20, fields: [{ name: "custom.field", dtype: 1, shape: [1_000_000], offset: 0 }] },
    });
    const body = validSnapshot({
      capture: { connecting: false, started: startedWith([stream]), status: null, stopped: null, error_message: null, binary_generation: null },
    });
    expect(() => parseSnapshot(body)).toThrow(CapturePreviewParseError);
  });

  it("accepts an unaligned-but-valid field layout (packed, not padded)", () => {
    const stream = trackingStreamJson({
      layout: {
        payload_size: 21,
        fields: [
          { name: "head.present", dtype: 1, shape: [], offset: 0 },
          { name: "head.position", dtype: 2, shape: [3], offset: 1 },
          { name: "controller_left.trigger_value", dtype: 3, shape: [], offset: 13 },
        ],
      },
    });
    const body = validSnapshot({
      capture: { connecting: false, started: startedWith([stream]), status: null, stopped: null, error_message: null, binary_generation: null },
    });
    expect(() => parseSnapshot(body)).not.toThrow();
  });

  it("tolerates an unrecognized extra field name alongside known fields", () => {
    const stream = trackingStreamJson({
      layout: {
        payload_size: 14,
        fields: [
          { name: "head.present", dtype: 1, shape: [], offset: 0 },
          { name: "head.position", dtype: 2, shape: [3], offset: 1 },
          { name: "vendor.extra", dtype: 1, shape: [], offset: 13 },
        ],
      },
    });
    const body = validSnapshot({
      capture: { connecting: false, started: startedWith([stream]), status: null, stopped: null, error_message: null, binary_generation: null },
    });
    expect(() => parseSnapshot(body)).not.toThrow();
  });
});

describe("parseBinaryHeader", () => {
  function buildHeader(overrides: Partial<{ generation: number; kind: number; reserved: number; streamId: number; sequence: bigint }> = {}): ArrayBuffer {
    const buf = new ArrayBuffer(BINARY_HEADER_SIZE + 4);
    const view = new DataView(buf);
    view.setUint32(0, overrides.generation ?? 1, true);
    view.setUint8(4, overrides.kind ?? 1);
    view.setUint8(5, overrides.reserved ?? 0);
    view.setUint16(6, overrides.streamId ?? 0, true);
    view.setBigUint64(8, overrides.sequence ?? 42n, true);
    view.setBigUint64(16, 1000n, true);
    view.setBigUint64(24, 2000n, true);
    return buf;
  }

  it("parses a well-formed header", () => {
    const parsed = parseBinaryHeader(buildHeader());
    expect(parsed.generation).toBe(1);
    expect(parsed.kind).toBe(1);
    expect(parsed.streamId).toBe(0);
    expect(parsed.sequence).toBe(42n);
    expect(parsed.payloadOffset).toBe(BINARY_HEADER_SIZE);
  });

  it("decodes a sequence above 2**53 without precision loss", () => {
    const big = 9007199254740993n; // 2**53 + 1
    const parsed = parseBinaryHeader(buildHeader({ sequence: big }));
    expect(parsed.sequence).toBe(big);
  });

  it("rejects a nonzero reserved byte", () => {
    expect(() => parseBinaryHeader(buildHeader({ reserved: 1 }))).toThrow(BinaryHeaderError);
  });

  it("rejects a message shorter than the header", () => {
    expect(() => parseBinaryHeader(new ArrayBuffer(10))).toThrow(BinaryHeaderError);
  });
});
