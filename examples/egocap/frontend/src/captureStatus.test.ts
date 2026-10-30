import { describe, expect, it } from "vitest";
import { CaptureStatusParseError, SCHEMA_VERSION, parseCaptureStatusResponse } from "./captureStatus";

function validBody(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: SCHEMA_VERSION,
    backend_instance_id: "b1",
    station_label: "Station 04",
    profile_label: "Headset and hands",
    mode: 0,
    session_id: "s1",
    connecting: false,
    status_type: 2,
    running: true,
    startup_stage: null,
    message: null,
    terminal: false,
    stop_reason: null,
    error_message: null,
    status: { updated_at_ns: "123", providers: [], devices: [], cameras: [] },
    ...overrides,
  };
}

function validDevice(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: "d1",
    path: null,
    type: "head",
    description: "Pico headset",
    provider_id: "provider-openxr",
    status: 3, // DeviceState.CONNECTED
    reason: 0, // StatusReason.NONE
    error: "",
    ...overrides,
  };
}

describe("parseCaptureStatusResponse", () => {
  it("accepts a well-formed response", () => {
    const parsed = parseCaptureStatusResponse(validBody());
    expect(parsed.session_id).toBe("s1");
  });

  it("rejects a response with an unsupported schema_version", () => {
    expect(() => parseCaptureStatusResponse(validBody({ schema_version: 2 }))).toThrow(
      CaptureStatusParseError,
    );
  });

  it("rejects an out-of-range mode rather than silently defaulting to a fabricated value", () => {
    expect(() => parseCaptureStatusResponse(validBody({ mode: 99 }))).toThrow(CaptureStatusParseError);
  });

  it("rejects a null entry inside status.devices instead of letting it reach rendering", () => {
    const body = validBody({
      status: { updated_at_ns: "123", providers: [], devices: [null], cameras: [] },
    });
    expect(() => parseCaptureStatusResponse(body)).toThrow(CaptureStatusParseError);
  });

  it("rejects a device with an out-of-range status enum value", () => {
    const body = validBody({
      status: { updated_at_ns: "123", providers: [], devices: [validDevice({ status: 999 })], cameras: [] },
    });
    expect(() => parseCaptureStatusResponse(body)).toThrow(CaptureStatusParseError);
  });

  it("accepts a well-formed device entry", () => {
    const body = validBody({
      status: { updated_at_ns: "123", providers: [], devices: [validDevice()], cameras: [] },
    });
    const parsed = parseCaptureStatusResponse(body);
    expect(parsed.status?.devices[0]?.description).toBe("Pico headset");
  });
});
