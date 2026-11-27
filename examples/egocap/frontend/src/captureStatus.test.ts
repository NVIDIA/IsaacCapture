// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  CaptureStatusParseError,
  NetworkCheckState,
  SCHEMA_VERSION,
  parseCaptureStatusResponse,
} from "./captureStatus";

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

function validCamera(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    stream_id: 1,
    name: "head_camera",
    passed: true,
    expected_fps: 30,
    usable_frames: 300,
    delivered_fps: 29.9,
    missing_frames: 0,
    max_capture_gap_ms: 40,
    max_arrival_gap_ms: 55,
    start_silence_ms: 10,
    end_silence_ms: 12,
    age_median_ms: 80,
    age_max_ms: 120,
    age_growth_ms: 1,
    reasons: [],
    ...overrides,
  };
}

function validCheck(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    state: 2, // NetworkCheckState.READY
    generation: "g1",
    endpoint: "capture host to station",
    legs_tested: ["media delivery"],
    legs_not_tested: ["recorder disk"],
    window_s: 10,
    valid_for_s: 60,
    phase: null,
    progress_s: null,
    completed_at_unix_ns: 1,
    reason: null,
    cameras: [validCamera()],
    ...overrides,
  };
}

function bodyWithStatus(status: Record<string, unknown>): Record<string, unknown> {
  return validBody({ status: { updated_at_ns: "123", providers: [], devices: [], cameras: [], ...status } });
}

describe("network_check parsing", () => {
  it("is null when the key is absent", () => {
    expect(parseCaptureStatusResponse(validBody()).status?.network_check).toBeNull();
  });

  it("is null when the key is null", () => {
    expect(parseCaptureStatusResponse(bodyWithStatus({ network_check: null })).status?.network_check).toBeNull();
  });

  it("parses a full check with its cameras", () => {
    const check = parseCaptureStatusResponse(bodyWithStatus({ network_check: validCheck() })).status?.network_check;
    expect(check?.state).toBe(NetworkCheckState.READY);
    expect(check?.legs_not_tested).toEqual(["recorder disk"]);
    expect(check?.cameras[0]?.name).toBe("head_camera");
    expect(check?.cameras[0]?.delivered_fps).toBe(29.9);
  });

  it("keeps null measurements and a null verdict as null rather than zero", () => {
    const camera = validCamera({ passed: null, delivered_fps: null, missing_frames: null, age_max_ms: null });
    const check = parseCaptureStatusResponse(bodyWithStatus({ network_check: validCheck({ cameras: [camera] }) })).status
      ?.network_check;
    expect(check?.cameras[0]?.passed).toBeNull();
    expect(check?.cameras[0]?.delivered_fps).toBeNull();
    expect(check?.cameras[0]?.missing_frames).toBeNull();
    expect(check?.cameras[0]?.age_max_ms).toBeNull();
  });

  it("parses the checking phase and progress", () => {
    const check = parseCaptureStatusResponse(
      bodyWithStatus({ network_check: validCheck({ state: 1, phase: "measuring", progress_s: 4 }) }),
    ).status?.network_check;
    expect(check?.state).toBe(NetworkCheckState.CHECKING);
    expect(check?.phase).toBe("measuring");
    expect(check?.progress_s).toBe(4);
  });

  it.each([
    ["an out-of-range state", validCheck({ state: 9 })],
    ["a non-array cameras", validCheck({ cameras: null })],
    ["a missing endpoint", validCheck({ endpoint: undefined })],
    ["a non-string legs_tested entry", validCheck({ legs_tested: [1] })],
    ["a non-boolean camera verdict", validCheck({ cameras: [validCamera({ passed: "yes" })] })],
    ["a string camera measurement", validCheck({ cameras: [validCamera({ delivered_fps: "30" })] })],
    ["a null camera entry", validCheck({ cameras: [null] })],
    ["non-array reasons", validCheck({ cameras: [validCamera({ reasons: "slow" })] })],
  ])("rejects %s", (_name, check) => {
    expect(() => parseCaptureStatusResponse(bodyWithStatus({ network_check: check }))).toThrow(CaptureStatusParseError);
  });

  it("rejects a network_check that is not an object", () => {
    expect(() => parseCaptureStatusResponse(bodyWithStatus({ network_check: "ready" }))).toThrow(
      CaptureStatusParseError,
    );
  });
});
