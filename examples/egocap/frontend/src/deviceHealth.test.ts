/** The operator health chips: short names, tones, reasons only for problems, and a retained feed never looks healthy. */
import { describe, expect, it } from "vitest";
import { DeviceState } from "./captureStatus";
import type { CaptureStatusResponse } from "./captureStatus";
import { chipsNeedingAttention, deriveHealthChips, deviceChipLabel, deviceTone, retainedTone } from "./deviceHealth";

const device = (id: string, status: number, error = "", reason = 0) => ({
  id,
  path: `/${id}`,
  type: "headset",
  description: `desc of ${id}`,
  provider_id: "p",
  status,
  reason,
  error,
});
const camera = (stream_id: number, name: string, status: number, error = "", reason = 0) => ({
  name,
  stream_id,
  status,
  reason,
  error,
  intrinsics: null,
});
const response = (devices: unknown[], cameras: unknown[]) =>
  ({ status: { providers: [], devices, cameras } }) as unknown as CaptureStatusResponse;

describe("deviceChipLabel", () => {
  it.each([
    ["openxr/headset", "Headset"],
    ["openxr/controller/left", "Controller L"],
    ["openxr/controller/right", "Controller R"],
    ["manus/hand/left", "Glove L"],
    ["manus/hand/right", "Glove R"],
  ])("names %s as %s", (id, label) => {
    expect(deviceChipLabel({ id, description: "ignored" })).toBe(label);
  });

  it("keeps an unrecognised device's own description so a new kind never vanishes", () => {
    expect(deviceChipLabel({ id: "acme/tracker", description: "Acme tracker" })).toBe("Acme tracker");
    expect(deviceChipLabel({ id: "acme/tracker", description: "" })).toBe("acme/tracker");
  });
});

describe("deriveHealthChips", () => {
  it("lists devices then cameras, and gives a healthy one no detail", () => {
    const chips = deriveHealthChips(
      response([device("openxr/headset", DeviceState.CONNECTED)], [camera(1, "head", DeviceState.CONNECTED)]),
      true,
    );
    expect(chips.map((c) => [c.label, c.tone, c.detail])).toEqual([
      ["Headset", "positive", null],
      ["head", "positive", null],
    ]);
  });

  it("carries a problem's own error, falling back to the reason, and tones it by severity", () => {
    const [lost, flaky, failed] = deriveHealthChips(
      response(
        [
          device("openxr/controller/left", DeviceState.DISCONNECTED, "controller pose lost"),
          device("manus/hand/left", DeviceState.DEGRADED, "", 4),
        ],
        [camera(2, "left_wrist", DeviceState.FAILED, "stream closed")],
      ),
      true,
    );
    expect(lost).toMatchObject({ tone: "critical", detail: "controller pose lost" });
    expect(flaky.tone).toBe("warning");
    expect(flaky.detail).not.toBeNull(); // the reason label stands in for a missing error
    expect(failed).toMatchObject({ tone: "critical", detail: "stream closed" });
  });

  it("never shows a retained (stale or final) response as healthy, but still flags real faults", () => {
    const chips = deriveHealthChips(
      response(
        [device("openxr/headset", DeviceState.CONNECTED), device("manus/hand/right", DeviceState.FAILED, "dongle gone")],
        [],
      ),
      false,
    );
    expect(chips.map((c) => c.tone)).toEqual(["neutral", "critical"]);
  });

  it("is empty before anything is reported", () => {
    expect(deriveHealthChips(null, true)).toEqual([]);
    expect(deriveHealthChips(response([], []), true)).toEqual([]);
  });
});

describe("chipsNeedingAttention", () => {
  it("keeps warnings and faults only", () => {
    const chips = deriveHealthChips(
      response(
        [
          device("openxr/headset", DeviceState.CONNECTED),
          device("openxr/controller/left", DeviceState.DEGRADED),
          device("openxr/controller/right", DeviceState.DISABLED),
        ],
        [camera(1, "head", DeviceState.UNAVAILABLE)],
      ),
      true,
    );
    expect(chipsNeedingAttention(chips).map((c) => c.label)).toEqual(["Controller L", "head"]);
  });
});

describe("tones", () => {
  it("maps every DeviceState and falls back to neutral", () => {
    expect(deviceTone(DeviceState.CONNECTED)).toBe("positive");
    expect(deviceTone(DeviceState.DEGRADED)).toBe("warning");
    expect(deviceTone(DeviceState.FAILED)).toBe("critical");
    expect(deviceTone(DeviceState.DISABLED)).toBe("neutral");
    expect(retainedTone("positive", false)).toBe("neutral");
    expect(retainedTone("critical", false)).toBe("critical");
    expect(retainedTone("positive", true)).toBe("positive");
  });
});
