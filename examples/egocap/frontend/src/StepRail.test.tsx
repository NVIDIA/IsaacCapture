// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The start-of-day rail: how each step's state reads, which step is current, and what a step opens. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { SetupStep } from "./setupApi";
import { SETUP_NOT_TESTED, SETUP_PASSED } from "./setupTestDoubles";
import { StepRail, currentStep } from "./StepRail";

afterEach(cleanup);

const step = (name: RegExp) => screen.getByRole("button", { name });

describe("currentStep", () => {
  it("is the first step that is neither done nor skipped", () => {
    expect(currentStep(SETUP_NOT_TESTED.steps)).toBe("devices");
    expect(currentStep(SETUP_PASSED.steps)).toBe("record");
    const allDone: SetupStep[] = SETUP_PASSED.steps.map((s) => ({ ...s, status: s.id === "calibration" ? "skipped" : "done" }));
    expect(currentStep(allDone)).toBeNull();
  });
});

describe("StepRail", () => {
  it("renders nothing while the flow could not be read", () => {
    const { container } = render(<StepRail steps={null} onOpen={vi.fn()} />);
    expect(container.firstChild).toBeNull();
    cleanup();
    expect(render(<StepRail steps={[]} onOpen={vi.fn()} />).container.firstChild).toBeNull();
  });

  it("shows which steps are done and which remain, and marks the one to deal with next", () => {
    render(<StepRail steps={SETUP_NOT_TESTED.steps} onOpen={vi.fn()} />);
    expect(screen.getByRole("navigation", { name: "Start-of-day steps" })).toBeTruthy();
    expect(screen.getAllByRole("listitem").map((item) => item.textContent)).toEqual([
      "✓StationBench A",
      "2DevicesRun the connection test",
      "✓Calibration",
      "4Record",
      "5Wrap up",
    ]);
    expect(step(/Step 1, Station: done\. Bench A/).getAttribute("aria-current")).toBeNull();
    const devices = step(/Step 2, Devices: to do\. Run the connection test/);
    expect(devices.getAttribute("aria-current")).toBe("step");
    expect(devices.className).toContain("step--current");
    expect(step(/Step 4, Record: to do$/).getAttribute("aria-current")).toBeNull();
  });

  it("marks a step that needs attention and one that does not apply to this station", () => {
    const steps: SetupStep[] = [
      { id: "station", label: "Station", status: "done", detail: "Bench A" },
      { id: "devices", label: "Devices", status: "attention", detail: "left wrist: cam_left: unplugged" },
      { id: "calibration", label: "Calibration", status: "skipped", detail: "the station has no cameras assigned" },
    ];
    render(<StepRail steps={steps} onOpen={vi.fn()} />);
    const devices = step(/Step 2, Devices: needs attention\. Left wrist: cam_left: unplugged/);
    expect(devices.textContent).toBe("!DevicesLeft wrist: cam_left: unplugged");
    expect(devices.title).toBe("Left wrist: cam_left: unplugged");
    expect(devices.getAttribute("aria-current")).toBe("step");
    const calibration = step(/Step 3, Calibration: not needed\. The station has no cameras assigned/);
    expect(calibration.textContent).toBe("–CalibrationThe station has no cameras assigned");
    expect(calibration.className).toContain("step--skipped");
  });

  it("opens the place each step is dealt with", () => {
    const onOpen = vi.fn();
    render(<StepRail steps={SETUP_PASSED.steps} onOpen={onOpen} />);
    for (const name of [/Station/, /Devices/, /Calibration/, /Record/, /Wrap up/]) fireEvent.click(step(name));
    expect(onOpen.mock.calls.map(([id]) => id)).toEqual(["station", "devices", "calibration", "record", "wrap_up"]);
  });
});
