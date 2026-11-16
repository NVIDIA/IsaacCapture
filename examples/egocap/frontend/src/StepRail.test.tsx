// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The start-of-day rail: how each step's state reads, which step is current, and what a step opens. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { SetupStep } from "./setupApi";
import { SETUP_HELD, SETUP_READY } from "./setupTestDoubles";
import { StepRail, currentStep } from "./StepRail";

afterEach(cleanup);

const step = (name: RegExp) => screen.getByRole("button", { name });
const items = () => screen.getAllByRole("listitem");

describe("currentStep", () => {
  it("is the first step that is neither done nor skipped", () => {
    expect(currentStep(SETUP_HELD.steps)).toBe("devices");
    expect(currentStep(SETUP_READY.steps)).toBe("record");
    const allDone: SetupStep[] = SETUP_READY.steps.map((s): SetupStep => ({ ...s, status: s.id === "devices" ? "skipped" : "done" }));
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

  it("shows the four steps, which are done and which remain, and marks the one to deal with next", () => {
    render(<StepRail steps={SETUP_HELD.steps} onOpen={vi.fn()} />);
    expect(screen.getByRole("navigation", { name: "Start-of-day steps" })).toBeTruthy();
    expect(items().map((item) => item.textContent)).toEqual([
      "✓StationBench A",
      "!DevicesLeft wrist: cam_left: no recent frame",
      "3Record: to do",
      "4Wrap up",
    ]);
    expect(step(/Step 1, Station: done\. Bench A/).getAttribute("aria-current")).toBeNull();
    const devices = step(/Step 2, Devices: needs attention\. Left wrist: cam_left: no recent frame/);
    expect(devices.title).toBe("Left wrist: cam_left: no recent frame");
    expect(devices.getAttribute("aria-current")).toBe("step");
    expect(devices.className).toContain("step--current");
    expect(step(/Step 4, Wrap up: to do$/).getAttribute("aria-current")).toBeNull();
  });

  it("marks a step that does not apply to this station", () => {
    const steps: SetupStep[] = [
      { id: "station", label: "Station", status: "done", detail: "Bench A" },
      { id: "devices", label: "Devices", status: "skipped", detail: "no station is in use" },
    ];
    render(<StepRail steps={steps} onOpen={vi.fn()} />);
    const devices = step(/Step 2, Devices: not needed\. No station is in use/);
    expect(devices.textContent).toBe("–DevicesNo station is in use");
    expect(devices.className).toContain("step--skipped");
  });

  it("shows Record as a plain item with the same look, not a control", () => {
    const onOpen = vi.fn();
    render(<StepRail steps={SETUP_READY.steps} onOpen={onOpen} />);
    expect(screen.queryByRole("button", { name: /Record/ })).toBeNull();
    expect(screen.getAllByRole("button")).toHaveLength(3);
    const record = items()[2].firstElementChild as HTMLElement;
    expect(record.tagName).toBe("SPAN");
    expect(record.tabIndex).toBe(-1);
    expect(record.className).toBe("step step--todo step--current step--static");
    // It is still the step the operator is on, and says so to a screen reader.
    expect(record.getAttribute("aria-current")).toBe("step");
    expect(record.textContent).toBe("3Record: to do");
    fireEvent.click(record);
    expect(onOpen).not.toHaveBeenCalled();
  });

  it("carries the Record step's reason when it has one", () => {
    const steps = SETUP_READY.steps.map((s): SetupStep => (s.id === "record" ? { ...s, status: "attention", detail: "set a task to enable Record" } : s));
    render(<StepRail steps={steps} onOpen={vi.fn()} />);
    const record = items()[2].firstElementChild as HTMLElement;
    expect(record.textContent).toBe("!Record: needs attentionSet a task to enable Record");
    expect(record.title).toBe("Set a task to enable Record");
  });

  it("opens the place each of the other steps is dealt with", () => {
    const onOpen = vi.fn();
    render(<StepRail steps={SETUP_READY.steps} onOpen={onOpen} />);
    for (const name of [/Station/, /Devices/, /Wrap up/]) fireEvent.click(step(name));
    expect(onOpen.mock.calls.map(([id]) => id)).toEqual(["station", "devices", "wrap_up"]);
  });
});
