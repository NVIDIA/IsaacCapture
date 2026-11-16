// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The start-of-day checks dialog: what it shows before and after the connection test, and running the test. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SettingsHttpError } from "./settingsApi";
import { SetupDialog, type SetupDialogProps } from "./SetupDialog";
import type { SetupStatus } from "./setupApi";
import { SETUP_NOT_TESTED, SETUP_PASSED } from "./setupTestDoubles";

afterEach(cleanup);

function renderDialog(overrides: Partial<SetupDialogProps> = {}) {
  const props: SetupDialogProps = {
    open: true,
    onClose: vi.fn(),
    setup: SETUP_NOT_TESTED,
    controlToken: "tok",
    onChanged: vi.fn(),
    runTest: vi.fn(async () => SETUP_PASSED),
    ...overrides,
  };
  const utils = render(<SetupDialog {...props} />);
  return { props, rerender: (next: Partial<SetupDialogProps>) => utils.rerender(<SetupDialog {...props} {...next} />) };
}

const dialog = () => screen.getByRole("dialog", { name: "Start-of-day checks" });
const rowsOf = (region: HTMLElement) => within(region).getAllByRole("listitem").map((row) => row.textContent);

describe("SetupDialog", () => {
  it("renders nothing while closed", () => {
    renderDialog({ open: false });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("says recording is held and why, before the test has been run", () => {
    renderDialog();
    expect(screen.getByRole("status").textContent).toBe("Recording is held: run the device connection test before recording.");
    const test = within(dialog()).getByRole("region", { name: "Device connection test" });
    expect(within(test).getByText("Not run yet for this station in this capture session.")).toBeTruthy();
    expect(within(test).getByRole("button", { name: "Run test" })).toBeTruthy();
    expect(rowsOf(within(dialog()).getByRole("region", { name: "Calibration and system" }))).toEqual([
      "✓Camera calibration passed",
      "–Glove plugin not neededGloves are not part of this deployment",
      "✓Free disk space passed211.3 GB free",
      "–Camera producer not neededCameras are read by CaptureServer",
    ]);
  });

  it("runs the test with the control token, shows each device, and tells the page", async () => {
    const { props } = renderDialog();
    fireEvent.click(screen.getByRole("button", { name: "Run test" }));
    expect(await screen.findByText(/✓ Passed at .* for Bench A/)).toBeTruthy();
    expect(props.runTest).toHaveBeenCalledWith("tok");
    expect(props.onChanged).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("status").textContent).toBe("Station setup allows recording.");
    expect(rowsOf(screen.getByRole("region", { name: "Tracking" }))).toEqual(["✓Pico headset passed"]);
    expect(rowsOf(screen.getByRole("region", { name: "Cameras" }))).toEqual([
      "✓Ego (head): cam_head passed",
      "–Right wrist not neededNot used by this station",
    ]);
    expect(screen.getByRole("button", { name: "Run again" })).toBeTruthy();
  });

  it("counts the failures and gives each one's reason", async () => {
    const failed: SetupStatus = {
      ...SETUP_PASSED,
      blocked_reason: "the device connection test failed: Left wrist: cam_left: unplugged",
      connection_test: {
        ran_at: 1_773_050_520,
        station: "Bench A",
        passed: false,
        items: [
          { id: "device:controller:1", group: "tracking", label: "Left controller", status: "fail", detail: "battery empty" },
          { id: "camera:wrist_left", group: "cameras", label: "Left wrist: cam_left", status: "fail", detail: "unplugged" },
        ],
      },
    };
    renderDialog({ setup: failed });
    expect(screen.getByText(/2 failed at .* for Bench A/)).toBeTruthy();
    expect(rowsOf(screen.getByRole("region", { name: "Tracking" }))).toEqual(["✗Left controller failedBattery empty"]);
    expect(screen.getByRole("status").textContent).toContain("Recording is held: the device connection test failed");
  });

  it("prefers a newer read from the page over the answer of the last run", async () => {
    const { rerender } = renderDialog();
    fireEvent.click(screen.getByRole("button", { name: "Run test" }));
    await screen.findByText(/✓ Passed at/);
    rerender({ setup: { ...SETUP_NOT_TESTED } }); // the capture session restarted: the result no longer applies
    await waitFor(() => expect(screen.getByText("Not run yet for this station in this capture session.")).toBeTruthy());
  });

  it("says the checks only advise while no station is in use", () => {
    renderDialog({ setup: { ...SETUP_PASSED, enforced: false } });
    expect(screen.getByRole("status").textContent).toBe("No station is in use, so these checks advise and do not hold recording.");
  });

  it("shows the checks without the control token but cannot run the test", () => {
    const { props } = renderDialog({ controlToken: null });
    const run = screen.getByRole("button", { name: "Run test" }) as HTMLButtonElement;
    expect(run.disabled).toBe(true);
    expect(screen.getByText("Enter the control token in the recording panel to run the test.")).toBeTruthy();
    fireEvent.click(run);
    expect(props.runTest).not.toHaveBeenCalled();
  });

  it("shows why the test could not be run and stays usable", async () => {
    const runTest = vi.fn(async () => {
      throw new SettingsHttpError("control token rejected");
    });
    const { props } = renderDialog({ runTest });
    fireEvent.click(screen.getByRole("button", { name: "Run test" }));
    expect(await screen.findByText("Not run: control token rejected")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Run test" }) as HTMLButtonElement).disabled).toBe(false);
    expect(props.onChanged).not.toHaveBeenCalled();
  });

  it("says so when the backend did not answer", () => {
    renderDialog({ setup: null });
    expect(screen.getByText("The checks are unavailable: the backend did not answer.")).toBeTruthy();
  });

  it("closes on Escape, the Close button and the backdrop, and owns focus while open", () => {
    const { props } = renderDialog();
    expect(document.activeElement).toBe(dialog());
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    fireEvent.click(document.querySelector(".settings-backdrop") as Element);
    expect(props.onClose).toHaveBeenCalledTimes(3);
  });
});
