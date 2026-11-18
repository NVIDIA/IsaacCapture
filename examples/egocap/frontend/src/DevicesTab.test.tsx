// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Devices tab: the verdict, the live checklist and how often it is read, the calibration requirement, and the calibration tools. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { DEVICES_REFRESH_MS, DevicesTab, type DevicesTabProps } from "./DevicesTab";
import { SettingsHttpError } from "./settingsApi";
import { OPERATOR_SETTINGS_FIXTURE, TOOLS_FIXTURE } from "./settingsTestDoubles";
import type { SetupStatus } from "./setupApi";
import { SETUP_ADVISED, SETUP_HELD, SETUP_READY, SETUP_UNCALIBRATED } from "./setupTestDoubles";

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

/** Let the pending read settle, then advance the fake clock. */
async function settle(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

const TOOLS: DevicesTabProps["stationSetup"] = { kind: "loaded", value: { tools: TOOLS_FIXTURE, operator: OPERATOR_SETTINGS_FIXTURE } };

/** The tab over a backend whose answer a test can swap between reads. */
async function renderTab(first: SetupStatus | Error, overrides: Partial<DevicesTabProps> = {}) {
  const state = { answer: first };
  const loadSetup = vi.fn(async () => {
    if (state.answer instanceof Error) throw state.answer;
    return state.answer;
  });
  const saveRequireCalibration = vi.fn(async (_required: boolean) => undefined);
  const props: DevicesTabProps = { setup: null, controlToken: "tok", loadSetup, stationSetup: TOOLS, saveRequireCalibration, ...overrides };
  const utils = render(<DevicesTab {...props} />);
  await settle();
  return { state, loadSetup, saveRequireCalibration, rerender: (next: Partial<DevicesTabProps>) => utils.rerender(<DevicesTab {...props} {...next} />) };
}

/** The dialog's station setup with the calibration requirement as saved. */
const withRequirement = (required: boolean): DevicesTabProps["stationSetup"] => ({ kind: "loaded", value: { tools: TOOLS_FIXTURE, operator: { ...OPERATOR_SETTINGS_FIXTURE, require_calibration: required } } });
const requirement = () => within(screen.getByRole("region", { name: "Calibration requirement" })).getByRole("checkbox") as HTMLInputElement;

const rowsOf = (group: string) => within(screen.getByRole("region", { name: group })).getAllByRole("listitem").map((row) => row.textContent);

describe("the verdict", () => {
  it("says recording is held and why while a required device fails", async () => {
    await renderTab(SETUP_HELD);
    const verdict = screen.getByRole("status");
    expect(verdict.textContent).toBe("Recording is held: Left wrist: cam_left: no recent frame.");
    expect(verdict.className).toContain("setup-verdict--held");
  });

  it("says station setup allows recording once nothing fails", async () => {
    await renderTab(SETUP_READY);
    const verdict = screen.getByRole("status");
    expect(verdict.textContent).toBe("Station setup allows recording.");
    expect(verdict.className).toContain("setup-verdict--clear");
  });

  it("says the checks only advise while no station is in use", async () => {
    await renderTab({ ...SETUP_READY, enforced: false });
    const verdict = screen.getByRole("status");
    expect(verdict.textContent).toBe("No station is in use, so these checks advise and do not hold recording.");
    expect(verdict.className).toBe("setup-verdict");
  });
});

describe("a check whose outcome this page does not know", () => {
  it("is shown as unrecognised with its reason, never as passed or not needed", async () => {
    const [first, ...rest] = SETUP_READY.checks.filter((check) => check.group === "system");
    await renderTab({ ...SETUP_READY, checks: [{ ...first, status: "unknown", detail: "waiting for the vendor service" }, ...rest] });
    const row = within(screen.getByRole("region", { name: "System" })).getAllByRole("listitem")[0];
    expect(row.className).toBe("check-row check-row--unknown");
    expect(row.querySelector(".check-mark")?.textContent).toBe("?");
    expect(row.querySelector(".visually-hidden")?.textContent).toBe(" outcome not recognised by this page");
    expect(row.querySelector(".check-detail")?.textContent).toBe("Waiting for the vendor service");
  });
});

describe("a check that only warns", () => {
  it("has its own mark, colour class and spoken outcome, and its reason", async () => {
    await renderTab(SETUP_ADVISED);
    expect(rowsOf("System")).toEqual(["!Camera calibration advisoryMissing head_intrinsics.json, head_extrinsics.json", "✓Free disk space passed211.3 GB free"]);
    const advised = within(screen.getByRole("region", { name: "System" })).getAllByRole("listitem")[0];
    expect(advised.className).toBe("check-row check-row--warn");
    expect(advised.querySelector(".visually-hidden")?.textContent).toBe(" advisory");
  });

  it("leaves the verdict saying recording is allowed", async () => {
    await renderTab(SETUP_ADVISED);
    const verdict = screen.getByRole("status");
    expect(verdict.textContent).toBe("Station setup allows recording.");
    expect(verdict.className).toContain("setup-verdict--clear");
  });
});

describe("the calibration requirement", () => {
  it("is off by default, beside the checklist, with what Off means", async () => {
    await renderTab(SETUP_ADVISED);
    const card = screen.getByRole("region", { name: "Calibration requirement" });
    expect(card.previousElementSibling?.getAttribute("aria-label")).toBe("Checklist");
    expect(within(card).getByLabelText("Require calibration files before recording")).toBe(requirement());
    expect(requirement().checked).toBe(false);
    expect(requirement().disabled).toBe(false);
    expect(within(card).getByText("Off: missing calibration files are shown as a warning and takes are allowed.").id).toBe(requirement().getAttribute("aria-describedby"));
  });

  it("saves the new choice and reads the checklist again at once", async () => {
    const { state, loadSetup, saveRequireCalibration, rerender } = await renderTab(SETUP_ADVISED);
    expect(loadSetup).toHaveBeenCalledTimes(1);
    state.answer = SETUP_UNCALIBRATED;
    fireEvent.click(requirement());
    expect(saveRequireCalibration).toHaveBeenCalledWith(true);
    // The dialog then carries the saved value; no timer has run.
    rerender({ stationSetup: withRequirement(true) });
    await settle();
    expect(loadSetup).toHaveBeenCalledTimes(2);
    expect(requirement().checked).toBe(true);
    expect(screen.getByRole("status").textContent).toBe("Recording is held: Camera calibration: missing head_intrinsics.json, head_extrinsics.json.");
    expect(rowsOf("System")[0]).toBe("✗Camera calibration failedMissing head_intrinsics.json, head_extrinsics.json");
  });

  it("shows why a save was refused and goes back to the saved choice", async () => {
    const saveRequireCalibration = vi.fn(async () => {
      throw new SettingsHttpError("control token is invalid");
    });
    const { loadSetup } = await renderTab(SETUP_ADVISED, { saveRequireCalibration });
    fireEvent.click(requirement());
    expect(requirement().checked).toBe(true); // shown while the save is on its way
    await settle();
    expect(requirement().checked).toBe(false);
    expect(within(screen.getByRole("region", { name: "Calibration requirement" })).getByText("Not saved: control token is invalid")).toBeTruthy();
    expect(loadSetup).toHaveBeenCalledTimes(1);
  });

  it("cannot be changed without the control token, and says where to enter it", async () => {
    const { saveRequireCalibration } = await renderTab(SETUP_ADVISED, { controlToken: null, stationSetup: withRequirement(true) });
    expect(requirement().checked).toBe(true);
    expect(requirement().disabled).toBe(true);
    expect(within(screen.getByRole("region", { name: "Calibration requirement" })).getByText(/Enter the control token in the recording panel/)).toBeTruthy();
    expect(saveRequireCalibration).not.toHaveBeenCalled();
  });

  it("is absent until the dialog's station setup has loaded", async () => {
    await renderTab(SETUP_READY, { stationSetup: { kind: "loading" } });
    expect(screen.queryByRole("region", { name: "Calibration requirement" })).toBeNull();
  });
});

describe("the checklist", () => {
  it("groups the checks as Tracking, Cameras and System, each with its outcome and reason", async () => {
    await renderTab(SETUP_HELD);
    const checklist = screen.getByRole("region", { name: "Checklist" });
    expect(within(checklist).getAllByRole("heading", { level: 4 }).map((heading) => heading.textContent)).toEqual(["Tracking", "Cameras", "System"]);
    expect(rowsOf("Tracking")).toEqual(["✓Pico headset passed"]);
    expect(rowsOf("Cameras")).toEqual([
      "✓Ego (head): cam_head passed",
      "✗Left wrist: cam_left failedNo recent frame",
      "–Right wrist not neededNot used by this station",
    ]);
    expect(rowsOf("System")).toEqual(["–Camera calibration not neededMock capture has nothing to calibrate", "✓Free disk space passed211.3 GB free"]);
    // The outcome word is for a screen reader; the mark carries it on screen.
    const failed = within(screen.getByRole("region", { name: "Cameras" })).getAllByRole("listitem")[1];
    expect(failed.className).toBe("check-row check-row--fail");
    expect(failed.querySelector(".check-mark")?.getAttribute("aria-hidden")).toBe("true");
    expect(failed.querySelector(".visually-hidden")?.textContent).toBe(" failed");
  });

  it("leaves out a group the station has nothing in", async () => {
    await renderTab({ ...SETUP_READY, checks: SETUP_READY.checks.filter((check) => check.group !== "tracking") });
    expect(screen.queryByRole("region", { name: "Tracking" })).toBeNull();
    expect(screen.getByRole("region", { name: "Cameras" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "System" })).toBeTruthy();
  });

  it("says so when the backend reports no checks at all", async () => {
    await renderTab({ ...SETUP_READY, checks: [] });
    expect(within(screen.getByRole("region", { name: "Checklist" })).getByText("Nothing is being checked yet.")).toBeTruthy();
  });

  it("is read every two seconds with the control token, so a fixed device shows by itself", async () => {
    const { state, loadSetup } = await renderTab(SETUP_HELD);
    expect(DEVICES_REFRESH_MS).toBe(2_000);
    expect(loadSetup).toHaveBeenCalledTimes(1);
    expect(loadSetup).toHaveBeenLastCalledWith(expect.any(AbortSignal), "tok");
    state.answer = SETUP_READY;
    await settle(1_999);
    expect(loadSetup).toHaveBeenCalledTimes(1);
    await settle(1);
    expect(loadSetup).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("status").textContent).toBe("Station setup allows recording.");
    expect(rowsOf("Cameras")[1]).toBe("✓Left wrist: cam_left passed");
    await settle(DEVICES_REFRESH_MS);
    expect(loadSetup).toHaveBeenCalledTimes(3);
  });

  it("stops reading once the tab is no longer shown", async () => {
    const { loadSetup } = await renderTab(SETUP_READY);
    cleanup();
    await settle(3 * DEVICES_REFRESH_MS);
    expect(loadSetup).toHaveBeenCalledTimes(1);
  });

  it("offers nothing that runs a test", async () => {
    await renderTab(SETUP_HELD);
    expect(within(screen.getByRole("region", { name: "Checklist" })).queryByRole("button")).toBeNull();
    expect(screen.getAllByRole("button").map((button) => button.textContent)).toEqual(["Open", "Open", "Open"]);
  });

  it("does not fall back to the page's older flow once its own read has answered and then fails", async () => {
    const { state } = await renderTab(SETUP_READY, { setup: SETUP_HELD });
    expect(screen.getByRole("status").textContent).toBe("Station setup allows recording.");
    state.answer = new Error("offline");
    await settle(DEVICES_REFRESH_MS);
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.queryByRole("region", { name: "Checklist" })).toBeNull();
    expect(screen.getByText("The checks are unavailable: the backend did not answer.")).toBeTruthy();
    // The next read that answers brings the list back.
    state.answer = SETUP_HELD;
    await settle(DEVICES_REFRESH_MS);
    expect(screen.getByRole("status").textContent).toBe("Recording is held: Left wrist: cam_left: no recent frame.");
  });

  it("shows the page's flow until its own read answers, and says so when neither has one", async () => {
    await renderTab(new Error("offline"), { setup: SETUP_HELD });
    expect(screen.getByRole("status").textContent).toBe("Recording is held: Left wrist: cam_left: no recent frame.");
    cleanup();
    await renderTab(new Error("offline"));
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByText("The checks are unavailable: the backend did not answer.")).toBeTruthy();
  });
});

describe("the calibration tools", () => {
  it("lists each tool with its description, why it is unavailable, and a disabled Open button", async () => {
    await renderTab(SETUP_READY);
    const tools = screen.getByRole("region", { name: "Calibration tools" });
    expect(within(tools).getAllByRole("listitem").map((row) => row.querySelector(".settings-list-name")?.textContent)).toEqual([
      "Headset calibration",
      "Glove calibration",
      "Fisheye rectification",
    ]);
    expect(within(tools).getByText("Intrinsics and extrinsics of the headset cameras. — not wired in this build.")).toBeTruthy();
    const buttons = within(tools).getAllByRole("button", { name: "Open" }) as HTMLButtonElement[];
    expect(buttons.every((button) => button.disabled)).toBe(true);
    expect(buttons[0].getAttribute("aria-describedby")).toBe("tool-headset_calibration-why");
  });

  it("opens a tool this build can launch, unless it is for another camera type", async () => {
    const tools = TOOLS_FIXTURE.map((tool) => ({ ...tool, available: true, unavailable_reason: null }));
    await renderTab(SETUP_READY, { stationSetup: { kind: "loaded", value: { tools, operator: { camera_type: "oak_d", require_calibration: false } } } });
    const buttons = screen.getAllByRole("button", { name: "Open" }) as HTMLButtonElement[];
    expect(buttons.map((button) => button.disabled)).toEqual([false, false, true]);
    expect(screen.getByText(/Only fisheye lenses need this; rectilinear cameras do not\. — for SENSING GMSL cameras stations\./)).toBeTruthy();
  });

  it("says why the tools cannot be listed and still shows the checklist", async () => {
    await renderTab(SETUP_READY, { stationSetup: { kind: "error", message: "tools are not available yet" } });
    expect(screen.getByText("Calibration tools unavailable: tools are not available yet")).toBeTruthy();
    expect(screen.getByRole("region", { name: "Checklist" })).toBeTruthy();
    cleanup();
    await renderTab(SETUP_READY, { stationSetup: { kind: "loading" } });
    expect(screen.getByText("Loading…")).toBeTruthy();
  });
});
