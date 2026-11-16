// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The Devices tab: the verdict, the live checklist and how often it is read, and the calibration tools. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen, within } from "@testing-library/react";
import { DEVICES_REFRESH_MS, DevicesTab, type DevicesTabProps } from "./DevicesTab";
import { OPERATOR_SETTINGS_FIXTURE, TOOLS_FIXTURE } from "./settingsTestDoubles";
import type { SetupStatus } from "./setupApi";
import { SETUP_HELD, SETUP_READY } from "./setupTestDoubles";

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
  render(<DevicesTab setup={null} controlToken="tok" loadSetup={loadSetup} stationSetup={TOOLS} {...overrides} />);
  await settle();
  return { state, loadSetup };
}

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
    await renderTab(SETUP_READY, { stationSetup: { kind: "loaded", value: { tools, operator: { camera_type: "oak_d" } } } });
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
