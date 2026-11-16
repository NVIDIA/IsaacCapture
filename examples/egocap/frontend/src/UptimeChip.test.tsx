// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The uptime chip: what it shows in each state, how it counts between loads, and when it reads again. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { UPTIME_FIXTURE } from "./dayReportTestDoubles";
import type { StationUptime } from "./stationsApi";
import { dayHasTake, formatActivity, formatTotal } from "./stationTime";
import { UPTIME_REFRESH_MS, UptimeChip, downSeconds } from "./UptimeChip";

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

/** Before the first take of the day: nothing counted, nothing blocking. */
const READY: StationUptime = { ...UPTIME_FIXTURE, state: "ready", reason: null, active_seconds: 0, down_seconds: 0, blockers: [] };
const RECORDING: StationUptime = { ...UPTIME_FIXTURE, state: "recording", reason: null, blockers: UPTIME_FIXTURE.blockers.slice(0, 1) };

/** Let the pending load settle, then advance the fake clock. */
async function settle(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

/** The chip over a backend whose answer a test can swap between loads. */
async function renderChip(first: StationUptime | Error, stationLabel: string | null = "Bench A") {
  const state = { answer: first };
  const load = vi.fn(async () => {
    if (state.answer instanceof Error) throw state.answer;
    return state.answer;
  });
  const utils = render(<UptimeChip stationLabel={stationLabel} load={load} />);
  await settle();
  return { state, load, rerender: (label: string | null) => utils.rerender(<UptimeChip stationLabel={label} load={load} />) };
}

const chip = () => screen.getByRole("img");
const summary = () => document.querySelector(".station-uptime-summary");

describe("formatActivity, formatTotal, dayHasTake and downSeconds", () => {
  it("shows a duration in minutes and seconds, then hours and minutes", () => {
    expect(formatActivity(0)).toBe("0m 00s");
    expect(formatActivity(125.9)).toBe("2m 05s");
    expect(formatActivity(7_500)).toBe("2h 05m");
    expect(formatActivity(-5)).toBe("0m 00s");
  });

  it("shows a day's total to the minute", () => {
    expect(formatTotal(0)).toBe("0m");
    expect(formatTotal(59.9)).toBe("<1m");
    expect(formatTotal(720)).toBe("12m");
    expect(formatTotal(3_599)).toBe("59m");
    expect(formatTotal(7_500)).toBe("2h 05m");
    expect(formatTotal(-5)).toBe("0m");
  });

  it("knows the day has a take once anything was counted", () => {
    expect(dayHasTake(READY)).toBe(false);
    expect(dayHasTake({ ...READY, active_seconds: 4 })).toBe(true);
    expect(dayHasTake({ ...READY, down_seconds: 4 })).toBe(true);
    expect(dayHasTake({ ...READY, blockers: UPTIME_FIXTURE.blockers })).toBe(true);
  });

  it("counts a down time from the backend's own figures plus the time since they arrived", () => {
    expect(downSeconds(UPTIME_FIXTURE, 0)).toBe(300);
    expect(downSeconds(UPTIME_FIXTURE, 60)).toBe(360);
    expect(downSeconds(UPTIME_FIXTURE, -3)).toBe(300);
    expect(downSeconds(READY, 60)).toBeNull();
    expect(downSeconds({ ...UPTIME_FIXTURE, since: null }, 60)).toBeNull();
  });
});

describe("UptimeChip", () => {
  it("says Ready in a positive tone, with no totals before the day's first take", async () => {
    await renderChip(READY);
    expect(chip().textContent).toBe("Ready");
    expect(chip().className).toBe("uptime-chip uptime-chip--ready");
    expect(chip().getAttribute("aria-label")).toBe("Station ready");
    expect(chip().hasAttribute("title")).toBe(false);
    expect(summary()).toBeNull();
  });

  it("says Recording in the accent tone, with the day's active and down time and its blockers", async () => {
    await renderChip(RECORDING);
    expect(chip().textContent).toBe("Recording");
    expect(chip().className).toBe("uptime-chip uptime-chip--recording");
    expect(chip().getAttribute("aria-label")).toBe("Station recording");
    expect(summary()?.textContent).toBe("active 2h 05m · down 17m · 1 blocker");
  });

  it("says Down for how long in a warning tone, with the reason as its title and in its name", async () => {
    await renderChip(UPTIME_FIXTURE);
    expect(chip().textContent).toBe("Down · 5m 00s");
    expect(chip().className).toBe("uptime-chip uptime-chip--down");
    expect(chip().title).toBe("Left wrist: cam_left: no recent frame");
    expect(chip().getAttribute("aria-label")).toBe("Station down for 5m 00s: Left wrist: cam_left: no recent frame");
    expect(summary()?.textContent).toBe("active 2h 05m · down 17m · 2 blockers");
  });

  it("leaves out the blockers when there are none", async () => {
    await renderChip({ ...RECORDING, blockers: [] });
    expect(summary()?.textContent).toBe("active 2h 05m · down 17m");
  });

  it("keeps the down time counting between loads", async () => {
    await renderChip({ ...UPTIME_FIXTURE, since: UPTIME_FIXTURE.as_of - 59 });
    expect(chip().textContent).toBe("Down · 0m 59s");
    await settle(2_000);
    expect(chip().textContent).toBe("Down · 1m 01s");
    expect(chip().getAttribute("aria-label")).toBe("Station down for 1m 01s: Left wrist: cam_left: no recent frame");
  });

  it("is not a control: nothing in the strip can be pressed", async () => {
    await renderChip(UPTIME_FIXTURE);
    expect(screen.queryByRole("button")).toBeNull();
    expect(chip().tabIndex).toBe(-1);
  });

  it("reads again every fifteen seconds and when the station in use changes", async () => {
    const { state, load, rerender } = await renderChip(READY);
    expect(UPTIME_REFRESH_MS).toBe(15_000);
    expect(load).toHaveBeenCalledTimes(1);
    state.answer = UPTIME_FIXTURE;
    await settle(14_999);
    expect(load).toHaveBeenCalledTimes(1);
    await settle(1);
    expect(load).toHaveBeenCalledTimes(2);
    expect(chip().textContent).toBe("Down · 5m 00s");
    state.answer = RECORDING;
    rerender("Cart B");
    await settle();
    expect(load).toHaveBeenCalledTimes(3);
    expect(chip().textContent).toBe("Recording");
  });

  it("reads again at once when its refresh key changes, without waiting for the timer", async () => {
    const state = { answer: READY };
    const load = vi.fn(async () => state.answer);
    const { rerender } = render(<UptimeChip stationLabel="Bench A" refreshKey="IDLE||3" load={load} />);
    await settle();
    expect(chip().textContent).toBe("Ready");
    state.answer = RECORDING;
    rerender(<UptimeChip stationLabel="Bench A" refreshKey="RECORDING||3" load={load} />);
    await settle();
    expect(load).toHaveBeenCalledTimes(2);
    expect(chip().textContent).toBe("Recording");
    // The same key is no reason to read.
    rerender(<UptimeChip stationLabel="Bench A" refreshKey="RECORDING||3" load={load} />);
    await settle();
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("shows the station now in use, not an answer for the last one that arrives late", async () => {
    const pending: Array<(uptime: StationUptime) => void> = [];
    const load = vi.fn(() => new Promise<StationUptime>((resolve) => pending.push(resolve)));
    const { rerender, unmount } = render(<UptimeChip stationLabel="Bench A" load={load} />);
    rerender(<UptimeChip stationLabel="Cart B" load={load} />);
    expect(pending).toHaveLength(2);
    pending[1](RECORDING);
    await settle();
    pending[0](UPTIME_FIXTURE); // Bench A's read answers late
    await settle();
    expect(chip().textContent).toBe("Recording");
    unmount();
  });

  it("keeps the last answer when an abandoned read fails late", async () => {
    const pending: Array<{ resolve: (uptime: StationUptime) => void; reject: (reason: Error) => void }> = [];
    const load = vi.fn(() => new Promise<StationUptime>((resolve, reject) => pending.push({ resolve, reject })));
    const { rerender } = render(<UptimeChip stationLabel="Bench A" load={load} />);
    rerender(<UptimeChip stationLabel="Cart B" load={load} />);
    pending[1].resolve(READY);
    await settle();
    pending[0].reject(new Error("offline"));
    await settle();
    expect(chip().textContent).toBe("Ready");
  });

  it("renders nothing while the uptime cannot be read, and recovers on the next refresh", async () => {
    const { state } = await renderChip(new Error("offline"));
    expect(screen.queryByRole("img")).toBeNull();
    expect(document.querySelector(".station-uptime")).toBeNull();
    state.answer = READY;
    await settle(UPTIME_REFRESH_MS);
    expect(chip().textContent).toBe("Ready");
  });
});
