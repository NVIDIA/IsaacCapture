// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** The station activity toggle: what it shows through a day, how it counts between loads, and its refusals. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { SettingsHttpError } from "./settingsApi";
import { StationActivityToggle, formatActivity, liveTotals, type ActivityBackend } from "./StationActivityToggle";
import type { StationActivity } from "./stationsApi";

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const NOT_STARTED: StationActivity = {
  station: "Bench A",
  date: "2026-03-09",
  state: "inactive",
  since: null,
  active_seconds: 0,
  inactive_seconds: 0,
  blocker_count: 0,
  notes: "",
  as_of: 1_000,
};
const ACTIVE: StationActivity = { ...NOT_STARTED, state: "active", since: 1_000, active_seconds: 3_720, inactive_seconds: 300, blocker_count: 1 };
const BLOCKED: StationActivity = { ...ACTIVE, state: "inactive", since: 4_720, inactive_seconds: 45, blocker_count: 2 };

/** A backend whose answers a test can swap between calls. */
function fakeBackend(first: StationActivity | Error) {
  const state = { answer: first as StationActivity | Error, next: ACTIVE as StationActivity | Error };
  const backend: ActivityBackend = {
    load: vi.fn(async () => {
      if (state.answer instanceof Error) throw state.answer;
      return state.answer;
    }),
    change: vi.fn(async () => {
      if (state.next instanceof Error) throw state.next;
      return state.next;
    }),
  };
  return { backend, state };
}

/** Let the pending load or change settle, then advance the fake clock. */
async function settle(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function renderToggle(first: StationActivity | Error, props: { controlToken?: string | null; stationLabel?: string | null } = {}) {
  const fake = fakeBackend(first);
  const view = (label: string | null) => (
    <StationActivityToggle stationLabel={label} controlToken={props.controlToken === undefined ? "tok" : props.controlToken} backend={fake.backend} />
  );
  const utils = render(view(props.stationLabel === undefined ? "Bench A" : props.stationLabel));
  await settle();
  return { ...fake, rerender: (label: string | null) => utils.rerender(view(label)) };
}

const toggle = () => screen.getByRole("button") as HTMLButtonElement;

describe("formatActivity and liveTotals", () => {
  it("shows minutes and seconds, then hours and minutes", () => {
    expect(formatActivity(0)).toBe("0m 00s");
    expect(formatActivity(125.9)).toBe("2m 05s");
    expect(formatActivity(3_720)).toBe("1h 02m");
    expect(formatActivity(-5)).toBe("0m 00s");
  });

  it("counts the open segment onto the side that is open, and nothing before the day is started", () => {
    expect(liveTotals(ACTIVE, 60)).toEqual({ active: 3_780, inactive: 300 });
    expect(liveTotals(BLOCKED, 60)).toEqual({ active: 3_720, inactive: 105 });
    expect(liveTotals(NOT_STARTED, 600)).toEqual({ active: 0, inactive: 0 });
    expect(liveTotals(ACTIVE, -3)).toEqual({ active: 3_720, inactive: 300 });
  });
});

describe("StationActivityToggle", () => {
  it("starts the day on the first press", async () => {
    const { backend } = await renderToggle(NOT_STARTED);
    expect(toggle().textContent).toBe("Not started");
    expect(toggle().getAttribute("aria-pressed")).toBe("false");
    expect(toggle().title).toBe("Mark the station active");
    fireEvent.click(toggle());
    await settle();
    expect(backend.change).toHaveBeenCalledWith("tok", { state: "active" });
    expect(toggle().textContent).toBe("Active · 1h 02m");
    expect(toggle().getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText("blocked 5m 00s · 1 blocker")).toBeTruthy();
  });

  it("keeps the open segment counting between loads", async () => {
    await renderToggle({ ...ACTIVE, active_seconds: 59 });
    expect(toggle().textContent).toBe("Active · 0m 59s");
    await settle(2_000);
    expect(toggle().textContent).toBe("Active · 1m 01s");
  });

  it("marks the station blocked and shows both totals", async () => {
    const { backend, state } = await renderToggle(ACTIVE);
    state.next = BLOCKED;
    fireEvent.click(toggle());
    await settle();
    expect(backend.change).toHaveBeenCalledWith("tok", { state: "inactive" });
    expect(toggle().textContent).toBe("Blocked · 0m 45s");
    expect(toggle().title).toBe("Mark the station active");
    expect(screen.getByText("active 1h 02m · 2 blockers")).toBeTruthy();
  });

  it("reads the totals again each minute and when the station in use changes", async () => {
    const { backend, state, rerender } = await renderToggle(ACTIVE);
    expect(backend.load).toHaveBeenCalledTimes(1);
    state.answer = BLOCKED;
    await settle(60_000);
    expect(backend.load).toHaveBeenCalledTimes(2);
    expect(toggle().textContent).toContain("Blocked");
    state.answer = NOT_STARTED;
    rerender("Cart B");
    await settle();
    expect(backend.load).toHaveBeenCalledTimes(3);
    expect(toggle().textContent).toBe("Not started");
  });

  it("shows the state without the control token but cannot change it", async () => {
    const { backend } = await renderToggle(ACTIVE, { controlToken: null });
    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe("Enter the control token in the recording panel to change this");
    fireEvent.click(toggle());
    expect(backend.change).not.toHaveBeenCalled();
  });

  it("shows why a change was refused and keeps the last known state", async () => {
    const { state } = await renderToggle(ACTIVE);
    state.next = new SettingsHttpError("could not save station activity");
    fireEvent.click(toggle());
    await settle();
    expect(screen.getByText("Not saved: could not save station activity")).toBeTruthy();
    expect(toggle().textContent).toContain("Active");
    expect(toggle().disabled).toBe(false);
  });

  it("renders nothing while the activity cannot be read, and recovers on the next refresh", async () => {
    const { state } = await renderToggle(new Error("offline"));
    expect(screen.queryByRole("button")).toBeNull();
    state.answer = ACTIVE;
    await settle(60_000);
    expect(toggle().textContent).toContain("Active");
  });
});
