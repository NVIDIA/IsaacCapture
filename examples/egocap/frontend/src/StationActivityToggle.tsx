// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The station's active/inactive toggle, beside the station name in the status strip.
 *
 * The operator marks the station active while it is collecting and inactive while it is blocked; the backend
 * keeps the day's intervals and writes the daily activity log. Nothing is counted until the first press of the
 * day. The open segment keeps counting here between loads, from the totals the backend computed and the time
 * since this browser received them, so no clock comparison between the two machines is involved.
 */
import { useEffect, useReducer, useState } from "react";
import { changeStationActivity, fetchStationActivity, type ActivityState, type StationActivity } from "./stationsApi";

/** The calls the toggle makes; the page uses the real ones and tests pass fakes. */
export interface ActivityBackend {
  load: (signal: AbortSignal) => Promise<StationActivity>;
  change: (controlToken: string, change: { state: ActivityState }) => Promise<StationActivity>;
}

export const REAL_ACTIVITY_BACKEND: ActivityBackend = { load: fetchStationActivity, change: changeStationActivity };

/** How often the totals are read again, so a change made in another tab or a new day shows up. */
const REFRESH_MS = 60_000;

/** A duration as the operator reads it: hours and minutes once there are hours, else minutes and seconds. */
export function formatActivity(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${minutes}m ${String(whole % 60).padStart(2, "0")}s`;
}

/** Today's totals with the open segment counted up to now. */
export function liveTotals(activity: StationActivity, secondsSinceReceipt: number): { active: number; inactive: number } {
  // Before the first press of the day no segment is open, so nothing accrues.
  const open = activity.since === null ? 0 : Math.max(0, secondsSinceReceipt);
  return {
    active: activity.active_seconds + (activity.state === "active" ? open : 0),
    inactive: activity.inactive_seconds + (activity.state === "inactive" ? open : 0),
  };
}

export interface StationActivityToggleProps {
  /** The station label in effect; activity is kept per station, so a change of it loads that station's day. */
  stationLabel: string | null;
  /** The run's control token once entered; without it the toggle shows the state but cannot change it. */
  controlToken: string | null;
  backend?: ActivityBackend;
}

/** Show whether the station is active and for how long today, and flip it on a press. */
export function StationActivityToggle({ stationLabel, controlToken, backend = REAL_ACTIVITY_BACKEND }: StationActivityToggleProps) {
  const [loaded, setLoaded] = useState<{ activity: StationActivity; receivedAt: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [refresh, bumpRefresh] = useReducer((n: number) => n + 1, 0);
  const [, tick] = useReducer((n: number) => n + 1, 0);
  /** Keep an answer with the moment it arrived, which is what the live count is measured from. */
  const accept = (activity: StationActivity) => setLoaded({ activity, receivedAt: performance.now() });

  useEffect(() => {
    const controller = new AbortController();
    backend.load(controller.signal).then(
      (activity) => {
        if (!controller.signal.aborted) accept(activity);
      },
      () => {
        // The strip stays usable without it; the next refresh tries again.
        if (!controller.signal.aborted) setLoaded(null);
      },
    );
    return () => controller.abort();
  }, [backend, stationLabel, refresh]);

  useEffect(() => {
    const id = setInterval(bumpRefresh, REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  // Re-render once a second only while a segment is open; an untouched station has nothing to count.
  const counting = loaded !== null && loaded.activity.since !== null;
  useEffect(() => {
    if (!counting) return undefined;
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [counting]);

  if (loaded === null) return null;
  const { activity, receivedAt } = loaded;
  const totals = liveTotals(activity, (performance.now() - receivedAt) / 1000);
  const active = activity.state === "active";
  const started = activity.since !== null;
  /** Flip the state; the backend's answer replaces what is shown. */
  const toggle = () => {
    if (controlToken === null || busy) return;
    setBusy(true);
    setFailure(null);
    backend.change(controlToken, { state: active ? "inactive" : "active" }).then(
      (next) => {
        accept(next);
        setBusy(false);
      },
      (error: unknown) => {
        setFailure(error instanceof Error ? error.message : String(error));
        setBusy(false);
      },
    );
  };
  const label = !started ? "Not started" : active ? `Active · ${formatActivity(totals.active)}` : `Blocked · ${formatActivity(totals.inactive)}`;
  const hint = controlToken === null ? "Enter the control token in the recording panel to change this" : active ? "Mark the station blocked" : "Mark the station active";
  return (
    <span className="station-activity">
      <button
        type="button"
        className={`station-activity-toggle station-activity-toggle--${active ? "active" : started ? "blocked" : "idle"}`}
        aria-pressed={active}
        aria-label={`Station ${active ? "active" : started ? "blocked" : "not started"}: ${hint}`}
        title={hint}
        disabled={controlToken === null || busy}
        onClick={toggle}
      >
        <span className="station-activity-dot" aria-hidden="true" />
        {label}
      </button>
      {started && (
        <span className="station-activity-summary">
          {/* The side not on the button, so both totals are always in view. */}
          {active ? `blocked ${formatActivity(totals.inactive)}` : `active ${formatActivity(totals.active)}`}
          {activity.blocker_count > 0 && ` · ${activity.blocker_count} blocker${activity.blocker_count === 1 ? "" : "s"}`}
        </span>
      )}
      {failure !== null && <span className="station-activity-summary tone-warning">Not saved: {failure}</span>}
    </span>
  );
}
