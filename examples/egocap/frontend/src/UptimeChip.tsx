// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The station's uptime chip, beside the station name in the status strip.
 *
 * Read-only: the backend decides by itself whether the station can record and logs the day's timeline of it.
 * The chip says ready, recording, or down and for how long; beside it, once the day has a take, are the active
 * and down time between the first take and the last. How long the station has been down keeps counting here
 * between loads, from the figure the backend computed and the time since this browser received it, so no clock
 * comparison between the two machines is involved.
 */
import { useEffect, useReducer, useState } from "react";
import { sentence } from "./sentence";
import { fetchStationUptime, type StationUptime, type UptimeState } from "./stationsApi";
import { dayHasTake, formatActivity, formatTotal } from "./stationTime";

/** How often the uptime is read again when nothing prompts it, so a station that went down or came back shows without a reload. */
export const UPTIME_REFRESH_MS = 15_000;

/** Each state as the chip words it. */
const STATE_WORDS: Record<UptimeState, string> = { ready: "Ready", recording: "Recording", down: "Down" };

/** How long the station has been down, counted up to now; null when it is not down or the start is unknown. */
export function downSeconds(uptime: StationUptime, secondsSinceReceipt: number): number | null {
  if (uptime.state !== "down" || uptime.since === null) return null;
  return Math.max(0, uptime.as_of - uptime.since) + Math.max(0, secondsSinceReceipt);
}

export interface UptimeChipProps {
  /** The station label in effect; uptime is kept per station, so a change of it loads that station's day. */
  stationLabel: string | null;
  /** Anything whose change means the station's state may have: the chip reads again at once instead of on its timer. */
  refreshKey?: unknown;
  load?: (signal: AbortSignal) => Promise<StationUptime>;
}

/** Show whether the station can record right now and, once the day has a take, its active and down time. */
export function UptimeChip({ stationLabel, refreshKey, load = fetchStationUptime }: UptimeChipProps) {
  // Kept with the station it was read for: until that station's own answer arrives, another station's state,
  // totals and clock must not stand beside its name.
  const [answer, setLoaded] = useState<{ uptime: StationUptime; receivedAt: number; stationLabel: string | null } | null>(null);
  const loaded = answer !== null && answer.stationLabel === stationLabel ? answer : null;
  const [refresh, bumpRefresh] = useReducer((n: number) => n + 1, 0);
  const [, tick] = useReducer((n: number) => n + 1, 0);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(
      (uptime) => {
        // Kept with the moment it arrived, which is what the live count is measured from.
        if (!controller.signal.aborted) setLoaded({ uptime, receivedAt: performance.now(), stationLabel });
      },
      () => {
        // The strip stays usable without it; the next refresh tries again.
        if (!controller.signal.aborted) setLoaded(null);
      },
    );
    return () => controller.abort();
  }, [load, stationLabel, refreshKey, refresh]);

  useEffect(() => {
    const id = setInterval(bumpRefresh, UPTIME_REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  // Re-render once a second only while a down time is counting; the other states show no clock.
  const counting = loaded !== null && downSeconds(loaded.uptime, 0) !== null;
  useEffect(() => {
    if (!counting) return undefined;
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [counting]);

  if (loaded === null) return null;
  const { uptime, receivedAt } = loaded;
  const down = downSeconds(uptime, (performance.now() - receivedAt) / 1000);
  const reason = uptime.state === "down" && uptime.reason !== null ? sentence(uptime.reason) : null;
  const elapsed = down === null ? null : formatActivity(down);
  const blockers = uptime.blockers.length;
  return (
    <span className="station-uptime">
      <span
        className={`uptime-chip uptime-chip--${uptime.state}`}
        // Named as a whole, so the reason is read with the state instead of being left to a hover.
        role="img"
        aria-label={`Station ${uptime.state}${elapsed === null ? "" : ` for ${elapsed}`}${reason === null ? "" : `: ${reason}`}`}
        title={reason ?? undefined}
      >
        <span className="uptime-chip-dot" aria-hidden="true" />
        {STATE_WORDS[uptime.state]}
        {elapsed !== null && ` · ${elapsed}`}
      </span>
      {dayHasTake(uptime) && (
        <span className="station-uptime-summary">
          active {formatTotal(uptime.active_seconds)} · down {formatTotal(uptime.down_seconds)}
          {blockers > 0 && ` · ${blockers} blocker${blockers === 1 ? "" : "s"}`}
        </span>
      )}
    </span>
  );
}
