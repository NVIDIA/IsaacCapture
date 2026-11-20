// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** How the station's time is worded, shared by the status strip's chip and the end-of-day dialog. */
import type { StationUptime } from "./stationsApi";

/** A duration as the operator reads it: hours and minutes once there are hours, else minutes and seconds. */
export function formatActivity(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${minutes}m ${String(whole % 60).padStart(2, "0")}s`;
}

/** A total for the day, to the minute: it is read at a glance, where seconds would only be noise. */
export function formatTotal(seconds: number): string {
  // Under a minute is not nothing: "0m" beside a listed blocker would read as a contradiction.
  if (seconds > 0 && seconds < 60) return "<1m";
  const minutes = Math.floor(Math.max(0, seconds) / 60);
  return minutes >= 60 ? `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m` : `${minutes}m`;
}

/** Whether the uptime shows the day has a take, for a reader with no take count: the totals span the first take
 * to the last, so before one they are all zero. A reader that has the day's report goes by its count instead. */
export function dayHasTake(uptime: StationUptime): boolean {
  return uptime.active_seconds > 0 || uptime.down_seconds > 0 || uptime.blockers.length > 0;
}
