// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** A day's report for tests: one session whose takes ended every way a take can, and the station's uptime. */
import type { DayReportReply, ReportEpisode, Tally, TakeDisposition } from "./dayReportApi";
import type { StationUptime } from "./stationsApi";

const SESSION = "session_20260309-081500_abc123";

/** One take of the fixture session. */
function take(index: number, disposition: TakeDisposition, seconds: number, extra: Partial<ReportEpisode> = {}): ReportEpisode {
  return {
    index,
    episode_id: `${SESSION}/episode_${index}`,
    disposition,
    duration_seconds: seconds,
    task: "Pour water.",
    task_details: "",
    operator: "alex",
    issues: [],
    qc_decision: null,
    location: `${SESSION}/episode_${index}`,
    ...extra,
  };
}

const TALLY: Tally = {
  kept: { count: 2, seconds: 3_725 },
  flagged: { count: 1, seconds: 8 },
  rejected: { count: 1, seconds: 3 },
  discarded: { count: 0, seconds: 0 },
  pending: { count: 1, seconds: 4 },
  recording: { count: 0, seconds: 0 },
};

/** Epoch seconds of a local clock time on the fixture's day, so the times a test reads do not depend on its time zone. */
function localTime(hour: number, minute: number): number {
  return new Date(2026, 2, 9, hour, minute).getTime() / 1000;
}

/** Bench A's uptime: down since 14:05 for the reason that already blocked it once in the morning. */
export const UPTIME_FIXTURE: StationUptime = {
  station: "Bench A",
  date: "2026-03-09",
  state: "down",
  reason: "Left wrist: cam_left: no recent frame",
  since: localTime(14, 5),
  active_seconds: 7_500,
  down_seconds: 1_020,
  blockers: [
    { reason: "Left wrist: cam_left: no recent frame", start: localTime(11, 2), end: localTime(11, 14), seconds: 720 },
    { reason: "Left wrist: cam_left: no recent frame", start: localTime(14, 5), end: null, seconds: 300 },
  ],
  notes: "left glove battery",
  as_of: localTime(14, 10),
};

/** Bench A's day: two kept takes, one flagged, one rejected and one interrupted, and its uptime. */
export const DAY_REPORT_FIXTURE: DayReportReply = {
  report: {
    date: "2026-03-09",
    station: "Bench A",
    totals: { episodes: 5, seconds: 3_740, by_disposition: TALLY },
    sessions: [
      {
        session_id: SESSION,
        station: "Bench A",
        started_at: "2026-03-09T08:15:00",
        by_disposition: TALLY,
        episodes: [
          take(0, "kept", 3_600),
          take(1, "kept", 125, { task_details: "Left hand, blue cup." }),
          take(2, "flagged", 8, { issues: ["sustained_freeze:pico_controller/head_pose"], qc_decision: "FLAGGED_BAD" }),
          take(3, "rejected", 3, { issues: ["camera_min_fps:head_human_camera"], qc_decision: "REJECTED_CAMERA" }),
          take(4, "pending", 4),
        ],
        problems: [],
      },
    ],
  },
  uptime: UPTIME_FIXTURE,
  today: "2026-03-09",
  station_in_use: "Bench A",
};
