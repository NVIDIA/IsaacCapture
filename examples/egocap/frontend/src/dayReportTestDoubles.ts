// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** A day's report for tests: one session whose takes ended every way a take can. */
import type { DayReportReply, ReportEpisode, Tally, TakeDisposition } from "./dayReportApi";

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

/** Bench A's day: two kept takes, one flagged, one rejected and one interrupted, and its activity. */
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
  activity: {
    station: "Bench A",
    date: "2026-03-09",
    state: "inactive",
    since: 1_000,
    active_seconds: 18_720,
    inactive_seconds: 2_400,
    blocker_count: 2,
    notes: "left glove battery",
    as_of: 2_000,
  },
  today: "2026-03-09",
  station_in_use: "Bench A",
};
