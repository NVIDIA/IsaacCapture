// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The end-of-day report (`/api/day-report`): today's takes for the station in use, by session, with what
 * became of each; the station's uptime; saving the report on the station; and correcting a task after
 * recording. Every call carries the run's control token: the report names operators and tasks, and a
 * correction rewrites recorded labels.
 */
import { SettingsHttpError, SettingsParseError } from "./settingsApi";
import { parseUptime, type StationUptime } from "./stationsApi";

/** What became of a take. `recording` is the one being written right now. */
export type TakeDisposition = "kept" | "flagged" | "rejected" | "discarded" | "pending" | "recording";

/** The dispositions in the order they are listed. */
export const DISPOSITIONS: readonly TakeDisposition[] = ["kept", "flagged", "rejected", "discarded", "pending", "recording"];

/** A count and a duration for each disposition; every disposition is present. */
export type Tally = Record<TakeDisposition, { count: number; seconds: number }>;

/** One take as the report lists it. */
export interface ReportEpisode {
  index: number;
  episode_id: string;
  disposition: TakeDisposition;
  duration_seconds: number;
  task: string;
  task_details: string;
  operator: string;
  issues: string[];
  qc_decision: string | null;
  /** Where the take's data is, relative to the recording output: `<session or group>/episode_<n>`. */
  location: string;
}

/** One recording session's takes of the day. */
export interface ReportSession {
  session_id: string;
  station: string | null;
  /** When the session was opened, as local `YYYY-MM-DDTHH:MM:SS`; null when its name does not say. */
  started_at: string | null;
  by_disposition: Tally;
  episodes: ReportEpisode[];
  /** Manifests that could not be read: takes the day cannot account for. */
  problems: string[];
}

export interface DayReport {
  date: string;
  /** The station the report is limited to; null when it is not limited to one. */
  station: string | null;
  totals: { episodes: number; seconds: number; by_disposition: Tally };
  sessions: ReportSession[];
}

/** `GET /api/day-report`, and with `saved` the reply to saving it. */
export interface DayReportReply {
  report: DayReport;
  /** The station's active and down time today, its blockers and the operator's notes. */
  uptime: StationUptime;
  today: string;
  station_in_use: string;
  /** Present after a save: the files written on the station. */
  saved?: { report: string; activity_log: string };
}

/** A task correction for a range of one session's episodes. */
export interface Correction {
  /** The session or routed group holding the episodes, relative to the recording output. */
  session: string;
  /** One episode number or a range, such as `3` or `2-7`. */
  episodes: string;
  task: string;
}

/** `POST /api/day-report/correct`. */
export interface CorrectionReply {
  session: string;
  updated: string[];
  /** Episodes of the range that hold no recorded data to relabel. */
  missing: string[];
  config_updated: boolean;
}

/** The backend's public reason for a refusal, or the HTTP status when it gave none. */
async function failureOf(response: Response): Promise<SettingsHttpError> {
  let detail = `HTTP ${response.status}`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // A non-JSON error body keeps the status-code message.
  }
  return new SettingsHttpError(detail);
}

/** Checks the parts the dialog reads without guarding. */
function parseReply(value: unknown): DayReportReply {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  const report = record.report as Partial<DayReport> | undefined;
  if (typeof report !== "object" || report === null || !Array.isArray(report.sessions) || typeof report.totals !== "object" || report.totals === null) {
    throw new SettingsParseError("day report is malformed");
  }
  parseUptime(record.uptime);
  return value as DayReportReply;
}

/** The headers of a change: the credential and the JSON type. */
function jsonHeaders(controlToken: string): Record<string, string> {
  return { "content-type": "application/json", "X-Control-Token": controlToken };
}

/** Read today's report for the station in use. */
export async function fetchDayReport(controlToken: string, signal?: AbortSignal): Promise<DayReportReply> {
  const response = await fetch("/api/day-report", { signal, headers: { "X-Control-Token": controlToken } });
  if (!response.ok) throw await failureOf(response);
  return parseReply(await response.json());
}

/** Save today's report on the station and bring its activity log up to date. */
export async function saveDayReport(controlToken: string): Promise<DayReportReply> {
  const response = await fetch("/api/day-report/save", { method: "POST", headers: jsonHeaders(controlToken), body: "{}" });
  if (!response.ok) throw await failureOf(response);
  return parseReply(await response.json());
}

/** Correct the task of a range of a session's episodes. */
export async function correctSession(controlToken: string, correction: Correction): Promise<CorrectionReply> {
  const response = await fetch("/api/day-report/correct", { method: "POST", headers: jsonHeaders(controlToken), body: JSON.stringify(correction) });
  if (!response.ok) throw await failureOf(response);
  const body = (await response.json()) as Partial<CorrectionReply>;
  if (!Array.isArray(body.updated) || !Array.isArray(body.missing)) throw new SettingsParseError("correction reply is malformed");
  return body as CorrectionReply;
}
