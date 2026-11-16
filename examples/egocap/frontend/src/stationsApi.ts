// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The station registry (`/api/stations`) and the station's uptime for today (`/api/station-uptime`).
 *
 * Reading either needs nothing; every change carries the run's control token, as recording commands do.
 */
import { SettingsHttpError, SettingsParseError } from "./settingsApi";

/** One station as the backend lists it. */
export interface StationRecord {
  id: string;
  name: string;
  camera_type: string | null;
  /** Which offered camera fills each role; null where unassigned. Every role is present. */
  camera_roles: Record<string, string | null>;
  /** Why this station cannot record yet, most important first; empty when it can. */
  problems: string[];
}

/** `GET /api/stations`, and the reply to every registry change. */
export interface StationListing {
  stations: StationRecord[];
  /** The id of the station in use, or null when none is: the deployment's own station label then applies. */
  selected: string | null;
  roles: string[];
  camera_types: string[];
  /** The station label recordings carry right now. */
  label: string;
  /** Why the station in use cannot change now (a take is being recorded), or null. */
  locked_reason: string | null;
  /** Set when the saved registry could not be read. */
  warning?: string;
}

/** The fields a station can be created or changed with; a role change may name only some roles. */
export interface StationChange {
  name?: string;
  camera_type?: string | null;
  camera_roles?: Record<string, string | null>;
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

/** Checks the fields the page reads without guarding; the rest is trusted, since the two halves ship together. */
function parseListing(value: unknown): StationListing {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  if (!Array.isArray(record.stations) || !Array.isArray(record.roles) || typeof record.label !== "string") {
    throw new SettingsParseError("station listing is malformed");
  }
  return value as StationListing;
}

/** Sends one request and parses the listing every station route answers with. */
async function listingFrom(path: string, init: RequestInit): Promise<StationListing> {
  const response = await fetch(path, init);
  if (!response.ok) throw await failureOf(response);
  return parseListing(await response.json());
}

/** The headers of a change: the credential, and the JSON type when a body follows. */
function changeHeaders(controlToken: string, withBody: boolean): Record<string, string> {
  return withBody ? { "content-type": "application/json", "X-Control-Token": controlToken } : { "X-Control-Token": controlToken };
}

/** Load the registry. No credential is sent: the page reads the headset's address from the start-of-day flow. */
export function fetchStations(signal?: AbortSignal): Promise<StationListing> {
  return listingFrom("/api/stations", { signal });
}

/** Add a station. */
export function addStation(controlToken: string, fields: StationChange): Promise<StationListing> {
  return listingFrom("/api/stations", { method: "POST", headers: changeHeaders(controlToken, true), body: JSON.stringify(fields) });
}

/** Change the named fields of one station. */
export function updateStation(controlToken: string, id: string, changes: StationChange): Promise<StationListing> {
  return listingFrom(`/api/stations/${encodeURIComponent(id)}`, { method: "PUT", headers: changeHeaders(controlToken, true), body: JSON.stringify(changes) });
}

/** Delete one station. */
export function deleteStation(controlToken: string, id: string): Promise<StationListing> {
  return listingFrom(`/api/stations/${encodeURIComponent(id)}`, { method: "DELETE", headers: changeHeaders(controlToken, false) });
}

/** Put a station in use, or none for null. */
export function selectStation(controlToken: string, id: string | null): Promise<StationListing> {
  return listingFrom("/api/stations/select", { method: "POST", headers: changeHeaders(controlToken, true), body: JSON.stringify({ id }) });
}

export type UptimeState = "ready" | "recording" | "down";

/** One stretch of the day in which the station could not record. */
export interface UptimeBlocker {
  reason: string;
  /** Epoch seconds the stretch began. */
  start: number;
  /** Epoch seconds it ended; null while it is still open. */
  end: number | null;
  seconds: number;
}

/**
 * `GET /api/station-uptime`: whether the station in use can record right now, and today's timeline of it.
 * The backend keeps the timeline by itself; the totals and blockers cover the time between the first and the
 * last take of the day, so they are zero and empty before the first take.
 */
export interface StationUptime {
  station: string;
  date: string;
  state: UptimeState;
  /** Why the station cannot record; set only while it is down. */
  reason: string | null;
  /** Epoch seconds the current state began; without it no elapsed time is shown. */
  since: number | null;
  active_seconds: number;
  down_seconds: number;
  blockers: UptimeBlocker[];
  /** The operator's notes for the day. */
  notes: string;
  /** Epoch seconds the figures were computed at. */
  as_of: number;
}

const UPTIME_STATES: readonly unknown[] = ["ready", "recording", "down"];

/** Checks the fields the status strip and the end-of-day dialog render from. */
export function parseUptime(value: unknown): StationUptime {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  const numeric = typeof record.active_seconds === "number" && typeof record.down_seconds === "number" && typeof record.as_of === "number";
  if (!numeric || !UPTIME_STATES.includes(record.state) || !Array.isArray(record.blockers) || typeof record.notes !== "string") {
    throw new SettingsParseError("station uptime is malformed");
  }
  return value as StationUptime;
}

/** Load today's uptime for the station in use. */
export async function fetchStationUptime(signal?: AbortSignal): Promise<StationUptime> {
  const response = await fetch("/api/station-uptime", { signal });
  if (!response.ok) throw await failureOf(response);
  return parseUptime(await response.json());
}

/** Save the day's notes. Notes are all that can be set: the state is the backend's own. */
export async function saveStationNotes(controlToken: string, notes: string): Promise<StationUptime> {
  const response = await fetch("/api/station-uptime", { method: "POST", headers: changeHeaders(controlToken, true), body: JSON.stringify({ notes }) });
  if (!response.ok) throw await failureOf(response);
  return parseUptime(await response.json());
}
