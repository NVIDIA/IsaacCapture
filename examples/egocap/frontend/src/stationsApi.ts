// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The station registry, the station's activity for today and the headset address
 * (`backend/egocap_web/station_routes.py`).
 *
 * Reading the registry and the activity needs nothing; a station's headset address and this host's network
 * addresses are returned only with the run's control token, and every change carries it, as recording
 * commands do.
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
  /** The capture host address last entered in the headset. Present only when the request carried the token. */
  headset_address?: string | null;
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
  headset_address?: string | null;
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

/** Load the registry; with the control token each station also carries its headset address. */
export function fetchStations(signal?: AbortSignal, controlToken?: string | null): Promise<StationListing> {
  return listingFrom("/api/stations", { signal, headers: controlToken ? { "X-Control-Token": controlToken } : {} });
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

/** One address the headset could be given. */
export interface HostAddress {
  address: string;
  /** The network interface holding the address, or null when it is not one of the backend host's. */
  interface: string | null;
  kind: string;
}

/** `GET /api/headset-address`. */
export interface HeadsetAddress {
  /** Best first: wireless before wired. Empty when the host has no usable address. */
  candidates: HostAddress[];
  /** The address saved for the station in use, or null. */
  saved: string | null;
  /** The saved address is no longer one the host has, so the headset needs the new one. */
  changed: boolean;
}

/** Load the addresses to enter in the headset. */
export async function fetchHeadsetAddress(controlToken: string, signal?: AbortSignal): Promise<HeadsetAddress> {
  const response = await fetch("/api/headset-address", { signal, headers: { "X-Control-Token": controlToken } });
  if (!response.ok) throw await failureOf(response);
  const body = (await response.json()) as Partial<HeadsetAddress>;
  if (!Array.isArray(body.candidates)) throw new SettingsParseError("headset address is malformed");
  return body as HeadsetAddress;
}

export type ActivityState = "active" | "inactive";

/** `GET /api/station-activity`: today's active and inactive time for the station in use. */
export interface StationActivity {
  station: string;
  date: string;
  state: ActivityState;
  /** Epoch seconds the current state began, or null before the first change of the day. */
  since: number | null;
  active_seconds: number;
  inactive_seconds: number;
  /** How many times the station went from active to inactive today. */
  blocker_count: number;
  notes: string;
  /** Epoch seconds the totals were computed at. */
  as_of: number;
}

/** Checks the fields the toggle renders from. */
function parseActivity(value: unknown): StationActivity {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  const numeric = typeof record.active_seconds === "number" && typeof record.inactive_seconds === "number";
  if (!numeric || (record.state !== "active" && record.state !== "inactive") || typeof record.station !== "string") {
    throw new SettingsParseError("station activity is malformed");
  }
  return value as StationActivity;
}

/** Load today's activity. */
export async function fetchStationActivity(signal?: AbortSignal): Promise<StationActivity> {
  const response = await fetch("/api/station-activity", { signal });
  if (!response.ok) throw await failureOf(response);
  return parseActivity(await response.json());
}

/** Set the state, the blocker notes, or both. */
export async function changeStationActivity(controlToken: string, change: { state?: ActivityState; notes?: string }): Promise<StationActivity> {
  const response = await fetch("/api/station-activity", { method: "POST", headers: changeHeaders(controlToken, true), body: JSON.stringify(change) });
  if (!response.ok) throw await failureOf(response);
  return parseActivity(await response.json());
}
