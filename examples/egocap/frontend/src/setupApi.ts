// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day flow (`GET /api/setup`): where each step stands, the live checklist of what the station in
 * use needs, and what holds recording. Reading it needs nothing; with the run's control token the reply also
 * says which address the headset connects to.
 */
import { SettingsHttpError, SettingsParseError } from "./settingsApi";

export type StepStatus = "done" | "todo" | "attention" | "skipped";
export type StepId = "station" | "devices" | "record" | "wrap_up";

/** One step of the flow. */
export interface SetupStep {
  id: StepId;
  label: string;
  status: StepStatus;
  /** What the step is waiting for, what is wrong, or what it settled on; null when there is nothing to say. */
  detail: string | null;
}

export type CheckStatus = "pass" | "fail" | "skip";
export type CheckGroup = "tracking" | "cameras" | "system";

/** One device or check, and how it stands right now. */
export interface CheckItem {
  id: string;
  group: CheckGroup;
  label: string;
  status: CheckStatus;
  detail: string | null;
}

/** Where the headset connects, for the station in use. */
export interface HeadsetLink {
  /** The capture host address to enter in the headset; null when none could be determined. */
  address: string | null;
  /** The address remembered for the station in use, or null. */
  saved: string | null;
  /** The remembered address is no longer the host's and the headset is not connected. */
  changed: boolean;
}

/** `GET /api/setup`. */
export interface SetupStatus {
  steps: SetupStep[];
  /** Every device and check the backend judges, continuously. */
  checks: CheckItem[];
  /** Why station setup holds a take right now, or null. */
  blocked_reason: string | null;
  /** False while no station is in use: the checklist then advises and holds nothing. */
  enforced: boolean;
  /** Null unless the request carried a valid control token. */
  headset: HeadsetLink | null;
}

/** Checks the parts the page reads without guarding. */
function parseSetup(value: unknown): SetupStatus {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  if (!Array.isArray(record.steps) || !Array.isArray(record.checks) || typeof record.enforced !== "boolean") {
    throw new SettingsParseError("setup status is malformed");
  }
  return value as SetupStatus;
}

/** Read where the flow stands; with the control token the reply carries the headset address. */
export async function fetchSetup(signal?: AbortSignal, controlToken?: string | null): Promise<SetupStatus> {
  const response = await fetch("/api/setup", { signal, headers: controlToken ? { "X-Control-Token": controlToken } : {} });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // A non-JSON error body keeps the status-code message.
    }
    throw new SettingsHttpError(detail);
  }
  return parseSetup(await response.json());
}
