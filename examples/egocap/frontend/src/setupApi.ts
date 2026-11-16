// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day flow (`backend/egocap_web/setup_routes.py`): where each step stands, the last device
 * connection test, the system checks, and what holds recording. Reading it needs nothing; running the test
 * carries the run's control token, since its result decides whether a take may start.
 */
import { SettingsHttpError, SettingsParseError } from "./settingsApi";

export type StepStatus = "done" | "todo" | "attention" | "skipped";
export type StepId = "station" | "devices" | "calibration" | "record" | "wrap_up";

/** One step of the flow. */
export interface SetupStep {
  id: StepId;
  label: string;
  status: StepStatus;
  /** What the step is waiting for, what is wrong, or what it settled on; null when there is nothing to say. */
  detail: string | null;
}

export type CheckStatus = "pass" | "fail" | "skip";

/** One device or check, and how it stands. */
export interface CheckItem {
  id: string;
  /** "tracking", "cameras" or "system". */
  group: string;
  label: string;
  status: CheckStatus;
  detail: string | null;
}

/** The last run of the device connection test, while it still describes this station and capture session. */
export interface ConnectionTest {
  /** Epoch seconds the test ran at. */
  ran_at: number;
  station: string;
  passed: boolean;
  items: CheckItem[];
}

/** `GET /api/setup`, and the reply to running the test. */
export interface SetupStatus {
  steps: SetupStep[];
  connection_test: ConnectionTest | null;
  system: CheckItem[];
  /** Why station setup holds a take right now, or null. */
  blocked_reason: string | null;
  /** False while no station is in use: the flow then advises and holds nothing. */
  enforced: boolean;
}

/** Checks the parts the page reads without guarding. */
function parseSetup(value: unknown): SetupStatus {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  if (!Array.isArray(record.steps) || !Array.isArray(record.system) || typeof record.enforced !== "boolean") {
    throw new SettingsParseError("setup status is malformed");
  }
  return value as SetupStatus;
}

/** One request to a setup route, answered with the flow or the backend's reason. */
async function setupFrom(path: string, init: RequestInit): Promise<SetupStatus> {
  const response = await fetch(path, init);
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

/** Read where the flow stands. */
export function fetchSetup(signal?: AbortSignal): Promise<SetupStatus> {
  return setupFrom("/api/setup", { signal });
}

/** Test every device the station in use needs, now. */
export function runConnectionTest(controlToken: string): Promise<SetupStatus> {
  return setupFrom("/api/setup/connection-test", { method: "POST", headers: { "X-Control-Token": controlToken } });
}
