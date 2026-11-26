// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day flow (`GET /api/setup`): where each step stands, the live checklist of what the station in
 * use needs, and what holds recording. Reading it needs nothing; with the run's control token the reply also
 * says which address the headset connects to.
 */
import { READ_TIMEOUT_MS, withTimeout } from "./requestTimeout";
import { SettingsHttpError, SettingsParseError } from "./settingsApi";

export type StepStatus = "done" | "todo" | "attention" | "skipped";

const STEP_STATUSES: readonly unknown[] = ["done", "todo", "attention", "skipped"];
export type StepId = "station" | "devices" | "record" | "wrap_up";

/** One step of the flow. */
export interface SetupStep {
  /** Which step this is; the rail opens the place a step is dealt with by it. */
  id: StepId;
  /** The step's name as the rail shows it. */
  label: string;
  /** Where the step stands. */
  status: StepStatus;
  /** What the step is waiting for, what is wrong, or what it settled on; null when there is nothing to say. */
  detail: string | null;
}

/** How a check stands: `warn` is a finding the deployment chose not to hold recording for, and `unknown` is an
 * outcome this page has no name for, which a newer backend may send. */
export type CheckStatus = "pass" | "fail" | "warn" | "skip" | "unknown";

/** The outcomes the backend sends; `unknown` is only ever given by this page. */
const CHECK_STATUSES: readonly unknown[] = ["pass", "fail", "warn", "skip"];
export type CheckGroup = "tracking" | "cameras" | "system";

/** One device or check, and how it stands right now. */
export interface CheckItem {
  /** A stable name for the check, unique in the list. */
  id: string;
  /** The heading the check is listed under. */
  group: CheckGroup;
  /** The device or check as the operator knows it. */
  label: string;
  /** How it stands right now. */
  status: CheckStatus;
  /** Why it did not simply pass, or null. */
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
  /** The flow's steps, in order. */
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
  if (!record.checks.every((check) => typeof check === "object" && check !== null)) {
    throw new SettingsParseError("setup status has a malformed check");
  }
  if (!record.steps.every((step) => typeof step === "object" && step !== null)) {
    throw new SettingsParseError("setup status has a malformed step");
  }
  // An outcome this page has no mark for is shown as exactly that. Calling it "not needed" would hide why a take
  // is held, and refusing the whole reply would blank the rail and the list over one row.
  const checks = record.checks.map((check: CheckItem) => (CHECK_STATUSES.includes(check.status) ? check : { ...check, status: "unknown" as const }));
  // A step in a state this page cannot draw asks for a look, which is the one reading that is never falsely calm.
  const steps = record.steps.map((step: SetupStep) => (STEP_STATUSES.includes(step.status) ? step : { ...step, status: "attention" as const }));
  return { ...(value as SetupStatus), steps, checks };
}

/** Read where the flow stands; with the control token the reply carries the headset address. */
export function fetchSetup(signal?: AbortSignal, controlToken?: string | null): Promise<SetupStatus> {
  // Bounded: the page reads this on a timer that waits for each answer, so one that never came would stop it.
  return withTimeout(signal, READ_TIMEOUT_MS, async (bounded) => {
    const response = await fetch("/api/setup", { signal: bounded, headers: controlToken ? { "X-Control-Token": controlToken } : {} });
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
  });
}
