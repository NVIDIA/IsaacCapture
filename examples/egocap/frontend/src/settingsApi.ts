// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * `GET /api/settings`: the backend's effective configuration, read-only.
 *
 * The backend resolves these once at startup from its environment and station file
 * (`backend/egocap_web/settings_routes.py`); the page shows them so an operator can tell what a
 * station is running with without reading its launch shell. Nothing here is written back: the
 * browser is never a configuration source.
 */

export interface CameraSelectionSetting {
  name: string;
  type: string;
}

export interface EffectiveSettings {
  capture: {
    server_url: string;
    station_label: string;
    profile_label: string;
    sources: string[];
    plugins: string[];
    cameras: CameraSelectionSetting[];
  };
  preview: { camera_sample_hz: number; tracking_sample_hz: number };
  station: {
    path: string | null;
    error: string | null;
    id: string | null;
    operator_name: string | null;
    auto_fill_operator_name: boolean;
    task_description: string | null;
    data_dir: string | null;
  };
  recording: { output_root: string };
  visualization: {
    error: string | null;
    enabled: boolean;
    host: string | null;
    port: number | null;
    public_url: string | null;
    device: string | null;
    head_model: string | null;
    head_model_scale: number | null;
  };
  livekit: { credentials_configured: boolean };
  recordings_browser: { enabled: boolean };
  task_files: { dir: string; available: boolean };
  origins: { allowed: string[]; trusted_hostnames: string[] | null };
}

/** The backend answered, but not with 200: `message` carries its `detail` or the HTTP status. */
export class SettingsHttpError extends Error {}

/** The body was not the expected shape; the page shows nothing rather than guessing. */
export class SettingsParseError extends Error {}

const GROUPS = [
  "capture",
  "preview",
  "station",
  "recording",
  "visualization",
  "livekit",
  "recordings_browser",
  "task_files",
  "origins",
] as const;

/** Checks the body is an object with every group present as an object; field types are trusted from there,
 * since the two halves ship together and the render tolerates a missing leaf (it shows a dash). */
export function parseSettings(value: unknown): EffectiveSettings {
  if (typeof value !== "object" || value === null) throw new SettingsParseError("settings must be an object");
  const record = value as Record<string, unknown>;
  for (const group of GROUPS) {
    const member = record[group];
    if (typeof member !== "object" || member === null) throw new SettingsParseError(`settings.${group} is missing`);
  }
  return value as EffectiveSettings;
}

/**
 * `GET`/`PUT /api/operator-settings`: the operator's own station setup, kept by the backend so every browser
 * sees the same value. A change carries the run's control token, as recording commands do.
 */
export interface OperatorSettings {
  camera_type: string | null;
  /** The accepted values, from the backend, so the page keeps no second copy of the list. */
  camera_types: string[];
  /** Which offered camera (by its session name) fills each role; null where unassigned. Every role is present. */
  camera_roles: Record<string, string | null>;
  /** The roles, in display order, from the backend. */
  roles: string[];
  /** The catalog the recording panel's task picker offers, or null for free text only. */
  task_file: string | null;
  /** The task (by catalog name) preselected while the current task is still unset. */
  preselected_task: string | null;
}

/** Operator-facing names for the camera types the backend accepts; an unknown value shows its raw name. */
export const CAMERA_TYPE_LABELS: Record<string, string> = {
  sensing_gmsl: "SENSING GMSL cameras — published from the Jetson over LiveKit",
  oak_d: "OAK-D cameras — USB, read by CaptureServer",
};

/** One entry of `GET /api/tools`: an operator tool this build knows about and whether it can launch it. */
export interface ToolInfo {
  id: string;
  name: string;
  description: string;
  /** The camera type the tool is for, or null when it applies to every station. */
  applies_to: string | null;
  available: boolean;
  unavailable_reason: string | null;
}

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

function parseOperatorSettings(value: unknown): OperatorSettings {
  if (typeof value !== "object" || value === null) throw new SettingsParseError("operator settings must be an object");
  const record = value as Record<string, unknown>;
  if (record.camera_type !== null && typeof record.camera_type !== "string") {
    throw new SettingsParseError("operator settings camera_type must be a string or null");
  }
  if (!Array.isArray(record.camera_types)) throw new SettingsParseError("operator settings camera_types must be a list");
  if (typeof record.camera_roles !== "object" || record.camera_roles === null || !Array.isArray(record.roles)) {
    throw new SettingsParseError("operator settings camera_roles/roles are missing");
  }
  return value as OperatorSettings;
}

export async function fetchOperatorSettings(signal?: AbortSignal): Promise<OperatorSettings> {
  const response = await fetch("/api/operator-settings", { signal });
  if (!response.ok) throw await failureOf(response);
  return parseOperatorSettings(await response.json());
}

/** Operator-facing names for the camera roles; an unknown role shows its raw name. */
export const ROLE_LABELS: Record<string, string> = {
  ego: "Ego (head)",
  wrist_left: "Left wrist",
  wrist_right: "Right wrist",
};

/** The fields a `PUT` may change; the backend keeps the stored value of every field left out. A `camera_roles`
 * change may name only some roles. */
export type OperatorSettingsChange = Partial<Pick<OperatorSettings, "camera_type" | "task_file" | "preselected_task">> & {
  camera_roles?: Record<string, string | null>;
};

export async function saveOperatorSettings(controlToken: string, changes: OperatorSettingsChange): Promise<OperatorSettings> {
  const response = await fetch("/api/operator-settings", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ control_token: controlToken, ...changes }),
  });
  if (!response.ok) throw await failureOf(response);
  return parseOperatorSettings(await response.json());
}

/** One task as a catalog defines it; only `name` and `instruction` are always present. */
export interface TaskDefinition {
  name: string;
  instruction: string;
  studio?: string;
  scenario?: string;
  primitives?: string[];
  props?: string;
  left_grasp?: string;
  right_grasp?: string;
}

export interface TaskFileContents {
  name: string;
  format: string;
  tasks: TaskDefinition[];
}

/** `GET /api/task-files`: every catalog name, and which of them the operator imported. */
export interface TaskFileListing {
  files: string[];
  imported: string[];
}

/** What an import takes: a CSV path on the backend's machine, or the file's name and text from this browser. */
export type TaskFileSource = { path: string } | { name: string; content: string };

function parseListing(value: unknown): TaskFileListing {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  if (!Array.isArray(record.files) || !Array.isArray(record.imported)) throw new SettingsParseError("task file listing is malformed");
  return { files: record.files as string[], imported: record.imported as string[] };
}

export async function fetchTaskFiles(signal?: AbortSignal): Promise<TaskFileListing> {
  const response = await fetch("/api/task-files", { signal });
  if (!response.ok) throw await failureOf(response);
  return parseListing(await response.json());
}

export async function fetchTaskFile(name: string, signal?: AbortSignal): Promise<TaskFileContents> {
  const response = await fetch(`/api/task-files/${encodeURIComponent(name)}`, { signal });
  if (!response.ok) throw await failureOf(response);
  const body = (await response.json()) as Partial<TaskFileContents>;
  if (typeof body.name !== "string" || !Array.isArray(body.tasks)) throw new SettingsParseError("task file is malformed");
  return body as TaskFileContents;
}

export async function importTaskFile(controlToken: string, source: TaskFileSource): Promise<TaskFileListing & { file: TaskFileContents }> {
  const response = await fetch("/api/task-files/import", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ control_token: controlToken, ...source }),
  });
  if (!response.ok) throw await failureOf(response);
  const body = (await response.json()) as Record<string, unknown>;
  return { ...parseListing(body), file: body.file as TaskFileContents };
}

export async function removeTaskFile(controlToken: string, name: string): Promise<TaskFileListing> {
  const response = await fetch(`/api/task-files/${encodeURIComponent(name)}`, {
    method: "DELETE",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ control_token: controlToken }),
  });
  if (!response.ok) throw await failureOf(response);
  return parseListing(await response.json());
}

export async function fetchTools(signal?: AbortSignal): Promise<ToolInfo[]> {
  const response = await fetch("/api/tools", { signal });
  if (!response.ok) throw await failureOf(response);
  const body = (await response.json()) as { tools?: unknown };
  if (!Array.isArray(body.tools)) throw new SettingsParseError("tools must be a list");
  return body.tools as ToolInfo[];
}

export async function fetchSettings(signal?: AbortSignal): Promise<EffectiveSettings> {
  const response = await fetch("/api/settings", { signal });
  if (!response.ok) throw await failureOf(response);
  return parseSettings(await response.json());
}
