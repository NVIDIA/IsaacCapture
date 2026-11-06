// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Operator-facing health of the tracking devices and cameras: one short chip per device, derived from the capture
 * status only. Shared by the page's status bar and its session details so both agree on what is healthy.
 */

import {
  DeviceState,
  deviceStateLabel,
  statusReasonLabel,
  type CameraStatus,
  type CaptureStatusResponse,
  type DeviceStatus,
} from "./captureStatus";

export type Tone = "neutral" | "positive" | "warning" | "critical";

// Keyed directly off DeviceState's own numeric values, not its label text --
// cameras and tracking devices both use DeviceState, so one table covers both.
const DEVICE_TONE: Partial<Record<number, Tone>> = {
  [DeviceState.CONNECTED]: "positive",
  [DeviceState.FAILED]: "critical",
  [DeviceState.DISCONNECTED]: "critical",
  [DeviceState.UNAVAILABLE]: "critical",
  [DeviceState.DEGRADED]: "warning",
  [DeviceState.INITIALIZING]: "warning",
  [DeviceState.UNKNOWN]: "warning",
  [DeviceState.DISABLED]: "neutral",
};

/** The semantic tone for a tracking device or camera's DeviceState. */
export function deviceTone(status: DeviceStatus["status"] | CameraStatus["status"]): Tone {
  return DEVICE_TONE[status] ?? "neutral";
}

/** Downgrades a positive ("current and healthy") tone to neutral when the data behind it is retained rather than
 * live. Non-positive tones (warning/critical) are left alone: a real fault is still worth flagging even in retained
 * data. */
export function retainedTone(tone: Tone, live: boolean): Tone {
  return !live && tone === "positive" ? "neutral" : tone;
}

/** One device or camera as the operator reads it: a short name, how healthy it is, and why when it is not. */
export interface HealthChip {
  key: string;
  label: string;
  tone: Tone;
  /** The state word ("Connected", "Disconnected", ...). */
  state: string;
  /** The device's own error or reason, only when there is one worth showing; null for a healthy device. */
  detail: string | null;
}

/** A short operator label for a tracking device. The ids are the capture server's own (``openxr/headset``,
 * ``openxr/controller/left``, ``manus/hand/right``); anything unrecognised keeps its description so a new device
 * kind still shows up rather than vanishing. */
export function deviceChipLabel(device: Pick<DeviceStatus, "id" | "description">): string {
  const side = device.id.endsWith("/left") ? "L" : device.id.endsWith("/right") ? "R" : null;
  if (device.id.includes("headset")) return "Headset";
  if (device.id.includes("controller")) return side === null ? "Controller" : `Controller ${side}`;
  if (device.id.includes("hand")) return side === null ? "Glove" : `Glove ${side}`;
  return device.description || device.id;
}

/** The reason shown on a chip: nothing for a healthy device, otherwise its error text or its status reason. */
function detailOf(tone: Tone, error: string, reason: DeviceStatus["reason"]): string | null {
  if (tone === "positive") return null;
  if (error !== "") return error;
  const label = statusReasonLabel(reason);
  return label === "None" ? null : label;
}

/** Every tracking device then every camera, in the order the capture status reports them. ``live`` is false for a
 * retained (stale or final) response, which never keeps the healthy look. */
export function deriveHealthChips(response: CaptureStatusResponse | null, live: boolean): HealthChip[] {
  const devices = response?.status?.devices ?? [];
  const cameras = response?.status?.cameras ?? [];
  const chips: HealthChip[] = [];
  for (const d of devices) {
    const tone = retainedTone(deviceTone(d.status), live);
    chips.push({
      key: `device:${d.id}`,
      label: deviceChipLabel(d),
      tone,
      state: deviceStateLabel(d.status),
      detail: detailOf(tone, d.error, d.reason),
    });
  }
  for (const c of cameras) {
    const tone = retainedTone(deviceTone(c.status), live);
    chips.push({
      key: `camera:${c.stream_id}`,
      label: c.name,
      tone,
      state: deviceStateLabel(c.status),
      detail: detailOf(tone, c.error, c.reason),
    });
  }
  return chips;
}

/** The chips that need the operator's attention (anything not healthy), for the bar's one-line summary. */
export function chipsNeedingAttention(chips: readonly HealthChip[]): HealthChip[] {
  return chips.filter((chip) => chip.tone === "warning" || chip.tone === "critical");
}
