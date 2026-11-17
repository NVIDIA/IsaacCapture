// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** Start-of-day flow bodies for tests: a station in use held by a camera that stopped, the same station ready,
 * and the same station ready on real cameras whose calibration files are missing but not required. */
import type { CheckItem, SetupStatus } from "./setupApi";

const SYSTEM: CheckItem[] = [
  { id: "calibration", group: "system", label: "Camera calibration", status: "skip", detail: "mock capture has nothing to calibrate" },
  { id: "disk", group: "system", label: "Free disk space", status: "pass", detail: "211.3 GB free" },
];

/** A station is in use and its left wrist camera has stopped delivering frames, so recording is held. */
export const SETUP_HELD: SetupStatus = {
  steps: [
    { id: "station", label: "Station", status: "done", detail: "Bench A" },
    { id: "devices", label: "Devices", status: "attention", detail: "Left wrist: cam_left: no recent frame" },
    { id: "record", label: "Record", status: "todo", detail: null },
    { id: "wrap_up", label: "Wrap up", status: "todo", detail: null },
  ],
  checks: [
    { id: "device:headset:0", group: "tracking", label: "Pico headset", status: "pass", detail: null },
    { id: "camera:ego", group: "cameras", label: "Ego (head): cam_head", status: "pass", detail: null },
    { id: "camera:wrist_left", group: "cameras", label: "Left wrist: cam_left", status: "fail", detail: "no recent frame" },
    { id: "camera:wrist_right", group: "cameras", label: "Right wrist", status: "skip", detail: "not used by this station" },
    ...SYSTEM,
  ],
  blocked_reason: "Left wrist: cam_left: no recent frame",
  enforced: true,
  // As a read without the control token answers.
  headset: null,
};

/** The same station with every device delivering: nothing holds recording. */
export const SETUP_READY: SetupStatus = {
  ...SETUP_HELD,
  steps: SETUP_HELD.steps.map((step) => (step.id === "devices" ? { ...step, status: "done", detail: null } : step)),
  checks: SETUP_HELD.checks.map((check) => (check.status === "fail" ? { ...check, status: "pass", detail: null } : check)),
  blocked_reason: null,
};

const MISSING_CALIBRATION = "missing head_intrinsics.json, head_extrinsics.json";

/** The ready station without its calibration files, in a deployment that does not require them: a warning that
 * holds nothing, with the Devices step done. */
export const SETUP_ADVISED: SetupStatus = {
  ...SETUP_READY,
  steps: SETUP_READY.steps.map((step) => (step.id === "devices" ? { ...step, detail: "Camera calibration: missing: head_camera_intrinsics.json; takes are not held" } : step)),
  checks: SETUP_READY.checks.map((check) => (check.id === "calibration" ? { ...check, status: "warn", detail: MISSING_CALIBRATION } : check)),
};

/** The same station in a deployment that requires the files: the same finding fails and holds recording. */
export const SETUP_UNCALIBRATED: SetupStatus = {
  ...SETUP_READY,
  steps: SETUP_READY.steps.map((step) => (step.id === "devices" ? { ...step, status: "attention", detail: `Camera calibration: ${MISSING_CALIBRATION}` } : step)),
  checks: SETUP_READY.checks.map((check) => (check.id === "calibration" ? { ...check, status: "fail", detail: MISSING_CALIBRATION } : check)),
  blocked_reason: `Camera calibration: ${MISSING_CALIBRATION}`,
};
