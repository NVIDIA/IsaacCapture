// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** Start-of-day flow bodies for tests: a station in use before its connection test, and after it passed. */
import type { CheckItem, SetupStatus } from "./setupApi";

const SYSTEM: CheckItem[] = [
  { id: "calibration", group: "system", label: "Camera calibration", status: "pass", detail: null },
  { id: "gloves", group: "system", label: "Glove plugin", status: "skip", detail: "gloves are not part of this deployment" },
  { id: "disk", group: "system", label: "Free disk space", status: "pass", detail: "211.3 GB free" },
  { id: "producer", group: "system", label: "Camera producer", status: "skip", detail: "cameras are read by CaptureServer" },
];

/** A station is in use and its devices have not been tested yet, so recording is held. */
export const SETUP_NOT_TESTED: SetupStatus = {
  steps: [
    { id: "station", label: "Station", status: "done", detail: "Bench A" },
    { id: "devices", label: "Devices", status: "todo", detail: "run the connection test" },
    { id: "calibration", label: "Calibration", status: "done", detail: null },
    { id: "record", label: "Record", status: "todo", detail: null },
    { id: "wrap_up", label: "Wrap up", status: "todo", detail: null },
  ],
  connection_test: null,
  system: SYSTEM,
  blocked_reason: "run the device connection test before recording",
  enforced: true,
};

/** The same station after its connection test passed: nothing holds recording. */
export const SETUP_PASSED: SetupStatus = {
  ...SETUP_NOT_TESTED,
  steps: SETUP_NOT_TESTED.steps.map((step) => (step.id === "devices" ? { ...step, status: "done", detail: "connection test passed" } : step)),
  connection_test: {
    ran_at: 1_773_050_520,
    station: "Bench A",
    passed: true,
    items: [
      { id: "device:headset:0", group: "tracking", label: "Pico headset", status: "pass", detail: null },
      { id: "camera:ego", group: "cameras", label: "Ego (head): cam_head", status: "pass", detail: null },
      { id: "camera:wrist_right", group: "cameras", label: "Right wrist", status: "skip", detail: "not used by this station" },
    ],
  },
  blocked_reason: null,
};
