// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** A complete `/api/settings` body for tests: a development default with one camera and LiveKit credentials set. */
import type { EffectiveSettings, OperatorSettings, TaskFileContents, TaskFileListing, ToolInfo } from "./settingsApi";

export const OPERATOR_SETTINGS_FIXTURE: OperatorSettings = {
  camera_type: null,
  camera_types: ["sensing_gmsl", "oak_d"],
  camera_roles: { ego: null, wrist_left: null, wrist_right: null },
  roles: ["ego", "wrist_left", "wrist_right"],
  task_file: null,
  preselected_task: null,
};

export const TASK_FILES_FIXTURE: TaskFileListing = { files: ["Kitchen.csv"], imported: [] };

export const KITCHEN_FIXTURE: TaskFileContents = {
  name: "Kitchen.csv",
  format: "studio_csv",
  tasks: [
    { name: "wash_cup", instruction: "Wash the cup in the sink.", studio: "A", scenario: "Sink" },
    { name: "pour_water", instruction: "Pour water into the cup.", studio: "A", scenario: "Counter" },
  ],
};

export const TOOLS_FIXTURE: ToolInfo[] = [
  {
    id: "headset_calibration",
    name: "Headset calibration",
    description: "Intrinsics and extrinsics of the headset cameras.",
    applies_to: null,
    available: false,
    unavailable_reason: "not wired in this build",
  },
  {
    id: "glove_calibration",
    name: "Glove calibration",
    description: "Hand-tracking glove calibration against the headset controllers.",
    applies_to: null,
    available: false,
    unavailable_reason: "not wired in this build",
  },
  {
    id: "fisheye_rectification",
    name: "Fisheye rectification",
    description: "Rectification maps for the three SENSING GMSL cameras. Only fisheye lenses need this; rectilinear cameras do not.",
    applies_to: "sensing_gmsl",
    available: false,
    unavailable_reason: "not wired in this build",
  },
];

export const SETTINGS_FIXTURE: EffectiveSettings = {
  capture: {
    server_url: "ws://127.0.0.1:8082",
    station_label: "Station 01",
    profile_label: "Headset and hands",
    sources: ["head", "controllers", "hands"],
    plugins: [],
    cameras: [{ name: "head_human_camera", type: "human" }],
  },
  preview: { camera_sample_hz: 10, tracking_sample_hz: 10 },
  station: {
    path: null,
    error: null,
    id: null,
    operator_name: null,
    auto_fill_operator_name: false,
    task_description: null,
    data_dir: null,
  },
  recording: { output_root: "/home/op/egocap_data/recordings" },
  visualization: {
    error: null,
    enabled: true,
    host: "127.0.0.1",
    port: 8085,
    public_url: null,
    device: "pico-4-ultra",
    head_model: null,
    head_model_scale: 1,
  },
  livekit: { credentials_configured: true },
  recordings_browser: { enabled: false },
  task_files: { dir: "/repo/examples/egocap/data/taskfiles", available: true },
  origins: { allowed: [], trusted_hostnames: null },
};
