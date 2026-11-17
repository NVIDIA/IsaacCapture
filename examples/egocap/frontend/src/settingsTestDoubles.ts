// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** A complete `/api/settings` body for tests: a development default with one camera. */
import type { EffectiveSettings, OperatorSettings, TaskFileContents, TaskFileListing, ToolInfo } from "./settingsApi";
import type { StationChange, StationListing, StationRecord } from "./stationsApi";

export const OPERATOR_SETTINGS_FIXTURE: OperatorSettings = {
  camera_type: null,
  camera_types: ["sensing_gmsl", "oak_d"],
  camera_roles: { ego: null, wrist_left: null, wrist_right: null },
  roles: ["ego", "wrist_left", "wrist_right"],
  task_file: null,
  preselected_task: null,
  require_calibration: false,
};

/** A registry with no stations: the configuration-free default, where the deployment's own label applies. */
export const STATIONS_FIXTURE: StationListing = {
  stations: [],
  selected: null,
  roles: ["ego", "wrist_left", "wrist_right"],
  camera_types: ["sensing_gmsl", "oak_d"],
  label: "Station 01",
  locked_reason: null,
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
  recording: { output_root: "/var/egocap/recordings" },
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
  recordings_browser: { enabled: false },
  task_files: { dir: "/repo/examples/egocap/data/taskfiles", available: true },
  origins: { allowed: [], trusted_hostnames: null },
};

/** An in-memory station registry that answers like the backend's, so a change shows up in the next listing. */
export function fakeStationRegistry(initial: StationListing = STATIONS_FIXTURE) {
  let listing = initial;
  let nextId = initial.stations.length + 1;
  /** Replace one station and return the listing, as every backend change does. */
  const replace = (id: string, change: (station: StationRecord) => StationRecord): StationListing => {
    listing = { ...listing, stations: listing.stations.map((station) => (station.id === id ? change(station) : station)) };
    return listing;
  };
  /** The label recordings carry: the station in use, else the deployment's own. */
  const relabel = (): StationListing => {
    const inUse = listing.stations.find((station) => station.id === listing.selected);
    listing = { ...listing, label: inUse?.name ?? STATIONS_FIXTURE.label };
    return listing;
  };
  return {
    get listing() {
      return listing;
    },
    set listing(next: StationListing) {
      listing = next;
    },
    /** The station in use, or null. */
    inUse: (): StationRecord | null => listing.stations.find((station) => station.id === listing.selected) ?? null,
    addStation: async (_token: string, fields: StationChange): Promise<StationListing> => {
      const station: StationRecord = {
        id: `st_${nextId++}`,
        name: fields.name ?? "",
        camera_type: fields.camera_type ?? null,
        camera_roles: { ego: null, wrist_left: null, wrist_right: null, ...fields.camera_roles },
        problems: [],
      };
      listing = { ...listing, stations: [...listing.stations, station] };
      return listing;
    },
    updateStation: async (_token: string, id: string, changes: StationChange): Promise<StationListing> => {
      replace(id, (station) => ({ ...station, ...changes, camera_roles: { ...station.camera_roles, ...changes.camera_roles } }));
      return relabel();
    },
    deleteStation: async (_token: string, id: string): Promise<StationListing> => {
      listing = { ...listing, stations: listing.stations.filter((station) => station.id !== id), selected: listing.selected === id ? null : listing.selected };
      return relabel();
    },
    selectStation: async (_token: string, id: string | null): Promise<StationListing> => {
      listing = { ...listing, selected: id };
      return relabel();
    },
  };
}
