<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# recording

The recording module library: configuration I/O, episode storage/layout, video writers, QC and
studio/task-library lookups. A Python module library, not a runnable application — nothing here
is launched, it's imported, by the recording service (`recording_service/`) and the backend. See
the root [`README.md`](../README.md) for install, and [`../DESIGN.md`](../DESIGN.md) for how
this fits into the overall architecture.

This package never opens a physical camera or reads a device SDK directly — a camera is a JPEG
byte stream and tracking is a sample, both arriving over CaptureServer's own WebSocket protocol
(not implemented here). CaptureServer, device SDKs, and any UI toolkit are all out of
scope for this package.

## Configuration — `egocap/core/config_io.py`

Schema-agnostic YAML load/save (`ConfigIO.load`/`save`, plus comment-preserving
`load_with_comments`/`save_with_comments` via `ruamel.yaml` when available, falling back to
plain `yaml` otherwise). `_get_default_config()` returns the recorded-dataset schema the
downstream data-loading and training-conversion pipeline requires:

```yaml
recording:
  data_dir: "~/egocap_data/recordings"
  record_fps: 60
  stream_fps: 60
  video_frame_type: RGBCameraFrame
  numeric_frame_type: NumericFrame
video_sources: []    # one entry per recorded camera: name, type, frame_type, latency
numeric_sources: []  # one entry per recorded tracking source: name, type, frame_type, latency
```

**Where recordings go.** This section describes `config_io.py`'s own library behavior, not the
backend's — the backend (`backend/egocap_web/app.py`) resolves its recording output root from
the deployment station YAML's `recording.data_dir` when configured, else the fixed development
default `./recordings`; it does not use `EGOCAP_DATA_DIR` for the output root. Within this
library's own generated-config path, an existing config file's `recording.data_dir` is used
verbatim; `EGOCAP_DATA_DIR` (default `~/egocap_data`) only fills in the generated default config
when no config file exists or it is empty. `EGOCAP_CALIBRATION_DIR` locates the calibration JSONs
and already has a default (`<data root>/calibration`, where `<data root>` is `EGOCAP_DATA_DIR`'s
resolved value — so the backend does read `EGOCAP_DATA_DIR` for this, just not for the output
root) — `EGOCAP_CALIBRATION_DIR` only needs setting when calibration lives somewhere else.
Calibration copying is best-effort either way: a file simply absent at that location is skipped
with a warning, recording still proceeds.

`video_sources`/`numeric_sources` declare the *recorded dataset*, not how to open a device — no
device handle, index, or launcher path belongs in them. See
`egocap/config/data_recording/pico_manus_config.yaml` and
`egocap/stations/ego_stations.example.yaml` for populated templates.

## Studio / task-library lookups — `egocap/studio/`

- `studio_lookup.py` — `build_description_to_studio()`/`known_studios_from_task_library_dir()`
  parse every CSV directly under one task-library directory (default `egocap/task_library/`,
  skipping `Summary.csv`) into a `{normalized_description: studio}` map and the set of known
  studio names; `lookup_studio()` resolves a free-text task description against it.
- `task_library_data.py` — `load_task_library()`/`lookup_task_description()`: the same CSVs,
  keyed by studio, for task-name-to-description lookup.
- `studio_path_names.py` — `sanitize_studio_dir_name()`/`canonicalize_studio_dir_name()`: turn a
  studio label into a safe directory segment and resolve it against a known spelling
  case-insensitively. `_unmatched` is the reserved fallback bucket for a description that
  matches nothing.
- `mid_train_data.py` — the CSV-backed "Mid Train" task/primitive/grasp lookup, a separate
  instruction mode from the diverse-language task library above; routes to a single `mid_train`
  studio folder.

No task-library CSV or grasp-thumbnail assets ship in this tree — these modules return empty
results against a missing directory rather than raising, which is not the same as the features
actually having data to work with. Provisioning real/fixture CSVs is a deployment's own step.

## Recording — `egocap/recording/`

`episode_store.py`'s `EpisodeStore` owns per-episode Zarr/MP4 storage: appending numeric rows
and video frames, and the dataset-level `studio`/Mid-Train `inverse`/`props` zarr attrs. Images
are recorded as received, with no rectification applied.

`episode_layout.py` owns the on-disk layout: pending/final directory naming, studio/timestamp
routing (`assign_output`/`finalize_episode_dir` route each finalized episode to
`<eod_root>/<Studio>/<timestamp>/`, falling back to `_unmatched/` when the task description
matches no known studio), and the durable `write_json_atomic`/`write_yaml_atomic` primitives the
rest of the package's atomic writes build on.

`video_writer_backends.py` and `constants.py` are the local RGB video writer backends
(OpenCV/FFmpeg/NVENC) and shared recording-layout constants.

`episode_error_log.py`'s `log_episode_errors()` appends one JSONL record per QC/guardrail
failure to `<session_root>/errors.jsonl` — a tail-able, greppable record independent of the Zarr
store.

## Analytics — `egocap/analytics/`

- `station_status_log.py`'s `StationStatusTracker` is a per-station, per-day active/inactive
  time tracker: `toggle`/`set_state` persist to `<log_dir>/<date>.json`,
  `compute_totals`/`save_daily_log` produce the operator-facing daily activity JSON
  (`<station>_<date>_activity.json`).
- `count_daily_trajectories.py`'s `count_trajectories()`/`compute_station_data_breakdown()` walk
  `<data_dir>/<Studio>/<timestamp>/episode_*`, grouping by session and attributing each episode
  to the day it was actually recorded (not the session folder's own date, which can predate a
  midnight-crossing episode). A standalone library function for offline reporting — the backend
  and recording service do not call it.

## EOD session tooling — `egocap/eod/`, `egocap/data_processing/`

Standalone library functions for offline session/episode metadata maintenance — the backend and
recording service do not call them:

- `eod_studio_session_move.py`'s `relocate_session_to_canonical_studio()` moves a session folder
  from `_unmatched/` (or a mis-filed studio) to its canonical `<eod>/<Studio>/<timestamp>/`
  location when the operator corrects the Studio metadata after the fact;
  `refresh_session_egocap_zarr_aggregates()` recomputes `num_episodes`/`episode_lengths`/
  `total_data_time_hours` from the session's finalized episodes.
- `update_episode_task_description.py`'s `update_episode_task_description()` updates
  `task_description`/`task_name`/`studio` on a range of episodes (dataset- and episode-level
  Zarr attrs, plus `config.yaml`'s `meta`), optionally relocating the session first.

## Data layout and format

What the recording service writes into each routed group:

```
<recording_root>/
├── <Studio>/
│   └── <YYYY-MM-DD-HH-MM-SS>/
│       ├── zarr.json              # dataset-level attrs
│       ├── config.yaml             # {"meta": {"egocap_station_name": ...}} -- station identity
│       │                           # only, NOT the fuller video_sources/numeric_sources/recording
│       │                           # schema `core/config_io.py`'s default config illustrates
│       ├── head_camera_intrinsics.json, head_camera_extrinsics.json
│       ├── left_wrist_camera_intrinsics.json, right_wrist_camera_intrinsics.json
│       ├── wrist_calibration_offsets.json
│       ├── episode_0/, episode_1/, ...
│       └── ...
└── _unmatched/                    # descriptions matching no known studio
    └── ...
```

Calibration files appear only when present at `EGOCAP_CALIBRATION_DIR` (which always has a
resolved value, see above -- absence there is a skip, not a configuration requirement); the
service does not write a separate `data_check_config.yaml`.

A data-loading pipeline's source-config reader requires `video_sources`/`numeric_sources`,
source names, frame types/defaults, and latency to run its alignment step; the per-group
`config.yaml` above does not carry those fields, so this reader needs that configuration
supplied separately, not read from the recorded output. A Zarr-to-LeRobot converter reads fixed,
hardcoded group paths instead: `pico_controller/{left,right,head}_pose_interp` and
`manus_glove/{left,right}_hand_pose_interp`. Renaming a wire stream is harmless as long as the
backend still maps it to the established recorded source name; renaming the recorded group
itself breaks conversion.

Recorded numeric sources, exactly as written by `backend/egocap_web/recording_adapter.py`:

| Source group | Field | Shape | dtype | Meaning |
| --- | --- | --- | --- | --- |
| `pico_controller` | `head_pose`, `left_pose`, `right_pose` | `(4, 4)` | `float64` | Homogeneous transforms. |
| `pico_controller` | `left_trigger`, `right_trigger` | scalar | `float64` | Analog trigger values. |
| `pico_controller` | `a_button`, `b_button`, `x_button`, `y_button` | scalar | `bool` | Controller buttons. |
| `manus_glove` | `left_hand_pose`, `right_hand_pose` | `(25, 4, 4)` | `float64` | Root-relative transform stacks, one per hand joint. |

Both groups carry `capture_time`/`receive_time` as seconds (converted from CaptureServer's
nanosecond timestamps). Camera groups each have a matching `.mp4`, plus capture time, receive
time, and intrinsics when provided. A `manus_glove` row is omitted entirely for a tick where
either hand's wrist is untracked — never backfilled with a stale or fabricated zero pose. The
OpenXR-to-this-skeleton hand-joint mapping is implemented but not yet validated against real
hardware (see `capture_server/README.md`'s "implemented but not exercised on hardware" note).

Calibration assets are copied into every session under their established filenames when
available: head camera intrinsics and extrinsics, left/right wrist camera intrinsics (wrist
cameras have no separate extrinsics file), and the wrist-to-hand `wrist_calibration_offsets.json`;
a session recorded without calibration present still produces valid video/tracking data, just
without those sidecar files.
