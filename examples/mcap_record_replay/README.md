<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MCAP record / replay

The viewer examples record DeviceIO tracking to an MCAP file and replay it
into a viser 3D view. Live viewers are included for watching without recording.

```bash
uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.record_hand      # 5 s
python -m isaaccapture_examples.mcap_record_replay.replay_hand      # newest take
```

Recordings are written to `./recordings/` relative to where you run the
command; a replay viewer given no path picks the newest matching recording there
(e.g. `replay_hand` picks the newest `hands_*.mcap`) and exits if none
match -- no silent fallback to a wrong-type recording.

`uv run` (instead of an activated venv) needs an explicit `--python 3.11` --
without it, `uv` may resolve against the system Python and fail to match the
lockfile.

The live and replay viewers bind every interface, so a browser on another
machine can reach them at `http://<this-host>:8080`. Pass `--host 127.0.0.1` to
keep a viewer local. Replay viewers repeat until Ctrl+C; no loop flag is needed.

| Channel | Live | Record | Replay |
| --- | --- | --- | --- |
| Hands | `live_hand` | `record_hand` | `replay_hand` |
| Controllers | `live_controller` | `record_controller` | `replay_controller` |
| Full body | `live_full_body` | `record_full_body` | `replay_full_body` |
| Raw hand joint SE3 poses | `live_joint_se3_pose` | `record_joint_se3_pose` | `replay_joint_se3_pose` |
| VIVE SE3 trackers | — | `record_se3_vive` | `replay_se3_vive` |
| EGO structured data | — | [EGO recording CLI](../../src/plugins/ego/README.md) | [replay_ego MCAP](../../src/plugins/ego/README.md#replay-structured-ego-data) (headless; stops at EOF) |

`record_*` takes an optional duration in seconds and an optional output path.
These recorders need a live OpenXR runtime; replay needs only the file.

The EGO recording wrapper captures stereo video and available IMU streams by
default. Audio is opt-in through `--audio` or `audio = true` in a TOML config;
`--no-audio` overrides a config that enables it. Raw PCM capture, storage and
export are supported, but the tested Ethernet path has an SDK audio timestamp
cadence anomaly. Audio/video and five-stream physical synchronization have not
passed acceptance. See [optional audio and synchronization limits](../../src/plugins/ego/README.md#optional-audio-and-synchronization-limits).

The current `replay_ego` example requires an Audio channel alongside camera,
IMU, calibration and any enabled diagnostic channels, so use it with an
audio-enabled recording. A recording without audio is valid, but this example reports its
missing Audio channel. Each tracker reads its next record according to the
existing NV update behavior; one update need not represent one physical instant.

For selected-stream sample association, install the built `isaaccapture` package
with its `ego` extra and use the recommended four-stream offline audit:

```bash
uv pip install 'isaaccapture[ego]'
python -m isaaccapture_examples.mcap_record_replay.analyze_ego_sync \
  recordings/ego_run --required-streams left_rgb right_rgb accel gyro \
  --output recordings/ego_run/sync_audit.json
```

Include `audio` only when the recording contains accessible audio media and an
independently validated audio time model. Selecting it does not establish
physical synchronization. A valid recorded time model is used by default; a
later recording-bound model requires explicit `--alignment PATH`. Use
`--reference-stream right_rgb`, `--epoch N`, `--start-time-ns N` and
`--end-time-ns N` to select frame windows in aligned nanoseconds. The JSON
contains an association index, coverage, gaps and declared validation evidence.
It does not decode or play media, and an unverified/incomplete result exits
nonzero while preserving its report.

Fit a recording-bound model from independently annotated physical events:

```bash
python -m isaaccapture_examples.mcap_record_replay.calibrate_ego_sync \
  recordings/ego_run/events.json --output recordings/ego_run/alignment.json
```

Both commands require a fresh output path. Calibration uses separate training
and held-out events over at least 30 minutes, includes measurement uncertainty,
and does not open a camera or detect events. See the
[EGO time-alignment workflow](../../src/plugins/ego/README.md#sample-time-alignment-and-offline-reading)
for the manifest/model contracts and the Python `iter_sync_windows` API.

A C++ recorder lives in `cpp/`. Docs:
`docs/source/references/mcap_record_replay.rst`.
