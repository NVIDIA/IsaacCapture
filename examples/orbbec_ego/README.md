<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Orbbec EGO

Structured replay for recordings made with the
[Orbbec EGO camera plugin](../../src/plugins/orbbec_ego/README.md). Replay reads
every structured EGO record back from the MCAP file; the video stays in the
elementary streams under `raw/`.

```text
EGO camera -> orbbec_ego_camera plugin -> metadata.mcap -> replay
```

## Prerequisites

- **To record:** an Orbbec EGO camera and the `orbbec_ego_camera` plugin, built
  against the EGO SDK release `v2.10.6-EGO-20261019`.
  `./src/plugins/orbbec_ego/install.sh` downloads that SDK and builds the plugin;
  a CMake build needs `-DBUILD_PLUGIN_ORBBEC_EGO_CAMERA=ON` and
  `-DORBBEC_SDK_ROOT=<sdk>`. The install tree carries this example only when
  that option is on.
- **To replay:** only the recording and the Isaac Capture Python package; no
  camera or SDK.

## Run

In a venv set up as in [Running an example](../README.md#running-an-example):

```bash
uv pip install -e ./examples/orbbec_ego
python -m isaaccapture_examples.orbbec_ego.replay "$RUN/metadata.mcap"
```

`$RUN` is the directory given to `orbbec_ego.sh record --output`. Replay prints
one line per record, then a count per stream, and exits nonzero when a stream
has no records.

No sample recording ships with the repository, so there is no hardware-free
run: make a recording on the camera first, then replay it on any machine.

The [EGO device page](../../docs/source/device/cameras/orbbec_ego.rst) covers recording workflows
and stream selection.

## Troubleshooting

- **`Replay incomplete; no records for: Accel, Gyro`:** the recording has no
  IMU. Replay expects both eyes, IMU, calibration and, unless diagnostics were
  off, `DeviceState`.
- **TeleopSession recording, streams missing:** name the trackers as in the
  plugin README's "TeleopSession: one MCAP with other trackers". Pass
  `--record-device-diagnostics off` when the session recorded no diagnostic
  tracker.
- **`Requested and recorded diagnostic policies disagree`:** drop
  `--record-device-diagnostics`; replay reads the policy from the recording.
- **Only `metadata.mcap.partial` exists:** the capture did not finish; see the
  device page's troubleshooting.
