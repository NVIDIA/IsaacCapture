<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Orbbec EGO

Python helpers for recordings made with the
[Orbbec EGO camera plugin](../../src/plugins/ego/README.md).

- `recording`: `configure_recording` adds the EGO calibration tracker to a
  `TeleopSessionConfig`; `embedded_recording` produces one MCAP that holds the session's
  records and the plugin's embedded media.
- `replay`: replays every structured EGO tracker from a recording.
- `sync`, `analyze_sync`: associate video, IMU and audio samples offline and audit a
  recording's coverage.
- `calibration`, `calibrate_sync`: fit a time model from annotated events.

Install the example into an environment that has the built `isaaccapture` package:

```bash
uv pip install -e ./examples/orbbec_ego
```

Then run, for example:

```bash
python -m isaaccapture_examples.orbbec_ego.replay "$RUN/metadata.mcap"
python -m isaaccapture_examples.orbbec_ego.analyze_sync "$RUN" \
    --required-streams left_rgb right_rgb accel gyro --output "$RUN/sync_audit.json"
```

The [EGO device page](../../docs/source/device/cameras/ego.rst) covers recording layouts,
time models and stream selection.
