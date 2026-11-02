<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Orbbec EGO

Python helpers for recordings made with the
[Orbbec EGO camera plugin](../../src/plugins/orbbec_ego/README.md).

- `recording`: `configure_recording` adds the EGO calibration tracker to a
  `TeleopSessionConfig`; `embedded_recording` produces one MCAP that holds the session's
  records and the plugin's embedded media.
- `replay`: replays every structured EGO tracker from a recording.

Install the example into an environment that has the built `isaaccapture` package:

```bash
uv pip install -e ./examples/orbbec_ego
```

Then replay a recording:

```bash
python -m isaaccapture_examples.orbbec_ego.replay "$RUN/metadata.mcap"
```

The [EGO device page](../../docs/source/device/cameras/orbbec_ego.rst) covers recording layouts
and stream selection.
