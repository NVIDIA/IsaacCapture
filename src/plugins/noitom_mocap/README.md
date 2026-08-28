<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Noitom Mocap Plugin

Optional plugin that reads Noitom Hybrid Data Server data through MocapApi and
publishes IsaacTeleop full-body samples over OpenXR tensor data.

The plugin converts Noitom avatar joints to the existing `FullBodyPose`
layout and publishes tensor identifier `full_body` in collection
`noitom_mocap`. Consumers use `FullBodyTracker()` with the `body.noitom`
vendor selected; vendor parameters carry the collection ID and maximum sample
size from the Noitom-specific source.

Positions are read from the Noitom SDK in centimeters and published in meters.
MCAP timestamps use the local monotonic clock for local/common time fields and
the Noitom avatar posture timestamp for the raw device clock when the SDK
provides it.

## Build

The SDK is fetched from `https://github.com/pnmocap/MocapApi`; it is not checked
into this repository.

```bash
cmake -B build -DBUILD_PLUGIN_NOITOM_MOCAP=ON
cmake --build build --target noitom_mocap_plugin --parallel
cmake --install build
```

For offline builds:

```bash
cmake -B build \
  -DBUILD_PLUGIN_NOITOM_MOCAP=ON \
  -DNOITOM_MOCAP_API_ROOT=/path/to/MocapApi
```

## Run

Start CloudXR/OpenXR first, keep Axis Studio or the Noitom Data Server running,
then let IsaacTeleop's plugin manager launch the plugin. Before installing,
edit [`plugin.yaml`](plugin.yaml) and set `--host` and `--port` to the Hybrid
Data Server TCP endpoint. Run `cmake --install build` again after changing the
file so the installed plugin configuration is updated.

```bash
./install/plugins/noitom_mocap/noitom_mocap_plugin
```

Print frames received through DeviceIO:

```bash
python src/plugins/noitom_mocap/tools/noitom_mocap_printer.py --duration=10
```

Record through the Noitom example wrapper and replay with the shared MCAP
script:

```bash
uv pip install -e ./examples/noitom
python -m isaaccapture_examples.noitom.record_noitom_full_body \
  10 examples/noitom/recordings/noitom_full_body.mcap

uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.replay_full_body \
  examples/noitom/recordings/noitom_full_body.mcap
```
