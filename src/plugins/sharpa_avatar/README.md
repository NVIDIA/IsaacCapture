<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Avatar glove

This plugin connects Sharpa Avatar gloves to Isaac Teleop. It publishes
hand poses and glove joint data through the standard Isaac Teleop interfaces,
and forwards haptic commands to the glove motors.

The accompanying sample shows both hands in a viser browser view. Bringing a
fingertip close to the thumb vibrates that finger:

```text
gloves -> avatar_hand_plugin -> TeleopSession -> viser and haptics
```

## Prerequisites

- Linux x86_64 (Ubuntu 22.04 or 24.04)
- A built Isaac Teleop checkout
- An Avatar SDK package from Sharpa, extracted to a local directory
- Sharpa Avatar gloves connected through the USB dongle or wired Ethernet

The SDK is distributed separately because it is not publicly downloadable. It
does not need to be installed as a system package: the plugin installer stages
the required headers, libraries, and runtime data from the extracted SDK tree.
Those files remain outside the git repository.

The plugin talks to the gloves directly, so do not run `avatar-backend`, Avatar
Desktop, or `avatar_hand_tracker_printer` at the same time.

## Install

On a development machine where the SDK package is already available at
`/opt/avatar-sdk`, run this from the Isaac Teleop root:

```bash
./src/plugins/sharpa_avatar/install.sh
```

To use an SDK package extracted elsewhere, point the installer at its root:

```bash
AVATAR_SDK_ROOT=/path/to/avatar-sdk ./src/plugins/sharpa_avatar/install.sh
```

The SDK package must provide `include/avatar_sdk/AvatarSDK.h`, `lib/`,
`share/sdk_config.json`, `share/hand_fk`, and `share/wave-sdk`. The source-tree
layout `src/hand_fk/data` is also accepted for the hand FK data. The installer
copies these files into the gitignored plugin vendor directory, then builds the
plugin and places its executable, configuration, and runtime dependencies under
`install/`.

## Run the sample

Install the small Python-only sample dependencies once:

```bash
uv pip install viser numpy
```

Then launch the sample from the repository root:

```bash
.venv/bin/python src/plugins/sharpa_avatar/tools/sharpa_avatar_sample.py
```

The command starts CloudXR and the glove plugin, serves a viser view at
http://127.0.0.1:8080, and enables pinch feedback. The left hand is cyan and
the right hand is orange. The terminal also reports the 22-DoF RAW and ROBOT
joint streams. Pass `--no-viz` for terminal and haptic only.

Useful options:

```text
--transport=wired             use wired Ethernet instead of the USB dongle
--no-haptic                   disable pinch feedback
--no-viz                      terminal + haptic only (no browser view)
--host / --port               viser bind (default 127.0.0.1:8080)
--no-launch-plugin            connect to an already running plugin
--no-launch-cloudxr-runtime   connect to an already running CloudXR runtime
--world-frame                 display the unmodified OpenXR poses
```

Run the installed copy with:

```bash
.venv/bin/python install/plugins/sharpa_avatar/tools/sharpa_avatar_sample.py
```

## Published data

| Device or collection | Contents |
|---|---|
| `/hand/left`, `/hand/right` | OpenXR 26-joint hand poses from Avatar HUMAN data |
| `avatar_raw_left`, `avatar_raw_right` | RAW 22-DoF joint state |
| `avatar_robot_left`, `avatar_robot_right` | ROBOT 22-DoF joint state |
| `avatar_glove_haptic` | Per-finger vibration commands |

The gloves do not provide a world-space wrist pose. Without another tracked
wrist source, the sample places both skeletons in a stable local display frame.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| Avatar SDK headers are not found | Extract the Sharpa-provided SDK package and set `AVATAR_SDK_ROOT` to its root |
| CMake 3.24 or newer is required | Install a newer CMake in the Isaac Teleop environment and rerun `install.sh` |
| `No module named isaacteleop` | Activate the Isaac Teleop environment and install its wheel |
| `No module named viser` | Run `uv pip install viser` in the same environment |
| Plugin binary is not found | Run `install.sh`, or pass `--plugin-search-path install/plugins` |
| Hand and joint streams stay offline | Power on the gloves and stop any other process using the Avatar SDK |
| No vibration | Keep `haptic` in `--datasets` and close a fingertip toward the thumb |
| CloudXR cannot start | Source `~/.cloudxr/run/cloudxr.env`, or allow the sample to launch CloudXR |
