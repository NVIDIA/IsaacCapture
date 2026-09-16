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

- Linux x86_64 (Ubuntu 22.04)
- A built Isaac Teleop checkout
- Sharpa Avatar gloves connected through the USB dongle or wired Ethernet

The installer retrieves the production `avatar-sdk` package, currently pinned
to `1.7.3-17`, from Sharpa's signed production APT repository. It never selects
the `avatar-sdk-dev` or `avatar-sdk-beta` channels. The SDK remains external under
`/opt/avatar-sdk`; its headers, libraries, and data are not copied into this
repository or the plugin installation.

The plugin talks to the gloves directly, so do not run `avatar-backend`, Avatar
Desktop, or `avatar_hand_tracker_printer` at the same time.

## Install

Run this from the Isaac Teleop root:

```bash
./src/plugins/sharpa_avatar/install.sh
```

If `/opt/avatar-sdk` is absent, the installer configures the same production
APT channel used by the Sharpa host application and installs the pinned SDK.
To install only that dependency, run:

```bash
./src/plugins/sharpa_avatar/install_avatar_sdk.sh
```

If a development or beta SDK is already installed, the dependency installer
stops instead of replacing it implicitly. Remove that package explicitly before
installing production. The plugin build accepts an existing SDK under `/opt` but
prints its version and warns when its `BUILD_TYPE` is not `Production`.

Install the device rules once on the host, then unplug and reconnect the glove
or dongle:

```bash
./src/plugins/sharpa_avatar/install_udev_rules.sh
```

udev does not run inside containers, so this command must run on the host.
The plugin follows the Sharpa host application layout and uses the SDK,
configuration, and runtime assets directly from `/opt/avatar-sdk`. Transport
selection, including wired Ethernet, is controlled by
`/opt/avatar-sdk/share/sdk_config.json`.

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
| Avatar SDK installation fails | Check access to Sharpa's production APT endpoint and rerun `install_avatar_sdk.sh` |
| USB glove is not detected | Run `install_udev_rules.sh` on the host, then unplug and reconnect the glove or dongle |
| CMake 3.24 or newer is required | Install a newer CMake in the Isaac Teleop environment and rerun `install.sh` |
| `No module named isaacteleop` | Activate the Isaac Teleop environment and install its wheel |
| `No module named viser` | Run `uv pip install viser` in the same environment |
| Plugin binary is not found | Run `install.sh`, or pass `--plugin-search-path install/plugins` |
| Hand and joint streams stay offline | Power on the gloves and stop any other process using the Avatar SDK |
| No vibration | Keep `haptic` in `--datasets` and close a fingertip toward the thumb |
| CloudXR cannot start | Source `~/.cloudxr/run/cloudxr.env`, or allow the sample to launch CloudXR |
