<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Avatar Glove

C++ plugin that connects Sharpa Avatar gloves to Isaac Teleop. It publishes
OpenXR hand poses and glove joint state, and consumes inbound haptic commands.

The plugin talks to the gloves directly, so do not run `avatar-backend` or
Avatar Desktop at the same time.

```text
gloves -> avatar_hand_plugin -> OpenXR / DeviceIO
```

The visualization example is separate:
[`examples/sharpa_avatar`](../../../examples/sharpa_avatar/README.md).

## Prerequisites

- Linux x86_64 (Ubuntu 22.04)
- A built Isaac Teleop checkout
- Sharpa Avatar gloves connected through the USB dongle or wired Ethernet

The installer retrieves a pinned production `avatar-sdk` version from Sharpa's
signed production APT repository, without selecting the `avatar-sdk-dev` or
`avatar-sdk-beta` channels. To move to another release, bump
`production_version` in `install_avatar_sdk.sh`. The SDK remains external; its
headers, libraries, and data are not copied into this repository or the plugin
installation.

## Install

Run this from the Isaac Teleop root:

```bash
./src/plugins/sharpa_avatar/install.sh
```

If the SDK is absent from the default root (`/opt/avatar-sdk`), the installer
configures the same production APT channel used by the Sharpa host application
and installs the pinned SDK. With a custom `AVATAR_SDK_ROOT`, the tree must
already be complete. To install only that dependency, run:

```bash
./src/plugins/sharpa_avatar/install_avatar_sdk.sh
```

If a development or beta SDK is already installed, the dependency installer
stops instead of replacing it implicitly. Remove that package explicitly before
installing production. `install.sh` and CMake verify the SDK by calling
`install_avatar_sdk.sh --check`: the default root must hold the pinned
production package, while a custom root is checked for a complete tree and only
warns on a non-production build.

CMake is Linux-only for this plugin. The SDK root comes from
`-DAVATAR_SDK_ROOT`, then `$AVATAR_SDK_ROOT`, then `/opt/avatar-sdk`.
`-DBUILD_PLUGIN_SHARPA_AVATAR=ON` without a usable SDK under the selected root
skips the plugin and the rest of Isaac Teleop still configures. `install.sh`
installs the SDK first, then configures with that flag.

Install the device rules once on the host, then unplug and reconnect the glove
or dongle:

```bash
./src/plugins/sharpa_avatar/install_udev_rules.sh
```

udev does not run inside containers, so this command must run on the host.
The plugin follows the Sharpa host application layout and uses the SDK,
configuration, and runtime assets directly from the selected SDK root
(`AVATAR_SDK_ROOT`, default `/opt/avatar-sdk`). Transport selection, including
wired Ethernet, is controlled by `<sdk-root>/share/sdk_config.json`.

## Run

`TeleopSession` launches the installed binary through `plugin.yaml`. To start
it yourself after CloudXR is up:

```bash
./install/plugins/sharpa_avatar/avatar_hand_plugin
```

## Published data

| Device or collection | Contents |
|---|---|
| `/hand/left`, `/hand/right` | OpenXR 26-joint hand poses from Avatar HUMAN data |
| `avatar_raw_left`, `avatar_raw_right` | RAW 22-DoF joint state |
| `avatar_robot_left`, `avatar_robot_right` | ROBOT 22-DoF joint state |
| `avatar_glove_haptic` | Per-finger vibration commands |

The gloves do not provide a world-space wrist pose. Downstream consumers that
need a display frame (the example) place both skeletons locally; a headset
wrist source is used when one is available.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| Avatar SDK installation fails | Check access to Sharpa's production APT endpoint and rerun `install_avatar_sdk.sh` |
| CMake skipped the Sharpa Avatar plugin | The SDK is missing; run `install_avatar_sdk.sh` and reconfigure, or use `install.sh` |
| CMake rejected the Avatar SDK | Install the pinned production package with `install_avatar_sdk.sh`, or point `AVATAR_SDK_ROOT` at a complete SDK tree |
| USB glove is not detected | Run `install_udev_rules.sh` on the host, then unplug and reconnect the glove or dongle |
| CMake 3.24 or newer is required | Install a newer CMake in the Isaac Teleop environment and rerun `install.sh` |
| Plugin binary is not found | Run `install.sh` |
| Hand and joint streams stay offline | Power on the gloves and stop any other process using the Avatar SDK |
| No vibration | Keep `haptic` in `--datasets` and close a fingertip toward the thumb |
