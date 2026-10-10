<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SpaceMouse Plugin

Reads a 3Dconnexion SpaceMouse from `/dev/hidraw*` and pushes `SpaceMouseOutput` (translation,
rotation, buttons, connected) via OpenXR. Read it with `SpaceMouseTracker`, `SpaceMouseSource`,
or any reader using the same `collection_id`. Linux only.

The plugin is also bundled in the `isaaccapture` package: a `TeleopSession` starts it with
`PluginConfig(plugin_name="spacemouse", plugin_root_id="spacemouse", search_paths=[BUNDLED_PLUGINS_DIR])`.

Validated devices: SpaceMouse Compact, SpaceMouse Wireless, SpaceNavigator, SpaceNavigator for
Notebooks, 3Dconnexion Universal Receiver.

## Device access

`/dev/hidraw*` is root-only by default. Install the udev rule once on the host (not in a
container), then unplug and replug the SpaceMouse:

```bash
./install_udev_rules.sh
```

The rule (`70-spacemouse.rules`) makes SpaceMouse HID nodes read-only for all users; no group
membership is needed.

## Usage

```bash
./spacemouse_plugin [--device=/dev/hidrawN [--combined-report]] [--collection-id=ID]
```

- **--device**: HID node to read. By default the first validated model is discovered under
  `/sys/class/hidraw`, and rediscovered after it is unplugged.
- **--combined-report**: With `--device`, the device sends translation and rotation in one report
  (3Dconnexion Universal Receiver). Discovery sets this automatically.
- **--collection-id**: Default `spacemouse`. Match this when creating `SpaceMouseTracker`.
- **--plugin-root-id**: Set by the plugin launcher; enables device status reporting.

The plugin pushes at 90 Hz. While no device is open it keeps pushing with `connected` false and
zero axes, so a reader never holds the last deflection of an unplugged device. A deflected device
reports every 16 ms even when held still, so an open one that goes silent for 250 ms (a lost
wireless link) has its axes zeroed too.

## Data

- `translation`, `rotation`: `[x, y, z]` in `[-1, 1]`, in the device's axis order.
- `buttons`: one entry per button, 1 while held.

## Device status

With `--plugin-root-id`, the plugin reports device `/spacemouse` as `CONNECTED`,
`DISCONNECTED` (no SpaceMouse found), or `FAILED` (found but cannot be opened, usually because the
udev rule is missing).
