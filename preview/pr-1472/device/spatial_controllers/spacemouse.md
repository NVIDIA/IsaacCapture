<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SpaceMouse Plugin

C++ plugin that streams a **3Dconnexion SpaceMouse** (a 6-DoF cap plus buttons) into the
standard DeviceIO path. It reads the device’s raw HID node (`/dev/hidraw*`), so it needs no
3Dconnexion driver, and pushes six axes and the button state at 90 Hz. Source and plugin README:
[src/plugins/spacemouse/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/src/plugins/spacemouse/README.md).

Validated devices: SpaceMouse Compact, SpaceMouse Wireless, SpaceNavigator, SpaceNavigator for
Notebooks, and the 3Dconnexion Universal Receiver.

This plugin follows the [Add a New Device](../add_device.md) pattern (FlatBuffer schema → plugin `SchemaPusher` →
tracker → MCAP).

> ##### On this page
> 
> * [Components](#components)
> * [Device access](#device-access)
> * [Build](#build)
> * [Usage](#usage)
> * [Published data](#published-data)
> * [Retargeters](#retargeters)
> * [Device status](#device-status)
> * [Troubleshooting](#troubleshooting)
> * [Tests](#tests)

## Components

- **Schema** — [src/core/schema/fbs/spacemouse.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/spacemouse.fbs)
  (`SpaceMouseOutput` / `SpaceMouseOutputRecord`).
- **Plugin** — [src/plugins/spacemouse](https://github.com/NVIDIA/IsaacCapture/tree/main/src/plugins/spacemouse): `core/` parses HID reports, discovers the
  device and reconnects after an unplug; `app/` is the OpenXR push loop.
- **Tracker** — `SpaceMouseTracker`, generated at configure time from the `spacemouse` entry
  in [src/core/deviceio_trackers/trackers.toml](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml).
- **Source node** — `SpaceMouseSource` turns the tracker’s data into
  `spacemouse_translation`, `spacemouse_rotation` and `spacemouse_buttons` for retargeters.

## Device access

`/dev/hidraw*` nodes are readable by root only by default. Install the plugin’s udev rule once
**on the host** (udev does not run in containers), then unplug and replug the SpaceMouse:

```bash
./src/plugins/spacemouse/install_udev_rules.sh
```

From a wheel install, run the copy bundled with the plugin:

```bash
"$(python -c 'from isaaccapture.teleop_session_manager import BUNDLED_PLUGINS_DIR; print(BUNDLED_PLUGINS_DIR)')"/spacemouse/install_udev_rules.sh
```

The rule ([src/plugins/spacemouse/70-spacemouse.rules](https://github.com/NVIDIA/IsaacCapture/blob/main/src/plugins/spacemouse/70-spacemouse.rules)) makes SpaceMouse HID nodes
read-only for all users. No group membership is needed.

## Build

Linux only. Complete the [source-build prerequisites](../../development/build_from_source/index.md)
first. The plugin builds with the default plugin set:

```bash
cmake -B build
cmake --build build --target spacemouse_plugin --parallel
```

## Usage

Start the plugin with the session through `TeleopSessionConfig.plugins`, from the copy bundled
with `isaaccapture` (wheel installs included), and read it with `SpaceMouseSource`, as
[examples/teleop/python/isaaccapture_examples/teleop/spacemouse_printer_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop/python/isaaccapture_examples/teleop/spacemouse_printer_example.py) does:

```python
from isaaccapture.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
from isaaccapture.teleop_session_manager import (
    BUNDLED_PLUGINS_DIR,
    PluginConfig,
    TeleopSessionConfig,
)

config = TeleopSessionConfig(
    app_name="MyApp",
    pipeline=pipeline_reading(SpaceMouseSource(name="spacemouse")),
    plugins=[
        PluginConfig(
            plugin_name="spacemouse",
            plugin_root_id="spacemouse",
            search_paths=[BUNDLED_PLUGINS_DIR],
            plugin_args=[],  # e.g. ["--device=/dev/hidraw3"]
        )
    ],
)
```

Without `--device` the plugin opens the first validated model under `/sys/class/hidraw` and
finds it again after it is unplugged. For a Universal Receiver at an explicit path, add
`--combined-report`.

The plugin can also run on its own, for example on another machine pushing to the same runtime,
next to any reader of the same collection:

```bash
./install/plugins/spacemouse/spacemouse_plugin
./install/plugins/spacemouse/spacemouse_plugin --device=/dev/hidraw3
```

## Published data

The plugin pushes `SpaceMouseOutput` to collection `spacemouse` (`--collection-id` changes
it; match it in `SpaceMouseSource` / `SpaceMouseTracker`):

- `translation`, `rotation` — `[x, y, z]` in `[-1, 1]`, in the device’s axis order.
- `buttons` — one entry per button, 1 while held.
- `connected` — whether the plugin has the device open.

While no device is open the plugin keeps pushing with `connected` false and zero axes, so a
reader never holds the last deflection of an unplugged device; `SpaceMouseSource` then reports
its outputs as absent. An open device that stops reporting for 250 ms while deflected has its
axes zeroed. Recordings use MCAP channels `spacemouse` and `spacemouse_tracked`.

## Retargeters

`SpaceMouseToSe3RelRetargeter` (end-effector delta), `SpaceMouseToSe2Retargeter` (base
velocity) and `SpaceMouseGripperRetargeter` (toggle on the left button) consume
`SpaceMouseSource`.

## Device status

When `TeleopSession` starts the plugin, it reports device `<collection_id>/spacemouse`
(`spacemouse/spacemouse` by default) through
[Device provider monitoring](../../references/device_provider_monitoring.md): `CONNECTED`, `DISCONNECTED` when no SpaceMouse
is found, or `FAILED` when one is found but cannot be opened.

## Troubleshooting

- **Status FAILED, or “Cannot open /dev/hidrawN (Permission denied)”** — install the udev rule
  and replug the device.
- **Status DISCONNECTED with the device plugged in** — check `cat
  /sys/class/hidraw/*/device/uevent | grep HID_NAME`; an unlisted model can be read with
  `--device`.
- **Axes move but buttons never change** — some receivers report buttons on another interface;
  try `--device` on the device’s other `hidraw` node.

## Tests

`spacemouse_plugin_tests` covers HID report parsing, discovery against a fake sysfs tree, and
reconnecting through a FIFO standing in for the device node.
