<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand FFG Glove plugin

Streams the 21 joint sensors from one or two RealHand FFG Gloves as the standard Isaac Teleop
`JointStateOutput` schema. The plugin scans supported USB serial devices, queries each glove for
its firmware-reported hand side, and publishes independent `realhand_ffg_glove_left` and
`realhand_ffg_glove_right` tensor collections. A single connected glove is supported; the absent side
remains disconnected and does not produce samples.

The serial protocol implementation follows the Apache-2.0
[RealHand FFG Glove pure-Python SDK](https://github.com/RealHand-Robotics/FFG_realhand_pure_python).
The external Python SDK is not redistributed or required at runtime.

## Build and run

The plugin is enabled by default on POSIX systems:

```bash
cmake -S . -B build
cmake --build build --target realhand_ffg_glove_plugin
./build/src/plugins/realhand_ffg_glove/realhand_ffg_glove_plugin
```

Use `-DBUILD_PLUGIN_REALHAND_FFG_GLOVE=OFF` to exclude it. The POSIX serial backend is not built by
default on Windows.

By default it scans `/dev/ttyUSB*`, `/dev/ttyACM*`, `/dev/ttyXRUSB*`, and `/dev/ttyOBC*` at
2,000,000, 460,800, 1,000,000, and 921,600 baud. Optional overrides are available:

```bash
realhand_ffg_glove_plugin --ports=/dev/ttyUSB0,/dev/ttyUSB1 --baudrates=2000000,460800
```

On Linux, the user must be able to open the serial device. Common distributions grant this via
the `dialout` group:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. Do not run the Isaac Teleop process as root.

The consumer side uses `JointStateSource` with collection ID `realhand_ffg_glove_left` or
`realhand_ffg_glove_right` and joint names `sensor_0` through `sensor_20`. Mechanical-hand mapping and
calibration belong to the RealHand retargeters rather than this device plugin.
