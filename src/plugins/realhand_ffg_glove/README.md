<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand FFG Glove plugin

Publishes one or two RealHand FFG Gloves as standard 26-joint OpenXR hands at
`/hand/left` and `/hand/right`, using the official `HandInjector` and `WristPoseSource`
utilities. `HandsSource` feeds compatible `HandInput` retargeters independently of
the target robot. Raw sensor collections are for calibration and diagnostics only.
The plugin queries each glove's firmware-reported hand side.
A single glove is supported; the absent side remains inactive.

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

Automatic discovery only considers `/dev/serial/by-id/` entries whose names identify
RealHand, RealForce, or FFG hardware. Generic USB UART names do not identify a glove;
explicitly select the glove ports in that case:

```bash
realhand_ffg_glove_plugin --ports=/dev/ttyUSB0,/dev/ttyUSB1 --baudrates=2000000,460800
```

Use only verified glove ports and close other applications reading the same devices.

On Linux, the user must be able to open the serial device. Common distributions grant this via
the `dialout` group:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. Do not run the IsaacCapture process as root.

## Human calibration and tracking

Without human calibration, raw sensor collections remain available for calibration
capture and diagnostics, but the human-hand outputs remain inactive.
They use `JointStateSource`, IDs `realhand_ffg_glove_left` / `realhand_ffg_glove_right`,
and names `sensor_0` through `sensor_20` (radians).

```bash
pip install -e ./examples/teleop
python -m isaaccapture_examples.teleop.realhand_ffg_glove_calibration \
  --side left --output /path/to/ffg_human.yml \
  --plugin-path install/plugins --realhand-ffg-glove-ports=/dev/ttyUSB0
realhand_ffg_glove_plugin --ports=/dev/ttyUSB0 \
  --calibration=/path/to/ffg_human.yml --wrist-source=controller
```

Capture open palm, full fist, thumb-only curl, thumb opposition, and finger spread,
then the four thumb-to-fingertip contacts, a second set of those contacts, and a
repeated open palm. Each step waits for Enter, allows 8 seconds to prepare, and
records a 6-second hold. Calibration requires NumPy and SciPy. Failed captures
leave the previous file intact.
Calibration is side-specific and robot-independent. Files use
`realhand-ffg-glove-human-calibration-v1`; legacy robot calibration is rejected.
The PC hand demos share this workflow and reuse one calibration for L6, O6, and L20.
Recalibrate after re-wearing or power-cycling a glove. Human joint poses are
estimated from glove angles; `--hand-scale=1.0` adjusts the estimated hand size.

`--wrist-source=auto` prefers native hand tracking, then controller aim pose; `controller`
and `handtracking` select a source explicitly. Default offsets support PICO 4 Ultra
controllers mounted upright on the RealHand FFG hand-back bracket, with mirrored
left/right installation. Override a different mount with
`--left-aim-to-wrist=px,py,pz,qx,qy,qz,qw` and the corresponding right option
(meters in the aim frame, XYZW quaternion). In controller mode, lost or jumping
wrist tracking makes the hand inactive until tracking recovers.

See `docs/source/device/realhand_ffg_glove.rst` for calibration poses and coordinate details.
For verification, see the official [Hand CTS](https://nvidia.github.io/IsaacCapture/main/conformance/hand.html).
