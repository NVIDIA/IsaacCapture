<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand FFG Glove plugin

Reads one or two USB serial gloves and publishes standard 26-joint OpenXR hands
at `/hand/left` and `/hand/right` using `HandInjector` and `WristPoseSource`.
Firmware identifies each glove's side; an absent glove never controls the other
hand. Human reconstruction is independent of any robot model.

The protocol follows the Apache-2.0
[RealHand FFG Glove pure-Python SDK](https://github.com/RealHand-Robotics/FFG_realhand_pure_python).
That SDK is not redistributed or required by the native plugin.

## Build

The plugin is optional and disabled by default. The serial backend requires POSIX:

```bash
cmake -S . -B build -DBUILD_PLUGIN_REALHAND_FFG_GLOVE=ON
cmake --build build --target realhand_ffg_glove_plugin
cmake --install build --component realhand_ffg_glove
```

Install the built Isaac Capture package using the
[source-build guide](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html#install-the-isaaccapture-pip-package),
then install the self-contained example with `uv pip install -e ./examples/realhand_ffg_glove`.

## Calibrate, Then Run

On Ubuntu, USB access commonly requires membership of `dialout`. After
`sudo usermod -aG dialout "$USER"`, log out and back in. Do not run Isaac Capture
as root. Close any existing readers of the selected ports.

```bash
python -m isaaccapture_examples.realhand_ffg_glove.realhand_ffg_glove_calibration \
  --side both --output /path/to/ffg_human.yml --plugin-path install/plugins \
  --realhand-ffg-glove-ports=/dev/ttyUSB0,/dev/ttyUSB1
```

The calibration command can acquire raw sensors before human calibration exists.
Without calibration the plugin publishes raw sensor collections, but no active
OpenXR hand. The same human calibration feeds any compatible hand retargeter.
Recalibrate after re-wearing, power-cycling, or shifting a glove.
Pose instructions and troubleshooting are on the
[device page](../../../docs/source/device/hand_tracking/realhand_ffg_glove.rst).

After calibration, start the runtime and the plugin:

```bash
python -m isaaccapture.cloudxr.service start
source ~/.cloudxr/run/cloudxr.env
./install/plugins/realhand_ffg_glove/realhand_ffg_glove_plugin \
  --calibration=/path/to/ffg_human.yml \
  --ports=/dev/ttyUSB0,/dev/ttyUSB1 --wrist-source=controller
```

Restricted automatic discovery probes USB serial identifiers containing
RealHand, RealForce, or FFG. Generic UART identifiers require explicit verified
ports. Connected devices are queried independently; lost or jumping controller
tracking makes the human hand inactive until stable tracking returns.

Raw collections `realhand_ffg_glove_left` and `realhand_ffg_glove_right` contain
`sensor_0` through `sensor_20` in radians for calibration and diagnostics only.
Robot hands consume the standard `HandsSource` output, not these raw sensors.

## Wrist Source and CTS

`--wrist-source=auto` prefers optical hand tracking, then controller aim pose.
`controller` and `hand_tracking` select one source explicitly. The default
controller-to-wrist transforms assume mirrored PICO 4 Ultra controllers upright
on the RealHand hand-back bracket. Other mounts can use
`--left-aim-to-wrist=px,py,pz,qx,qy,qz,qw` and `--right-aim-to-wrist=...`.

The CTS recorder starts plugins without command-line flags. Set these defaults
in its shell; explicit plugin flags take precedence:

```bash
export REALHAND_FFG_GLOVE_CALIBRATION=/absolute/path/to/ffg_human.yml
export REALHAND_FFG_GLOVE_PORTS=/dev/ttyUSB0,/dev/ttyUSB1
export REALHAND_FFG_GLOVE_WRIST_SOURCE=controller
bash src/python/isaaccapture/cts/hand/capture/record.sh realhand_ffg_glove \
  --plugin realhand_ffg_glove_plugin
```

Prepare the official CTS environments and record from the pushed source commit
with its installed plugin and wheel. Keep the generated provenance unchanged.
See the [Hand CTS workflow](https://nvidia.github.io/IsaacCapture/main/conformance/hand.html).
