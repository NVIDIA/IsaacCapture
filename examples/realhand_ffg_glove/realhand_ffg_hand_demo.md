<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand FFG Hand Validation

By default, this optional Isaac Sim example displays the plugin's two **human 26-joint OpenXR
skeletons**, not a robot retargeter. Left is blue, right is coral. Wrists remain
fixed in the viewer; fingers follow the FFG stream directly. No P7, robot IK,
mechanical-hand collision, or headset connection is involved. Isaac Sim and
the local OpenXR runtime are required; they are not added as package dependencies.

The skeleton is estimated from glove angles and approximate human dimensions.
This demo checks device reconstruction before introducing robot-hand retargeting.
It does not test physical contacts or mechanical-hand accuracy.

## Local checkout

Activate your Isaac Sim Python environment first, then substitute your checkout path:

```bash
cd /path/to/IsaacCapture
bash examples/realhand_ffg_glove/run_realhand_ffg_hand_demo.sh --ports=/dev/ttyUSB0
```

For two gloves use `--ports=/dev/ttyUSB0,/dev/ttyUSB1`. Firmware identifies left
and right; port numbering does not determine the hand side. Only verified FFG
ports should be passed. USB access still requires Linux serial permissions.
Omitting `--ports` enables the plugin's restricted discovery of USB serial
identifiers containing RealHand, RealForce, or FFG. Generic USB UART identifiers
require explicit verified ports; omission never defaults to `/dev/ttyUSB0`.

The wrapper builds the current FFG plugin, stages it in a user cache, and uses
the locally built Python package. Its separate DeviceIO process supports a
different Python ABI from Isaac Sim. Set `REALHAND_CAPTURE_PYTHON` to another
matching Isaac Capture interpreter when needed; the default is the checkout's
`.venv/bin/python`, set up using the main build guide. Default build configuration is
`Release`; override `REALHAND_CAPTURE_BUILD_CONFIG` for a different configuration.

The background CloudXR/OpenXR runtime starts automatically or attaches to an
existing compatible service. Do not launch another FFG reader on the same USB
port. The bundled `.env` uses the official [standalone CloudXR profile](https://isaac-sim.github.io/IsaacLab/develop/source/features/isaac_teleop.html#cloudxr-environment-profiles)
(`quest3`) with push-device injection enabled. No PICO client is needed.
If an existing runtime uses another profile, stop it before starting this demo;
the launcher cannot reconfigure a service that is already running.
If using a separately configured compatible
runtime, source its environment and pass `--no-launch-cloudxr-runtime`.
The CloudXR service outlives the viewer. Stop it when finished using the
Isaac Capture interpreter and the built package:

```bash
PYTHONPATH=build/python_package/Release \
  .venv/bin/python -m isaaccapture.cloudxr.service stop
```

The Isaac Capture interpreter also needs the package's `cloudxr` extra.
The offline calibration fitter requires NumPy and SciPy in that interpreter.

## PC Calibration Window

Use the **RealHand FFG Calibration** window in Isaac Sim.
If closed, reopen it from **Window > RealHand FFG Calibration**.

1. Select Left, Right, or Both according to the connected gloves.
2. Click **Start Calibration**.
3. Click **Capture Pose**, release the mouse during the 8-second **PREPARE** countdown,
   then keep the displayed pose still for the 6-second **HOLD**.
4. Complete the 14-step sequence. The complete file is validated, saved, and applied automatically.

The PC window and `realhand_ffg_glove_calibration` command use the same human
calibration workflow used before CTS: open palm, fist, thumb-only curl, thumb
opposition, finger spread, four thumb-to-fingertip contacts, a second set of those
contacts, and repeated open palm. See the [device page](../../docs/source/device/hand_tracking/realhand_ffg_glove.rst)
for pose instructions. Both interfaces preserve repeated samples for fitting and
require at least 120 samples per hold. Calibration is not a CTS test result.

For Both, hold the same pose on both hands simultaneously.
The calibration describes the human hand, not a target robot.
Fingers remain independently driven; contact poses train the calibration, not runtime snapping.

**Cancel Capture** cancels the whole calibration without replacing the
saved file. Saving happens only after all captures and fitting succeed.

The viewer uses solid cylindrical bones between joint spheres. Palms face up;
only the viewer orientation is changed, not the input or calibration coordinates.

The default file is `~/.config/realhand/ffg_human_demo.yml`. Use `--calibration`
to select another **human-calibration-v1** file. Stable-release robot-specific
FFG calibration files are not compatible. Each side has its own measurements;
calibrating one side preserves the other. Invalid or cancelled captures do not
replace a working file. Existing calibrations, including files used for CTS, are
loaded without refitting or rewriting. Recalibrate after re-wearing or power-cycling the gloves.
A missing or stale hand stream is hidden, not frozen.

## Rendering Test Without Hardware

Generate fixtures from the production C++ skeleton and display them without
starting OpenXR or accessing a USB device:

```bash
cmake --build build --target realhand_ffg_skeleton_fixtures
build/tests/cpp/plugins/realhand_ffg_glove/fixtures/realhand_ffg_skeleton_fixtures \
  --write-skeleton-fixtures /tmp/ffg_skeleton.yml
PYTHONPATH=examples/realhand_ffg_glove/python python -m isaaccapture_examples.realhand_ffg_glove.realhand_ffg_hand_demo \
  --test-frames /tmp/ffg_skeleton.yml
```

Live calibration is disabled in fixture mode.
