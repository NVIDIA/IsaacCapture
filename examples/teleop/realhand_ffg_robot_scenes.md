<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# L6 / O6 / L20 FFG Pose Verification Scenes

Each scene contains the selected model's left and right mesh hands above a small
workbench. Wrists are fixed and palms face upward. The displayed joints follow
FFG -> calibrated standard OpenXR hand -> official `DexHandRetargeter`.
There is no raw-glove robot mapping or gesture snapping.

These are **kinematic pose tests**, not grasping or contact-physics tests.
No dynamics or self-collision response is simulated. This isolates retargeting
quality from actuator lag and contacts; the full robot environment is needed
to verify those effects.

## Setup

From the IsaacCapture checkout, use an Isaac Sim Python environment with
`numpy`, `scipy`, `trimesh`, `pyyaml`, and `Pillow`.
Set `REALHAND_CAPTURE_PYTHON` to a separate interpreter containing the locally
built IsaacCapture package and its `retargeters` extra. Its Python ABI must match
the CMake build. The launcher uses the checkout's `.venv/bin/python` by
default and builds/installs the FFG plugin automatically.

Download the verified public meshes once:

```bash
PYTHONPATH=examples/teleop/python \
python -m isaaccapture_examples.teleop.fetch_realhand_assets
```

## Three Scenes

Run one scene at a time, closing an existing FFG demo before launching another.
An existing demo may own the serial ports; never run two glove readers together.

```bash
bash examples/teleop/run_realhand_ffg_hand_demo.sh --hand-model l6
```

```bash
bash examples/teleop/run_realhand_ffg_hand_demo.sh --hand-model o6
```

```bash
bash examples/teleop/run_realhand_ffg_hand_demo.sh --hand-model l20
```

Native discovery is used by default. If your USB adapter has no recognized
FFG identifier, append your verified glove ports, for example
`--ports=/dev/ttyUSB0,/dev/ttyUSB1`. Do not select unrelated serial devices.
USB permission errors are shown in the PC window; the launcher does not use sudo.

The emulated local OpenXR runtime is started by the existing launcher. A headset
and controller are not required for these fixed-wrist tests.
The demo uses `--wrist-source=auto` so a missing wrist tracker does not suppress
the local finger skeleton. This does not change the tracked-wrist requirements of CTS.

## Calibration

The **RealHand FFG Calibration** PC window is shared with the human skeleton demo.
All three models use the same 14-step human calibration used before CTS, also
available through `realhand_ffg_glove_calibration`. There is no L20-specific sequence.
Choose Left, Right, or Both; click **Start Calibration**, then **Capture Pose** for
each step. Each capture has 8 seconds to prepare and a 6-second hold.
See the [shared calibration instructions](realhand_ffg_hand_demo.md#pc-calibration-window).
Complete all steps even when the mechanical hand cannot reproduce every human contact.

The default file is `~/.config/realhand/ffg_human_demo.yml`. An existing valid
human calibration can be reused across all three models; model-specific legacy
FFG files cannot. Append `--calibration /path/to/human.yml` to select another file.
An existing CTS human calibration is loaded unchanged, without automatic refitting.
Saving restarts only the device session and activates the new calibration.
Reopen a closed window from **Window > RealHand FFG Calibration**.

A missing glove shows **not found** and its robot hand stays open. Connected
gloves are independent; testing only one glove is supported.

## Offline Verification

Generate native skeleton fixtures and render them without opening USB or CloudXR:

```bash
build/tests/cpp/plugins/realhand_ffg_glove/test_realhand_ffg_glove_protocol \
  --write-skeleton-fixtures /tmp/ffg-skeletons.yml
bash examples/teleop/run_realhand_ffg_hand_demo.sh \
  --hand-model l20 --test-frames /tmp/ffg-skeletons.yml \
  --headless --frames 90 --screenshot /tmp/l20-ffg.png
```
