<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand FFG Glove examples

Robot-independent calibration, diagnostics, and an Isaac Sim viewer for the
FFG Glove's standard 26-joint OpenXR human skeleton. This package contains no
robot models or robot-specific retargeting.

## Installation

Follow the Isaac Capture source-build package installation guide, enabling the
optional glove plugin:

```bash
cmake -S . -B build -DBUILD_PLUGIN_REALHAND_FFG_GLOVE=ON
cmake --build build
cmake --install build
uv pip install -e ./examples/realhand_ffg_glove
```

Use that build's interpreter and locally built Isaac Capture package. The
optional viewer runs in Isaac Sim's Python environment, which supplies
`isaacsim`, `omni`, and `pxr`; its DeviceIO worker uses the Isaac Capture
interpreter. Neither calibration nor diagnostics requires Isaac Sim.

## Calibration

Close other glove readers and select verified USB ports:

```bash
python -m isaaccapture_examples.realhand_ffg_glove.realhand_ffg_glove_calibration \
  --side both --plugin-path install/plugins \
  --realhand-ffg-glove-ports=/dev/ttyUSB0,/dev/ttyUSB1 \
  --output /path/to/ffg_human.yml
```

The CLI and PC window share the same endpoint and repeated-contact sequence.
The resulting human calibration is independent of the consuming robot. See
the [FFG device page](../../docs/source/device/hand_tracking/realhand_ffg_glove.rst)
for pose instructions, USB permissions, status diagnostics, and the official
CTS recording command.

## Commands

Run modules as `python -m isaaccapture_examples.realhand_ffg_glove.<module>`.
Each command accepts `--help`:

| Module | Purpose |
| --- | --- |
| `realhand_ffg_glove_calibration` | Capture human-hand calibration. |
| `realhand_ffg_diagnostics` | Record raw sensor and tracking diagnostics. |
| `realhand_ffg_hand_demo` | View human skeletons and calibrate in a PC window. |
| `realhand_ffg_hand_demo_bridge` | DeviceIO worker for the viewer. |

The [viewer guide](realhand_ffg_hand_demo.md) describes the two-interpreter
launch. Calibration remains a PC-window action; no XR UI or controller-button
calibration binding is installed.
