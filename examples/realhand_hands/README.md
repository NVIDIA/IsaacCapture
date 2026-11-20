<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand L6 / O6 / L20 hands

Mesh-free hand kinematics and per-side vector-retargeting YAMLs for the official
Isaac Capture `DexHandRetargeter`. The example consumes standard OpenXR hand
tracking or controller triggers, without requiring a particular arm or glove.

## Installation

Build and install Isaac Capture following its source-build package guide, then:

```bash
uv pip install -e ./examples/realhand_hands
python -m isaaccapture_examples.realhand_hands.hand_example --help
```

In-tree installation and tests are opt-in with
`-DBUILD_EXAMPLE_REALHAND_HANDS=ON`.

## Run

```bash
python -m isaaccapture_examples.realhand_hands.hand_example --hand-model l6 --input-mode handtracking
python -m isaaccapture_examples.realhand_hands.hand_example --hand-model o6 --input-mode handtracking
python -m isaaccapture_examples.realhand_hands.hand_example --hand-model l20 --input-mode handtracking
```

Use `--input-mode controller` for independent linear trigger-to-posture commands.
`--dry-run` builds the graph without OpenXR. The standalone CLI launches CloudXR
and prints finger targets; it does not start Isaac Sim or command real hardware.

Any compatible glove plugin can provide the same `HandsSource` stream as optical
tracking. Start and calibrate that plugin using its own device instructions;
this package neither reads glove sensors nor fits a glove calibration.

## Reuse

`hand_config.get_hand_config(model, side)` returns a public
`DexHandRetargeterConfig`. Both URDF and YAML are installed inside this wheel.
The unmodified official solver uses relative bone and fingertip vectors; there
is no contact snapping, gesture classification, or private solver access.

`hand_example.build_pipeline(mode, model)` returns `left_hand` and `right_hand`
outputs. Their scalar joint order comes from `get_realhand_profile(model).joint_names(side)`.
Official `JointRateLimiter` nodes limit target changes using graph time.
The consuming simulator or controller binds these names to its robot joints.

The packaged URDFs are kinematic models for retargeting, not rendered or
collision-enabled simulation assets. Robot physics and self-collision belong
to the consuming environment. No robot assets, drivers, or confidential CAD
files are downloaded by this example.
