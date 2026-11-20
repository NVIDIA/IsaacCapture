<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RealHand P7 dual-arm example

Controller or standard OpenXR hand wrists produce P7 left/right end-effector
targets through the official `Se3AbsRetargeter` and `EePoseRateLimiter`.
No glove implementation or dexterous-hand model is required.

## Installation

Build and install Isaac Capture using its source-build package guide, then:

```bash
uv pip install -e ./examples/realhand_p7
python -m isaaccapture_examples.realhand_p7.p7_bimanual_example --help
```

For in-tree installation and tests, configure with
`-DBUILD_EXAMPLE_REALHAND_P7=ON`. This optional example is disabled by default.

## Run

```bash
python -m isaaccapture_examples.realhand_p7.p7_bimanual_example --input-mode controller
python -m isaaccapture_examples.realhand_p7.p7_bimanual_example --input-mode handtracking
```

The standalone command launches CloudXR and prints the target poses; it does not
start a robot simulator or command real hardware. `--dry-run` validates the graph
without OpenXR. A standard hand source can come from optical tracking or a
separately installed compatible glove plugin, without changing the graph.

## Integration

`build_pipeline(mode)` returns the graph and its action order. Supply
`world_T_anchor` at each step; in Isaac Lab, `IsaacTeleopDevice` supplies this
stage-to-world transform. The CLI defaults to P7 +X right, +Y forward, +Z up;
`--world-t-anchor` accepts an explicit row-major 4x4 matrix. Tool rotation is a
separate wrist-to-end-effector alignment, not a substitute for an anchor.

Output is 14 values: left XYZ + XYZW quaternion, then right XYZ + XYZW quaternion.
The consuming environment owns inverse kinematics, joint limits, collision
checking, and robot physics. It can combine these targets with independently
retargeted fingers. This example does not implement an arm driver or IK solver.
