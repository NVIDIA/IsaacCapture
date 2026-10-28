<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Teleop Examples

Each example reads hands, controllers or a device through `TeleopSession` and
prints the result; most retarget it first. Each connects through CloudXR and
starts the runtime itself if none is serving.

Install `isaaccapture` into the active environment first, as in the
[Quick Start](https://nvidia.github.io/IsaacCapture/main/teleop/quick_start.html),
then install this example from the repository root:

```bash
uv pip install -e ./examples/teleop
python -m isaaccapture_examples.teleop.gripper_retargeting_example_simple
```

Every module runs the same way, as
`python -m isaaccapture_examples.teleop.<module>`; `--help` lists its options.

| Module | What it shows |
| --- | --- |
| `gripper_retargeting_example_simple` | Controller trigger to gripper command, with the minimal `TeleopSession` setup. |
| `gripper_retargeting_example` | A hand-built gripper and end-effector graph combined into one action. |
| `isaac_lab_gripper_example` | Thumb-index pinch to gripper command (hand tracking). |
| `se3_retargeting_example` | Hand or controller pose to an absolute or relative end-effector pose. |
| `dex_bimanual_example` | Both hands to robot hand joints, with one `DexHandRetargeter` per hand. |
| `full_bimanual_reordering_example` | Arms and hands flattened into one Isaac Lab action tensor. |
| `g1_trihand_retargeting_example` | Motion controllers to G1 three-finger hand joints. |
| `locomotion_retargeting_example` | Controller sticks to robot base commands. |
| `foot_pedal_locomotion_example` | A 3-axis foot pedal to robot base commands. |
| `joint_space_device_example` | An SO-101 leader arm to an Isaac Lab action. |
| `gamepad_printer_example` | Gamepad axes and buttons, and what the gamepad retargeters make of them. |
| `spacemouse_printer_example` | SpaceMouse axes and buttons. |
| `keyboard_printer_example` | Keys from a small GLFW window. |
| `keyboard_terminal_example` | Keys from the terminal, with no window. |

The bimanual examples default to the dex-retargeting YAML and URDF files in
[`config/dex_retargeting/`](python/isaaccapture_examples/teleop/config/dex_retargeting/),
which install with the package. Their `--enable-tuning` window needs ImGui from
the `ui` extra; `imgui` 2.0.0 does not build on Python 3.13:

```bash
uv pip install -e './examples/teleop[ui]'
```

The retargeters these examples use are documented in
[Retargeting](https://nvidia.github.io/IsaacCapture/main/references/retargeting/index.html).
