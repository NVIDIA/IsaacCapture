<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Capture integrations

Each subdirectory wires Isaac Capture to a specific third-party device, robot or
framework. Unlike [`../examples/`](../examples/README.md), which teaches the
Isaac Capture API with nothing but a headset or a mock source, an integration
is only useful with the system it targets.

| Integration | Connects Isaac Capture to |
|-------------|---------------------------|
| [`teleop_ros2`](teleop_ros2/README.md) | ROS 2 (reference publisher consumed by Isaac ROS Teleop) |
| [`vehicle_teleop`](vehicle_teleop/README.md) | Logitech steering wheel and pedals, driving a Panda vehicle worker |
| [`noitom`](noitom/README.md) | Noitom full-body mocap and an Isaac Lab G1 task |
| [`oglo_tactile`](oglo_tactile/README.md) | OGLO tactile gloves alongside a Meta Quest |
| [`sharpa_avatar`](sharpa_avatar/README.md) | Sharpa AvatarExo gloves, with pinch haptics |
| [`rebot`](rebot/SKILL.md) | reBot B601-RS arm bring-up on a Jetson AGX Orin |

Integrations follow the layout, packaging and import rules in
[`examples/README.md`](../examples/README.md). Most install the same way:

```bash
uv pip install -e ./integrations/<name>
python -m isaaccapture_examples.<name>
```

`vehicle_teleop` runs from wrapper scripts that call `.venv/bin/python` inside
its directory, and `rebot` has no importable Python; their own docs cover setup.

The Python distributions keep the `isaaccapture-examples-<name>` names and the
`isaaccapture_examples` namespace, so moving a directory here changes no
import path.
