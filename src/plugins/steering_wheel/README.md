<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Steering Wheel Plugin

Reads a Linux joystick device from `/dev/input/js*` and pushes `SteeringWheelOutput` via OpenXR. Use with `SteeringWheelTracker` and the same `collection_id`.

## Usage

```bash
./steering_wheel_plugin [device_path] [collection_id] [steering_axis] [throttle_axis] [brake_axis] [clutch_axis]
```

- **device_path**: Default `/dev/input/js0`.
- **collection_id**: Default `steering_wheel`.
- **axis indexes**: Generic defaults are steering `0`, throttle `1`, brake `2`, clutch disabled with `-1`. Pass explicit axis indexes for wheel-specific layouts.

Axis values are normalized joystick values in `[-1, 1]`. The plugin does not convert pedal axes into pressed fractions; retargeters own that mapping.

The RemoteTeleop wrapper can read `config/steering_wheel_config.yaml` and pass the resulting axis indexes to this plugin. On the current G920 setup that maps steering `0`, throttle `2`, brake `3`, and clutch `1`.
