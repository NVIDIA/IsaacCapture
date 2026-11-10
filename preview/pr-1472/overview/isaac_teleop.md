<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Isaac Teleop

**Isaac Teleop** is the teleoperation component of [Isaac Capture](../index.md).
It combines a standardized device interface, a graph-based retargeting pipeline,
and a data collection workflow for simulated and real-world robots.

![Isaac Teleop robot teleoperation](_static/isaac-teleop-hero.jpg)

Key features:

- **Unified teleoperation workflows** for simulation and real robots, including
  ROS 2 and Isaac Lab integrations.
- **Standardized device interfaces** for XR headsets, gloves, pedals, and body
  trackers, backed by Isaac Capture’s shared device plugins.
- **Flexible retargeting** to different robot embodiments.
- **Session management** for device tracking, retargeting, and robot control.
- **Visualization with Televiz** for streaming robot camera and sensor feeds,
  plus 3D reconstruction content, to XR headsets.
- **Recording and replay** of tracking data for repeatable sessions and dataset
  collection.

Start with [Quick Start](../teleop/quick_start.md), then use [Teleop Session](../teleop/teleop_session.md)
and [Teleop Control State Machine](../teleop/control_state_machine.md) to integrate your application.
The [LeRobot and SO-101](../integrations/lerobot/index.md) guides cover SO-101 data collection and training.

Isaac Teleop uses Isaac Capture’s [Spatial Trackers and Device Data](../device/trackers/index.md),
[Retargeting Interface](../references/retargeting/index.md), and [MCAP Recording & Replay](../references/mcap_record_replay.md).
See [Architecture](architecture.md) for how these shared capabilities relate to the Ego
Pipeline.
