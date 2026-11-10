<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture

Isaac Capture provides shared capture infrastructure for teleoperation and
egocentric data. It is a library used by applications and integration
environments, including Isaac Lab and Sim, Isaac ROS, and other ROS stacks.

![Isaac Capture contains Isaac Teleop and the Ego Pipeline above shared device plugins, retargeters, and data utilities. Integration environments use the library; GitHub and PyPI distribute it.](_static/isaac-capture-architecture.png)

Isaac Capture components, shared infrastructure, and integration environments.

## Capture components

**Isaac Teleop** provides robot-in-the-loop capture. [Isaac Teleop](isaac_teleop.md) introduces
its workflows, including [Teleop Session](../teleop/teleop_session.md) for connecting input devices,
retargeting, and robot control. Remote Assist is also part of the teleoperation
component; the [Out-of-Band Teleop Control](../references/oob_teleop_control.md) reference documents its
control transport, rather than a complete operator workflow.

The **Ego Pipeline** covers human and egocentric data capture. **Ego Capture**
and **Ego Post-Processing** are its two stages. They share the underlying device
and data infrastructure with Isaac Teleop.

## Shared infrastructure

### Device plugins

The [Spatial Trackers and Device Data](../device/trackers/index.md) interfaces provide typed access to device data.
Plugins connect hardware such as gloves, body trackers, and cameras through a
common transport. XR headsets provide poses and input through OpenXR.
See [Add a New Device](../device/add_device.md) for plugin development and [Ecosystem](ecosystem.md) for
individual device capabilities.

### Retargeters

The [Retargeting Interface](../references/retargeting/index.md) framework maps device inputs to robot
embodiments through graph-based source nodes and retargeters. Hand retargeters
and body-tracking inputs can be combined with application-specific mappings.
The framework is shared; Isaac Teleop uses it to produce robot actions.

### Data utilities

Shared [FlatBuffers schemas](https://github.com/NVIDIA/IsaacCapture/tree/main/src/core/schema/fbs) define device data.
[MCAP Recording & Replay](../references/mcap_record_replay.md) covers recording and playback;
[LeRobot and SO-101](../integrations/lerobot/index.md) covers LeRobot data collection workflows.

### Visualization and streaming

[Televiz](../development/televiz.md) (`isaaccapture.viz`) composites camera and
sensor feeds and 3D content for XR headsets or desktop windows. It supports
per-eye stereo views and GPU frame submission, and can share an OpenXR session
with device tracking. See [Camera Streaming](../references/camera_streaming.md) for streaming
camera data and [CloudXR Service](../references/cloudxr.md) for the CloudXR service.

## Integration and distribution

Applications use Isaac Capture in simulation and on real robots. Integration
guides are organized under [Isaac Lab & Sim](../integrations/isaac_lab_sim.md) and
[Isaac ROS](../integrations/isaac_ros.md); the ecosystem also includes research and custom
ROS integrations. Integration names in the diagram do not imply that every
workflow or device is supported.

The source is distributed through [GitHub](https://github.com/NVIDIA/IsaacCapture),
and Python packages through [PyPI](https://pypi.org/project/isaaccapture/).
See [Build from Source](../development/build_from_source/index.md) for a source build.
