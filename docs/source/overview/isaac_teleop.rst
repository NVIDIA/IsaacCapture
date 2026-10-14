.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Isaac Teleop
============

**Isaac Teleop** is the teleoperation component of :doc:`Isaac Capture </index>`.
It combines a standardized device interface, a graph-based retargeting pipeline,
and a data collection workflow for simulated and real-world robots.

.. figure:: /_static/isaac-teleop-hero.jpg
   :width: 100%
   :alt: Isaac Teleop robot teleoperation

Key features:

- **Unified teleoperation workflows** for simulation and real robots, including
  ROS 2 and Isaac Lab integrations.
- **Standardized device interfaces** for XR headsets, gloves, pedals, and body
  trackers, backed by Isaac Capture's shared device plugins.
- **Flexible retargeting** to different robot embodiments.
- **Session management** for device tracking, retargeting, and robot control.
- **Visualization with Televiz** for streaming robot camera and sensor feeds,
  plus 3D reconstruction content, to XR headsets.
- **Recording and replay** of tracking data for repeatable sessions and dataset
  collection.

Start with :doc:`/teleop/quick_start`, then use :doc:`/teleop/teleop_session`
and :doc:`/teleop/control_state_machine` to integrate your application.
The :doc:`/teleop/lerobot/index` guides cover SO-101 data collection and training.

Isaac Teleop uses Isaac Capture's :doc:`/device/trackers`,
:doc:`/references/retargeting/index`, and :doc:`/references/mcap_record_replay`.
See :doc:`architecture` for how these shared capabilities relate to the Ego
Pipeline.
