.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Architecture
============

Isaac Capture provides shared capture infrastructure for teleoperation and
egocentric data. It is a library used by applications and integration
environments, including Isaac Lab and Sim, Isaac ROS, and other ROS stacks.

.. figure:: /_static/isaac-capture-architecture.png
   :alt: Isaac Capture contains Isaac Teleop and the Ego Pipeline above shared device plugins, retargeters, and data utilities. Integration environments use the library; GitHub and PyPI distribute it.

   Isaac Capture components, shared infrastructure, and integration environments.

Capture components
------------------

**Isaac Teleop** provides robot-in-the-loop capture. :doc:`isaac_teleop` introduces
its workflows, including :doc:`/teleop/teleop_session` for connecting input devices,
retargeting, and robot control. Remote Assist is also part of the teleoperation
component; the :doc:`/references/oob_teleop_control` reference documents its
control transport, rather than a complete operator workflow.

The **Ego Pipeline** covers human and egocentric data capture. **Ego Capture**
and **Ego Post-Processing** are its two stages. They share the underlying device
and data infrastructure with Isaac Teleop.

Shared infrastructure
---------------------

Device plugins
~~~~~~~~~~~~~~

The :doc:`/device/trackers/index` interfaces provide typed access to device data.
Plugins connect hardware such as gloves, body trackers, and cameras through a
common transport. XR headsets provide poses and input through OpenXR.
See :doc:`/device/add_device` for plugin development and :doc:`ecosystem` for
individual device capabilities.

Retargeters
~~~~~~~~~~~

The :doc:`/references/retargeting/index` framework maps device inputs to robot
embodiments through graph-based source nodes and retargeters. Hand retargeters
and body-tracking inputs can be combined with application-specific mappings.
The framework is shared; Isaac Teleop uses it to produce robot actions.

Data utilities
~~~~~~~~~~~~~~

Shared :code-dir:`FlatBuffers schemas <src/core/schema/fbs>` define device data.
:doc:`/references/mcap_record_replay` covers recording and playback;
:doc:`/integrations/lerobot/index` covers LeRobot data collection workflows.

Visualization and streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`Televiz </development/televiz>` (``isaaccapture.viz``) composites camera and
sensor feeds and 3D content for XR headsets or desktop windows. It supports
per-eye stereo views and GPU frame submission, and can share an OpenXR session
with device tracking. See :doc:`/references/camera_streaming` for streaming
camera data and :doc:`/references/cloudxr` for the CloudXR service.

Integration and distribution
----------------------------

Applications use Isaac Capture in simulation and on real robots. Integration
guides are organized under :doc:`/integrations/isaac_lab_sim` and
:doc:`/integrations/isaac_ros`; the ecosystem also includes research and custom
ROS integrations. Integration names in the diagram do not imply that every
workflow or device is supported.

The source is distributed through `GitHub <https://github.com/NVIDIA/IsaacCapture>`_,
and Python packages through `PyPI <https://pypi.org/project/isaaccapture/>`_.
See :doc:`/development/build_from_source/index` for a source build.
