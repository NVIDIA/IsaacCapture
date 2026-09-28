.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Welcome to Isaac Teleop
=======================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: references/license.html
   :alt: License
   :class: license-badge

**Isaac Teleop** (`GitHub repo <https://github.com/NVIDIA/IsaacTeleop>`_) is the unified framework for high-fidelity
egocentric and robot data collection. It provides a standardized device interface, a flexible
graph-based retargeting pipeline, and a streamlined data collection workflow
that work seamlessly across simulated and real-world robots.

.. figure:: _static/isaac-teleop-hero.jpg
   :width: 100%
   :alt: Isaac Teleop

Key features:

- **Unified stack** for sim and real teleoperation (ROS2, Isaac Sim, Isaac Lab)
- **Standardized device interface** for XR headsets, gloves, foot pedals, body trackers
- **Flexible retargeting** to different robot embodiments
- **Plugin system** for adding new devices and use cases
- **Visualization module (Televiz)** for streaming robot camera and sensor feeds, plus 3D reconstruction content to XR headsets
- **Markerless hand reconstruction** from egocentric video into 4D hand and camera poses (monocular supported, stereo on the roadmap)

.. tip::

   **Just want to get running?** Follow the :doc:`getting_started/quick_start` how-to guide for
   installation and first-run steps.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Overview
   :titlesonly:

   Isaac Teleop <self>
   overview/architecture
   overview/ecosystem

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/quick_start
   getting_started/build_from_source/index
   getting_started/teleop_session
   getting_started/teleop_control_state_machine
   getting_started/televiz
   getting_started/lerobot/index
   getting_started/contributing

.. toctree::
   :maxdepth: 2
   :caption: Devices

   device/trackers
   device/add_device
   device/joint_space
   device/body_tracking
   device/haptic_feedback
   device/manus
   device/oak
   device/oglo
   device/wuji_glove
   device/sharpa_avatar
   device/haptikos

.. toctree::
   :maxdepth: 2
   :caption: References

   references/requirements
   references/build
   references/generated_trackers
   references/retargeting/index
   references/camera_streaming
   references/mcap_record_replay
   references/cloudxr
   references/rig
   references/oob_teleop_control
   references/egocentric_hand_reconstruction
   references/license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
